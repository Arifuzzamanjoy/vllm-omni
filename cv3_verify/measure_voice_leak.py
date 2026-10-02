#!/usr/bin/env python3
"""Measure whether CosyVoice3 answers a request in another request's voice.

Subcommands
  run        register K voices, send N seeded speech requests round-robin over
             them at concurrency C, save every WAV plus per-request metadata.
  score      score run directories: speaker ID with two independent scorers
             (CAM++ campplus.onnx shipped with the model, and WavLM-SV),
             Whisper WER, duration ratio. Writes scores.csv + summary.json.
  refmatrix  reference-vs-reference cosine matrix for both scorers.
  report     aggregate scored runs into report.md / cells.csv / confusion.
  gate       positive-control check: exit 3 (and write FINDINGS.md) if no
             scorer finds a wrong voice in any C>1 run of the given commit.

An output is "wrong" for a scorer when argmax cosine over all K references is
not the requested voice. A leak is only counted when both scorers name the
same wrong voice ("agree").
"""

import argparse
import asyncio
import csv
import hashlib
import io
import json
import math
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parent

# English targets of different lengths, so requests finish at different steps
# and queued requests get admitted into running batches (#8288).
TEXTS = [
    "Thank you so much for the birthday gift you sent me last week.",
    "The weather is lovely today, with warm sunshine and a gentle breeze, so we decided to walk in the park.",
    "Please remember to bring your notebook and a pen to the meeting tomorrow morning.",
    "After the long winter, the farmers were glad to see the first green shoots appear in the fields, "
    "and the children ran outside to play until the sun went down behind the hills.",
]

SCORERS = ("campplus", "wavlm")


# ---------------------------------------------------------------- helpers ---

def load_voices(path: str | Path) -> list[dict]:
    path = Path(path)
    voices = json.loads(path.read_text())["voices"]
    for v in voices:
        v["wav_path"] = str((path.parent / v["wav"]).resolve())
        if v.get("heldout_wav"):
            v["heldout_path"] = str((path.parent / v["heldout_wav"]).resolve())
    return voices


def plan_requests(n: int, voices: list[dict], seed: int) -> list[dict]:
    """Round-robin voices; text index decorrelated from voice index."""
    k = len(voices)
    return [
        {
            "idx": i,
            "voice": voices[i % k]["name"],
            "text_idx": (i // k) % len(TEXTS),
            "text": TEXTS[(i // k) % len(TEXTS)],
            "seed": seed * 1000 + i,
        }
        for i in range(n)
    ]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def fisher_greater(k1: int, n1: int, k0: int, n0: int) -> float:
    """One-sided Fisher exact p that group 1 rate > group 0 rate."""
    if n1 == 0 or n0 == 0:
        return float("nan")
    K, N = k1 + k0, n1 + n0
    denom = math.comb(N, n1)
    return sum(math.comb(K, x) * math.comb(N - K, n1 - x) for x in range(k1, min(K, n1) + 1)) / denom


def normalize_words(text: str) -> list[str]:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return [w.strip("'") for w in text.split() if w.strip("'")]


def wer(ref: str, hyp: str) -> float:
    r, h = normalize_words(ref), normalize_words(hyp)
    if not r:
        return float("nan")
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


def read_audio_16k(path_or_bytes) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes | bytearray) else str(path_or_bytes)
    wav, sr = sf.read(src, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != 16000:
        g = math.gcd(sr, 16000)
        wav = resample_poly(wav, 16000 // g, sr // g).astype(np.float32)
    return wav


def git_head(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------- run ---

async def _register(client, base: str, voices: list[dict]) -> dict:
    r = await client.get(f"{base}/v1/audio/voices")
    r.raise_for_status()
    body = r.json()
    existing = set()
    for key in ("voices", "uploaded_voices", "data"):
        for item in body.get(key, []) or []:
            existing.add((item.get("name") if isinstance(item, dict) else str(item)).lower())
    result = {}
    for v in voices:
        if v["name"].lower() in existing:
            result[v["name"]] = "already-registered"
            continue
        with open(v["wav_path"], "rb") as f:
            files = {"audio_sample": (Path(v["wav_path"]).name, f.read(), "audio/wav")}
        data = {"name": v["name"], "consent": "cv3-verify-public-dataset", "ref_text": v["ref_text"]}
        r = await client.post(f"{base}/v1/audio/voices", files=files, data=data)
        if r.status_code != 200 or not r.json().get("success", False):
            raise RuntimeError(f"voice upload {v['name']} failed: {r.status_code} {r.text[:500]}")
        result[v["name"]] = "registered"
    return result


async def _one(client, sem, base, model, req, wav_dir, timeout, t_origin):
    payload = {"model": model, "input": req["text"], "voice": req["voice"],
               "response_format": "wav", "seed": req["seed"]}
    async with sem:
        t0 = time.perf_counter()
        rec = dict(req, t_start=round(t0 - t_origin, 4))
        try:
            r = await client.post(f"{base}/v1/audio/speech", json=payload, timeout=timeout)
            t1 = time.perf_counter()
            rec.update(status=r.status_code, latency_s=round(t1 - t0, 4), t_end=round(t1 - t_origin, 4))
            ctype = r.headers.get("content-type", "")
            if r.status_code == 200 and "json" not in ctype:
                fname = f"{req['idx']:04d}_{req['voice']}.wav"
                (wav_dir / fname).write_bytes(r.content)
                rec.update(wav=fname, bytes=len(r.content), sha256=hashlib.sha256(r.content).hexdigest())
            else:
                rec["error"] = f"HTTP {r.status_code}: {r.text[:500]}"
        except Exception as e:  # noqa: BLE001 - every failure is data here
            rec.update(status=-1, error=f"{type(e).__name__}: {e}", t_end=round(time.perf_counter() - t_origin, 4))
    return rec


async def _run_async(args, voices, reqs, out: Path) -> dict:
    import httpx

    wav_dir = out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    limits = httpx.Limits(max_connections=args.concurrency + 4)
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        reg = await _register(client, args.base, voices)
        sem = asyncio.Semaphore(args.concurrency)
        t_origin = time.perf_counter()
        recs = await asyncio.gather(*[_one(client, sem, args.base, args.model, r, wav_dir, args.timeout, t_origin)
                                      for r in reqs])
        wall = time.perf_counter() - t_origin
    with open(out / "requests.jsonl", "w") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"registration": reg, "wall_s": round(wall, 3),
            "n_ok": sum(1 for r in recs if "wav" in r), "n_err": sum(1 for r in recs if "wav" not in r)}


def cmd_run(args) -> int:
    voices = load_voices(args.voices)
    reqs = plan_requests(args.n, voices, args.seed)
    meta = {
        "label": args.label, "commit": args.commit, "mode": args.mode, "overlay": args.overlay,
        "K": len(voices), "C": args.concurrency, "seed": args.seed, "N": args.n,
        "voices_file": str(Path(args.voices).resolve()), "voices": [v["name"] for v in voices],
        "base": args.base, "model": args.model, "kit_commit": git_head(KIT),
        "texts": TEXTS, "request_seed_rule": "seed*1000 + idx",
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if args.dry_run:
        print(json.dumps(meta, ensure_ascii=False, indent=1))
        for r in reqs[: min(len(reqs), 2 * len(voices))]:
            print(f"  req {r['idx']:3d} voice={r['voice']:18s} seed={r['seed']} text#{r['text_idx']}")
        print(f"  ... {len(reqs)} requests, concurrency {args.concurrency}")
        return 0
    out = Path(args.out)
    if (out / "requests.jsonl").exists():
        sys.exit(f"{out} already has results; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    res = asyncio.run(_run_async(args, voices, reqs, out))
    meta.update(res, finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"{out}: ok={res['n_ok']} err={res['n_err']} wall={res['wall_s']}s")
    return 0 if res["n_ok"] > 0 else 2


# ----------------------------------------------------------------- scorers ---

class Scorers:
    def __init__(self, model_dir: str, device: str, asr_model: str, wavlm_model: str):
        import onnxruntime as ort
        import torch

        self.torch = torch
        self.device = device
        self.campplus = ort.InferenceSession(str(Path(model_dir) / "campplus.onnx"),
                                             providers=["CPUExecutionProvider"])
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.wavlm_fe = AutoFeatureExtractor.from_pretrained(wavlm_model)
        self.wavlm = WavLMForXVector.from_pretrained(wavlm_model).to(device).eval()
        self.asr_model = asr_model
        self._asr = None
        self.versions = {"wavlm_model": wavlm_model, "asr_model": asr_model,
                         "campplus": str(Path(model_dir) / "campplus.onnx")}

    def embed_campplus(self, wav16: np.ndarray) -> np.ndarray:
        import torchaudio.compliance.kaldi as kaldi

        feat = kaldi.fbank(self.torch.from_numpy(wav16).unsqueeze(0), num_mel_bins=80, dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        e = self.campplus.run(None, {self.campplus.get_inputs()[0].name: feat.unsqueeze(0).numpy()})[0].flatten()
        return e / np.linalg.norm(e)

    def embed_wavlm(self, wav16: np.ndarray) -> np.ndarray:
        inp = self.wavlm_fe(wav16, sampling_rate=16000, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            e = self.wavlm(**inp).embeddings[0].float().cpu().numpy()
        return e / np.linalg.norm(e)

    def embed(self, wav16: np.ndarray) -> dict[str, np.ndarray]:
        return {"campplus": self.embed_campplus(wav16), "wavlm": self.embed_wavlm(wav16)}

    def transcribe(self, wavs16: list[np.ndarray]) -> list[str]:
        if self._asr is None:
            from transformers import pipeline

            self._asr = pipeline("automatic-speech-recognition", model=self.asr_model, device=self.device)
        out = self._asr([{"raw": w, "sampling_rate": 16000} for w in wavs16], batch_size=8,
                        generate_kwargs={"language": "english", "task": "transcribe"})
        return [o["text"].strip() for o in out]


def reference_embeddings(sc: Scorers, voices: list[dict]) -> dict:
    refs = {s: {} for s in SCORERS}
    held = {s: {} for s in SCORERS}
    for v in voices:
        e = sc.embed(read_audio_16k(v["wav_path"]))
        for s in SCORERS:
            refs[s][v["name"]] = e[s]
        if v.get("heldout_path"):
            e = sc.embed(read_audio_16k(v["heldout_path"]))
            for s in SCORERS:
                held[s][v["name"]] = e[s]
    return {"ref": refs, "heldout": held}


def ref_matrix(emb: dict, names: list[str]) -> dict:
    out = {}
    for s in SCORERS:
        R = np.stack([emb["ref"][s][n] for n in names])
        m = {"names": names, "ref_vs_ref": np.round(R @ R.T, 4).tolist()}
        if all(n in emb["heldout"][s] for n in names):
            H = np.stack([emb["heldout"][s][n] for n in names])
            m["heldout_vs_ref"] = np.round(H @ R.T, 4).tolist()
            hr = H @ R.T
            m["heldout_top1_correct"] = int(sum(int(np.argmax(hr[i]) == i) for i in range(len(names))))
        off = (R @ R.T)[~np.eye(len(names), dtype=bool)]
        m["max_offdiag_ref"] = round(float(off.max()), 4) if off.size else None
        out[s] = m
    return out


def score_run(sc: Scorers, run_dir: Path, ref_cache: dict) -> dict:
    meta = json.loads((run_dir / "meta.json").read_text())
    voices = load_voices(meta["voices_file"])
    names = [v["name"] for v in voices]
    key = meta["voices_file"]
    if key not in ref_cache:
        ref_cache[key] = reference_embeddings(sc, voices)
    emb = ref_cache[key]
    recs = [json.loads(l) for l in (run_dir / "requests.jsonl").read_text().splitlines() if l.strip()]
    ok = [r for r in recs if r.get("wav")]
    wavs = [read_audio_16k(run_dir / "wavs" / r["wav"]) for r in ok]
    hyps = sc.transcribe(wavs) if wavs else []
    rows = []
    for r, w, hyp in zip(ok, wavs, hyps):
        e = sc.embed(w)
        row = {k: r[k] for k in ("idx", "voice", "text_idx", "seed", "latency_s", "t_start", "t_end", "wav", "sha256")}
        row["duration_s"] = round(len(w) / 16000, 3)
        for s in SCORERS:
            cos = {n: float(e[s] @ emb["ref"][s][n]) for n in names}
            pred = max(cos, key=cos.get)
            others = [c for n, c in cos.items() if n != r["voice"]]
            row[f"{s}_pred"] = pred
            row[f"{s}_cos_expected"] = round(cos[r["voice"]], 4)
            row[f"{s}_margin"] = round(cos[r["voice"]] - max(others), 4) if others else float("nan")
            row[f"{s}_wrong"] = int(pred != r["voice"])
            row[f"{s}_cos_all"] = json.dumps({n: round(c, 4) for n, c in cos.items()})
        row["wrong_both_agree"] = int(row["campplus_wrong"] and row["wavlm_wrong"]
                                      and row["campplus_pred"] == row["wavlm_pred"])
        row["wrong_any"] = int(row["campplus_wrong"] or row["wavlm_wrong"])
        row["asr_text"] = hyp
        row["wer"] = round(wer(TEXTS[r["text_idx"]], hyp), 4)
        rows.append(row)
    # Duration ratio: against the median for the same (text, voice) in this run.
    groups: dict = {}
    for row in rows:
        groups.setdefault((row["text_idx"], row["voice"]), []).append(row["duration_s"])
    for row in rows:
        med = statistics.median(groups[(row["text_idx"], row["voice"])])
        row["dur_ratio_run_median"] = round(row["duration_s"] / med, 4) if med else float("nan")
    errors = [r for r in recs if not r.get("wav")]
    if rows:
        with open(run_dir / "scores.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    summ = summarize(rows, names)
    summ.update(meta={k: meta[k] for k in ("label", "commit", "mode", "overlay", "K", "C", "seed", "N")},
                n_requests=len(recs), n_errors=len(errors),
                error_examples=[e.get("error", "")[:300] for e in errors[:3]],
                scorer_versions=sc.versions, reference_matrix=ref_matrix(emb, names))
    (run_dir / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    return summ


def summarize(rows: list[dict], names: list[str]) -> dict:
    n = len(rows)
    out = {"n_scored": n}
    for col in ("campplus_wrong", "wavlm_wrong", "wrong_both_agree", "wrong_any"):
        k = sum(r[col] for r in rows)
        lo, hi = wilson(k, n)
        out[col] = {"k": k, "n": n, "rate": round(k / n, 4) if n else None,
                    "wilson95": [round(lo, 4), round(hi, 4)]}
    for s in SCORERS:
        conf = {a: {b: 0 for b in names} for a in names}
        for r in rows:
            conf[r["voice"]][r[f"{s}_pred"]] += 1
        out[f"confusion_{s}"] = conf
    wers = [r["wer"] for r in rows if not math.isnan(r["wer"])]
    durs = [r["dur_ratio_run_median"] for r in rows]
    out["wer_mean"] = round(statistics.mean(wers), 4) if wers else None
    out["wer_gt_0.2"] = sum(1 for x in wers if x > 0.2)
    out["dur_ratio_gt_1.3"] = sum(1 for x in durs if x > 1.3)
    out["min_margin"] = {s: round(min((r[f"{s}_margin"] for r in rows), default=float("nan")), 4) for s in SCORERS}
    return out


def _scorers_from_args(args) -> Scorers:
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return Scorers(args.model_dir, device, args.asr_model, args.wavlm_model)


def cmd_score(args) -> int:
    sc = _scorers_from_args(args)
    cache: dict = {}
    for d in args.run_dirs:
        d = Path(d)
        if not (d / "requests.jsonl").exists():
            print(f"skip {d}: no requests.jsonl")
            continue
        s = score_run(sc, d, cache)
        print(f"{d}: n={s['n_scored']} err={s['n_errors']} "
              + " ".join(f"{c}={s[c]['k']}/{s[c]['n']}" for c in ("campplus_wrong", "wavlm_wrong", "wrong_both_agree")))
    return 0


def cmd_refmatrix(args) -> int:
    sc = _scorers_from_args(args)
    out = {}
    for vf in args.voices:
        voices = load_voices(vf)
        names = [v["name"] for v in voices]
        out[Path(vf).name] = ref_matrix(reference_embeddings(sc, voices), names)
    text = json.dumps(out, indent=1)
    if args.out:
        Path(args.out).write_text(text)
    for vf, mats in out.items():
        for s, m in mats.items():
            print(f"\n{vf} [{s}] ref-vs-ref cosine (max off-diagonal {m['max_offdiag_ref']})")
            print(_fmt_matrix(m["names"], m["ref_vs_ref"]))
            if "heldout_vs_ref" in m:
                print(f"{vf} [{s}] held-out utterance (rows) vs reference (cols); "
                      f"top-1 correct {m['heldout_top1_correct']}/{len(m['names'])}")
                print(_fmt_matrix(m["names"], m["heldout_vs_ref"]))
    return 0


def _fmt_matrix(names, mat) -> str:
    short = [n.split("_", 1)[-1][:9] for n in names]
    lines = [" " * 10 + " ".join(f"{s:>9s}" for s in short)]
    for s, row in zip(short, mat):
        lines.append(f"{s:>9s} " + " ".join(f"{x:9.3f}" for x in row))
    return "\n".join(lines)


# ------------------------------------------------------------------ report ---

def _load_summaries(root: Path) -> list[dict]:
    out = []
    for p in sorted(root.glob("**/summary.json")):
        s = json.loads(p.read_text())
        s["run_dir"] = str(p.parent)
        out.append(s)
    return out


def _cell_key(m: dict) -> tuple:
    return (m["commit"], m["mode"], m.get("overlay") or "-", m["K"], m["C"])


def cmd_report(args) -> int:
    root = Path(args.runs_root)
    summ = _load_summaries(root)
    cells: dict = {}
    for s in summ:
        cells.setdefault(_cell_key(s["meta"]), []).append(s)
    rows = []
    for key in sorted(cells):
        ss = cells[key]
        agg = {"commit": key[0], "mode": key[1], "overlay": key[2], "K": key[3], "C": key[4],
               "seeds": ",".join(str(s["meta"]["seed"]) for s in ss),
               "n_requests": sum(s["n_requests"] for s in ss), "n_errors": sum(s["n_errors"] for s in ss)}
        for col in ("campplus_wrong", "wavlm_wrong", "wrong_both_agree"):
            k = sum(s[col]["k"] for s in ss)
            n = sum(s[col]["n"] for s in ss)
            lo, hi = wilson(k, n)
            agg[col] = k
            agg[col + "_n"] = n
            agg[col + "_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        wers = [s["wer_mean"] for s in ss if s["wer_mean"] is not None]
        agg["wer_mean"] = round(statistics.mean(wers), 4) if wers else ""
        agg["wer_gt_0.2"] = sum(s["wer_gt_0.2"] for s in ss)
        agg["dur_ratio_gt_1.3"] = sum(s["dur_ratio_gt_1.3"] for s in ss)
        rows.append(agg)
    # Leak test: C>1 cell vs the C=1 cell of the same (commit, mode, overlay-free, K).
    base = {(r["commit"], r["mode"], r["K"]): r for r in rows if r["C"] == 1 and r["overlay"] == "-"}
    for r in rows:
        b = base.get((r["commit"], r["mode"], r["K"]))
        if r["C"] > 1 and b:
            p = fisher_greater(r["wrong_both_agree"], r["wrong_both_agree_n"],
                               b["wrong_both_agree"], b["wrong_both_agree_n"])
            r["fisher_p_vs_C1"] = f"{p:.2e}"
            r["leak"] = "YES" if (r["wrong_both_agree"] > 0 and p < args.alpha) else "no"
        else:
            r["fisher_p_vs_C1"] = ""
            r["leak"] = "" if r["C"] == 1 else "no-C1-baseline"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out / "cells.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_dir", "commit", "mode", "overlay", "K", "C", "seed", "n_requests", "n_errors",
                    "campplus_wrong", "wavlm_wrong", "wrong_both_agree", "wer_mean", "dur_ratio_gt_1.3"])
        for s in summ:
            m = s["meta"]
            w.writerow([s["run_dir"], m["commit"], m["mode"], m.get("overlay") or "-", m["K"], m["C"], m["seed"],
                        s["n_requests"], s["n_errors"], s["campplus_wrong"]["k"], s["wavlm_wrong"]["k"],
                        s["wrong_both_agree"]["k"], s["wer_mean"], s["dur_ratio_gt_1.3"]])
    md = ["# CosyVoice3 voice-leak measurement report", "",
          f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} from `{root}`.", "",
          "Wrong = argmax cosine over all K references is not the requested voice. "
          "`agree` = both scorers name the same wrong voice. "
          f"Leak = agree count > 0 and one-sided Fisher p < {args.alpha} vs the C=1 cell. "
          "CIs are Wilson 95%.", "",
          "| commit | mode | overlay | K | C | seeds | req | err | CAM++ wrong | WavLM wrong | agree | agree CI "
          "| Fisher p | leak | WER mean | WER>0.2 | dur>1.3x |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['commit']} | {r['mode']} | {r['overlay']} | {r['K']} | {r['C']} | {r['seeds']} | "
                  f"{r['n_requests']} | {r['n_errors']} | {r['campplus_wrong']}/{r['campplus_wrong_n']} | "
                  f"{r['wavlm_wrong']}/{r['wavlm_wrong_n']} | {r['wrong_both_agree']}/{r['wrong_both_agree_n']} | "
                  f"{r['wrong_both_agree_ci']} | {r['fisher_p_vs_C1']} | {r['leak']} | {r['wer_mean']} | "
                  f"{r['wer_gt_0.2']} | {r['dur_ratio_gt_1.3']} |")
    for key in sorted(cells):
        ss = cells[key]
        if key[3] < 8 and not args.all_confusions:
            continue
        for sname in SCORERS:
            names = list(ss[0][f"confusion_{sname}"].keys())
            tot = {a: {b: sum(s[f"confusion_{sname}"][a][b] for s in ss) for b in names} for a in names}
            md += ["", f"### Confusion {sname}: {key[0]} {key[1]} overlay={key[2]} K={key[3]} C={key[4]} "
                       "(rows requested, cols predicted)", "", "```",
                   _fmt_matrix(names, [[tot[a][b] for b in names] for a in names]).replace(".000", "    "), "```"]
    errs = [s for s in summ if s["n_errors"]]
    if errs:
        md += ["", "## Request errors", ""]
        for s in errs:
            md.append(f"- `{s['run_dir']}`: {s['n_errors']} errors, e.g. `{(s['error_examples'] or [''])[0][:200]}`")
    (out / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


def cmd_gate(args) -> int:
    summ = [s for s in _load_summaries(Path(args.runs_root)) if s["meta"]["commit"].startswith(args.commit)]
    hi = [s for s in summ if s["meta"]["C"] > 1]
    k_cam = sum(s["campplus_wrong"]["k"] for s in hi)
    k_wlm = sum(s["wavlm_wrong"]["k"] for s in hi)
    n = sum(s["n_scored"] for s in hi)
    print(f"gate {args.commit}: C>1 runs={len(hi)} scored={n} campplus_wrong={k_cam} wavlm_wrong={k_wlm}")
    if not hi or n == 0:
        print("gate: no scored C>1 runs for the positive control")
        return 4
    if k_cam == 0 and k_wlm == 0:
        lo, hi_ci = wilson(0, n)
        Path(args.findings).write_text(
            "# FINDINGS: positive control did not reproduce\n\n"
            f"At {args.commit} (before #8224) no scorer found a wrong voice in {n} scored outputs at C>1 "
            f"(Wilson 95% upper bound {hi_ci:.4f}). Without a positive control the later commits cannot be "
            "compared, so the plan stopped here.\n")
        return 3
    return 0


# -------------------------------------------------------------------- main ---

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--base", default="http://127.0.0.1:8091")
    r.add_argument("--model", default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    r.add_argument("--voices", required=True)
    r.add_argument("--n", type=int, default=120)
    r.add_argument("--concurrency", "-c", type=int, default=8)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--timeout", type=float, default=300.0)
    r.add_argument("--out")
    r.add_argument("--label", default="")
    r.add_argument("--commit", default="unknown")
    r.add_argument("--mode", default="async", choices=["async", "no-async"])
    r.add_argument("--overlay", default="")
    r.add_argument("--dry-run", action="store_true")

    def scorer_args(sp):
        sp.add_argument("--model-dir", required=True, help="local CosyVoice3 snapshot (contains campplus.onnx)")
        sp.add_argument("--device", default=None)
        sp.add_argument("--asr-model", default="openai/whisper-small")
        sp.add_argument("--wavlm-model", default="microsoft/wavlm-base-plus-sv")

    s = sub.add_parser("score")
    s.add_argument("run_dirs", nargs="+")
    scorer_args(s)

    m = sub.add_parser("refmatrix")
    m.add_argument("--voices", nargs="+", required=True)
    m.add_argument("--out")
    scorer_args(m)

    rp = sub.add_parser("report")
    rp.add_argument("--runs-root", required=True)
    rp.add_argument("--out-dir", required=True)
    rp.add_argument("--alpha", type=float, default=0.01)
    rp.add_argument("--all-confusions", action="store_true")

    g = sub.add_parser("gate")
    g.add_argument("--runs-root", required=True)
    g.add_argument("--commit", required=True)
    g.add_argument("--findings", required=True)

    args = p.parse_args()
    if args.cmd == "run" and not args.dry_run and not args.out:
        p.error("run needs --out unless --dry-run")
    return {"run": cmd_run, "score": cmd_score, "refmatrix": cmd_refmatrix,
            "report": cmd_report, "gate": cmd_gate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
