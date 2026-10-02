#!/usr/bin/env python3
"""Fetch the reference voices used by the CosyVoice3 voice-leak kit.

K=2: the two official CosyVoice prompt clips (Apache-2.0).
K=8: one 5-10 s LibriTTS-R test-clean utterance per speaker (CC BY 4.0),
     plus a second, held-out utterance of the same speaker that is never sent
     to the server. The held-out clip gives the same-speaker / different-
     utterance cosine (scorer noise floor) and is what mock_server.py returns.

Audio goes to cv3_verify/voices/ (git-excluded). The JSON manifests
(voices_k2.json, voices_k8.json) are rewritten only with --write-manifests;
the committed manifests pin the selected utterance IDs, and a plain run
re-downloads exactly those IDs and checks the transcripts still match.
"""

import argparse
import io
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

KIT = Path(__file__).resolve().parent
REPO = KIT.parent
VOICE_DIR = KIT / "voices"

COSYVOICE_ASSET = "https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/"
K2 = [
    {
        "name": "k2_zero_shot",
        "file": "zero_shot_prompt.wav",
        "local_copy": "tests/assets/cosyvoice3/zero_shot_prompt.wav",
        "ref_text": "希望你以后能够做的比我还好呦。",
        "gender": "F",
    },
    {
        "name": "k2_cross_lingual",
        "file": "cross_lingual_prompt.wav",
        "local_copy": None,
        "ref_text": "在那之后，完全收购那家公司，因此保持管理层的一致性，利益与即将加入家族的资产保持一致。这就是我们有时不买下全部的原因。",
        "gender": "M",
    },
]

LIBRITTS_SPEAKERS = [("237", "F"), ("2961", "F"), ("5142", "F"), ("6829", "F"),
                     ("1188", "M"), ("7127", "M"), ("8224", "M"), ("8230", "M")]
MIN_S, MAX_S = 5.0, 10.0
TARGET_SR = 24000


def _mono(wav: np.ndarray) -> np.ndarray:
    return wav.mean(axis=1) if wav.ndim == 2 else wav


def _resample(wav: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return wav
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr, target)
    return resample_poly(wav, target // g, sr // g).astype(np.float32)


def prepare_k2() -> list[dict]:
    out = []
    for v in K2:
        dst = VOICE_DIR / "k2" / v["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if v["local_copy"] and (REPO / v["local_copy"]).is_file():
            shutil.copyfile(REPO / v["local_copy"], dst)
            origin = f"{v['local_copy']} (repo copy of {COSYVOICE_ASSET}{v['file']})"
        elif not dst.is_file():
            dst.write_bytes(urllib.request.urlopen(COSYVOICE_ASSET + v["file"], timeout=60).read())
            origin = COSYVOICE_ASSET + v["file"]
        else:
            origin = COSYVOICE_ASSET + v["file"]
        info = sf.info(str(dst))
        out.append({
            "name": v["name"],
            "wav": str(dst.relative_to(KIT)),
            "ref_text": v["ref_text"],
            "gender": v["gender"],
            "source": "FunAudioLLM/CosyVoice asset/" + v["file"],
            "origin": origin,
            "license": "Apache-2.0",
            "sample_rate": info.samplerate,
            "duration_s": round(info.frames / info.samplerate, 3),
            "heldout_wav": None,
            "heldout_text": None,
        })
    return out


def _iter_libritts():
    from datasets import Audio, load_dataset

    ds = load_dataset("mythicinfinity/libritts_r", "clean", split="test.clean", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    for row in ds:
        yield row


def _decode(row) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=False)
    return _mono(wav), sr


def prepare_k8(pinned: list[dict] | None) -> list[dict]:
    want = {spk for spk, _ in LIBRITTS_SPEAKERS}
    pinned_ids = {}
    if pinned:
        for v in pinned:
            pinned_ids[v["speaker_id"]] = (v["utterance_id"], v["heldout_utterance_id"])
    picks: dict[str, list] = {spk: [] for spk in want}
    for row in _iter_libritts():
        spk = str(row["speaker_id"])
        if spk not in want:
            continue
        uid = row["id"]
        if pinned_ids:
            if uid not in pinned_ids[spk]:
                continue
            picks[spk].append(row)
        else:
            if len(picks[spk]) >= 2:
                continue
            wav, sr = _decode(row)
            dur = len(wav) / sr
            if not (MIN_S <= dur <= MAX_S):
                continue
            # Single-chapter-heading or very short texts make poor references.
            if len(row["text_normalized"].split()) < 8:
                continue
            picks[spk].append(row)
        if all(len(p) >= 2 for p in picks.values()):
            break
    out = []
    for spk, gender in LIBRITTS_SPEAKERS:
        rows = picks[spk]
        if len(rows) < 2:
            sys.exit(f"speaker {spk}: found {len(rows)} usable utterances, need 2")
        if pinned_ids:
            rows.sort(key=lambda r: pinned_ids[spk].index(r["id"]))
        files = []
        for role, row in zip(("ref", "heldout"), rows[:2]):
            wav, sr = _decode(row)
            wav = _resample(wav, sr, TARGET_SR)
            dst = VOICE_DIR / "k8" / f"{spk}_{role}_{row['id']}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(dst), wav, TARGET_SR, subtype="PCM_16")
            files.append((dst, row, len(wav) / TARGET_SR))
        (ref, rrow, rdur), (held, hrow, _) = files
        out.append({
            "name": f"k8_{gender.lower()}{spk}",
            "wav": str(ref.relative_to(KIT)),
            "ref_text": rrow["text_normalized"],
            "gender": gender,
            "speaker_id": spk,
            "utterance_id": rrow["id"],
            "heldout_utterance_id": hrow["id"],
            "source": "mythicinfinity/libritts_r clean/test.clean",
            "license": "CC BY 4.0",
            "sample_rate": TARGET_SR,
            "duration_s": round(rdur, 3),
            "heldout_wav": str(held.relative_to(KIT)),
            "heldout_text": hrow["text_normalized"],
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--write-manifests", action="store_true",
                   help="select utterances afresh and rewrite voices_k2.json / voices_k8.json")
    args = p.parse_args()
    VOICE_DIR.mkdir(exist_ok=True)

    k2 = prepare_k2()
    k8_path = KIT / "voices_k8.json"
    pinned = None if args.write_manifests or not k8_path.is_file() else json.loads(k8_path.read_text())["voices"]
    k8 = prepare_k8(pinned)

    if args.write_manifests or pinned is None:
        (KIT / "voices_k2.json").write_text(json.dumps({"voices": k2}, ensure_ascii=False, indent=2) + "\n")
        k8_path.write_text(json.dumps({"voices": k8}, ensure_ascii=False, indent=2) + "\n")
        print("wrote voices_k2.json, voices_k8.json")
    else:
        for new, old in zip(k8, pinned):
            if (new["utterance_id"], new["ref_text"]) != (old["utterance_id"], old["ref_text"]):
                sys.exit(f"pinned utterance mismatch for speaker {old['speaker_id']}")
        print("re-downloaded pinned utterances; manifests unchanged")
    for v in k2 + k8:
        print(f"{v['name']:22s} {v['duration_s']:5.2f}s  {v['wav']}")


if __name__ == "__main__":
    main()
    # The datasets streaming worker thread can crash CPython's finalizer
    # (PyGILState_Release) after all files are written; skip finalization.
    sys.stdout.flush()
    os._exit(0)
