#!/usr/bin/env python3
"""Combined analysis of the CosyVoice3 voice-leak runs in $OUT/runs.

Per cell (commit x mode x overlay x K x C): wrong-voice counts for each scorer
and for both-agree, Wilson 95% CI, one-sided Fisher p vs the C=1 cell of the
same commit/mode/K (overlay cells use the plain-config C=1 cell of the same
commit), WER > 0.2 count, and duration outliers against the C=1 baseline
median of the same (text, voice).

Mechanism check (inference from client-side timing only, not from server
batch state): for every both-agree wrong output at C>1, compare the voice it
came out in with
  earliest  - the voice of the earliest-started request still in flight when
              this request was sent (the request most likely to sit in batch
              row 0; fits the element[0] fallback in to_payload_element),
  adjacent  - the voice of the request sent immediately before or after it
              (fits conditioning rows shifted by one).
"""

import csv
import glob
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # cv3_verify/
from measure_voice_leak import fisher_greater, wilson  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/cv3_out")
DEST = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/ubuntu/evidence")
DUR_LIMIT = 1.3
WER_LIMIT = 0.2


def load_runs():
    runs = []
    for d in sorted(glob.glob(str(OUT / "runs/*/K*_C*_s*"))):
        d = Path(d)
        if not (d / "scores.csv").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        rows = list(csv.DictReader(open(d / "scores.csv")))
        reqs = [json.loads(l) for l in open(d / "requests.jsonl") if l.strip()]
        runs.append({"dir": d, "meta": meta, "rows": rows, "reqs": reqs})
    return runs


def key(m):
    return (m["commit"], m["mode"], m.get("overlay") or "-", m["K"], m["C"])


def mechanism(run):
    rows = run["rows"]
    by_idx = {int(r["idx"]): r for r in rows}
    # in-flight info from all requests (including failed ones)
    reqs = {int(q["idx"]): q for q in run["reqs"]}
    out = defaultdict(int)
    for r in rows:
        if r["wrong_both_agree"] != "1":
            continue
        out["n"] += 1
        i = int(r["idx"])
        s = float(r["t_start"])
        pred = r["campplus_pred"]
        inflight = [q for j, q in reqs.items() if j != i and q["t_start"] <= s < q.get("t_end", math.inf)]
        overlapping = [q for j, q in reqs.items() if j != i and q["t_start"] < float(r["t_end"]) and q.get("t_end", math.inf) > s]
        earliest = min(inflight, key=lambda q: (q["t_start"], q["idx"]))["voice"] if inflight else None
        adj = {reqs[j]["voice"] for j in (i - 1, i + 1) if j in reqs}
        e, a = pred == earliest, pred in adj
        out["pred_voice_in_flight"] += any(q["voice"] == pred for q in overlapping)
        # Null: the wrong voice is a uniformly random *other* voice among the
        # requests overlapping this one. Expected matches per category:
        cand = {q["voice"] for q in overlapping} - {r["voice"]}
        if cand:
            out["null_earliest"] += (earliest in cand) / len(cand)
            out["null_adjacent"] += len(adj & cand) / len(cand)
        # Submission-order distance to the nearest overlapping request in the predicted voice.
        d = [abs(q["idx"] - i) for q in overlapping if q["voice"] == pred]
        if d:
            out[f"offset_{min(min(d), 9)}"] += 1
        out["earliest_only"] += e and not a
        out["adjacent_only"] += a and not e
        out["both"] += e and a
        out["neither"] += not e and not a
    return out


def main():
    runs = load_runs()
    cells = defaultdict(list)
    for r in runs:
        cells[key(r["meta"])].append(r)

    # C=1 duration baselines per (commit, mode, K): median per (text_idx, voice)
    base_dur = {}
    for k, rs in cells.items():
        if k[4] == 1 and k[2] == "-":
            g = defaultdict(list)
            for r in rs:
                for row in r["rows"]:
                    g[(row["text_idx"], row["voice"])].append(float(row["duration_s"]))
            base_dur[(k[0], k[1], k[3])] = {kk: statistics.median(v) for kk, v in g.items()}

    table = []
    for k in sorted(cells):
        rs = cells[k]
        rows = [row for r in rs for row in r["rows"]]
        n_req = sum(len(r["reqs"]) for r in rs)
        n = len(rows)
        rec = {"commit": k[0], "mode": k[1], "overlay": k[2], "K": k[3], "C": k[4],
               "seeds": ",".join(str(r["meta"]["seed"]) for r in rs), "requests": n_req,
               "errors": n_req - n, "scored": n}
        for col in ("campplus_wrong", "wavlm_wrong", "wrong_both_agree"):
            rec[col] = sum(int(x[col]) for x in rows)
        lo, hi = wilson(rec["wrong_both_agree"], n)
        rec["agree_rate"] = round(rec["wrong_both_agree"] / n, 4) if n else ""
        rec["agree_ci95"] = f"[{lo:.3f}, {hi:.3f}]" if n else ""
        b = cells.get((k[0], k[1], "-", k[3], 1))
        if k[4] > 1 and b:
            bk = sum(int(x["wrong_both_agree"]) for r in b for x in r["rows"])
            bn = sum(len(r["rows"]) for r in b)
            rec["fisher_p_vs_C1"] = f"{fisher_greater(rec['wrong_both_agree'], n, bk, bn):.2e}"
        else:
            rec["fisher_p_vs_C1"] = ""
        wers = [float(x["wer"]) for x in rows if x["wer"] not in ("", "nan")]
        rec["wer_mean"] = round(statistics.mean(wers), 4) if wers else ""
        rec["wer_gt_0.2"] = sum(w > WER_LIMIT for w in wers)
        bd = base_dur.get((k[0], k[1], k[3])) or base_dur.get((k[0], "async", k[3]))
        rec["dur_baseline"] = f"{k[0]} C=1" if bd else ""
        if not bd:
            # C=1 outputs are byte-identical across c85f1a4 and bbee488 (same
            # per-request seeds), so another commit's C=1 cell is a valid baseline.
            other = [v for (c, m, kk), v in base_dur.items() if m == k[1] and kk == k[3]]
            if other:
                bd = other[0]
                rec["dur_baseline"] = "other-commit C=1"
        if bd:
            ratios = [float(x["duration_s"]) / bd[(x["text_idx"], x["voice"])] for x in rows
                      if (x["text_idx"], x["voice"]) in bd]
            rec["dur_gt_1.3x_C1"] = sum(q > DUR_LIMIT for q in ratios)
            rec["dur_ratio_max"] = round(max(ratios), 2) if ratios else ""
        else:
            rec["dur_gt_1.3x_C1"] = rec["dur_ratio_max"] = ""
        mech = defaultdict(int)
        if k[4] > 1:
            for r in rs:
                for kk, v in mechanism(r).items():
                    mech[kk] += v
        for kk in ("pred_voice_in_flight", "earliest_only", "adjacent_only", "both", "neither"):
            rec["mech_" + kk] = mech.get(kk, 0) if k[4] > 1 else ""
        rec["mech_earliest_obs"] = (mech.get("earliest_only", 0) + mech.get("both", 0)) if k[4] > 1 else ""
        rec["mech_earliest_null"] = round(mech.get("null_earliest", 0), 1) if k[4] > 1 else ""
        rec["mech_adjacent_obs"] = (mech.get("adjacent_only", 0) + mech.get("both", 0)) if k[4] > 1 else ""
        rec["mech_adjacent_null"] = round(mech.get("null_adjacent", 0), 1) if k[4] > 1 else ""
        rec["mech_offset_hist"] = (" ".join(f"{o}:{mech[f'offset_{o}']}" for o in range(1, 10) if mech.get(f"offset_{o}"))
                                   if k[4] > 1 else "")
        table.append(rec)

    DEST.mkdir(parents=True, exist_ok=True)
    with open(DEST / "cells_combined.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    # no-async / server status per commit, from findings and runs
    status = []
    tags = sorted(Path(t).name for t in glob.glob(str(OUT / "runs/*")))
    owner = {}
    for p in glob.glob(str(OUT / "findings/*.txt")):
        name = Path(p).name
        # Assign each finding to the longest server tag it starts with.
        match = [t for t in tags if name.startswith(t + "_")]
        if match:
            owner.setdefault(max(match, key=len), []).append(name)
    for tagdir in sorted(glob.glob(str(OUT / "runs/*"))):
        tag = Path(tagdir).name
        fnd = sorted(owner.get(tag, []))
        nrun = len([p for p in glob.glob(f"{tagdir}/K*/scores.csv")])
        status.append({"server": tag, "scored_runs": nrun, "findings": " ".join(fnd) or "-"})
    with open(DEST / "server_status.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["server", "scored_runs", "findings"])
        w.writeheader()
        w.writerows(status)

    print("| commit | mode | overlay | K | C | seeds | req | err | CAM++ | WavLM | agree | rate | 95% CI | Fisher p | WER>0.2 | dur>1.3x | max dur | in-flight | earliest | adjacent | both | neither |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in table:
        print("| " + " | ".join(str(r[c]) for c in (
            "commit", "mode", "overlay", "K", "C", "seeds", "requests", "errors", "campplus_wrong",
            "wavlm_wrong", "wrong_both_agree", "agree_rate", "agree_ci95", "fisher_p_vs_C1", "wer_gt_0.2",
            "dur_gt_1.3x_C1", "dur_ratio_max", "mech_pred_voice_in_flight", "mech_earliest_only",
            "mech_adjacent_only", "mech_both", "mech_neither")) + " |")
    print()
    print("Mechanism (inference from client timing): observed vs expected if the wrong voice were a random other in-flight voice")
    for r in table:
        if r["C"] != 1 and r["wrong_both_agree"]:
            print(f"  {r['commit']} {r['mode']} {r['overlay']} K={r['K']}: leaks={r['wrong_both_agree']} "
                  f"earliest obs={r['mech_earliest_obs']} null={r['mech_earliest_null']} | "
                  f"adjacent obs={r['mech_adjacent_obs']} null={r['mech_adjacent_null']} | "
                  f"offset hist (9=9+) {r['mech_offset_hist']}")
    print()
    for s in status:
        print(f"{s['server']}: scored runs {s['scored_runs']}; findings {s['findings']}")


if __name__ == "__main__":
    main()
