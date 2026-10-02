# Evidence: CosyVoice3 voice leak under concurrency (vllm-omni #8235, #8288)

Start with FINDINGS.md.

| path | content |
|---|---|
| FINDINGS.md | results, mechanism check, crashes, caveats |
| analysis.txt, cells_combined.csv | combined per-cell table (both scorers, Wilson CI, Fisher p, WER, duration, mechanism counts) |
| server_status.csv | per server: scored runs and recorded failures |
| report/ | report.md, cells.csv, runs.csv written by measure_voice_leak.py report |
| runs/<server>/<run>/ | meta.json, requests.jsonl, scores.csv, summary.json per run (no audio) |
| audio/ | 15 MODEL-GENERATED clips with reference clips, listed in audio/README.md |
| findings/ | tracebacks: --no-async-chunk at three commits, prefix-caching talker crash |
| superseded/ | first positive-control attempt (FlashInfer sampler needs nvcc) |
| env/<commit>/ | env.txt (vllm version, vllm_omni path check, sampler setting), pip freeze, nvidia-smi, install log |
| commits.txt, commits_full.txt | commit hashes measured, kit branch and commits |
| versions.txt | GPU, driver, OS, packages, model and scorer snapshots |
| commands.txt | commands run, in order |
| reference_matrices.{json,txt} | reference-vs-reference and held-out cosine matrices, both scorers |
| sampler_deviation.md | why VLLM_USE_FLASHINFER_SAMPLER=0 |
| overlays/ | talker overlay deploy YAMLs (max_num_seqs=4, prefix caching) |
| logs/ | server and plan logs (gzip) |
| scripts/analyze.py | the combined analysis |
| drafts/ | unposted comment drafts for #8235, #8288, #8240, #8262 |

Kit: https://github.com/Arifuzzamanjoy/vllm-omni/tree/repro/cosyvoice3-voice-leak-8235
