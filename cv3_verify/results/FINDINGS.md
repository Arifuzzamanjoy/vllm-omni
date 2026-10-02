# CosyVoice3 voice leak under concurrency: measurement findings

Date: 2026-10-02. Hardware: 1x NVIDIA RTX 4090 24 GB, driver 580.95.05, Ubuntu 22.04.5.
Software: Python 3.12.15, vllm 0.30.0, torch 2.13.0+cu130, vllm-omni editable from each commit's worktree.
Model: FunAudioLLM/Fun-CosyVoice3-0.5B-2512 (snapshot 29e01c4e), stock `vllm_omni/deploy/cosyvoice3.yaml`
(talker max_num_seqs 8, FLASH_ATTN on both stages). Sampler: PyTorch top-k/top-p
(`VLLM_USE_FLASHINFER_SAMPLER=0`, see sampler_deviation.md).

Commits: c85f1a4 (before #8224, positive control), 1ea0ee3 (#8224 alone), bbee488 (current base).
Kit: https://github.com/Arifuzzamanjoy/vllm-omni/tree/repro/cosyvoice3-voice-leak-8235 (head f71f13c).

## Method (short)

K reference voices registered via `POST /v1/audio/voices`; N=120 `/v1/audio/speech`
requests per run, round-robin over voices, per-request seed `run_seed*1000 + idx`,
four English texts of different lengths, concurrency C. Each output is scored
against all K references by two independent speaker models (CAM++ `campplus.onnx`
from the model, WavLM-base-plus-SV); predicted voice = argmax cosine. A leak is
counted only when both scorers name the same wrong voice. Whisper-small WER and
duration ratio against the C=1 median of the same (text, voice) flag #8288-type
damage. Fisher p is one-sided against the C=1 cell of the same commit.

## Results: async_chunk (default) mode

| commit | config | K | C | requests | both scorers wrong | rate | Wilson 95% CI | Fisher p vs C=1 |
|---|---|---|---|---|---|---|---|---|
| c85f1a4 | stock | 2 | 1 | 120 | 0 | 0.0% | [0.000, 0.031] | |
| c85f1a4 | stock | 2 | 8 | 360 | **42** | **11.7%** | [0.087, 0.154] | 3.0e-06 |
| c85f1a4 | stock | 8 | 1 | 120 | 0 | 0.0% | [0.000, 0.031] | |
| c85f1a4 | stock | 8 | 8 | 360 | **90** | **25.0%** | [0.208, 0.297] | 2.2e-13 |
| 1ea0ee3 | stock | 2 | 8 | 360 | 0 | 0.0% | [0.000, 0.011] | (no C=1 cell by design) |
| bbee488 | stock | 2 | 1 | 120 | 0 | 0.0% | [0.000, 0.031] | |
| bbee488 | stock | 2 | 8 | 360 | 0 | 0.0% | [0.000, 0.011] | 1.0 |
| bbee488 | stock | 8 | 1 | 120 | 0 | 0.0% | [0.000, 0.031] | |
| bbee488 | stock | 8 | 8 | 360 | 0 | 0.0% | [0.000, 0.011] | 1.0 |
| bbee488 | talker max_num_seqs=4 | 8 | 8 | 360 | 0 | 0.0% | [0.000, 0.011] | 1.0 |
| bbee488 | talker prefix caching | 8 | 8 | 25 of 360 (crash) | 0 | 0.0% | [0.000, 0.133] | not meaningful |

Single-scorer counts: c85f1a4 K2 C8 CAM++ 42 / WavLM 43; K8 C8 CAM++ 95 / WavLM 92.
Everywhere else CAM++ found 0; WavLM flagged one output (bbee488 K2 C8 s3 idx 49, see below).
Per-run detail: report/report.md, runs.csv, cells_combined.csv.

## --no-async-chunk

| commit | result |
|---|---|
| c85f1a4 | server healthy; first request HTTP 500 in 1.3 s; stage 1 dead |
| 1ea0ee3 | same |
| bbee488 | same |

Same error at all three commits (findings/*_no-async_smoke.txt):
`RuntimeError: Error concatenating tensor list for key 'embed.speech_feat' under strategy
CONCAT_DIM0 ... Expected size 174 but got size 1500`, then in stage 1
`msgspec.ValidationError: cannot decode list into torch.Tensor - at $.embed.speech_token_len`.
No leak data exist for this mode.

## Talker prefix caching (bbee488 overlay)

Server healthy, `enable_prefix_caching=True` confirmed for stage 0 in the log. In the
first C=8 run, 25 requests succeeded, then the talker died:
`vectorized_gather_kernel ... Assertion 'ind >= 0 && ind < ind_dim_size' failed` →
`torch.AcceleratorError: CUDA error: device-side assert triggered`, surfacing in
`vllm_omni/core/prefix_cache/controller.py:304` (`mark_host_ready`, stream synchronize).
CUDA asserts are asynchronous, so the faulting kernel is not identified. Runs s2 and s3
hit the already-dead talker (0/120 each) and are not data; the plan's dead-server
check watches only the API process. findings/bbee488_async_prefix_died_K8_C8_s1.txt.

## Mechanism check (inference from client-side timing, not server batch state)

For each leaked output at C>1: is the voice it came out in that of the earliest-started
request still in flight when it was sent (most likely batch row 0, fits the
`element[0]` fallback at vllm_omni/utils/mm_outputs.py:241), or that of the request
sent immediately before/after it (fits conditioning rows shifted by one)? "null" is
the expected count if the wrong voice were a uniformly random other in-flight voice.

| cell | leaks | voice was in flight | earliest: observed / null | adjacent: observed / null | submission offset to source |
|---|---|---|---|---|---|
| c85f1a4 K=8 C=8 | 90 | 90/90 | 13 / 10.7 | **70 / 25.7** | 1:70, 2:13, 3:6, 4:1 |
| c85f1a4 K=2 C=8 | 42 | 42/42 | 20 / 20.0 | 42 / 42.0 | uninformative (only one other voice) |

At K=8 the leaked voice comes from the neighbouring request almost three times as often
as chance, and from the earliest in-flight request no more than chance. That fits
per-request conditioning landing one batch row off more than a fallback to row 0.
It is an inference: client timing approximates admission order, not the batch layout.

## #8288 signals (WER, duration)

No excess at any commit. Outputs with WER > 0.2 at C=8: c85f1a4 2/720, 1ea0ee3 0/360,
bbee488 1/720, max_num_seqs=4 0/360. Outputs > 1.3x the C=1 median duration: 4-7 per
360 at C=8, against 3-4 per 120 at C=1 (natural variation). Notable single outputs:

- c85f1a4 K2 C8 s3 idx 101: correct text plus "Blood Method Dave." at the end, the
  #8288 symptom; the same request at bbee488 is clean. One instance; not significant.
- idx 49 of K2 C8 s3 (seed 3049, cross-lingual voice): mostly non-speech, 9-11 s,
  ASR "Thanks." / "Thank you." at both c85f1a4 and bbee488. Fails the same way on
  both commits, so it looks like a sampling outcome for that seed, not batch corruption
  (not checked at C=1 with that seed). It is the one WavLM-only "wrong voice" flag.

max_num_seqs=4 with C=8 at bbee488 forces queued requests to be admitted into running
batches (the #8288 condition, FLASH_ATTN): 0 leaks, 0 WER outliers.

## Other observations

- C=1 outputs are byte-identical between c85f1a4 and bbee488 (240/240): #8224 does
  not change single-request output. C=8 outputs never match C=1 byte for byte at any
  commit (batch-dependent numerics), so byte identity is not used as a detector.
- Scorers: both place a held-out utterance of each K=8 speaker on the right speaker
  (8/8). Max off-diagonal reference cosine: CAM++ 0.70, WavLM 0.90 (K=8).

## Caveats and what was not verified

- PyTorch sampler, not FlashInfer (sampler_deviation.md).
- One seed at C=1 per (commit, K); 1ea0ee3 has no C=1 cell (C=1 is commit-independent
  here, see byte identity above).
- #8288 positive control is missing: c85f1a4 was not run with max_num_seqs=4, so the
  absence of #8288 damage at bbee488 is not shown to be caused by #8224.
- PRs #8240, #8343, #8262 were not built or run. #8240 targets the same aliasing that
  #8224 already handles through `_align_prompt_conditioning`.
- Mechanism counts come from client timestamps, not server batch state.
- Prefix-caching crash: faulting kernel not identified (would need CUDA_LAUNCH_BLOCKING=1).
- WavLM weights were loaded from the hub's safetensors-conversion ref (versions.txt);
  reference matrices are identical to the .bin weights.
- vllm-omni reports a version mismatch warning (0.1.devN vs vLLM 0.30.0) from
  setuptools-scm in worktrees without tags; all three commits loaded and served.
- RTX 4090 only; no other GPU class.
