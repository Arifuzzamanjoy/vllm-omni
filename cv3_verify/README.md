# CosyVoice3 voice-leak verification kit

Measures whether CosyVoice3 served by vLLM-Omni returns another in-flight
request's voice under concurrency (vllm-project/vllm-omni#8235) and whether
requests admitted mid-batch get corrupted (#8288: extra words, long audio).
Measurement only; nothing here changes `vllm_omni/`.

## Files

| file | purpose |
|---|---|
| `prepare_voices.py` | downloads the K=2 and K=8 reference voices (see ATTRIBUTION.md) |
| `voices_k2.json`, `voices_k8.json` | voice manifests: name, wav path, transcript, source IDs |
| `measure_voice_leak.py` | `run` / `score` / `refmatrix` / `report` / `gate` |
| `mock_server.py` | GPU-free stand-in for the speech API, with deliberate voice swaps |
| `run_plan.sh` | the 44-run matrix, one worktree per commit, one server per (commit, mode) |
| `requirements.txt` | extra packages on top of the vllm==0.30.0 + vllm-omni venv |

## Method

- Voices are registered with `POST /v1/audio/voices` (`audio_sample`, `name`,
  `consent`, `ref_text`) and requested by name through `/v1/audio/speech`
  (`response_format=wav`, per-request `seed = run_seed*1000 + idx`).
- N requests go round-robin over the K voices; four English texts of
  different lengths rotate independently of the voice, so requests finish at
  different steps and queued requests get admitted into running batches.
- Each output is embedded by two independent speaker models: CAM++
  (`campplus.onnx` from the model snapshot, Kaldi fbank, as in #8235) and
  WavLM-base-plus-SV. Predicted voice = argmax cosine over all K references.
- A request is a leak candidate when both scorers name the same wrong voice.
  A cell counts as leaking only if that count at C=8 is above the C=1 cell of
  the same commit/mode/K (one-sided Fisher exact, p < 0.01). Rates carry
  Wilson 95% CIs (the upper bound for 0/n is about 3.7/n).
- Whisper-small WER against the requested text and the duration ratio against
  the median of the same (text, voice) in the run flag #8288-style damage.

## Usage

```bash
.venv/bin/python cv3_verify/prepare_voices.py
.venv/bin/python cv3_verify/measure_voice_leak.py refmatrix --voices cv3_verify/voices_k2.json cv3_verify/voices_k8.json --model-dir <snapshot>
cv3_verify/run_plan.sh --dry-run
cv3_verify/run_plan.sh --phase control
cv3_verify/run_plan.sh --phase rest
```

Outputs (WAVs, CSVs, logs, findings) go to `$OUT` (default `/home/ubuntu/cv3_out`),
outside the repository.

## Offline test

```bash
.venv/bin/python cv3_verify/mock_server.py --voices cv3_verify/voices_k8.json --port 18091 --swap 5:k8_f237 --swap 17:k8_m8230
.venv/bin/python cv3_verify/measure_voice_leak.py run --base http://127.0.0.1:18091 --voices cv3_verify/voices_k8.json --n 24 -c 8 --out /tmp/mock/K8_C8_s1
.venv/bin/python cv3_verify/measure_voice_leak.py score /tmp/mock/K8_C8_s1 --model-dir <snapshot>
```

Requests 5 and 17 must come out as `wrong_both_agree=1`; all others correct.
