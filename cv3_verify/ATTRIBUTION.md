# Reference voice attribution

The audio itself is not committed; `prepare_voices.py` downloads it into
`cv3_verify/voices/`. Speech produced by the server during measurement is
model-generated and is not part of these datasets.

## K=2: official CosyVoice prompt clips

Source: FunAudioLLM/CosyVoice, `asset/` directory
(https://github.com/FunAudioLLM/CosyVoice), license Apache-2.0.

| name | file | sample rate | duration | transcript |
|---|---|---|---|---|
| k2_zero_shot | asset/zero_shot_prompt.wav (repo copy: tests/assets/cosyvoice3/zero_shot_prompt.wav) | 24000 | 3.48 s | 希望你以后能够做的比我还好呦。 |
| k2_cross_lingual | asset/cross_lingual_prompt.wav | 22050 | 13.75 s | 在那之后，完全收购那家公司，因此保持管理层的一致性，利益与即将加入家族的资产保持一致。这就是我们有时不买下全部的原因。 |

Both transcripts were checked against Whisper-small output of the clips.

## K=8: LibriTTS-R test-clean

Source: LibriTTS-R (Koizumi et al., 2023, "LibriTTS-R: A Restored Multi-Speaker
Text-to-Speech Corpus"), derived from LibriTTS (Zen et al., 2019) and LibriVox
recordings. License CC BY 4.0. Retrieved through the Hugging Face dataset
`mythicinfinity/libritts_r`, config `clean`, split `test.clean`, streamed.

Selection rule: first utterance in stream order per speaker with duration
5-10 s and at least 8 words; the next qualifying utterance of the same speaker
is kept as a held-out clip (never sent to the server; used only for the
scorer noise floor and by `mock_server.py`). Audio is mono, 24 kHz, 16-bit
WAV; transcripts are the dataset's `text_normalized` field.

| name | speaker | sex | reference utterance | held-out utterance | ref duration |
|---|---|---|---|---|---|
| k8_f237 | 237 | F | 237_134493_000002_000003 | 237_134493_000005_000003 | 5.52 s |
| k8_f2961 | 2961 | F | 2961_961_000004_000021 | 2961_961_000004_000029 | 7.08 s |
| k8_f5142 | 5142 | F | 5142_33396_000039_000001 | 5142_33396_000053_000000 | 5.76 s |
| k8_f6829 | 6829 | F | 6829_68771_000005_000001 | 6829_68771_000010_000004 | 7.64 s |
| k8_m1188 | 1188 | M | 1188_133604_000004_000005 | 1188_133604_000012_000003 | 8.12 s |
| k8_m7127 | 7127 | M | 7127_75946_000002_000001 | 7127_75946_000022_000001 | 5.08 s |
| k8_m8224 | 8224 | M | 8224_274384_000000_000001 | 8224_274384_000005_000001 | 5.64 s |
| k8_m8230 | 8230 | M | 8230_279154_000003_000007 | 8230_279154_000006_000002 | 8.84 s |

Held-out clips come from the same chapter as the reference clip.

## Scorer models (downloaded at run time, not redistributed)

- `campplus.onnx` shipped inside FunAudioLLM/Fun-CosyVoice3-0.5B-2512.
- `microsoft/wavlm-base-plus-sv` (WavLM x-vector speaker verification head).
- `openai/whisper-small` (ASR for WER).
