# Audio examples

Every file named MODEL-GENERATED_* is speech synthesized by Fun-CosyVoice3-0.5B-2512 served by vLLM-Omni during this measurement. It is not a recording of any person.
Files under reference/ are the public reference clips the voices were cloned from (see ATTRIBUTION below).

| file | run | idx | requested voice | CAM++ heard | WavLM heard | CAM++ margin | WER | ASR | note |
|---|---|---|---|---|---|---|---|---|---|
| MODEL-GENERATED_c85f1a4_K8_C8_leak_idx108_requested-k8_m1188_heard-k8_f6829.wav | c85f1a4_async/K8_C8_s1 | 108 | k8_m1188 | k8_f6829 | k8_f6829 | -0.5812 | 0.0 | The weather is lovely today with warm sunshine and a gentle breeze, so we decide | leak (both scorers) |
| MODEL-GENERATED_c85f1a4_K8_C8_leak_idx059_requested-k8_f6829_heard-k8_m1188.wav | c85f1a4_async/K8_C8_s1 | 59 | k8_f6829 | k8_m1188 | k8_m1188 | -0.5584 | 0.0 | After the long winter, the farmers were glad to see the first green shoots appea | leak (both scorers) |
| MODEL-GENERATED_c85f1a4_K8_C8_leak_idx077_requested-k8_m7127_heard-k8_m1188.wav | c85f1a4_async/K8_C8_s1 | 77 | k8_m7127 | k8_m1188 | k8_m1188 | -0.5178 | 0.0 | The weather is lovely today with warm sunshine and a gentle breeze, so we decide | leak (both scorers) |
| MODEL-GENERATED_bbee488_K8_C8_same-request_idx108_requested-k8_m1188_heard-k8_m1188.wav | bbee488_async/K8_C8_s1 | 108 | k8_m1188 | k8_m1188 | k8_m1188 | 0.3998 | 0.0 | The weather is lovely today with warm sunshine and a gentle breeze, so we decide | same request and seed as the c85f1a4 leak above |
| MODEL-GENERATED_bbee488_K8_C8_same-request_idx059_requested-k8_f6829_heard-k8_f6829.wav | bbee488_async/K8_C8_s1 | 59 | k8_f6829 | k8_f6829 | k8_f6829 | 0.2262 | 0.0 | After the long winter, the farmers were glad to see the first green shoots appea | same request and seed as the c85f1a4 leak above |
| MODEL-GENERATED_c85f1a4_K2_C8_leak_idx090_requested-k2_zero_shot_heard-k2_cross_lingual.wav | c85f1a4_async/K2_C8_s1 | 90 | k2_zero_shot | k2_cross_lingual | k2_cross_lingual | -0.8147 | 0.0 | The weather is lovely today with warm sunshine and a gentle breeze, so we decide | leak (both scorers) |
| MODEL-GENERATED_c85f1a4_K2_C8_leak_idx052_requested-k2_zero_shot_heard-k2_cross_lingual.wav | c85f1a4_async/K2_C8_s1 | 52 | k2_zero_shot | k2_cross_lingual | k2_cross_lingual | -0.7071 | 0.0 | Please remember to bring your notebook and a pen to the meeting tomorrow morning | leak (both scorers) |
| MODEL-GENERATED_bbee488_K2_C8_same-request_idx090_requested-k2_zero_shot_heard-k2_zero_shot.wav | bbee488_async/K2_C8_s1 | 90 | k2_zero_shot | k2_zero_shot | k2_zero_shot | 0.5535 | 0.0 | The weather is lovely today with warm sunshine and a gentle breeze, so we decide | same request and seed as the c85f1a4 leak above |
| MODEL-GENERATED_c85f1a4_K2_C8_extra-words_idx101_requested-k2_cross_lingual_heard-k2_cross_lingual.wav | c85f1a4_async/K2_C8_s3 | 101 | k2_cross_lingual | k2_cross_lingual | k2_cross_lingual | 0.7222 | 0.2143 | Please remember to bring your notebook and a pen to the meeting tomorrow morning | correct voice, extra words at the end (#8288-like), single instance |
| MODEL-GENERATED_bbee488_K2_C8_same-request_idx101_requested-k2_cross_lingual_heard-k2_cross_lingual.wav | bbee488_async/K2_C8_s3 | 101 | k2_cross_lingual | k2_cross_lingual | k2_cross_lingual | 0.7105 | 0.0 | Please remember to bring your notebook and a pen to the meeting tomorrow morning | same request and seed, bbee488 |
| MODEL-GENERATED_c85f1a4_K2_C8_babble_idx049_requested-k2_cross_lingual_heard-k2_cross_lingual.wav | c85f1a4_async/K2_C8_s3 | 49 | k2_cross_lingual | k2_cross_lingual | k2_zero_shot | 0.2048 | 1.0 | Thanks. | mostly non-speech, WER 1.0; same request also fails at bbee488 |
| MODEL-GENERATED_bbee488_K2_C8_babble_idx049_requested-k2_cross_lingual_heard-k2_cross_lingual.wav | bbee488_async/K2_C8_s3 | 49 | k2_cross_lingual | k2_cross_lingual | k2_zero_shot | 0.1354 | 0.8462 | Thank you. | mostly non-speech, WER 0.85; WavLM-only wrong flag |
| MODEL-GENERATED_bbee488_mns4_K8_C8_idx000_requested-k8_f237_heard-k8_f237.wav | bbee488_async_mns4/K8_C8_s1 | 0 | k8_f237 | k8_f237 | k8_f237 | 0.2519 | 0.0 | Thank you so much for the birthday gift you sent me last week. | max_num_seqs=4 talker, queued admission; correct |
| MODEL-GENERATED_bbee488_mns4_K8_C8_idx009_requested-k8_f2961_heard-k8_f2961.wav | bbee488_async_mns4/K8_C8_s1 | 9 | k8_f2961 | k8_f2961 | k8_f2961 | 0.3428 | 0.0 | The weather is lovely today with warm sunshine and a gentle breeze, so we decide | max_num_seqs=4 talker, queued admission; correct |
| MODEL-GENERATED_bbee488_prefix_K8_C8_pre-crash_idx000_requested-k8_f237_heard-k8_f237.wav | bbee488_async_prefix/K8_C8_s1 | 0 | k8_f237 | k8_f237 | k8_f237 | 0.2921 | 0.0 | Thank you so much for the birthday gift you sent me last week. | prefix caching on, before the talker crashed; correct |

## ATTRIBUTION

- k2_*: FunAudioLLM/CosyVoice asset/ (Apache-2.0).
- k8_*: LibriTTS-R test-clean via mythicinfinity/libritts_r (CC BY 4.0); utterance IDs in voices_k8.json / cv3_verify/ATTRIBUTION.md.
