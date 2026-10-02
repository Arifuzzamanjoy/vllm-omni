# Deviation: PyTorch top-k/top-p sampler instead of FlashInfer

vLLM 0.30.0 samples top-k/top-p with FlashInfer by default. FlashInfer
JIT-compiles its sampling kernel on first use and needs `nvcc`; this VM has no
CUDA toolkit, so the first c85f1a4 server died at startup:

    flashinfer/jit/cpp_ext.py get_cuda_path
    RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist

(traceback and log in superseded/). Every server in this measurement was
therefore started with `VLLM_USE_FLASHINFER_SAMPLER=0`, which selects vLLM's
PyTorch top-k/top-p sampler. The setting is identical for all three commits
and is recorded in env/<commit>/env.txt.

Why it should not affect the comparison: neither bug under test goes through
the sampler. #8235 is per-request conditioning routed to the wrong batch row
(to_payload_element / prompt conditioning alignment); #8288 is the prompt
rearrangement in CosyVoice3Model.embed_input_ids assuming a batch order. Both
are upstream of sampling.

What it does change: token sampling numerics, so audio is not comparable byte
for byte with a FlashInfer-sampler deployment, and leak rates on a default
deployment were not measured here.
