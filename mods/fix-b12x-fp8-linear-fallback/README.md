# fix-b12x-fp8-linear-fallback

Targeted runtime patch for the b12x vLLM fork (`vllm-node-b12x` image):
block-FP8 scaled-mm kernel selection for TP-sharded linears whose output
features are not a multiple of 128.

## Why

GLM-5.3-Flash (`glm5_next`) KDA layers merge q/k/v/beta/f_a into one
`MergedColumnParallelLinear` (`in_proj_qkvgfab`). With TP=4 the local
output is `3*2048 + 16 + 128 = 6288` features (6288 % 128 = 16). All
other linears in the model are 128-aligned.

- `--linear-backend b12x` filters the candidate list to B12X kernels
  only; B12X requires 128-aligned output, and the fork raises
  `ValueError: Failed to find a kernel ...` at kernel selection (no
  per-layer fallback).
- Without the flag (auto mode), `CutlassFp8BlockScaledMMKernel` is
  selected for N=6288 (FlashInfer block-scale is SM90-only, DeepGEMM
  needs N%64), but its `can_implement` never checks N. The SM120/121
  CUTLASS blockwise GEMM (tile N=128, incl. the swap-AB small-M config)
  then fails `gemm_op.can_implement(args)` at runtime:
  `cutlass_gemm_caller ... c3x/cutlass_gemm_caller.cuh:52, Invalid
  status`, killing the worker on the first forward pass.

## What it patches

Both patches target the installed package
(`/usr/local/lib/python3.12/dist-packages/vllm`, override with
`VLLM_PKG_DIR`):

1. `model_executor/kernels/linear/scaled_mm/cutlass.py` —
   `CutlassFp8BlockScaledMMKernel.can_implement` rejects output/input
   features that are not multiples of 128, matching the CUTLASS c3x
   constraint enforced at runtime.
2. `model_executor/kernels/linear/__init__.py` —
   `choose_scaled_mm_linear_kernel` retries against the full platform
   kernel list when the `--linear-backend`-filtered list cannot
   implement the layer (warning logged), instead of raising.

Effect for GLM-5.3-Flash at TP=4 with `--linear-backend b12x`: aligned
layers keep the B12X GEMM; only the 34 KDA `in_proj` layers (plus the
MTP draft's, if present) fall back through the default priority list
and land on Marlin block-FP8 (SM121 is CC >= 7.5).

## Behavior

`run.sh` applies `patch_vllm.py`, which:

- resolves the package root from `VLLM_PKG_DIR`
  (default: dist-packages path above),
- requires each anchor to occur exactly once and refuses to patch a
  changed source,
- skips files that already carry the replacement marker (safe to re-run
  on a freshly created container of the same image),
- compile-checks each patched file.

No other repository or container state is modified.
