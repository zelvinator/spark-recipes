#!/bin/bash
# Fix b12x fork block-FP8 scaled-mm kernel selection for TP-sharded linears
# whose output is not a multiple of 128 (GLM-5.3-Flash KDA in_proj: 6288 at
# TP=4). Two patches, both idempotent:
#   1. CutlassFp8BlockScaledMMKernel.can_implement must require 128-aligned
#      N/K — the SM90/SM12x CUTLASS blockwise GEMM enforces it at runtime
#      ("cutlass_gemm_caller ... Invalid status") but the Python side never
#      checked it, so the kernel was selected and then crashed the worker.
#   2. choose_scaled_mm_linear_kernel must fall back to the full platform
#      kernel list when the --linear-backend-forced kernels cannot implement
#      the layer, instead of failing the launch.
# See README.md.
set -e
MOD_DIR="$(dirname "$0")"
export VLLM_PKG_DIR="${VLLM_PKG_DIR:-/usr/local/lib/python3.12/dist-packages/vllm}"
python3 "$MOD_DIR/patch_vllm.py"
