#!/bin/bash
# Normalize ModelOpt NVFP4 exclusion names for GLM-5.3-Flash (glm5next):
# dealignai NVFP4 exports name the fused KDA projection
# '*.self_attn.in_proj_qkvbfg_a' while the runtime adapter builds
# 'in_proj_qkvgfab', so the BF16 KDA projection was built quantized and
# weight load aborted with `assert destination.shape == source.shape`.
# See patch_vllm.py for the full write-up.
set -e
MOD_DIR="$(dirname "$0")"
export VLLM_PKG_DIR="${VLLM_PKG_DIR:-/usr/local/lib/python3.12/dist-packages/vllm}"
python3 "$MOD_DIR/patch_vllm.py"
