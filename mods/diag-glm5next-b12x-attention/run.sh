#!/bin/bash
# Probe B12X sparse-MLA layer-3 state and the MLA v_up/o_proj post-attention
# projections for GLM-5Next.
#
# Patches B12xGLM5NextMLASparseBackend.forward_mqa, MLAAttention v_up, and
# MultiHeadLatentAttentionWrapper o_proj to report layer-3 tensor state.
# Run the recipe with --enforce-eager.
set -e
MOD_DIR="$(dirname "$0")"
export VLLM_PKG_DIR="${VLLM_PKG_DIR:-/usr/local/lib/python3.12/dist-packages/vllm}"
python3 "$MOD_DIR/patch_vllm.py"
