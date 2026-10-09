#!/bin/bash
# Report the first non-finite GLM-5.3 decoder-layer state.
#
# Patches Glm5NextModel.forward to print the first few layers whose
# hidden/residual/post/comb state contains a non-finite value, including the
# layer index and KDA/MLA kind. It also instruments the mHC decoder-layer
# path around layer 3 to identify the offending mHC/attention/MLP stage.
# Run the recipe with --enforce-eager.
set -e
MOD_DIR="$(dirname "$0")"
export VLLM_PKG_DIR="${VLLM_PKG_DIR:-/usr/local/lib/python3.12/dist-packages/vllm}"
python3 "$MOD_DIR/patch_vllm.py"
