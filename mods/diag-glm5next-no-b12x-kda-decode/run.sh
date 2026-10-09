#!/bin/bash
# Disable the B12X KDA decode path in GLM-5.3 to isolate degenerate output.
#
# The GLM KDA adapter forces enable_b12x_kda_decode = True. This mod flips it
# to False so pure-decode KDA batches use the reference Triton
# fused_recurrent_kda_packed_decode path instead of b12x.sequence.gdn_decode.
# Prefill is unchanged. Apply after mods/fix-glm5next-bf16-layers.
set -e
MOD_DIR="$(dirname "$0")"
export VLLM_PKG_DIR="${VLLM_PKG_DIR:-/usr/local/lib/python3.12/dist-packages/vllm}"
python3 "$MOD_DIR/patch_vllm.py"
