#!/bin/bash
# Unquantize the BF16 layers of GLM-5.3-Flash that Fp8Config mis-quantizes.
#
# The checkpoint's quantization_config.modules_to_not_convert lists the KDA
# projections and the lm_head (they are stored in BF16, no scales), but the
# entries use HF weight prefixes that never match the vLLM runtime prefixes
# under Fp8Config's default exact match mode. So Fp8LinearMethod is applied
# to them anyway and their weight_scale_inv stays at the -finfo.min
# sentinel (no scale tensors are serialized for them), producing garbage
# KDA outputs and garbage logits (degenerate repetition loops).
#
# Three force-patches plus one config normalization:
#   1. models/glm5next/nvidia/kda.py   - Glm5NextLinearAttention builds its
#      projections unquantized (temporarily nulls vllm_config.quant_config
#      for the duration of super().__init__).
#   2. models/glm5next/nvidia/model.py - Glm5NextForCausalLM builds lm_head
#      unquantized (kept for modelopt_mixed only, mirroring the MLA branch).
#   3. models/glm5next/nvidia/mtp.py   - MTP SharedHead (draft lm head)
#      built unquantized so speculative proposals are sane.
#   4. models/glm5next/nvidia/model.py - Glm5NextForCausalLM normalizes
#      quant_config.ignored_layers from HF prefixes to vLLM runtime
#      prefixes so Fp8Config skips the remaining BF16 linears.
# See README.md.
set -e
MOD_DIR="$(dirname "$0")"
export VLLM_PKG_DIR="${VLLM_PKG_DIR:-/usr/local/lib/python3.12/dist-packages/vllm}"
python3 "$MOD_DIR/patch_vllm.py"
