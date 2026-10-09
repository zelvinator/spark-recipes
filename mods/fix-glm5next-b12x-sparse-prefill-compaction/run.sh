#!/bin/bash
# Compact GLM-5Next B12X sparse-MLA selection indices into the first
# active_count entries of each row before the B12X sparse-MLA kernel
# consumes them. GLM prefill places tail selections at offset 2048+, so the
# un-compacted fallback conversion can give the kernel an empty prefix.
set -e
MOD_DIR="$(dirname "$0")"
export VLLM_PKG_DIR="${VLLM_PKG_DIR:-/usr/local/lib/python3.12/dist-packages/vllm}"
python3 "$MOD_DIR/patch_vllm.py"
