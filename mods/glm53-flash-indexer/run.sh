#!/bin/bash
# Installs the SM12x sparse-indexer topk gate for GLM-5.3 (Glm5Next) on GB10.
#
# Why: the vLLM build in radixark/vllm-glm53-flash:sm121-v8 selects
# torch.ops._C.persistent_topk for the sparse-attention indexer decode path
# whenever select_k is 512/1024/2048 (GLM-5.3: index_topk=2048, index_kpool=4
# -> select_k=512). That kernel cannot run on GB10 (sm_121): once a decode
# step's context grows its CTA demand exceeds num_sms*occupancy (48 on GB10),
# and its FilteredTopK fallback needs >=128KB smem per block while GB10 has
# 101376B. It crashes the worker under load, with or without CUDA graphs.
#
# The two files below are drop-in replacements for the image's vLLM copies;
# the only change is adding `not is_device_capability_family(120)` to the
# topk selection so SM12x falls back to top_k_per_row_* instead (verified on
# GB10 by Wpnx330/GLM-5.3-Flash-FP8-4x-DGX-Spark, patches/). Behavior on
# other architectures is unchanged.
set -euo pipefail

PREFIX="[glm53-flash-indexer]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
LAYER_ROOT="$PYTHON_ROOT/vllm/model_executor/layers"

echo "=== GLM-5.3 sparse-indexer SM12x topk-gate mod ==="

if [[ ! -d "$LAYER_ROOT" ]]; then
    echo "$PREFIX vLLM layers directory not found at $LAYER_ROOT" >&2
    exit 1
fi

for f in sparse_attn_indexer.py sparse_attn_indexer_kpool.py; do
    if [[ ! -f "$MOD_DIR/$f" ]]; then
        echo "$PREFIX required mod file not found: $MOD_DIR/$f" >&2
        exit 1
    fi
done

cp "$MOD_DIR/sparse_attn_indexer.py" "$LAYER_ROOT/sparse_attn_indexer.py"
cp "$MOD_DIR/sparse_attn_indexer_kpool.py" "$LAYER_ROOT/sparse_attn_indexer_kpool.py"

# Verify the installed gates are actually in place (an unpatched image copy
# would already contain one family-120 gate in sparse_attn_indexer.py, so a
# plain grep is not sufficient).
python3 - "$LAYER_ROOT" <<'PY'
import sys
from pathlib import Path

layer_root = Path(sys.argv[1])

plain = (layer_root / "sparse_attn_indexer.py").read_text()
start = plain.index("use_persistent_topk = (")
end = plain.index("if use_cooperative_topk:")
assert "is_device_capability_family(120)" in plain[start:end], (
    "sparse_attn_indexer.py: use_persistent_topk is not gated off for SM12x"
)

kpool = (layer_root / "sparse_attn_indexer_kpool.py").read_text()
start = kpool.index("select_k in (512, 1024, 2048)")
end = kpool.index("torch.ops._C.persistent_topk(", start)
assert "is_device_capability_family(120)" in kpool[start:end], (
    "sparse_attn_indexer_kpool.py: decode persistent_topk is not gated off for SM12x"
)
print("[glm53-flash-indexer] SM12x topk gates verified in both indexer modules")
PY

# Drop stale bytecode for the replaced modules.
find "$LAYER_ROOT/__pycache__" -maxdepth 1 -name 'sparse_attn_indexer*.pyc' -delete 2>/dev/null || true

echo "=== OK: SM12x (GB10) indexer topk now uses top_k_per_row_* ==="
