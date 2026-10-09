#!/usr/bin/env bash
# dsv41-boot10-patches mod — install the tonyd2wild boot10 DeepSeek-V4.1-Flash patch set
# over the vLLM tree inside the container (runs pre-engine via eugr mods plumbing).
#
# Source: github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark patch/ (boot 10),
# md5-verified against his patch/README.md table at adoption (2026-09-12).
#
# HARD REQUIREMENT: the container vLLM tree must match e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba
# (merge-base of the old dsv41-feat branch; served image vllm-dsv41:overlay5, nightly
# 8a728663c1c3eeace834a95f5654fa653cc1998c + overlays 3/4/5). The patches are whole-file
# replacements DERIVED FROM that tree: mounted on later commits (incl. post-merge main,
# which reshaped the indexer around attention.py) the engine boots clean and then emits
# a repeated garbage token with zero DSpark acceptance — see his docs/RECIPE.md
# "Pin the branch commit". This mod therefore md5-checks each destination against the
# pinned-tree manifest BEFORE overwriting and refuses to patch a drifted tree.
# Escape hatch: ALLOW_TREE_DRIFT=1 env (re-derive patches from patch/*/ diffs first).
set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(python3 - <<'PY'
import pathlib
import vllm

print(pathlib.Path(vllm.__file__).resolve().parent)
PY
)"

DRIFT=0
N=0
while read -r f rel pin; do
    [ -z "$f" ] && continue
    dest="${VLLM_DIR}/${rel#vllm/}"
    if [ ! -f "$dest" ]; then
        echo "ERROR: target missing in container (not the pinned tree?): $dest" >&2
        exit 3
    fi
    cur="$(md5sum "$dest" | cut -d' ' -f1)"
    if [ "$cur" != "$pin" ]; then
        echo "TREE DRIFT: $dest md5 $cur != pinned $pin" >&2
        DRIFT=1
        continue
    fi
    install -m 0644 "${MOD_DIR}/$f" "$dest"
    python3 -m py_compile "$dest"
    echo "installed $f -> $dest"
    N=$((N + 1))
done < "${MOD_DIR}/pinned-md5s.txt"

if [ "$DRIFT" = "1" ]; then
    if [ "${ALLOW_TREE_DRIFT:-0}" != "1" ]; then
        echo "Refusing to patch a drifted vLLM tree (some/all targets mismatched)." >&2
        echo "Rebuild the image from vLLM @ e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba" >&2
        echo "(or export ALLOW_TREE_DRIFT=1 after re-deriving patches; silent garbage risk)" >&2
        exit 3
    fi
    echo "ALLOW_TREE_DRIFT=1 — installing remaining files anyway" >&2
    while read -r f rel _pin; do
        [ -z "$f" ] && continue
        dest="${VLLM_DIR}/${rel#vllm/}"
        cmp -s "${MOD_DIR}/$f" "$dest" && { echo "unchanged $f" ; continue ; }
        install -m 0644 "${MOD_DIR}/$f" "$dest"
        python3 -m py_compile "$dest"
        echo "installed(drift-ok) $f -> $dest"
    done < "${MOD_DIR}/mounts.txt"
fi

echo "dsv41 boot10 patch set applied (${N} files fresh + drift handling) at ${VLLM_DIR}"
