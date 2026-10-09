"""Disable the B12X KDA decode path in GLM-5.3 to isolate degenerate output.

Target: models/glm5next (nvidia variant).

Glm5NextLinearAttention forces enable_b12x_kda_decode = True, which routes
every pure-decode KDA batch through b12x.sequence.gdn_decode instead of the
reference fused_recurrent_kda_packed_decode path. Setting the class attribute
to False makes _can_use_b12x_kda_decode() return False while leaving the
prefill backend unchanged.

This mod must run after mods/fix-glm5next-bf16-layers, whose anchor is the
same two class-attribute lines with the attribute set to True.
"""

import os
import sys

OLD = """    enable_b12x_kda_decode = True
    b12x_kda_null_state_index = 0"""
NEW = """    enable_b12x_kda_decode = False
    b12x_kda_null_state_index = 0"""
MARKER = "enable_b12x_kda_decode = False"
REL_PATH = "models/glm5next/nvidia/kda.py"
TAG = "[diag-glm5next-no-b12x-kda-decode]"


def main() -> int:
    root = os.environ.get("VLLM_PKG_DIR")
    if len(sys.argv) > 1:
        root = sys.argv[1]
    if not root:
        print(f"{TAG} ERROR: VLLM_PKG_DIR is not set and no path argument given.")
        return 1
    root = os.path.abspath(root)

    path = os.path.join(root, REL_PATH)
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
    except OSError as exc:
        print(f"{TAG} FAIL {REL_PATH}: cannot read file: {exc}")
        return 1

    if MARKER in content:
        print(f"{TAG} {REL_PATH}: already applied, skipping.")
        return 0

    count = content.count(OLD)
    if count != 1:
        print(
            f"{TAG} FAIL {REL_PATH}: anchor found {count} times "
            "(expected exactly 1); refusing to patch a changed source."
        )
        return 1

    patched = content.replace(OLD, NEW)
    try:
        compile(patched, path, "exec")
    except SyntaxError as exc:
        print(f"{TAG} FAIL {REL_PATH}: patched text does not compile: {exc}")
        return 1

    with open(path, "w", encoding="utf-8") as f:
        f.write(patched)
    print(f"{TAG} {REL_PATH}: patched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
