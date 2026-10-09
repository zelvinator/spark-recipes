"""Normalize ModelOpt NVFP4 exclusion names for GLM-5.3-Flash (glm5next).

dealignai NVFP4 exports (CRACK) list the fused KDA projection as
``*.self_attn.in_proj_qkvbfg_a`` in quantization_config.ignore, but the
vLLM runtime adapter builds that projection as ``in_proj_qkvgfab``
(models/glm5next/nvidia/kda.py), so the exclusion never matches and the
BF16 KDA projection is built quantized -> weight load aborts with
``assert destination.shape == source.shape``.

Verified on dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4 rev 745aac2f:
every ``*.self_attn.*`` tensor and every non-expert ``*.mlp.*`` tensor is
BF16 in the checkpoint; only routed experts are NVFP4. With the name
normalized, the fork's exclusion matching excludes the whole BF16 set
(verified via ModelOptNvFp4Config.from_config + apply_vllm_mapper +
is_layer_excluded in the vllm-node-b12x image).

Mirrors fix-glm5next-bf16-layers patch 4 (which does the same for the
Fp8Config ignored_layers list).
"""

import os
import sys

PATCHES = [
    (
        "models/glm5next/nvidia/model.py",
        """        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        self.model = Glm5NextModel(""",
        """        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        # [fix-glm5next-nvfp4-ignore] ModelOpt NVFP4 exports (dealignai
        # CRACK) name the fused KDA projection 'in_proj_qkvbfg_a'; this
        # adapter builds it as 'in_proj_qkvgfab', so the exclusion never
        # matches and the BF16 projection would load into an NVFP4 packed
        # parameter (assert destination.shape == source.shape). Normalize
        # before the model tree is built.
        if (
            quant_config is not None
            and getattr(quant_config, "exclude_modules", None)
        ):
            quant_config.exclude_modules = [
                "*.self_attn.in_proj_qkvgfab"
                if _exclude == "*.self_attn.in_proj_qkvbfg_a"
                else _exclude
                for _exclude in quant_config.exclude_modules
            ]
        self.model = Glm5NextModel(""",
        "[fix-glm5next-nvfp4-ignore]",
    ),
]

TAG = "[fix-glm5next-nvfp4-ignore]"


def main() -> int:
    root = os.environ.get("VLLM_PKG_DIR")
    if len(sys.argv) > 1:
        root = sys.argv[1]
    if not root:
        print(f"{TAG} ERROR: VLLM_PKG_DIR is not set and no path argument given.")
        return 1
    root = os.path.abspath(root)
    failed = False
    for rel_path, old, new, marker in PATCHES:
        path = os.path.join(root, rel_path)
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except OSError as exc:
            print(f"{TAG} FAIL {rel_path}: cannot read file: {exc}")
            failed = True
            continue
        if marker in content:
            print(f"{TAG} {rel_path}: already applied, skipping.")
            continue
        count = content.count(old)
        if count != 1:
            print(
                f"{TAG} FAIL {rel_path}: anchor found {count} times "
                "(expected exactly 1); refusing to patch a changed source."
            )
            failed = True
            continue
        patched = content.replace(old, new)
        try:
            compile(patched, path, "exec")
        except SyntaxError as exc:
            print(f"{TAG} FAIL {rel_path}: patched text does not compile: {exc}")
            failed = True
            continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(patched)
        print(f"{TAG} {rel_path}: patched.")

    if failed:
        print(f"{TAG} FAILED - see messages above.")
        return 1
    print(f"{TAG} Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
