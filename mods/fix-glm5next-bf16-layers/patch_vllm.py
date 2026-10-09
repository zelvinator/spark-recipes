"""Unquantize the BF16 layers of GLM-5.3-Flash that Fp8Config mis-quantizes.

Target: models/glm5next (nvidia variant).

GLM-5.3-Flash is a mixed checkpoint: MoE experts, dense MLPs and some MLA
projections are block-FP8 (weight_block_size [128,128]), while the KDA
projections, the MLA q_b/kv_b projections and the lm_head are BF16 and are
listed in ``quantization_config.modules_to_not_convert``.

Fp8Config.get_quant_method consults that list via is_layer_skipped, but the
list uses HF weight prefixes (``model.layers.N...``, bare ``lm_head``) while
the vLLM runtime prefixes are ``language_model.model.layers.N...`` /
``language_model.lm_head`` (the rename mapper only rewrites
``model.language_model.`` -> ``language_model.model.``). With the default
"exact" match mode nothing in the list ever matches, so Fp8LinearMethod is
applied to the BF16 layers. Their ``weight_scale_inv`` is initialized to
torch.finfo(float32).min and is never loaded from the checkpoint (no scales
are serialized for them), so the dequantized weights are garbage and the
model output degenerates (repetition loops).

Patches (all idempotent, anchor-strict):
  1. nvidia/kda.py      - Glm5NextLinearAttention.__init__ temporarily sets
                          vllm_config.quant_config to None while building the
                          layer, so all KDA projections (in_proj_qkvgfab,
                          g_a, g_b, f_b, o_proj, conv1d) are built
                          unquantized (BF16) as the checkpoint intends.
  2. nvidia/model.py    - Glm5NextForCausalLM lm_head is built unquantized
                          (kept quantized only for modelopt_mixed configs,
                          mirroring the existing MLA branch).
  3. nvidia/mtp.py      - the MTP SharedHead (draft lm head) is built
                           unquantized the same way, so speculative
                           proposals are sane.
  4. nvidia/model.py    - Glm5NextForCausalLM rewrites
                           quant_config.ignored_layers from HF prefixes to
                           vLLM runtime prefixes, so Fp8Config's own
                           is_layer_skipped check skips every BF16 linear
                           the checkpoint declares unquantized.

Every anchor must occur exactly once; a file that already contains the
replacement marker is skipped (safe to re-run in a fresh container).
Patched files must compile.
"""

import os
import sys

PATCHES = [
    (
        "models/glm5next/nvidia/kda.py",
        """    enable_b12x_kda_decode = True
    b12x_kda_null_state_index = 0""",
        """    enable_b12x_kda_decode = True
    b12x_kda_null_state_index = 0

    def __init__(self, config, vllm_config, prefix=""):
        # GLM-5.3-Flash checkpoints keep all KDA projections in BF16
        # (quantization_config.modules_to_not_convert), but that list uses
        # HF weight prefixes which never match the vLLM runtime prefixes,
        # so Fp8Config.get_quant_method quantizes these layers anyway. The
        # checkpoint serializes no scales for them, so weight_scale_inv
        # stays at the -finfo.min sentinel and the KDA output is garbage.
        # Build the layer unquantized instead (the MLA path already does
        # this for the same reason).
        original_quant_config = vllm_config.quant_config
        vllm_config.quant_config = None
        try:
            super().__init__(config, vllm_config, prefix)
        finally:
            vllm_config.quant_config = original_quant_config""",
        "weight_scale_inv",
    ),
    (
        "models/glm5next/nvidia/model.py",
        """        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )""",
        """        if get_pp_group().is_last_rank:
            # GLM-5.3-Flash serializes the lm_head in BF16 and lists it in
            # quantization_config.modules_to_not_convert, but that entry
            # never matches the runtime prefix (exact match), so build the
            # head unquantized (mirroring the MLA branch above).
            lm_head_quant_config = (
                quant_config
                if quant_config is not None
                and quant_config.get_name() == "modelopt_mixed"
                else None
            )
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=lm_head_quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )""",
        "lm_head_quant_config",
    ),
    (
        "models/glm5next/nvidia/mtp.py",
        """        self.shared_head = SharedHead(
            config=config, prefix=prefix, quant_config=quant_config
        )""",
        """        # The MTP head mirrors the base lm_head: BF16 in the checkpoint
        # (modules_to_not_convert), so build it unquantized too.
        shared_head_quant_config = (
            quant_config
            if quant_config is not None
            and quant_config.get_name() == "modelopt_mixed"
            else None
        )
        self.shared_head = SharedHead(
            config=config,
            prefix=prefix,
            quant_config=shared_head_quant_config,
        )""",
        "shared_head_quant_config",
    ),
    (
        "models/glm5next/nvidia/model.py",
        """        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        self.model = Glm5NextModel(""",
        """        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        # The checkpoint's modules_to_not_convert uses HF prefixes
        # (model.layers.N..., bare lm_head) that neither exact-match the
        # runtime prefixes nor survive apply_vllm_mapper (the mapper only
        # rewrites model.language_model.). Normalize them here so Fp8Config
        # skips every BF16 linear the checkpoint declares unquantized, not
        # just the ones force-patched above.
        if (
            quant_config is not None
            and quant_config.get_name() == "fp8"
            and getattr(quant_config, "ignored_layers", None)
        ):
            normalized_ignored_layers = []
            for _layer in quant_config.ignored_layers:
                if _layer.startswith("language_model."):
                    normalized_ignored_layers.append(_layer)
                elif _layer.startswith("model.language_model."):
                    normalized_ignored_layers.append(
                        "language_model.model."
                        + _layer[len("model.language_model."):]
                    )
                elif _layer.startswith("model."):
                    normalized_ignored_layers.append(
                        "language_model.model." + _layer[len("model."):]
                    )
                elif _layer == "lm_head":
                    normalized_ignored_layers.append("language_model.lm_head")
                else:
                    normalized_ignored_layers.append(_layer)
            quant_config.ignored_layers = normalized_ignored_layers
        self.model = Glm5NextModel(""",
        "normalized_ignored_layers",
    ),
]

TAG = "[fix-glm5next-bf16-layers]"


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
        print(f"{TAG} FAILED — see messages above.")
        return 1
    print(f"{TAG} Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
