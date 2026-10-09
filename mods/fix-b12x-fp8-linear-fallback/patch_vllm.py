"""Patch the b12x vLLM fork's block-FP8 scaled-mm kernel selection.

Targets:
  model_executor/kernels/linear/scaled_mm/cutlass.py
  model_executor/kernels/linear/__init__.py

Patch 1 (cutlass.py): CutlassFp8BlockScaledMMKernel.can_implement now
rejects layers whose output/input features are not multiples of 128. The
SM90/SM12x CUTLASS blockwise FP8 GEMM tiles are 128 wide in N (and the
swap-AB small-M config tiles 128 in the N dimension as well), so
gemm_op.can_implement(args) fails at runtime for other shapes and the
worker dies with "cutlass_gemm_caller ... Invalid status". Without this
check, a forced --linear-backend selection (or auto selection) happily
picks the kernel and crashes the engine on the first forward pass.

Patch 2 (linear/__init__.py): choose_scaled_mm_linear_kernel now retries
against the full platform kernel list when the --linear-backend-filtered
list cannot implement the layer, instead of raising ValueError. Aligned
layers still get the forced backend (b12x); only shapes it cannot serve
(e.g. GLM-5.3-Flash KDA in_proj_qkvgfab, 6288 output features per TP
shard) fall back through the normal priority list.

Strictness: every anchor must occur exactly once; a file that already
contains the replacement marker is skipped (safe to re-run in a fresh
container). Patched files must compile.
"""

import os
import sys

PATCHES = [
    (
        "model_executor/kernels/linear/scaled_mm/cutlass.py",
        """    @classmethod
    def can_implement(cls, config: FP8ScaledMMLinearLayerConfig):
        can_implement_base, reason = super().can_implement(config)
        if not can_implement_base:
            return can_implement_base, reason

        act_quant_desc = config.activation_quant_key.scale
        if act_quant_desc.group_shape != GroupShape(1, 128):
            return (
                False,
                "Supports only dynamic per token group activation "
                "quantization with group_shape=(1,128).",
            )
        return True, None""",
        """    @classmethod
    def can_implement(cls, config: FP8ScaledMMLinearLayerConfig):
        can_implement_base, reason = super().can_implement(config)
        if not can_implement_base:
            return can_implement_base, reason

        act_quant_desc = config.activation_quant_key.scale
        if act_quant_desc.group_shape != GroupShape(1, 128):
            return (
                False,
                "Supports only dynamic per token group activation "
                "quantization with group_shape=(1,128).",
            )
        out_features, in_features = config.weight_shape
        if out_features % 128 != 0 or in_features % 128 != 0:
            return (
                False,
                "CUTLASS block-FP8 GEMM requires 128-aligned "
                "output and input features.",
            )
        return True, None""",
        "CUTLASS block-FP8 GEMM requires 128-aligned",
    ),
    (
        "model_executor/kernels/linear/__init__.py",
        """    platform_kernels = possible_kernels.get(current_platform._enum, [])

    # Apply --linear-backend filtering when set.
    platform_kernels = _resolve_backend_kernels(platform_kernels, "scaled-mm")

    for kernel in platform_kernels:
        is_supported_and_can_implement, failure_reason = (
            is_supported_and_can_implement_kernel(kernel, config, compute_capability)
        )
        if is_supported_and_can_implement:
            return kernel
        failure_reason_list.append(failure_reason)

    raise ValueError(
        "Failed to find a kernel that can implement the "
        "ScaledMM linear layer. Reasons: \\n" + "\\n".join(failure_reason_list)
    )""",
        """    platform_kernels = possible_kernels.get(current_platform._enum, [])

    # Apply --linear-backend filtering when set.
    filtered_platform_kernels = _resolve_backend_kernels(
        platform_kernels, "scaled-mm"
    )

    # A forced backend whose kernels cannot implement this layer falls
    # back to the full platform kernel list instead of failing the launch.
    candidate_lists = (
        [filtered_platform_kernels, platform_kernels]
        if filtered_platform_kernels is not platform_kernels
        else [platform_kernels]
    )
    for candidate_kernels in candidate_lists:
        for kernel in candidate_kernels:
            is_supported_and_can_implement, failure_reason = (
                is_supported_and_can_implement_kernel(
                    kernel, config, compute_capability
                )
            )
            if is_supported_and_can_implement:
                return kernel
            failure_reason_list.append(failure_reason)
        if len(candidate_lists) > 1:
            logger.warning(
                "--linear-backend=%s cannot implement this ScaledMM layer; "
                "falling back to the default kernel list.",
                _get_linear_backend(),
            )

    raise ValueError(
        "Failed to find a kernel that can implement the "
        "ScaledMM linear layer. Reasons: \\n" + "\\n".join(failure_reason_list)
    )""",
        "falling back to the default kernel list.",
    ),
]

TAG = "[fix-b12x-fp8-linear-fallback]"


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
