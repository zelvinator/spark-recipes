"""Compact GLM-5Next B12X sparse-MLA selection indices before the kernel.

Targets:
  v1/attention/backends/mla/sparse_utils.py
  v1/attention/backends/mla/b12x_mla_sparse.py

GLM-5Next prefill produces 2051-wide request-local top-k selections whose
valid entries are not guaranteed to occupy the first `active_count` columns:
pooled history entries are written by selected-pool rank and tail entries sit
at offset 2048 and beyond. The B12X sparse-MLA kernel, however, consumes only
the first `active_count` entries of each row and treats the remainder as
unselected. Short rows can therefore have their only valid entry past the
consumed prefix, producing an empty attention set and non-finite output.

Patch 1 (sparse_utils.py): teach
`triton_convert_req_index_to_global_index` to compact valid entries into the
front of each row using the existing `COMPACT_TO_FRONT` kernel path.

Patch 2 (b12x_mla_sparse.py): enable that compaction in the GLM non-DCP
fallback conversion.

Strictness: every anchor must occur exactly once unless its replacement
marker is already present; each patch is independently idempotent. Patched
files must compile.
"""

import os
import sys

TAG = "[fix-glm5next-b12x-sparse-prefill-compaction]"

SPARSE_UTILS = "v1/attention/backends/mla/sparse_utils.py"
B12X_SPARSE = "v1/attention/backends/mla/b12x_mla_sparse.py"

SPARSE_SIGNATURE_OLD = """    prefill_workspace_request_ids: torch.Tensor | None = None,
    prefill_workspace_starts: torch.Tensor | None = None,
    return_valid_counts: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:"""
SPARSE_SIGNATURE_NEW = """    prefill_workspace_request_ids: torch.Tensor | None = None,
    prefill_workspace_starts: torch.Tensor | None = None,
    return_valid_counts: bool = False,
    compact_valid_to_front: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:"""
SPARSE_SIGNATURE_MARKER = "    compact_valid_to_front: bool = False,"

SPARSE_ALLOC_OLD = """    single_tile, block_n, tiles_per_row, num_warps = _remap_tiling(
        NUM_TOPK_TOKENS, BLOCK_N, return_valid_counts
    )

    # Ensure contiguous tensors on the same device
    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)

    valid_counts: torch.Tensor | None = None
    if return_valid_counts:
        # Zero-init only matters for the atomic accumulation path.
        alloc = torch.empty if single_tile else torch.zeros
        valid_counts = alloc(num_tokens, dtype=torch.int32, device=token_indices.device)"""
SPARSE_ALLOC_NEW = """    count_valid = return_valid_counts or compact_valid_to_front
    single_tile, block_n, tiles_per_row, num_warps = _remap_tiling(
        NUM_TOPK_TOKENS, BLOCK_N, count_valid
    )

    # Ensure contiguous tensors on the same device
    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    if compact_valid_to_front:
        # GLM5Next prefill compaction: pre-fill -1 so unwritten tail slots
        # stay invalid after valid entries are scattered to the front.
        out = torch.full_like(token_indices_c, -1)
    else:
        out = torch.empty_like(token_indices_c)

    valid_counts: torch.Tensor | None = None
    if count_valid:
        # Zero-init only matters for the atomic accumulation path.
        alloc = torch.empty if single_tile else torch.zeros
        valid_counts = alloc(num_tokens, dtype=torch.int32, device=token_indices.device)"""
SPARSE_ALLOC_MARKER = (
    "        # GLM5Next prefill compaction: pre-fill -1 so unwritten tail slots"
)

SPARSE_KERNEL_OLD = """        return_valid_counts,
        single_tile,
        False,  # COMPACT_TO_FRONT: keep input column == output column
        # DCP disabled (no-op de-interleave)"""
SPARSE_KERNEL_NEW = """        count_valid,
        single_tile,
        compact_valid_to_front,  # GLM5Next non-DCP selection compaction
        # DCP disabled (no-op de-interleave)"""
SPARSE_KERNEL_MARKER = (
    "        compact_valid_to_front,  # GLM5Next non-DCP selection compaction"
)

B12X_CALL_OLD = """                selected_indices, active_counts = (
                    triton_convert_req_index_to_global_index(
                        attn_metadata.req_id_per_token[:num_tokens],
                        attn_metadata.block_table,
                        topk_indices,
                        BLOCK_SIZE=attn_metadata.block_size,
                        BLOCK_STRIDE_ROWS=block_stride_rows,
                        NUM_TOPK_TOKENS=topk_indices.shape[1],
                        return_valid_counts=True,
                    )
                )"""
B12X_CALL_NEW = """                selected_indices, active_counts = (
                    triton_convert_req_index_to_global_index(
                        attn_metadata.req_id_per_token[:num_tokens],
                        attn_metadata.block_table,
                        topk_indices,
                        BLOCK_SIZE=attn_metadata.block_size,
                        BLOCK_STRIDE_ROWS=block_stride_rows,
                        NUM_TOPK_TOKENS=topk_indices.shape[1],
                        return_valid_counts=True,
                        compact_valid_to_front=True,
                    )
                )"""
B12X_CALL_MARKER = "                        compact_valid_to_front=True,"

PATCHES = {
    SPARSE_UTILS: (
        (SPARSE_SIGNATURE_OLD, SPARSE_SIGNATURE_NEW, SPARSE_SIGNATURE_MARKER),
        (SPARSE_ALLOC_OLD, SPARSE_ALLOC_NEW, SPARSE_ALLOC_MARKER),
        (SPARSE_KERNEL_OLD, SPARSE_KERNEL_NEW, SPARSE_KERNEL_MARKER),
    ),
    B12X_SPARSE: (
        (B12X_CALL_OLD, B12X_CALL_NEW, B12X_CALL_MARKER),
    ),
}


def main() -> int:
    root = os.environ.get("VLLM_PKG_DIR")
    if len(sys.argv) > 1:
        root = sys.argv[1]
    if not root:
        print(f"{TAG} ERROR: VLLM_PKG_DIR is not set and no path argument given.")
        return 1
    root = os.path.abspath(root)

    failed = False
    for rel_path, patches in PATCHES.items():
        path = os.path.join(root, rel_path)
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except OSError as exc:
            print(f"{TAG} FAIL {rel_path}: cannot read file: {exc}")
            failed = True
            continue

        patched = content
        file_failed = False
        changed = False
        for old, new, marker in patches:
            if marker in patched:
                print(f"{TAG} {rel_path}: patch already applied, skipping.")
                continue
            count = patched.count(old)
            if count != 1:
                print(
                    f"{TAG} FAIL {rel_path}: anchor found {count} times "
                    "(expected exactly 1); refusing to patch a changed source."
                )
                file_failed = True
                break
            patched = patched.replace(old, new)
            changed = True

        if file_failed:
            failed = True
            continue

        if changed:
            try:
                compile(patched, path, "exec")
            except SyntaxError as exc:
                print(f"{TAG} FAIL {rel_path}: patched text does not compile: {exc}")
                failed = True
                continue
            with open(path, "w", encoding="utf-8") as f:
                f.write(patched)
            print(f"{TAG} {rel_path}: patched.")
        else:
            print(f"{TAG} {rel_path}: already applied, skipping.")

    if failed:
        print(f"{TAG} FAILED — see messages above.")
        return 1
    print(f"{TAG} Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
