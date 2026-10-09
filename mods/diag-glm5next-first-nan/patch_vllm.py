"""Report the first non-finite tensor produced by a GLM-5.3 decoder layer.

Target: models/glm5next (nvidia variant).

The rev 8 build still produces degenerate output after the BF16-layer and
ignored-layer fixes. This diagnostic patches Glm5NextModel.forward so that,
outside CUDA-graph capture, the first few layers whose hidden/residual/post/comb
state contains a non-finite value are reported with their layer index and
attention kind (kda/mla).

The coarse run showed layer 3 (MLA) as the first bad layer with a bad
mHC residual stream. This diagnostic therefore also instruments the mHC
decoder-layer path around layer 3 to identify whether the non-finite state
appears after mHC pre, attention, mHC post/pre, or the MLP.

Run with --enforce-eager so the checks execute on every forward pass.
"""

import os
import sys

GLOBAL_OLD = "class Glm5NextMLP(nn.Module):"
GLOBAL_NEW = """_diag_glm5next_stage_reports = 0


class Glm5NextMLP(nn.Module):"""

OLD = """        for layer_idx, layer in enumerate(self._active_layers, start=self.start_layer):
            hidden_states, residual, post, comb = layer(
                positions, hidden_states, residual, post, comb
            )"""
NEW = """        _diag_glm5next_first_nan_reports = 0
        for layer_idx, layer in enumerate(self._active_layers, start=self.start_layer):
            hidden_states, residual, post, comb = layer(
                positions, hidden_states, residual, post, comb
            )
            if (
                _diag_glm5next_first_nan_reports < 8
                and not torch.cuda.is_current_stream_capturing()
                and (
                    not torch.isfinite(hidden_states).all()
                    or (residual is not None and not torch.isfinite(residual).all())
                    or (post is not None and not torch.isfinite(post).all())
                    or (comb is not None and not torch.isfinite(comb).all())
                )
            ):
                _diag_glm5next_first_nan_reports += 1
                print(
                    "[diag-glm5next-first-nan] "
                    f"layer={layer_idx} "
                    f"kind={getattr(layer, 'layer_kind', '?')} "
                    f"hidden_bad={bool(not torch.isfinite(hidden_states).all())} "
                    f"residual_bad={bool(residual is not None and not torch.isfinite(residual).all())} "
                    f"post_bad={bool(post is not None and not torch.isfinite(post).all())} "
                    f"comb_bad={bool(comb is not None and not torch.isfinite(comb).all())}",
                    flush=True,
                )"""

LAYER_PRE_OLD = """        # Attention needs the full token sequence; mHC above ran on the SP
        # shard. Gather for attention, scatter back afterward (DSv4 pattern).
        if self.is_sequence_parallel:
            x = sp_all_gather(x)[: positions.shape[0]]

        x = self.self_attn(
            hidden_states=x,
            positions=positions,
        )

        if self.is_sequence_parallel:
            x = sp_reduce_scatter(x)"""
LAYER_PRE_NEW = """        def _diag_glm5next_check_stage(stage, **tensors):
            global _diag_glm5next_stage_reports
            if (
                self.layer_idx != 3
                or _diag_glm5next_stage_reports >= 24
                or torch.cuda.is_current_stream_capturing()
            ):
                return
            bad = [
                name
                for name, value in tensors.items()
                if value is not None and not torch.isfinite(value).all()
            ]
            if bad:
                _diag_glm5next_stage_reports += 1
                print(
                    "[diag-glm5next-first-nan] "
                    f"layer={self.layer_idx} "
                    f"kind={self.layer_kind} "
                    f"stage={stage} "
                    f"bad={','.join(bad)}",
                    flush=True,
                )

        _diag_glm5next_check_stage(
            "after_mhc_pre",
            residual=residual,
            post=post,
            comb=comb,
            x=x,
        )

        # Attention needs the full token sequence; mHC above ran on the SP
        # shard. Gather for attention, scatter back afterward (DSv4 pattern).
        if self.is_sequence_parallel:
            x = sp_all_gather(x)[: positions.shape[0]]

        x = self.self_attn(
            hidden_states=x,
            positions=positions,
        )

        _diag_glm5next_check_stage("after_attn", x=x)

        if self.is_sequence_parallel:
            x = sp_reduce_scatter(x)

        _diag_glm5next_check_stage("after_attn_scatter", x=x)"""

LAYER_FFN_OLD = """        # Fuse post-attn hc_post + pre-FFN hc_pre (+ RMSNorm) into one kernel.
        residual, post, comb, x = self.hc_fused_post_pre(
            x,
            residual,
            post,
            comb,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            norm_weight=self.post_attention_layernorm.weight,
            norm_eps=self.post_attention_layernorm.variance_epsilon,
        )"""
LAYER_FFN_NEW = """        # Fuse post-attn hc_post + pre-FFN hc_pre (+ RMSNorm) into one kernel.
        residual, post, comb, x = self.hc_fused_post_pre(
            x,
            residual,
            post,
            comb,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            norm_weight=self.post_attention_layernorm.weight,
            norm_eps=self.post_attention_layernorm.variance_epsilon,
        )

        _diag_glm5next_check_stage(
            "after_mhc_post_pre",
            residual=residual,
            post=post,
            comb=comb,
            x=x,
        )"""

LAYER_MLP_OLD = """        # Fully Connected
        if self._mlp_is_moe:
            x = self.mlp(x, already_sequence_parallel=self.is_sequence_parallel)
        else:
            x = self.mlp(x)"""
LAYER_MLP_NEW = """        # Fully Connected
        if self._mlp_is_moe:
            x = self.mlp(x, already_sequence_parallel=self.is_sequence_parallel)
        else:
            x = self.mlp(x)

        _diag_glm5next_check_stage("after_mlp", x=x)"""

LAYER_FINAL_OLD = """        if self.layer_idx == self.num_hidden_layers - 1:
            x = self.hc_post(x, residual, post, comb)
            x = hc_contract(x, self.n)
            return x, None, None, None"""
LAYER_FINAL_NEW = """        if self.layer_idx == self.num_hidden_layers - 1:
            x = self.hc_post(x, residual, post, comb)
            x = hc_contract(x, self.n)
            _diag_glm5next_check_stage("after_final_contract", x=x)
            return x, None, None, None"""

MARKERS = (
    "_diag_glm5next_first_nan_reports",
    "_diag_glm5next_stage_reports",
)
PATCHES = (
    (GLOBAL_OLD, GLOBAL_NEW),
    (OLD, NEW),
    (LAYER_PRE_OLD, LAYER_PRE_NEW),
    (LAYER_FFN_OLD, LAYER_FFN_NEW),
    (LAYER_MLP_OLD, LAYER_MLP_NEW),
    (LAYER_FINAL_OLD, LAYER_FINAL_NEW),
)
REL_PATH = "models/glm5next/nvidia/model.py"
TAG = "[diag-glm5next-first-nan]"


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

    if any(marker in content for marker in MARKERS):
        print(f"{TAG} {REL_PATH}: already applied, skipping.")
        return 0

    patched = content
    for old, new in PATCHES:
        count = patched.count(old)
        if count != 1:
            print(
                f"{TAG} FAIL {REL_PATH}: anchor found {count} times "
                "(expected exactly 1); refusing to patch a changed source."
            )
            return 1
        patched = patched.replace(old, new)

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
