"""Probe the B12X sparse-MLA forward path and MLA post-attention projections.

Targets:
  v1/attention/backends/mla/b12x_mla_sparse.py
  model_executor/layers/attention/mla_attention.py
  model_executor/layers/mla.py

The layer-level diagnostic reports the first non-finite state after the
layer-3 attention call. This mod probes the B12X sparse-MLA kernel output,
the MLA v_up projection, and the outer o_proj projection in layer 3.

Run with --enforce-eager. Probes are armed after 300 seconds and limited to
the first few layer-3 forwards.
"""

import os
import sys

TAG = "[diag-glm5next-b12x-attention]"

B12X_REL_PATH = "v1/attention/backends/mla/b12x_mla_sparse.py"

B12X_HELPER_OLD = """@dataclass
class B12xMLASparseMetadata(AttentionMetadata):"""
B12X_HELPER_NEW = """import time as _diag_b12x_mla_time

_diag_b12x_mla_t0 = _diag_b12x_mla_time.monotonic()
_diag_b12x_mla_reports = 0
_diag_b12x_mla_state_reports = 0
_diag_b12x_mla_calls = 0
_diag_b12x_mla_name_reports = 0
_diag_b12x_mla_tag = "[diag-glm5next-b12x-attention]"


def _diag_b12x_mla_stats(t):
    try:
        f = t.detach().float()
        return (
            bool(not torch.isfinite(f).all().item()),
            float(f.min().item()),
            float(f.max().item()),
            float(f.abs().max().item()),
        )
    except Exception:
        return True, float("nan"), float("nan"), float("nan")


def _diag_b12x_mla_maybe_check(
    backend,
    *,
    num_tokens: int,
    attn_metadata,
    q_all: torch.Tensor,
    topk_indices: torch.Tensor,
    selected_indices: torch.Tensor,
    active_counts: torch.Tensor,
    cache_seq_lens: torch.Tensor,
    kv_cache_for_run: torch.Tensor,
    result,
    use_ckv_gather: bool,
    physical_selection,
    layer_ref=None,
) -> None:
    # diag-glm5next-b12x-attention: forward_mqa finite-state probe
    global _diag_b12x_mla_reports, _diag_b12x_mla_state_reports
    global _diag_b12x_mla_calls, _diag_b12x_mla_name_reports
    try:
        _layer_ref_name = getattr(layer_ref, "layer_name", None) if layer_ref is not None else None
        layer_name = (
            str(_layer_ref_name)
            if _layer_ref_name is not None
            else str(getattr(backend, "layer_name", type(backend).__name__))
        )
        layer_type = type(layer_ref).__name__ if layer_ref is not None else "?"
        _diag_b12x_mla_calls += 1
        elapsed = _diag_b12x_mla_time.monotonic() - _diag_b12x_mla_t0
        if (
            _diag_b12x_mla_calls > 100000
            or elapsed < 300
            or torch.cuda.is_current_stream_capturing()
        ):
            return
        if _diag_b12x_mla_name_reports < 8:
            _diag_b12x_mla_name_reports += 1
            print(
                f"{_diag_b12x_mla_tag} name-probe call={_diag_b12x_mla_calls} "
                f"layer={layer_name} "
                f"layer_type={layer_type} "
                f"tokens={num_tokens} "
                f"prefills={int(attn_metadata.num_prefills)} "
                f"decode_tokens={int(attn_metadata.num_decode_tokens)} "
                f"max_q={int(attn_metadata.max_query_len)}",
                flush=True,
            )
        if ".layers.3." not in layer_name:
            return
        res_tensor = result[0] if isinstance(result, tuple) else result
        q_bad = bool(not torch.isfinite(q_all).all().item())
        res_bad = bool(not torch.isfinite(res_tensor).all().item())
        ac = active_counts[:num_tokens]
        cl = cache_seq_lens[:num_tokens]
        zero_ac = int((ac == 0).sum().item())
        sel = selected_indices[:num_tokens]
        invalid_sel = int((sel == -1).sum().item())
        valid_sel = sel != -1
        has_valid = bool(valid_sel.any().item())
        sel_min = int(sel[valid_sel].min().item()) if has_valid else -1
        sel_max = int(sel[valid_sel].max().item()) if has_valid else -1
        kv_rows = None
        if (
            kv_cache_for_run is not None
            and kv_cache_for_run.ndim == 3
            and int(kv_cache_for_run.shape[-1]) > 0
        ):
            kv_rows = (
                (int(kv_cache_for_run.shape[0]) - 1)
                * (int(kv_cache_for_run.stride(0)) // int(kv_cache_for_run.shape[-1]))
                + int(kv_cache_for_run.shape[1])
            )
        if use_ckv_gather:
            source = "ckv_gather"
        elif physical_selection is not None:
            source = "physical"
        elif getattr(backend, "_is_glm_next", False):
            source = "glm_fallback"
        else:
            source = "plain"
        if _diag_b12x_mla_state_reports < 6:
            _diag_b12x_mla_state_reports += 1
            print(
                f"{_diag_b12x_mla_tag} b12x-state call={_diag_b12x_mla_calls} "
                f"layer={layer_name} "
                f"source={source} "
                f"tokens={num_tokens} "
                f"q_bad={q_bad} "
                f"res_bad={res_bad} "
                f"ac_zero={zero_ac} "
                f"ac[:8]={[int(v) for v in ac[:8].detach().cpu().tolist()]} "
                f"cl[:8]={[int(v) for v in cl[:8].detach().cpu().tolist()]} "
                f"sel_invalid={invalid_sel} "
                f"sel_min={sel_min} "
                f"sel_max={sel_max} "
                f"kv_flat_rows={kv_rows}",
                flush=True,
            )
        if not (q_bad or res_bad or zero_ac):
            return
        if _diag_b12x_mla_reports >= 12:
            return
        _diag_b12x_mla_reports += 1
        first_ac = ac[:8].detach().cpu().tolist()
        first_cl = cl[:8].detach().cpu().tolist()
        first_topk = (
            topk_indices[0, :8].detach().cpu().tolist() if num_tokens > 0 else []
        )
        first_sel = (
            selected_indices[0, :8].detach().cpu().tolist() if num_tokens > 0 else []
        )
        print(
            f"{_diag_b12x_mla_tag} b12x-bad call={_diag_b12x_mla_calls} "
            f"layer={layer_name} "
            f"layer_type={layer_type} "
            f"source={source} "
            f"tokens={num_tokens} "
            f"prefills={int(attn_metadata.num_prefills)} "
            f"decode_tokens={int(attn_metadata.num_decode_tokens)} "
            f"max_q={int(attn_metadata.max_query_len)} "
            f"q_bad={q_bad} "
            f"res_bad={res_bad} "
            f"ac_zero={zero_ac} "
            f"ac[:8]={first_ac} "
            f"cl[:8]={first_cl} "
            f"sel_invalid={invalid_sel} "
            f"sel_min={sel_min} "
            f"sel_max={sel_max} "
            f"kv_flat_rows={kv_rows} "
            f"topk[0,:8]={first_topk} "
            f"sel[0,:8]={first_sel}",
            flush=True,
        )
    except Exception as exc:
        print(f"{_diag_b12x_mla_tag} probe failed: {exc!r}", flush=True)


@dataclass
class B12xMLASparseMetadata(AttentionMetadata):"""
B12X_HELPER_MARKER = (
    "    # diag-glm5next-b12x-attention: forward_mqa finite-state probe"
)

B12X_DEL_OLD = """        del layer
        cache_page_size = int(kv_c_and_k_pe_cache.shape[1])"""
B12X_DEL_NEW = """        _diag_b12x_mla_layer_ref = layer
        del layer
        cache_page_size = int(kv_c_and_k_pe_cache.shape[1])"""
B12X_DEL_MARKER = "        _diag_b12x_mla_layer_ref = layer"

B12X_CALL_OLD = """        result = self._run(binding)
        if self.need_to_return_lse_for_decode:
            output, lse = result
            return output, lse
        assert isinstance(result, torch.Tensor)
        return result, None"""
B12X_CALL_NEW = """        result = self._run(binding)
        # diag-glm5next-b12x-attention: check layer-3 sparse-MLA state
        try:
            _diag_b12x_mla_maybe_check(
                self,
                num_tokens=num_tokens,
                attn_metadata=attn_metadata,
                q_all=q_all,
                topk_indices=topk_indices,
                selected_indices=selected_indices,
                active_counts=active_counts,
                cache_seq_lens=cache_seq_lens,
                kv_cache_for_run=kv_cache_for_run,
                result=result,
                use_ckv_gather=use_ckv_gather,
                physical_selection=locals().get("physical_selection"),
                layer_ref=locals().get("_diag_b12x_mla_layer_ref"),
            )
        except Exception as _diag_b12x_mla_call_exc:
            print(
                f"{_diag_b12x_mla_tag} probe call failed: "
                f"{_diag_b12x_mla_call_exc!r}",
                flush=True,
            )
        if self.need_to_return_lse_for_decode:
            output, lse = result
            return output, lse
        assert isinstance(result, torch.Tensor)
        return result, None"""
B12X_CALL_MARKER = (
    "        # diag-glm5next-b12x-attention: check layer-3 sparse-MLA state"
)

B12X_PATCHES = (
    (B12X_HELPER_OLD, B12X_HELPER_NEW, B12X_HELPER_MARKER),
    (B12X_DEL_OLD, B12X_DEL_NEW, B12X_DEL_MARKER),
    (B12X_CALL_OLD, B12X_CALL_NEW, B12X_CALL_MARKER),
)

MLA_ATTN_REL_PATH = "model_executor/layers/attention/mla_attention.py"

MLA_ATTN_HELPER_OLD = """class MLAAttention(nn.Module, AttentionLayerBase):"""
MLA_ATTN_HELPER_NEW = """import time as _diag_mla_vup_time

_diag_mla_vup_t0 = _diag_mla_vup_time.monotonic()
_diag_mla_vup_state_reports = 0
_diag_mla_vup_bad_reports = 0
_diag_mla_vup_tag = "[diag-glm5next-b12x-attention]"


def _diag_mla_vup_stats(t):
    try:
        f = t.detach().float()
        return (
            bool(not torch.isfinite(f).all().item()),
            float(f.min().item()),
            float(f.max().item()),
            float(f.abs().max().item()),
        )
    except Exception:
        return True, float("nan"), float("nan"), float("nan")


def _diag_mla_vup_maybe_check(layer, *, attn_out, vup_out) -> None:
    # diag-glm5next-b12x-attention: mla v_up probe
    global _diag_mla_vup_state_reports, _diag_mla_vup_bad_reports
    try:
        layer_name = str(getattr(layer, "layer_name", "?"))
        if ".layers.3." not in layer_name:
            return
        if (
            _diag_mla_vup_state_reports >= 6
            or _diag_mla_vup_bad_reports >= 12
            or _diag_mla_vup_time.monotonic() - _diag_mla_vup_t0 < 300
            or torch.cuda.is_current_stream_capturing()
        ):
            return
        attn_bad, attn_min, attn_max, attn_abs = _diag_mla_vup_stats(attn_out)
        vup_bad, vup_min, vup_max, vup_abs = _diag_mla_vup_stats(vup_out)
        _diag_mla_vup_state_reports += 1
        print(
            f"{_diag_mla_vup_tag} vup-state layer={layer_name} "
            f"attn_bad={attn_bad} "
            f"attn_min={attn_min:.6g} "
            f"attn_max={attn_max:.6g} "
            f"attn_abs={attn_abs:.6g} "
            f"vup_bad={vup_bad} "
            f"vup_min={vup_min:.6g} "
            f"vup_max={vup_max:.6g} "
            f"vup_abs={vup_abs:.6g}",
            flush=True,
        )
        if attn_bad or vup_bad:
            if _diag_mla_vup_bad_reports < 12:
                _diag_mla_vup_bad_reports += 1
                print(
                    f"{_diag_mla_vup_tag} vup-bad layer={layer_name} "
                    f"attn_bad={attn_bad} vup_bad={vup_bad}",
                    flush=True,
                )
    except Exception as exc:
        print(f"{_diag_mla_vup_tag} vup probe failed: {exc!r}", flush=True)


class MLAAttention(nn.Module, AttentionLayerBase):"""
MLA_ATTN_HELPER_MARKER = (
    "    # diag-glm5next-b12x-attention: mla v_up probe"
)

MLA_ATTN_CALL_OLD = """            # v_up projection
            self._v_up_proj(attn_out, out=mqa_output_slice)"""
MLA_ATTN_CALL_NEW = """            # v_up projection
            self._v_up_proj(attn_out, out=mqa_output_slice)
            # diag-glm5next-b12x-attention: mla v_up check
            try:
                _diag_mla_vup_maybe_check(
                    self,
                    attn_out=attn_out,
                    vup_out=mqa_output_slice,
                )
            except Exception as _diag_mla_vup_call_exc:
                print(
                    f"{_diag_mla_vup_tag} vup probe call failed: "
                    f"{_diag_mla_vup_call_exc!r}",
                    flush=True,
                )"""
MLA_ATTN_CALL_MARKER = (
    "            # diag-glm5next-b12x-attention: mla v_up check"
)

MLA_ATTN_PATCHES = (
    (MLA_ATTN_HELPER_OLD, MLA_ATTN_HELPER_NEW, MLA_ATTN_HELPER_MARKER),
    (MLA_ATTN_CALL_OLD, MLA_ATTN_CALL_NEW, MLA_ATTN_CALL_MARKER),
)

MLA_WRAPPER_REL_PATH = "model_executor/layers/mla.py"

MLA_WRAPPER_HELPER_OLD = """@PluggableLayer.register("multi_head_latent_attention")
class MultiHeadLatentAttentionWrapper(PluggableLayer):"""
MLA_WRAPPER_HELPER_NEW = """import time as _diag_mla_wrapper_time

_diag_mla_wrapper_t0 = _diag_mla_wrapper_time.monotonic()
_diag_mla_wrapper_state_reports = 0
_diag_mla_wrapper_bad_reports = 0
_diag_mla_wrapper_tag = "[diag-glm5next-b12x-attention]"


def _diag_mla_wrapper_stats(t):
    try:
        f = t.detach().float()
        return (
            bool(not torch.isfinite(f).all().item()),
            float(f.min().item()),
            float(f.max().item()),
            float(f.abs().max().item()),
        )
    except Exception:
        return True, float("nan"), float("nan"), float("nan")


def _diag_mla_wrapper_maybe_check(layer, *, attn_out, o_out) -> None:
    # diag-glm5next-b12x-attention: mla wrapper o_proj probe
    global _diag_mla_wrapper_state_reports, _diag_mla_wrapper_bad_reports
    try:
        inner = getattr(layer, "mla_attn", None)
        layer_name = str(getattr(inner, "layer_name", "?")) if inner is not None else "?"
        if ".layers.3." not in layer_name:
            return
        if (
            _diag_mla_wrapper_state_reports >= 6
            or _diag_mla_wrapper_bad_reports >= 12
            or _diag_mla_wrapper_time.monotonic() - _diag_mla_wrapper_t0 < 300
            or torch.cuda.is_current_stream_capturing()
        ):
            return
        attn_bad, attn_min, attn_max, attn_abs = _diag_mla_wrapper_stats(attn_out)
        o_bad, o_min, o_max, o_abs = _diag_mla_wrapper_stats(o_out)
        _diag_mla_wrapper_state_reports += 1
        print(
            f"{_diag_mla_wrapper_tag} oproj-state layer={layer_name} "
            f"attn_bad={attn_bad} "
            f"attn_min={attn_min:.6g} "
            f"attn_max={attn_max:.6g} "
            f"attn_abs={attn_abs:.6g} "
            f"o_bad={o_bad} "
            f"o_min={o_min:.6g} "
            f"o_max={o_max:.6g} "
            f"o_abs={o_abs:.6g}",
            flush=True,
        )
        if attn_bad or o_bad:
            if _diag_mla_wrapper_bad_reports < 12:
                _diag_mla_wrapper_bad_reports += 1
                print(
                    f"{_diag_mla_wrapper_tag} oproj-bad layer={layer_name} "
                    f"attn_bad={attn_bad} o_bad={o_bad}",
                    flush=True,
                )
    except Exception as exc:
        print(f"{_diag_mla_wrapper_tag} o_proj probe failed: {exc!r}", flush=True)


@PluggableLayer.register("multi_head_latent_attention")
class MultiHeadLatentAttentionWrapper(PluggableLayer):"""
MLA_WRAPPER_HELPER_MARKER = (
    "    # diag-glm5next-b12x-attention: mla wrapper o_proj probe"
)

MLA_WRAPPER_CALL_OLD = """        return self.o_proj(attn_out)[0]"""
MLA_WRAPPER_CALL_NEW = """        _diag_mla_wrapper_o_out = self.o_proj(attn_out)[0]
        # diag-glm5next-b12x-attention: mla wrapper o_proj check
        try:
            _diag_mla_wrapper_maybe_check(
                self,
                attn_out=attn_out,
                o_out=_diag_mla_wrapper_o_out,
            )
        except Exception as _diag_mla_wrapper_call_exc:
            print(
                f"{_diag_mla_wrapper_tag} o_proj probe call failed: "
                f"{_diag_mla_wrapper_call_exc!r}",
                flush=True,
            )
        return _diag_mla_wrapper_o_out"""
MLA_WRAPPER_CALL_MARKER = (
    "        # diag-glm5next-b12x-attention: mla wrapper o_proj check"
)

MLA_WRAPPER_PATCHES = (
    (MLA_WRAPPER_HELPER_OLD, MLA_WRAPPER_HELPER_NEW, MLA_WRAPPER_HELPER_MARKER),
    (MLA_WRAPPER_CALL_OLD, MLA_WRAPPER_CALL_NEW, MLA_WRAPPER_CALL_MARKER),
)

TARGETS = (
    (B12X_REL_PATH, B12X_PATCHES),
    (MLA_ATTN_REL_PATH, MLA_ATTN_PATCHES),
    (MLA_WRAPPER_REL_PATH, MLA_WRAPPER_PATCHES),
)


def apply_patches(root: str, rel_path: str, patches) -> int:
    path = os.path.join(root, rel_path)
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
    except OSError as exc:
        print(f"{TAG} FAIL {rel_path}: cannot read file: {exc}")
        return 1

    patched = content
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
            return 1
        patched = patched.replace(old, new)
        changed = True

    if changed:
        try:
            compile(patched, path, "exec")
        except SyntaxError as exc:
            print(f"{TAG} FAIL {rel_path}: patched text does not compile: {exc}")
            return 1
        with open(path, "w", encoding="utf-8") as f:
            f.write(patched)
        print(f"{TAG} {rel_path}: patched.")
    else:
        print(f"{TAG} {rel_path}: already applied, skipping.")
    return 0


def main() -> int:
    root = os.environ.get("VLLM_PKG_DIR")
    if len(sys.argv) > 1:
        root = sys.argv[1]
    if not root:
        print(f"{TAG} ERROR: VLLM_PKG_DIR is not set and no path argument given.")
        return 1
    root = os.path.abspath(root)

    for rel_path, patches in TARGETS:
        rc = apply_patches(root, rel_path, patches)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
