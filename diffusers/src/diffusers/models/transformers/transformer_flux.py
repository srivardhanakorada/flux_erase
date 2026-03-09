import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FluxTransformer2DLoadersMixin, FromOriginalModelMixin, PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, is_torch_npu_available, logging, scale_lora_layers, unscale_lora_layers
from ...utils.torch_utils import maybe_allow_in_graph
from .._modeling_parallel import ContextParallelInput, ContextParallelOutput
from ..attention import AttentionMixin, AttentionModuleMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..cache_utils import CacheMixin
from ..embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    apply_rotary_emb,
    get_1d_rotary_pos_embed,
)
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle

logger = logging.get_logger(__name__)

# ============================================================
# GenErase token-wise global state
# ============================================================
# Banks store raw token-aligned text-side value tensors: [1, L, H, Dh]
# Layout: block -> concept -> list[vt]

_FLUX_TARGET_VT_BANK_DUAL: Dict[int, Dict[str, List[torch.Tensor]]] = {}
_FLUX_TARGET_VT_BANK_SINGLE: Dict[int, Dict[str, List[torch.Tensor]]] = {}

_FLUX_RETAIN_VT_BANK_DUAL: Dict[int, Dict[str, List[torch.Tensor]]] = {}
_FLUX_RETAIN_VT_BANK_SINGLE: Dict[int, Dict[str, List[torch.Tensor]]] = {}

_FLUX_ANCHOR_VT_BANK_DUAL: Dict[int, Dict[str, List[torch.Tensor]]] = {}
_FLUX_ANCHOR_VT_BANK_SINGLE: Dict[int, Dict[str, List[torch.Tensor]]] = {}

# Per-block, per-token preserve bases / projectors
# B_j : [D, K_j], P_j : [D, D]
_FLUX_B_DUAL: Dict[int, Dict[int, torch.Tensor]] = {}
_FLUX_B_SINGLE: Dict[int, Dict[int, torch.Tensor]] = {}
_FLUX_P_DUAL: Dict[int, Dict[int, torch.Tensor]] = {}
_FLUX_P_SINGLE: Dict[int, Dict[int, torch.Tensor]] = {}

# Per-block, per-token target bases U_j and anchor bases A_j
# U_j : [D, R], A_j : [D, R]
_FLUX_U_DUAL: Dict[int, Dict[int, torch.Tensor]] = {}
_FLUX_U_SINGLE: Dict[int, Dict[int, torch.Tensor]] = {}
_FLUX_A_DUAL: Dict[int, Dict[int, torch.Tensor]] = {}
_FLUX_A_SINGLE: Dict[int, Dict[int, torch.Tensor]] = {}

_VT_DEDUP_COS_THR = 0.995
_SHARED_RETAIN_KEY = "__retain__"
_SHARED_ANCHOR_KEY = "__anchor__"


# ============================================================
# Reset helpers
# ============================================================
def flux_reset_vt_banks(reset_retain: bool = True):
    global _FLUX_TARGET_VT_BANK_DUAL, _FLUX_TARGET_VT_BANK_SINGLE
    global _FLUX_RETAIN_VT_BANK_DUAL, _FLUX_RETAIN_VT_BANK_SINGLE
    global _FLUX_ANCHOR_VT_BANK_DUAL, _FLUX_ANCHOR_VT_BANK_SINGLE
    global _FLUX_B_DUAL, _FLUX_B_SINGLE, _FLUX_P_DUAL, _FLUX_P_SINGLE
    global _FLUX_U_DUAL, _FLUX_U_SINGLE, _FLUX_A_DUAL, _FLUX_A_SINGLE

    _FLUX_TARGET_VT_BANK_DUAL.clear()
    _FLUX_TARGET_VT_BANK_SINGLE.clear()

    if reset_retain:
        _FLUX_RETAIN_VT_BANK_DUAL.clear()
        _FLUX_RETAIN_VT_BANK_SINGLE.clear()

    _FLUX_ANCHOR_VT_BANK_DUAL.clear()
    _FLUX_ANCHOR_VT_BANK_SINGLE.clear()

    _FLUX_B_DUAL.clear()
    _FLUX_B_SINGLE.clear()
    _FLUX_P_DUAL.clear()
    _FLUX_P_SINGLE.clear()
    _FLUX_U_DUAL.clear()
    _FLUX_U_SINGLE.clear()
    _FLUX_A_DUAL.clear()
    _FLUX_A_SINGLE.clear()


# ============================================================
# Generic tensor helpers
# ============================================================
def _cos_sim_flat(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    x = a.reshape(-1).to(torch.float32)
    y = b.reshape(-1).to(torch.float32)
    x = x / (x.norm() + eps)
    y = y / (y.norm() + eps)
    return float((x * y).sum().item())


def _append_vt_dedup(
    lst: List[torch.Tensor],
    vt_new: torch.Tensor,
    *,
    cos_thr: float = _VT_DEDUP_COS_THR,
    max_keep: int = 32,
):
    vt_new = vt_new.detach()

    if len(lst) == 0:
        lst.append(vt_new)
        return

    comparable = [old for old in lst if tuple(old.shape) == tuple(vt_new.shape)]

    if len(comparable) > 0:
        best = max(_cos_sim_flat(vt_new, old) for old in comparable)
        if best >= cos_thr:
            return

    lst.append(vt_new)

    if max_keep is not None and max_keep > 0 and len(lst) > max_keep:
        del lst[0 : (len(lst) - max_keep)]


def _bank_add_concept_vt(
    bank: Dict[int, Dict[str, List[torch.Tensor]]],
    block_index: int,
    concept: str,
    vt: torch.Tensor,
    *,
    max_keep: int,
    dedup_thr: float,
):
    if block_index not in bank:
        bank[block_index] = {}
    if concept not in bank[block_index]:
        bank[block_index][concept] = []
    _append_vt_dedup(bank[block_index][concept], vt, cos_thr=dedup_thr, max_keep=max_keep)


def _orth_columns(M: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    if M.numel() == 0 or M.shape[1] == 0:
        return M.new_zeros((M.shape[0], 0))
    Q, _ = torch.linalg.qr(M, mode="reduced")
    Q = Q / (Q.norm(dim=0, keepdim=True) + eps)
    return Q


def _extract_text_vt(vt: torch.Tensor, *, token_end: Optional[int] = None) -> torch.Tensor:
    assert vt.ndim == 4 and vt.shape[0] == 1
    L = int(vt.shape[1])
    if token_end is None:
        token_end = L
    token_end = int(max(1, min(token_end, L)))
    return vt[:, :token_end].detach().contiguous()


def _vt_token(vt: torch.Tensor, token_idx: int, *, eps: float = 1e-8) -> Optional[torch.Tensor]:
    L = int(vt.shape[1])
    if token_idx < 0 or token_idx >= L:
        return None
    v = vt[0, token_idx].reshape(-1).to(torch.float32).contiguous()
    n = v.norm()
    if n <= eps:
        return None
    return v / (n + eps)


def _token_range_from_vt(vt_list: List[torch.Tensor]) -> int:
    if len(vt_list) == 0:
        return 0
    return min(int(v.shape[1]) for v in vt_list)


def _project_out(v: torch.Tensor, P: Optional[torch.Tensor]) -> torch.Tensor:
    if P is None or P.numel() == 0:
        return v
    return v - P @ v


def _normalize(v: torch.Tensor, eps: float = 1e-8) -> Optional[torch.Tensor]:
    n = v.norm()
    if n <= eps:
        return None
    return v / (n + eps)
# ============================================================
# GenErase finalize: token-wise P_j / U_j / A_j
# ============================================================
def _build_tokenwise_preserve_projectors(
    bank: Dict[int, Dict[str, List[torch.Tensor]]],
    *,
    eps: float = 1e-8,
) -> Tuple[Dict[int, Dict[int, torch.Tensor]], Dict[int, Dict[int, torch.Tensor]]]:
    B_out: Dict[int, Dict[int, torch.Tensor]] = {}
    P_out: Dict[int, Dict[int, torch.Tensor]] = {}

    for blk, concept_map in bank.items():
        all_vts: List[torch.Tensor] = []
        for _, vt_list in concept_map.items():
            all_vts.extend(vt_list)
        if len(all_vts) == 0:
            continue

        Lmin = _token_range_from_vt(all_vts)
        if Lmin <= 1:
            continue

        B_out[blk] = {}
        P_out[blk] = {}

        for j in range(1, Lmin):
            cols: List[torch.Tensor] = []
            for _, vt_list in concept_map.items():
                for vt in vt_list:
                    d = _vt_token(vt, j, eps=eps)
                    if d is not None:
                        cols.append(d)

            if len(cols) == 0:
                continue

            M = torch.stack(cols, dim=1)
            Bj = _orth_columns(M, eps=eps)
            Pj = Bj @ Bj.t()
            B_out[blk][j] = Bj
            P_out[blk][j] = Pj

    return B_out, P_out

def _build_tokenwise_U_A(
    target_bank: Dict[int, Dict[str, List[torch.Tensor]]],
    anchor_bank: Dict[int, Dict[str, List[torch.Tensor]]],
    P_bank: Dict[int, Dict[int, torch.Tensor]],
    *,
    eps: float = 1e-8,
) -> Tuple[Dict[int, Dict[int, torch.Tensor]], Dict[int, Dict[int, torch.Tensor]]]:
    U_out: Dict[int, Dict[int, torch.Tensor]] = {}
    A_out: Dict[int, Dict[int, torch.Tensor]] = {}

    for blk, concept_map in target_bank.items():
        if blk not in P_bank:
            continue

        U_out[blk] = {}
        A_out[blk] = {}

        all_vts: List[torch.Tensor] = []
        for _, vt_list in concept_map.items():
            all_vts.extend(vt_list)
        if len(all_vts) == 0:
            continue

        Lmin = _token_range_from_vt(all_vts)
        for j in range(1, Lmin):
            Pj = P_bank.get(blk, {}).get(j, None)
            target_cols: List[torch.Tensor] = []
            concepts = list(concept_map.keys())

            for concept in concepts:
                rep = None
                for vt in concept_map[concept]:
                    d = _vt_token(vt, j, eps=eps)
                    if d is None:
                        continue
                    d = _project_out(d, Pj)
                    d = _normalize(d, eps=eps)
                    if d is not None:
                        rep = d
                        break
                if rep is not None:
                    target_cols.append(rep)

            if len(target_cols) == 0:
                continue

            Uj = _orth_columns(torch.stack(target_cols, dim=1), eps=eps)
            U_out[blk][j] = Uj

            anchor_cols: List[torch.Tensor] = []
            for concept in concepts:
                if concept not in anchor_bank.get(blk, {}):
                    continue
                rep = None
                for vt in anchor_bank[blk][concept]:
                    a = _vt_token(vt, j, eps=eps)
                    if a is None:
                        continue
                    a = _project_out(a, Pj)
                    a = a - Uj @ (Uj.t() @ a)
                    a = _normalize(a, eps=eps)
                    if a is not None:
                        rep = a
                        break
                if rep is not None:
                    anchor_cols.append(rep)

            if len(anchor_cols) > 0:
                Aj = torch.stack(anchor_cols, dim=1)
                Aj = Aj / (Aj.norm(dim=0, keepdim=True) + eps)
                A_out[blk][j] = Aj

    return U_out, A_out

def flux_finalize_cora_bases(*, eps: float = 1e-8):
    global _FLUX_B_DUAL, _FLUX_B_SINGLE, _FLUX_P_DUAL, _FLUX_P_SINGLE
    global _FLUX_U_DUAL, _FLUX_U_SINGLE, _FLUX_A_DUAL, _FLUX_A_SINGLE

    _FLUX_B_DUAL.clear()
    _FLUX_B_SINGLE.clear()
    _FLUX_P_DUAL.clear()
    _FLUX_P_SINGLE.clear()
    _FLUX_U_DUAL.clear()
    _FLUX_U_SINGLE.clear()
    _FLUX_A_DUAL.clear()
    _FLUX_A_SINGLE.clear()

    _FLUX_B_DUAL, _FLUX_P_DUAL = _build_tokenwise_preserve_projectors(_FLUX_RETAIN_VT_BANK_DUAL, eps=eps)
    _FLUX_B_SINGLE, _FLUX_P_SINGLE = _build_tokenwise_preserve_projectors(_FLUX_RETAIN_VT_BANK_SINGLE, eps=eps)

    _FLUX_U_DUAL, _FLUX_A_DUAL = _build_tokenwise_U_A(
        _FLUX_TARGET_VT_BANK_DUAL, _FLUX_ANCHOR_VT_BANK_DUAL, _FLUX_P_DUAL, eps=eps
    )
    _FLUX_U_SINGLE, _FLUX_A_SINGLE = _build_tokenwise_U_A(
        _FLUX_TARGET_VT_BANK_SINGLE, _FLUX_ANCHOR_VT_BANK_SINGLE, _FLUX_P_SINGLE, eps=eps
    )
# ============================================================
# Exact GenErase token-wise edit operator
# ============================================================
def _generase_edit_tokens(
    v_slice: torch.Tensor,
    *,
    P_map: Dict[int, torch.Tensor],
    U_map: Dict[int, torch.Tensor],
    A_map: Dict[int, torch.Tensor],
    tau: float,
    beta: float,
    eps: float,
) -> torch.Tensor:
    B, T, D = v_slice.shape
    out = v_slice.clone().to(torch.float32)

    for j in range(T):
        if j not in U_map:
            continue

        Pj = P_map.get(j, None)
        Uj = U_map[j]
        Aj = A_map.get(j, None)

        vj = out[:, j, :]

        if Pj is None:
            vpres = torch.zeros_like(vj)
            vfree = vj
        else:
            vpres = torch.einsum("de,be->bd", Pj, vj)
            vfree = vj - vpres

        tj = torch.einsum("dr,bd->br", Uj, vfree)
        rj = tj.abs().amax(dim=-1) / (vfree.norm(dim=-1) + eps)
        mask = (rj >= tau).to(vj.dtype).unsqueeze(-1)

        erased = torch.einsum("dr,br->bd", Uj, tj)
        replaced = 0.0
        if Aj is not None and Aj.numel() > 0:
            rr = min(Aj.shape[1], tj.shape[1])
            replaced = torch.einsum("dr,br->bd", Aj[:, :rr], beta * tj[:, :rr])

        vnew = vpres + (vfree - erased + replaced)
        out[:, j, :] = (1.0 - mask) * vj + mask * vnew

    return out.to(v_slice.dtype)
# ============================================================
# Projection helpers
# ============================================================
def _get_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)
    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)
    return query, key, value, encoder_query, encoder_key, encoder_value

def _get_fused_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
        encoder_query, encoder_key, encoder_value = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)
    return query, key, value, encoder_query, encoder_key, encoder_value

def _get_qkv_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states, encoder_hidden_states)
    return _get_projections(attn, hidden_states, encoder_hidden_states)

# ============================================================
# Attention processor with exact GenErase record/apply
# ============================================================
class FluxAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        dual_zero_text_value: bool = False,
        single_zero_text_value: bool = False,
        text_seq_len: Optional[int] = None,
        record_target_vt: bool = False,
        record_retain_vt: bool = False,
        record_anchor_vt: bool = False,
        apply_target_proj: bool = False,
        strength_tau: float = 0.1,
        anchor_strength: float = 0.5,
        proj_eps: float = 1e-8,
        record_concept: Optional[str] = None,
        use_anchors: bool = True,
        block_index: Optional[int] = None,
        target_block_indices: Optional[List[int]] = None,
        target_single_block_indices: Optional[List[int]] = None,
        vt_dedup_cos_thr: float = _VT_DEDUP_COS_THR,
        max_target_vt_per_block: int = 32,
        max_retain_vt_per_block: int = 32,
        max_anchor_vt_per_block: int = 32,
        proj_token_end: Optional[int] = None,
        detector_token_end: Optional[int] = None,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # ------------------------------------------------------------
        # Single-stream blocks
        # ------------------------------------------------------------
        if encoder_hidden_states is None and text_seq_len is not None and block_index is not None:
            target_single_block_indices = target_single_block_indices or []

            if single_zero_text_value:
                value[:, :text_seq_len] = 0.0

            if (record_retain_vt or record_target_vt or record_anchor_vt) and (block_index in target_single_block_indices):
                vt_single = value[:, :text_seq_len].detach()[:1].contiguous()
                vt_single = _extract_text_vt(vt_single, token_end=detector_token_end)

                if record_retain_vt:
                    retain_key = record_concept or _SHARED_RETAIN_KEY
                    _bank_add_concept_vt(
                        _FLUX_RETAIN_VT_BANK_SINGLE,
                        block_index,
                        retain_key,
                        vt_single,
                        max_keep=max_retain_vt_per_block,
                        dedup_thr=vt_dedup_cos_thr,
                    )

                if record_target_vt:
                    if not record_concept:
                        raise ValueError("record_target_vt=True requires joint_attention_kwargs['record_concept']")
                    _bank_add_concept_vt(
                        _FLUX_TARGET_VT_BANK_SINGLE,
                        block_index,
                        record_concept,
                        vt_single,
                        max_keep=max_target_vt_per_block,
                        dedup_thr=vt_dedup_cos_thr,
                    )

                if record_anchor_vt:
                    anchor_key = record_concept or _SHARED_ANCHOR_KEY
                    _bank_add_concept_vt(
                        _FLUX_ANCHOR_VT_BANK_SINGLE,
                        block_index,
                        anchor_key,
                        vt_single,
                        max_keep=max_anchor_vt_per_block,
                        dedup_thr=vt_dedup_cos_thr,
                    )

            if apply_target_proj and (block_index in target_single_block_indices):
                s = 0
                e = int(text_seq_len if proj_token_end is None else max(1, min(int(proj_token_end), text_seq_len)))
                v_txt = value[:, :text_seq_len].reshape(value.shape[0], text_seq_len, -1)
                v_slice = v_txt[:, s:e, :]

                P_map = _FLUX_P_SINGLE.get(block_index, {})
                U_map = _FLUX_U_SINGLE.get(block_index, {})
                A_map = _FLUX_A_SINGLE.get(block_index, {}) if use_anchors else {}

                if len(U_map) > 0:
                    v_slice2 = _generase_edit_tokens(
                        v_slice,
                        P_map=P_map,
                        U_map=U_map,
                        A_map=A_map,
                        tau=float(strength_tau),
                        beta=float(anchor_strength),
                        eps=float(proj_eps),
                    )
                    v_txt = torch.cat([v_txt[:, :s, :], v_slice2, v_txt[:, e:, :]], dim=1)
                    value_txt_new = v_txt.view(value.shape[0], text_seq_len, value.shape[2], value.shape[3])
                    value[:, :text_seq_len] = torch.nan_to_num(value_txt_new, nan=0.0, posinf=0.0, neginf=0.0)

        # ------------------------------------------------------------
        # Dual-stream blocks
        # ------------------------------------------------------------
        if attn.added_kv_proj_dim is not None and encoder_hidden_states is not None and block_index is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)
            target_block_indices = target_block_indices or []

            if (record_retain_vt or record_target_vt or record_anchor_vt) and (block_index in target_block_indices):
                vt_dual = encoder_value.detach()[:1].contiguous()
                vt_dual = _extract_text_vt(vt_dual, token_end=detector_token_end)

                if record_retain_vt:
                    retain_key = record_concept or _SHARED_RETAIN_KEY
                    _bank_add_concept_vt(
                        _FLUX_RETAIN_VT_BANK_DUAL,
                        block_index,
                        retain_key,
                        vt_dual,
                        max_keep=max_retain_vt_per_block,
                        dedup_thr=vt_dedup_cos_thr,
                    )

                if record_target_vt:
                    if not record_concept:
                        raise ValueError("record_target_vt=True requires joint_attention_kwargs['record_concept']")
                    _bank_add_concept_vt(
                        _FLUX_TARGET_VT_BANK_DUAL,
                        block_index,
                        record_concept,
                        vt_dual,
                        max_keep=max_target_vt_per_block,
                        dedup_thr=vt_dedup_cos_thr,
                    )

                if record_anchor_vt:
                    anchor_key = record_concept or _SHARED_ANCHOR_KEY
                    _bank_add_concept_vt(
                        _FLUX_ANCHOR_VT_BANK_DUAL,
                        block_index,
                        anchor_key,
                        vt_dual,
                        max_keep=max_anchor_vt_per_block,
                        dedup_thr=vt_dedup_cos_thr,
                    )

            if apply_target_proj and (block_index in target_block_indices):
                v = encoder_value.reshape(encoder_value.shape[0], encoder_value.shape[1], -1)
                s = 0
                e = int(v.shape[1] if proj_token_end is None else max(1, min(int(proj_token_end), v.shape[1])))
                v_slice = v[:, s:e, :]

                P_map = _FLUX_P_DUAL.get(block_index, {})
                U_map = _FLUX_U_DUAL.get(block_index, {})
                A_map = _FLUX_A_DUAL.get(block_index, {}) if use_anchors else {}

                if len(U_map) > 0:
                    v_slice2 = _generase_edit_tokens(
                        v_slice,
                        P_map=P_map,
                        U_map=U_map,
                        A_map=A_map,
                        tau=float(strength_tau),
                        beta=float(anchor_strength),
                        eps=float(proj_eps),
                    )
                    v = torch.cat([v[:, :s, :], v_slice2, v[:, e:, :]], dim=1)
                    encoder_value = v.view_as(encoder_value)
                    encoder_value = torch.nan_to_num(encoder_value, nan=0.0, posinf=0.0, neginf=0.0)

            if dual_zero_text_value:
                encoder_value = encoder_value * 0.0

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]],
                dim=1,
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states

        return hidden_states

# ============================================================
# Flux attention / blocks / model (unchanged except processor kwargs passthrough)
# ============================================================
class FluxAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = FluxAttnProcessor
    _available_processors = [FluxAttnProcessor]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        context_pre_only: Optional[bool] = None,
        pre_only: bool = False,
        elementwise_affine: bool = True,
        processor=None,
    ):
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias
        self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        if not self.pre_only:
            self.to_out = torch.nn.ModuleList(
                [torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias), torch.nn.Dropout(dropout)]
            )
        if added_kv_proj_dim is not None:
            self.norm_added_q = torch.nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = torch.nn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.to_add_out = torch.nn.Linear(self.inner_dim, query_dim, bias=out_bias)
        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states", "debug_tokens"}
        unused_kwargs = [k for k in kwargs.keys() if (k not in attn_parameters and k not in quiet_attn_parameters)]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"joint_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: v for k, v in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kwargs)

@maybe_allow_in_graph
class FluxSingleTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)
        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=FluxAttnProcessor(),
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = (joint_attention_kwargs or {}).copy()
        joint_attention_kwargs["text_seq_len"] = text_seq_len
        attn_output = self.attn(hidden_states=norm_hidden_states, image_rotary_emb=image_rotary_emb, **joint_attention_kwargs)
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states = gate.unsqueeze(1) * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
        return encoder_hidden_states, hidden_states

@maybe_allow_in_graph
class FluxTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, qk_norm: str = "rms_norm", eps: float = 1e-6):
        super().__init__()
        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)
        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=FluxAttnProcessor(),
            eps=eps,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)
        joint_attention_kwargs = joint_attention_kwargs or {}
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
            ip_attn_output = None
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs
        else:
            attn_output, context_attn_output = attention_outputs[0], attention_outputs[1]
            ip_attn_output = None
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output
        if ip_attn_output is not None:
            hidden_states = hidden_states + ip_attn_output
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states

class FluxPosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin

class FluxTransformer2DModel(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    FluxTransformer2DLoadersMixin,
    CacheMixin,
    AttentionMixin,
):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _cp_plan = {
        "": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "img_ids": ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
            "txt_ids": ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)
        text_time_guidance_cls = CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        self.time_text_embed = text_time_guidance_cls(embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim)
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_single_layers)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            joint_attention_kwargs = {}
            lora_scale = 1.0
        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective.")
        hidden_states = self.x_embedder(hidden_states)
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_text_embed(timestep, pooled_projections) if guidance is None else self.time_text_embed(timestep, guidance, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        ids = torch.cat((txt_ids, img_ids), dim=0)
        if is_torch_npu_available():
            freqs_cos, freqs_sin = self.pos_embed(ids.cpu())
            image_rotary_emb = (freqs_cos.npu(), freqs_sin.npu())
        else:
            image_rotary_emb = self.pos_embed(ids)
        for index_block, block in enumerate(self.transformer_blocks):
            ja = joint_attention_kwargs.copy()
            ja["block_index"] = index_block
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(block, hidden_states, encoder_hidden_states, temb, image_rotary_emb, ja)
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=ja,
                )
        for index_block, block in enumerate(self.single_transformer_blocks):
            ja = joint_attention_kwargs.copy()
            ja["block_index"] = index_block
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(block, hidden_states, encoder_hidden_states, temb, image_rotary_emb, ja)
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=ja,
                )
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
