# transformer_flux.py
# ================================================================
# FLUX concept eraser rewritten to match CORA/GenErase-style math.
#
# Key CORA ideas (per block j, per token):
#
# (1) Preserve/retain subspace projector from retain concepts:
#     Build V_ret,j (orthonormal columns) from retain exemplars.
#     P_j = V_ret,j V_ret,j^T
#     v_pres,j = P_j v_j
#     v_free,j = (I - P_j) v_j
#
# (2) Target concept subspace from MANY target exemplars (generalization):
#     For concept c, collect target directions d_{c,r,j} = (I - P_j) \tilde v_{c,r,j}
#     U_{c,j} = orth([ d_{c,1,j}, ..., d_{c,R,j} ])          (CORA Eq. 10)
#
# (3) Token-wise activation score (gate):
#     t_j = U_{c,j}^T v_free,j
#     r_j = ||t_j||_∞ / (||v_free,j||_2 + eps)               (CORA Eq. 11)
#     m_j = 1[r_j >= tau]
#
# (4) Erase-only (projection removal):
#     v_free,j' = v_free,j - gamma * m_j * U_{c,j} t_j
#     v_j'      = v_pres,j + v_free,j'
#
# (5) Erase-and-replace with per-concept anchors (CORA-style):
#     For concept c, build anchor basis A_{c,j} from anchor exemplars:
#       a_{c,j} = (I - P_j) \tilde v_anchor_{c,j}
#       A_{c,j} = orth([a_{c,j}])   (often rank-1 if one anchor)
#
#     Replace the removed component with anchor-aligned component using SAME coefficients t_j:
#       v_free,j' = v_free,j - gamma*m_j*(U_{c,j} t_j) + alpha*m_j*(A_{c,j} t_j)
#     where alpha is anchor_strength (your hyperparam).
#
# ----------------------------------------------------------------
# What changes vs your previous math:
# - OLD: solved least-squares over vt_list per token (Gram solve) and used a single "anchor_vector"
#        to skew the span. That is NOT CORA: it uses an explicit orthonormal U and a gate r_j.
# - NEW: explicitly constructs retain projector P_j, target basis U_{c,j}, anchor basis A_{c,j}
#        and applies CORA Eq(10)-(11) + erase (+ optional replace) with token-wise gating.
#
# Practical interface changes:
# - Recording now requires joint_attention_kwargs["record_concept"] for target/anchor recording.
# - Applying can specify joint_attention_kwargs["active_concept"] to use that concept's U/A.
#   If active_concept is None, we fall back to "union erase" across all concepts (no replace).
# ================================================================

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

# ---------------------------
# Globals / Banks
# ---------------------------

# Raw VT banks (stored as [1, L, H, Dh]) collected from exemplars.
# We keep these for reproducibility; bases are derived by flux_finalize_cora_bases().
_FLUX_TARGET_VT_BANK_DUAL: Dict[int, Dict[str, List[torch.Tensor]]] = {}      # block -> concept -> [vt_dual]
_FLUX_TARGET_VT_BANK_SINGLE: Dict[int, Dict[str, List[torch.Tensor]]] = {}    # block -> concept -> [vt_single]
_FLUX_RETAIN_VT_BANK_DUAL: Dict[int, List[torch.Tensor]] = {}                 # block -> [vt_dual]
_FLUX_RETAIN_VT_BANK_SINGLE: Dict[int, List[torch.Tensor]] = {}               # block -> [vt_single]
_FLUX_ANCHOR_VT_BANK_DUAL: Dict[int, Dict[str, List[torch.Tensor]]] = {}      # block -> concept -> [vt_dual_anchor]
_FLUX_ANCHOR_VT_BANK_SINGLE: Dict[int, Dict[str, List[torch.Tensor]]] = {}    # block -> concept -> [vt_single_anchor]

# CORA derived objects:
# Retain bases (orthonormal columns) V_ret,j and projectors P_j = V V^T (applied implicitly).
_FLUX_VRET_DUAL: Dict[int, torch.Tensor] = {}   # block -> [D, k]
_FLUX_VRET_SINGLE: Dict[int, torch.Tensor] = {} # block -> [D, k]

# Per concept target bases U_{c,j} and anchor bases A_{c,j}
_FLUX_U_DUAL: Dict[int, Dict[str, torch.Tensor]] = {}      # block -> concept -> [D, r]
_FLUX_U_SINGLE: Dict[int, Dict[str, torch.Tensor]] = {}    # block -> concept -> [D, r]
_FLUX_A_DUAL: Dict[int, Dict[str, torch.Tensor]] = {}      # block -> concept -> [D, rA]
_FLUX_A_SINGLE: Dict[int, Dict[str, torch.Tensor]] = {}    # block -> concept -> [D, rA]

# Union bases (for "erase everything in target set" without selecting a concept)
_FLUX_U_UNION_DUAL: Dict[int, torch.Tensor] = {}   # block -> [D, r_union]
_FLUX_U_UNION_SINGLE: Dict[int, torch.Tensor] = {} # block -> [D, r_union]

# Misc hyperparams defaults
_VT_DEDUP_COS_THR = 0.98
_FLUX_RETAIN_LAMBDA = 1.0  # used in old code; we keep it as a scaling on (I - lambda P), default 1.0 = true (I-P)


# ---------------------------
# Helpers: reset & append
# ---------------------------

def flux_reset_vt_banks(reset_retain: bool = True):
    """Reset all recorded banks and derived bases."""
    global _FLUX_TARGET_VT_BANK_DUAL, _FLUX_TARGET_VT_BANK_SINGLE
    global _FLUX_RETAIN_VT_BANK_DUAL, _FLUX_RETAIN_VT_BANK_SINGLE
    global _FLUX_ANCHOR_VT_BANK_DUAL, _FLUX_ANCHOR_VT_BANK_SINGLE
    global _FLUX_VRET_DUAL, _FLUX_VRET_SINGLE
    global _FLUX_U_DUAL, _FLUX_U_SINGLE, _FLUX_A_DUAL, _FLUX_A_SINGLE
    global _FLUX_U_UNION_DUAL, _FLUX_U_UNION_SINGLE

    _FLUX_TARGET_VT_BANK_DUAL.clear()
    _FLUX_TARGET_VT_BANK_SINGLE.clear()

    if reset_retain:
        _FLUX_RETAIN_VT_BANK_DUAL.clear()
        _FLUX_RETAIN_VT_BANK_SINGLE.clear()

    _FLUX_ANCHOR_VT_BANK_DUAL.clear()
    _FLUX_ANCHOR_VT_BANK_SINGLE.clear()

    _FLUX_VRET_DUAL.clear()
    _FLUX_VRET_SINGLE.clear()
    _FLUX_U_DUAL.clear()
    _FLUX_U_SINGLE.clear()
    _FLUX_A_DUAL.clear()
    _FLUX_A_SINGLE.clear()
    _FLUX_U_UNION_DUAL.clear()
    _FLUX_U_UNION_SINGLE.clear()


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
    if len(lst) == 0:
        lst.append(vt_new.detach())
        return
    best = max(_cos_sim_flat(vt_new, old) for old in lst)
    if best >= cos_thr:
        return
    lst.append(vt_new.detach())
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


def _bank_add_vt(
    bank: Dict[int, List[torch.Tensor]],
    block_index: int,
    vt: torch.Tensor,
    *,
    max_keep: int,
    dedup_thr: float,
):
    if block_index not in bank:
        bank[block_index] = []
    _append_vt_dedup(bank[block_index], vt, cos_thr=dedup_thr, max_keep=max_keep)


# ---------------------------
# CORA math primitives
# ---------------------------

def _vt_to_dir(vt: torch.Tensor, *, token_idx: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert vt [1,L,H,Dh] into a single direction d in R^D (D = H*Dh).
    We use token_idx=1 (the "detector" token after duplication), matching your duplicated-vt scheme.
    """
    v = vt[0, token_idx].reshape(-1).to(torch.float32).contiguous()
    return v / (v.norm() + eps)


def _orth_columns(M: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """
    Return an orthonormal basis spanning columns of M. M: [D, R].
    If R==0 returns empty [D,0].
    """
    if M.numel() == 0 or M.shape[1] == 0:
        return M.new_zeros((M.shape[0], 0))
    # QR gives Q with orthonormal columns.
    Q, _ = torch.linalg.qr(M, mode="reduced")
    # Numerical safety: normalize columns
    Q = Q / (Q.norm(dim=0, keepdim=True) + eps)
    return Q


def _project_with_basis(v: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Project v onto span(B). v: [..., D], B: [D, k] with orthonormal columns.
    proj = B (B^T v)
    """
    if B is None or B.numel() == 0 or B.shape[1] == 0:
        return torch.zeros_like(v)
    coeff = torch.einsum("dk,...d->...k", B, v)
    return torch.einsum("dk,...k->...d", B, coeff)


def _cora_score_and_coeff(v_free: torch.Tensor, U: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CORA Eq(11):
      t = U^T v_free
      r = ||t||_inf / (||v_free||_2 + eps)
    v_free: [B,T,D], U: [D,R]
    returns (t: [B,T,R], r: [B,T])
    """
    t = torch.einsum("dr,btd->btr", U, v_free)
    r = t.abs().amax(dim=-1) / (v_free.norm(dim=-1) + eps)
    return t, r


def _cora_erase_replace(
    v_slice: torch.Tensor,         # [B,T,D]
    *,
    Vret: Optional[torch.Tensor],  # [D,k] retain basis
    U: torch.Tensor,               # [D,R] target basis
    A: Optional[torch.Tensor],     # [D,R] anchor basis (can be rank-1 or match R)
    tau: float,
    gamma: float,
    anchor_strength: float,
    eps: float,
    use_replace: bool,
) -> torch.Tensor:
    """
    Apply CORA-style erase (and optional replace) to v_slice.

    Steps:
      v_pres = P v
      v_free = (I-P) v
      (t, r) from Eq(11)
      m = 1[r >= tau]
      erase: v_free' = v_free - gamma*m*(U t)
      replace (optional): v_free' += anchor_strength*m*(A t)

    Note on A t:
      If A has rank rA != R, we adapt:
        - If rA == R: A t is direct.
        - If rA == 1: we broadcast the scalar (sum of t) onto anchor direction.
        - Else: we project t down to rA by truncation.
    """
    v32 = v_slice.to(torch.float32)

    # Preserve split: v_pres = P v, v_free = (I-P) v
    if Vret is not None and Vret.numel() > 0 and Vret.shape[1] > 0:
        v_pres = _project_with_basis(v32, Vret)
        v_free = v32 - _FLUX_RETAIN_LAMBDA * v_pres  # lambda=1 -> true (I-P); keep lambda hook if desired
    else:
        v_pres = torch.zeros_like(v32)
        v_free = v32

    # Eq(11) gate based on U
    if U is None or U.numel() == 0 or U.shape[1] == 0:
        return v_slice  # nothing to do

    t, r = _cora_score_and_coeff(v_free, U, eps=eps)  # t: [B,T,R], r: [B,T]
    m = (r >= tau).to(v_free.dtype).unsqueeze(-1)      # [B,T,1]

    # Erase component: U t
    removed = torch.einsum("dr,btr->btd", U, t)        # [B,T,D]
    v_free2 = v_free - gamma * m * removed

    # Replace with anchor component (CORA-style)
    if use_replace and A is not None and A.numel() > 0 and A.shape[1] > 0:
        rA = A.shape[1]
        rU = U.shape[1]
        if rA == rU:
            added = torch.einsum("dr,btr->btd", A, t)
        elif rA == 1:
            # collapse coefficients to a scalar per token
            ts = t.sum(dim=-1, keepdim=True)  # [B,T,1]
            added = torch.einsum("d1,bt1->btd", A, ts)
        else:
            # truncate coeffs to min(rA,rU)
            rr = min(rA, rU)
            added = torch.einsum("dr,btr->btd", A[:, :rr], t[:, :, :rr])
        v_free2 = v_free2 + anchor_strength * m * added

    v_out = v_pres + v_free2
    return v_out.to(dtype=v_slice.dtype)


# ---------------------------
# Your detector construction (kept)
# ---------------------------

def _make_attn01_duplicated_vt(
    vt: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    fill_from: int = 1,
    zero_sot: bool = True,
    attn_mode: str = "col",          # "col" recommended
    normalize_detector: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Build a token-level "detector" direction from phrase tokens [0,1],
    then duplicate across [fill_from:].

    vt, q, k: [1, L, H, Dh] in head space (after unflatten + RMSNorm for q/k).

    This stays as your mechanism for extracting a stable direction d,
    but CORA now uses d only to build subspaces U/A (Eq. 10), not to do a per-token solve.
    """
    assert vt.ndim == 4 and vt.shape[0] == 1, f"Expected vt [1,L,H,Dh], got {tuple(vt.shape)}"
    assert q.shape == vt.shape and k.shape == vt.shape, f"q/k must match vt shape. got q={tuple(q.shape)}, k={tuple(k.shape)}, vt={tuple(vt.shape)}"
    L, H, Dh = vt.shape[1], vt.shape[2], vt.shape[3]
    if L < 2:
        raise ValueError(f"Need at least 2 tokens to use phrase tokens [0,1]. Got L={L}")

    qh = q.permute(0, 2, 1, 3)  # [1,H,L,Dh]
    kh = k.permute(0, 2, 1, 3)
    logits = torch.matmul(qh, kh.transpose(-1, -2)) / (Dh ** 0.5)
    A = F.softmax(logits, dim=-1)

    if attn_mode == "col":
        imp = A.mean(dim=-2)  # [1,H,L]
    elif attn_mode == "row":
        imp = A.mean(dim=-1)
    else:
        raise ValueError("attn_mode must be 'col' or 'row'")

    imp01 = imp[:, :, 0:2]  # [1,H,2]
    w = imp01 / imp01.sum(dim=-1, keepdim=True).clamp_min(eps)  # [1,H,2]

    vt01 = vt[:, 0:2, :, :].permute(0, 2, 1, 3)  # [1,H,2,Dh]
    d = (w.unsqueeze(-1) * vt01).sum(dim=-2, keepdim=True)      # [1,H,1,Dh]

    if normalize_detector:
        d = d / torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(eps)

    d = d.permute(0, 2, 1, 3)  # [1,1,H,Dh]

    vt2 = vt.clone()
    if L > fill_from:
        vt2[:, fill_from:, :, :] = d.expand(1, L - fill_from, H, Dh)
    if zero_sot:
        vt2[:, 0, :, :] = 0.0
    return vt2


# ---------------------------
# Retain basis construction
# ---------------------------

def _retain_basis_from_vt_list(v_list: Optional[List[torch.Tensor]], *, top_k: int = 3) -> Optional[torch.Tensor]:
    """
    Build V_ret,j via "stack SVD" across retain exemplars (kept close to your old projection_retain),
    but output an ORTHONORMAL basis V (columns), not a dense projector matrix.

    For each exemplar v (flattened to [D, L] or [D] depending), we take top_k left singular vectors,
    stack them, then orth again.
    """
    if v_list is None or len(v_list) == 0:
        return None

    U_stack = None
    for vt in v_list:
        # vt: [1,L,H,Dh] -> flatten token 1 direction (same as target/anchor)
        d = _vt_to_dir(vt)  # [D]
        # Treat it as a rank-1 "matrix": [D,1]
        M = d.unsqueeze(1)  # [D,1]
        # SVD of rank-1 is trivial, but keep structure for future (e.g., token-mean)
        u, _, _ = torch.linalg.svd(M, full_matrices=False)
        imp = u[:, : min(top_k, u.shape[1])]
        U_stack = imp if U_stack is None else torch.cat([U_stack, imp], dim=1)

    if U_stack is None:
        return None

    Vret = _orth_columns(U_stack)
    return Vret


def flux_finalize_cora_bases(
    *,
    retain_top_k: int = 3,
    eps: float = 1e-8,
):
    """
    After you finish recording retain/target/anchor exemplars, call this ONCE.

    It constructs:
      - V_ret,j for each block (dual + single)
      - U_{c,j} per concept and block (Eq. 10)
      - A_{c,j} per concept and block (anchors)
      - Union U for convenience if you don't specify active_concept
    """
    global _FLUX_VRET_DUAL, _FLUX_VRET_SINGLE
    global _FLUX_U_DUAL, _FLUX_U_SINGLE, _FLUX_A_DUAL, _FLUX_A_SINGLE
    global _FLUX_U_UNION_DUAL, _FLUX_U_UNION_SINGLE

    _FLUX_VRET_DUAL.clear()
    _FLUX_VRET_SINGLE.clear()
    _FLUX_U_DUAL.clear()
    _FLUX_U_SINGLE.clear()
    _FLUX_A_DUAL.clear()
    _FLUX_A_SINGLE.clear()
    _FLUX_U_UNION_DUAL.clear()
    _FLUX_U_UNION_SINGLE.clear()

    # ---- retain bases V_ret,j ----
    for blk, vlist in _FLUX_RETAIN_VT_BANK_DUAL.items():
        Vret = _retain_basis_from_vt_list(vlist, top_k=retain_top_k)
        if Vret is not None:
            _FLUX_VRET_DUAL[blk] = Vret

    for blk, vlist in _FLUX_RETAIN_VT_BANK_SINGLE.items():
        Vret = _retain_basis_from_vt_list(vlist, top_k=retain_top_k)
        if Vret is not None:
            _FLUX_VRET_SINGLE[blk] = Vret

    # helper: apply (I - P) to a direction d
    def free_dir(d: torch.Tensor, Vret: Optional[torch.Tensor]) -> torch.Tensor:
        if Vret is None or Vret.numel() == 0 or Vret.shape[1] == 0:
            return d
        proj = Vret @ (Vret.t() @ d)
        return d - _FLUX_RETAIN_LAMBDA * proj

    # ---- target U_{c,j} and anchor A_{c,j} ----
    # Dual
    for blk, concept_map in _FLUX_TARGET_VT_BANK_DUAL.items():
        Vret = _FLUX_VRET_DUAL.get(blk, None)
        _FLUX_U_DUAL.setdefault(blk, {})
        # Build U per concept
        for concept, vt_list in concept_map.items():
            dirs = []
            for vt in vt_list:
                d = _vt_to_dir(vt, eps=eps)        # \tilde v
                d = free_dir(d, Vret)              # (I - P) \tilde v
                d = d / (d.norm() + eps)
                dirs.append(d)
            if len(dirs) == 0:
                continue
            M = torch.stack(dirs, dim=1)          # [D, R]
            Uc = _orth_columns(M, eps=eps)        # Eq(10)
            _FLUX_U_DUAL[blk][concept] = Uc

    # Dual anchors A
    for blk, concept_map in _FLUX_ANCHOR_VT_BANK_DUAL.items():
        Vret = _FLUX_VRET_DUAL.get(blk, None)
        _FLUX_A_DUAL.setdefault(blk, {})
        for concept, vt_list in concept_map.items():
            dirs = []
            for vt in vt_list:
                a = _vt_to_dir(vt, eps=eps)
                a = free_dir(a, Vret)             # (I - P) anchor
                a = a / (a.norm() + eps)
                dirs.append(a)
            if len(dirs) == 0:
                continue
            M = torch.stack(dirs, dim=1)
            Ac = _orth_columns(M, eps=eps)
            _FLUX_A_DUAL[blk][concept] = Ac

    # Single
    for blk, concept_map in _FLUX_TARGET_VT_BANK_SINGLE.items():
        Vret = _FLUX_VRET_SINGLE.get(blk, None)
        _FLUX_U_SINGLE.setdefault(blk, {})
        for concept, vt_list in concept_map.items():
            dirs = []
            for vt in vt_list:
                d = _vt_to_dir(vt, eps=eps)
                d = free_dir(d, Vret)
                d = d / (d.norm() + eps)
                dirs.append(d)
            if len(dirs) == 0:
                continue
            M = torch.stack(dirs, dim=1)
            Uc = _orth_columns(M, eps=eps)
            _FLUX_U_SINGLE[blk][concept] = Uc

    for blk, concept_map in _FLUX_ANCHOR_VT_BANK_SINGLE.items():
        Vret = _FLUX_VRET_SINGLE.get(blk, None)
        _FLUX_A_SINGLE.setdefault(blk, {})
        for concept, vt_list in concept_map.items():
            dirs = []
            for vt in vt_list:
                a = _vt_to_dir(vt, eps=eps)
                a = free_dir(a, Vret)
                a = a / (a.norm() + eps)
                dirs.append(a)
            if len(dirs) == 0:
                continue
            M = torch.stack(dirs, dim=1)
            Ac = _orth_columns(M, eps=eps)
            _FLUX_A_SINGLE[blk][concept] = Ac

    # ---- Union bases (optional convenience) ----
    for blk, cmap in _FLUX_U_DUAL.items():
        allU = [U for U in cmap.values() if U is not None and U.numel() > 0 and U.shape[1] > 0]
        if len(allU) == 0:
            continue
        M = torch.cat(allU, dim=1)
        _FLUX_U_UNION_DUAL[blk] = _orth_columns(M, eps=eps)

    for blk, cmap in _FLUX_U_SINGLE.items():
        allU = [U for U in cmap.values() if U is not None and U.numel() > 0 and U.shape[1] > 0]
        if len(allU) == 0:
            continue
        M = torch.cat(allU, dim=1)
        _FLUX_U_UNION_SINGLE[blk] = _orth_columns(M, eps=eps)


# ---------------------------
# Attention projections helpers (unchanged)
# ---------------------------

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


# ---------------------------
# FluxAttnProcessor (CORA)
# ---------------------------

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
        # toggles
        dual_zero_text_value: bool = False,
        single_zero_text_value: bool = False,
        # text length
        text_seq_len: Optional[int] = None,
        # recording flags
        record_target_vt: bool = False,
        record_retain_vt: bool = False,
        record_anchor_vt: bool = False,
        # CORA apply
        apply_target_proj: bool = False,
        # CORA params
        strength_tau: float = 0.2,         # tau in Eq(11)
        strength_gamma: float = 1.0,       # gamma
        anchor_strength: float = 2.5,      # alpha
        proj_eps: float = 1e-8,
        # selection
        record_concept: Optional[str] = None,   # REQUIRED when record_target_vt or record_anchor_vt
        active_concept: Optional[str] = None,   # optional concept to apply erase+replace; if None -> union erase
        use_anchors: bool = True,               # if True and active_concept provided -> erase+replace
        # block routing
        block_index: Optional[int] = None,
        target_block_indices: Optional[List[int]] = None,
        target_single_block_indices: Optional[List[int]] = None,
        # vt record controls
        vt_dedup_cos_thr: float = _VT_DEDUP_COS_THR,
        max_target_vt_per_block: int = 8,
        max_retain_vt_per_block: int = 16,
        max_anchor_vt_per_block: int = 8,
        # token window
        proj_token_end: Optional[int] = None,
        # unused legacy args accepted by caller
        proj_strength: float = 1.0,
        proj_strength_tokens: Optional[torch.Tensor] = None,
        target_block_index: int = 0,
        **_unused,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # -----------------------
        # SINGLE STREAM (no encoder_hidden_states)
        # -----------------------
        if encoder_hidden_states is None and text_seq_len is not None and block_index is not None:
            target_single_block_indices = target_single_block_indices or []

            # Optional: zero text values (debug)
            if single_zero_text_value:
                value[:, :text_seq_len] = 0.0

            # ---- recording (single) ----
            if (record_retain_vt or record_target_vt or record_anchor_vt) and (block_index in target_single_block_indices):
                vt_single = value[:, :text_seq_len].detach()[:1].contiguous()
                q_txt = query[:, :text_seq_len].detach()[:1].contiguous()
                k_txt = key[:, :text_seq_len].detach()[:1].contiguous()

                vt_single = _make_attn01_duplicated_vt(vt_single, q_txt, k_txt, attn_mode="col")

                if record_retain_vt:
                    _bank_add_vt(
                        _FLUX_RETAIN_VT_BANK_SINGLE,
                        block_index,
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
                    if not record_concept:
                        raise ValueError("record_anchor_vt=True requires joint_attention_kwargs['record_concept']")
                    _bank_add_concept_vt(
                        _FLUX_ANCHOR_VT_BANK_SINGLE,
                        block_index,
                        record_concept,
                        vt_single,
                        max_keep=max_anchor_vt_per_block,
                        dedup_thr=vt_dedup_cos_thr,
                    )

            # ---- apply CORA (single) ----
            if apply_target_proj and (block_index in target_single_block_indices):
                # token window
                s = 0
                e = int(text_seq_len if proj_token_end is None else max(1, min(int(proj_token_end), text_seq_len)))

                v_txt = value[:, :text_seq_len].reshape(value.shape[0], text_seq_len, -1)  # [B,T,D]
                v_slice = v_txt[:, s:e, :]

                Vret = _FLUX_VRET_SINGLE.get(block_index, None)
                if active_concept is not None:
                    U = _FLUX_U_SINGLE.get(block_index, {}).get(active_concept, None)
                    A = _FLUX_A_SINGLE.get(block_index, {}).get(active_concept, None) if use_anchors else None
                    use_replace = bool(use_anchors and (A is not None))
                else:
                    U = _FLUX_U_UNION_SINGLE.get(block_index, None)
                    A = None
                    use_replace = False

                if U is not None and U.numel() > 0 and U.shape[1] > 0:
                    U = U.to(device=v_slice.device, dtype=torch.float32)
                    A = None if A is None else A.to(device=v_slice.device, dtype=torch.float32)
                    Vret = None if Vret is None else Vret.to(device=v_slice.device, dtype=torch.float32)

                    v_slice2 = _cora_erase_replace(
                        v_slice,
                        Vret=Vret,
                        U=U,
                        A=A,
                        tau=float(strength_tau),
                        gamma=float(strength_gamma),
                        anchor_strength=float(anchor_strength),
                        eps=float(proj_eps),
                        use_replace=use_replace,
                    )
                    v_txt = torch.cat([v_txt[:, :s, :], v_slice2, v_txt[:, e:, :]], dim=1)
                    value_txt_new = v_txt.view(value.shape[0], text_seq_len, value.shape[2], value.shape[3])
                    value[:, :text_seq_len] = torch.nan_to_num(value_txt_new, nan=0.0, posinf=0.0, neginf=0.0)

        # -----------------------
        # DOUBLE STREAM (with encoder_hidden_states)
        # -----------------------
        if attn.added_kv_proj_dim is not None and encoder_hidden_states is not None and block_index is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            target_block_indices = target_block_indices or []

            # ---- recording (dual) ----
            if (record_retain_vt or record_target_vt or record_anchor_vt) and (block_index in target_block_indices):
                vt_dual = encoder_value.detach()[:1].contiguous()
                q_txt = encoder_query.detach()[:1].contiguous()
                k_txt = encoder_key.detach()[:1].contiguous()

                vt_dual = _make_attn01_duplicated_vt(vt_dual, q_txt, k_txt, attn_mode="col")

                if record_retain_vt:
                    _bank_add_vt(
                        _FLUX_RETAIN_VT_BANK_DUAL,
                        block_index,
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
                    if not record_concept:
                        raise ValueError("record_anchor_vt=True requires joint_attention_kwargs['record_concept']")
                    _bank_add_concept_vt(
                        _FLUX_ANCHOR_VT_BANK_DUAL,
                        block_index,
                        record_concept,
                        vt_dual,
                        max_keep=max_anchor_vt_per_block,
                        dedup_thr=vt_dedup_cos_thr,
                    )

            # ---- apply CORA (dual) ----
            if apply_target_proj and (block_index in target_block_indices):
                v = encoder_value.reshape(encoder_value.shape[0], encoder_value.shape[1], -1)  # [B,T,D]
                s = 0
                e = int(v.shape[1] if proj_token_end is None else max(1, min(int(proj_token_end), v.shape[1])))
                v_slice = v[:, s:e, :]

                Vret = _FLUX_VRET_DUAL.get(block_index, None)
                if active_concept is not None:
                    U = _FLUX_U_DUAL.get(block_index, {}).get(active_concept, None)
                    A = _FLUX_A_DUAL.get(block_index, {}).get(active_concept, None) if use_anchors else None
                    use_replace = bool(use_anchors and (A is not None))
                else:
                    U = _FLUX_U_UNION_DUAL.get(block_index, None)
                    A = None
                    use_replace = False

                if U is not None and U.numel() > 0 and U.shape[1] > 0:
                    U = U.to(device=v_slice.device, dtype=torch.float32)
                    A = None if A is None else A.to(device=v_slice.device, dtype=torch.float32)
                    Vret = None if Vret is None else Vret.to(device=v_slice.device, dtype=torch.float32)

                    v_slice2 = _cora_erase_replace(
                        v_slice,
                        Vret=Vret,
                        U=U,
                        A=A,
                        tau=float(strength_tau),
                        gamma=float(strength_gamma),
                        anchor_strength=float(anchor_strength),
                        eps=float(proj_eps),
                        use_replace=use_replace,
                    )
                    v = torch.cat([v[:, :s, :], v_slice2, v[:, e:, :]], dim=1)
                    encoder_value = v.view_as(encoder_value)
                    encoder_value = torch.nan_to_num(encoder_value, nan=0.0, posinf=0.0, neginf=0.0)

            if dual_zero_text_value:
                encoder_value = encoder_value * 0.0

            # Merge dual+single streams for attention
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        # Rotary & attention
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


# ---------------------------
# Model code below unchanged (except processor class)
# ---------------------------

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
        quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states"}
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
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
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
        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
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
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
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

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )

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
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, temb, image_rotary_emb, ja
                )
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
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, temb, image_rotary_emb, ja
                )
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