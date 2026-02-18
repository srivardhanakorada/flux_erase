# ===========================
# PATCH: Dual-Zero + Single-Zero toggles for text VALUE vectors
# + Milestone A: record-only (print-once)
# + Milestone B: single-concept target projection (single dual block)  [backward compatible]
# + Milestone C: multi-block vt bank + multi-block projection + strength (VISIBLE edits)
# ===========================

# ===== Milestone A globals =====
_FLUX_RECORD_PRINT_COUNT = 0

# ===== Milestone B globals (kept for backward compatibility) =====
_FLUX_TARGET_VT = None               # stored target value direction [1, L_txt, H, Dh]
_FLUX_TARGET_VT_READY = False        # recorded once
_FLUX_TARGET_PROJ_APPLY_COUNT = 0    # throttle projection prints

# ===== Milestone C globals =====
_FLUX_TARGET_VT_BANK = {}            # dict[int, Tensor] : block_index -> [1, L_txt, H, Dh]
_FLUX_TARGET_VT_READY_SET = set()    # set[int] : which blocks are recorded
_FLUX_PROJ_PRINT_COUNT = 0           # throttle multi-block projection prints

# ===== Milestone D globals (single blocks) =====
_FLUX_SINGLE_VT_BANK = {}            # dict[int, Tensor] : single_block_index -> [1, L_txt, H, Dh]
_FLUX_SINGLE_VT_READY_SET = set()    # set[int]
_FLUX_SINGLE_PROJ_PRINT_COUNT = 0

import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def _make_lasttoken_duplicated_vt(
    vt: torch.Tensor,               # [1, L, H, Dh]
    skip_token0: bool = True,
    last_token_index: int = -1,     # -1 means last token
) -> torch.Tensor:
    """
    AdaVD-style token duplication:
    pick ONE token's value vector (default: last token) and broadcast it to all
    (non-special) token positions.

    vt expected shape: [1, L, H, Dh]
    """
    assert vt.ndim == 4 and vt.shape[0] == 1, f"Expected vt [1,L,H,Dh], got {tuple(vt.shape)}"
    L = vt.shape[1]

    # pick token index safely
    idx = last_token_index
    if idx < 0:
        idx = L + idx
    idx = max(0, min(idx, L - 1))

    # v_last: [1, 1, H, Dh]
    v_last = vt[:, idx:idx+1, :, :]

    vt2 = vt.clone()
    start_fill = 1 if skip_token0 else 0
    if L > start_fill:
        vt2[:, start_fill:, :, :] = v_last.expand(1, L - start_fill, vt.shape[2], vt.shape[3])

    if skip_token0:
        vt2[:, 0, :, :] = 0.0
    return vt2

def _make_avg_duplicated_vt(
    vt: torch.Tensor,               # [1, L, H, Dh]
    skip_token0: bool = True,
    avg_token_start: Optional[int] = None,
    avg_token_end: Optional[int] = None,
) -> torch.Tensor:
    """
    Take vt (target VALUEs) and replace token positions 1..L-1 with a single
    averaged token vector (average over a chosen token range).
    This matches the "token duplication" spirit: one concept direction broadcast
    across all (non-special) token positions.

    vt is expected to be batch=1 (we store templates as [1, L, H, Dh]).
    """
    assert vt.ndim == 4 and vt.shape[0] == 1, f"Expected vt [1,L,H,Dh], got {tuple(vt.shape)}"

    L = vt.shape[1]
    # choose averaging window
    s = 1 if skip_token0 else 0
    if avg_token_start is not None:
        s = int(avg_token_start)
    e = L
    if avg_token_end is not None:
        e = int(avg_token_end)

    s = max(0, min(s, L))
    e = max(s, min(e, L))

    # if empty range, just return original (safe fallback)
    if e <= s:
        return vt

    # average over tokens [s:e] -> [1, 1, H, Dh]
    vbar = vt[:, s:e, :, :].mean(dim=1, keepdim=True)

    vt2 = vt.clone()
    # broadcast vbar to all tokens except maybe token0
    start_fill = 1 if skip_token0 else 0
    if L > start_fill:
        vt2[:, start_fill:, :, :] = vbar.expand(1, L - start_fill, vt.shape[2], vt.shape[3])
    # ensure token0 zero if skipping
    if skip_token0:
        vt2[:, 0, :, :] = 0.0
    return vt2

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

class FluxAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version."
            )

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        # ===== toggles =====
        dual_zero_text_value: bool = False,
        single_zero_text_value: bool = False,
        text_seq_len: Optional[int] = None,
        # ===== Milestone A: record-only =====
        record_text_values: bool = False,
        record_max_print: int = 1,
        record_prefix: str = "",
        # ===== Milestone B/C: target projection toggles =====
        record_target_vt: bool = False,
        apply_target_proj: bool = False,
        target_block_index: int = 0,          # (B) apply/record at this dual block
        block_index: Optional[int] = None,    # provided by model forward
        skip_token0: bool = True,             # (B) skip token0 by default
        proj_eps: float = 1e-8,
        # ===== Milestone C: multi-block + stronger apply =====
        target_block_indices: Optional[List[int]] = None,  # record/apply set; if None -> [target_block_index]
        proj_strength: float = 1.0,                        # >1 => stronger/visible removal
        proj_token_start: Optional[int] = None,            # if None -> (skip_token0 ? 1 : 0)
        proj_token_end: Optional[int] = None,              # None => full length
        proj_min_vt_norm: float = 1e-6,                    # skip near-zero vt tokens (padding)
        # ===== Milestone D: single-block vt bank + projection =====
        target_single_block_index: int = 0,
        target_single_block_indices: Optional[List[int]] = None,  # if None -> [target_single_block_index]
        # ===== New: average-token vt duplication (AdaVD-style token duplication) =====
        avg_vt_tokens: bool = True,                 # turn on average-token duplication when recording vt
        avg_token_start: Optional[int] = None,       # if None -> proj_token_start (or 1)
        avg_token_end: Optional[int] = None,         # if None -> proj_token_end (or full length)
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        # unflatten into heads early
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # ===========================
        # Milestone A: print shapes ONCE
        # ===========================
        global _FLUX_RECORD_PRINT_COUNT
        if record_text_values and (_FLUX_RECORD_PRINT_COUNT < record_max_print):
            prefix = f"[{record_prefix}] " if record_prefix else ""
            if (attn.added_kv_proj_dim is not None) and (encoder_hidden_states is not None) and (encoder_value is not None):
                enc_v = encoder_value.unflatten(-1, (attn.heads, -1))
                print(
                    f"{prefix}DUAL encoder VALUE shape: {tuple(enc_v.shape)} "
                    f"(B,L_txt,H,Dh) heads={attn.heads} head_dim={attn.head_dim} block_index={block_index}"
                )
            elif (encoder_hidden_states is None) and (text_seq_len is not None):
                txt_v = value[:, :text_seq_len]
                print(
                    f"{prefix}SINGLE text VALUE shape: {tuple(txt_v.shape)} "
                    f"(B,L_txt,H,Dh) heads={attn.heads} head_dim={attn.head_dim} block_index={block_index}"
                )
            else:
                print(
                    f"{prefix}VALUE shape: {tuple(value.shape)} (B,L,H,Dh) "
                    f"heads={attn.heads} head_dim={attn.head_dim} block_index={block_index}"
                )
            _FLUX_RECORD_PRINT_COUNT += 1
        # ===========================

        # norms
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # ===== single-stream text-value zeroing =====
        if encoder_hidden_states is None and single_zero_text_value and (text_seq_len is not None):
            value[:, :text_seq_len] = 0.0

        # ===========================
        # Milestone D: SINGLE-block vt record/apply on text-prefix VALUE
        # ===========================
        if (encoder_hidden_states is None) and (text_seq_len is not None) and (block_index is not None):
            # determine active single blocks list
            if target_single_block_indices is None:
                target_single_block_indices = [target_single_block_index]

            # record: store vt for selected single blocks
            global _FLUX_SINGLE_VT_BANK, _FLUX_SINGLE_VT_READY_SET
            if record_target_vt and (block_index in target_single_block_indices) and (block_index not in _FLUX_SINGLE_VT_READY_SET):
                vt_single = value[:, :text_seq_len].detach()  # [B,L_txt,H,Dh]
                vt_single = vt_single[:1].contiguous()        # store template as batch=1
                # ---- NEW: average-token duplication ----
                if avg_vt_tokens:
                    # default averaging window: use avg_token_* if provided else fall back to proj_token_* semantics
                    a_s = avg_token_start if (avg_token_start is not None) else proj_token_start
                    a_e = avg_token_end   if (avg_token_end   is not None) else proj_token_end
                    # vt_single = _make_lasttoken_duplicated_vt(
                    #     vt_single,
                    #     skip_token0=skip_token0,
                    #     avg_token_start=a_s,
                    #     avg_token_end=a_e,
                    # )
                    vt_single = _make_lasttoken_duplicated_vt(
                        vt_single,
                        skip_token0=skip_token0,
                        last_token_index=-1,
                    )
                _FLUX_SINGLE_VT_BANK[block_index] = vt_single
                _FLUX_SINGLE_VT_READY_SET.add(block_index)
                print(f"[MIL_D] Recorded vt for SINGLE block {block_index}: {tuple(_FLUX_SINGLE_VT_BANK[block_index].shape)}")
            # apply projection
            do_apply_single = apply_target_proj and (block_index in target_single_block_indices) and (block_index in _FLUX_SINGLE_VT_BANK)
            if do_apply_single:
                vt = _FLUX_SINGLE_VT_BANK[block_index]
                if vt.device != value.device or vt.dtype != value.dtype:
                    vt = vt.to(device=value.device, dtype=value.dtype)

                # flatten heads: [B,L_txt,H,Dh] -> [B,L_txt,D]
                v_txt = value[:, :text_seq_len].reshape(value.shape[0], text_seq_len, -1)
                vt_flat = vt.reshape(1, text_seq_len, -1)  # [1,L_txt,D]

                # token range defaults (same semantics as Milestone C)
                if proj_token_start is None:
                    s = 1 if skip_token0 else 0
                else:
                    s = int(proj_token_start)
                s = max(0, min(s, text_seq_len))
                e = int(proj_token_end) if (proj_token_end is not None) else text_seq_len
                e = max(s, min(e, text_seq_len))

                if e > s:
                    v_slice = v_txt[:, s:e, :]
                    vt_slice = vt_flat[:, s:e, :]

                    vt_norm2 = (vt_slice * vt_slice).sum(-1, keepdim=True)  # [1,len,1]
                    mask = (vt_norm2 > proj_min_vt_norm).to(v_slice.dtype)
                    denom = vt_norm2.clamp_min(proj_eps)
                    alpha = (v_slice * vt_slice).sum(-1, keepdim=True) / denom
                    v_slice = v_slice - (proj_strength * alpha * vt_slice) * mask

                    v_txt = torch.cat([v_txt[:, :s, :], v_slice, v_txt[:, e:, :]], dim=1)

                # write back
                value_txt_new = v_txt.view(value.shape[0], text_seq_len, value.shape[2], value.shape[3])
                value[:, :text_seq_len] = torch.nan_to_num(value_txt_new, nan=0.0, posinf=0.0, neginf=0.0)

                global _FLUX_SINGLE_PROJ_PRINT_COUNT
                if _FLUX_SINGLE_PROJ_PRINT_COUNT < 10:
                    rms = value[:, :text_seq_len].float().pow(2).mean().sqrt().item()
                    print(f"[MIL_D] Proj@SINGLE block {block_index} | strength={proj_strength} | tokens=[{s}:{e}] | rms={rms:.6f}")
                    _FLUX_SINGLE_PROJ_PRINT_COUNT += 1
        # ===========================

        # ===== dual-stream path =====
        if attn.added_kv_proj_dim is not None and encoder_hidden_states is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            # ---------------------------
            # Milestone C: determine active blocks list
            # ---------------------------
            if target_block_indices is None:
                target_block_indices = [target_block_index]

            # ---------------------------
            # Milestone B: record single vt (kept)
            # ---------------------------
            # Only do Milestone-B single vt recording when Milestone-C list is NOT provided
            if target_block_indices is None:
                global _FLUX_TARGET_VT, _FLUX_TARGET_VT_READY
                if record_target_vt and (not _FLUX_TARGET_VT_READY) and (block_index == target_block_index):
                    _FLUX_TARGET_VT = encoder_value.detach()
                    _FLUX_TARGET_VT_READY = True
                    print(f"[MIL_B] Recorded target vt at dual block {block_index}: {tuple(_FLUX_TARGET_VT.shape)}")

            # ---------------------------
            # Milestone C: record vt bank (multi-block)
            # ---------------------------
            global _FLUX_TARGET_VT_BANK, _FLUX_TARGET_VT_READY_SET
            if record_target_vt and (block_index in target_block_indices) and (block_index not in _FLUX_TARGET_VT_READY_SET):
                vt_dual = encoder_value.detach()       # [B,L_txt,H,Dh]
                vt_dual = vt_dual[:1].contiguous()     # store template as batch=1
                # ---- NEW: average-token duplication ----
                if avg_vt_tokens:
                    a_s = avg_token_start if (avg_token_start is not None) else proj_token_start
                    a_e = avg_token_end   if (avg_token_end   is not None) else proj_token_end
                    # vt_dual = _make_lasttoken_duplicated_vt(
                    #     vt_dual,
                    #     skip_token0=skip_token0,
                    #     avg_token_start=a_s,
                    #     avg_token_end=a_e,
                    # )
                    vt_dual = _make_lasttoken_duplicated_vt(
                        vt_dual,
                        skip_token0=skip_token0, 
                        last_token_index=-1,
                    )
                _FLUX_TARGET_VT_BANK[block_index] = vt_dual
                _FLUX_TARGET_VT_READY_SET.add(block_index)
                print(f"[MIL_C] Recorded vt for dual block {block_index}: {tuple(vt_dual.shape)}")

            # ---------------------------
            # Milestone B/C: apply projection
            #   Priority: if vt bank has current block -> use C
            #   Else: fall back to B single vt for backward compatibility
            # ---------------------------
            do_apply_here = apply_target_proj and (block_index is not None) and (block_index in target_block_indices)

            if do_apply_here:
                # pick vt source
                vt_source = None
                if block_index in _FLUX_TARGET_VT_BANK:
                    vt_source = _FLUX_TARGET_VT_BANK[block_index]
                    tag = "MIL_C"
                elif _FLUX_TARGET_VT_READY and (block_index == target_block_index) and (_FLUX_TARGET_VT is not None):
                    vt_source = _FLUX_TARGET_VT
                    tag = "MIL_B"
                else:
                    vt_source = None

                if vt_source is not None:
                    vt = vt_source
                    if vt.device != encoder_value.device or vt.dtype != encoder_value.dtype:
                        vt = vt.to(device=encoder_value.device, dtype=encoder_value.dtype)

                    # token range defaults
                    if proj_token_start is None:
                        proj_token_start_eff = 1 if skip_token0 else 0
                    else:
                        proj_token_start_eff = int(proj_token_start)

                    # flatten heads: [B, L, H, Dh] -> [B, L, D]
                    v = encoder_value.reshape(encoder_value.shape[0], encoder_value.shape[1], -1)
                    vt_flat = vt.reshape(1, vt.shape[1], -1)  # broadcast over batch

                    L = v.shape[1]
                    s = max(0, min(proj_token_start_eff, L))
                    e = int(proj_token_end) if (proj_token_end is not None) else L
                    e = max(s, min(e, L))

                    if e > s:
                        v_slice = v[:, s:e, :]
                        vt_slice = vt_flat[:, s:e, :]

                        # skip padding/near-zero vt tokens (common with max_length=512)
                        vt_norm2 = (vt_slice * vt_slice).sum(-1, keepdim=True)  # [1, len, 1]
                        mask = (vt_norm2 > proj_min_vt_norm).to(v_slice.dtype)

                        denom = vt_norm2.clamp_min(proj_eps)
                        alpha = (v_slice * vt_slice).sum(-1, keepdim=True) / denom  # [B, len, 1]

                        v_slice = v_slice - (proj_strength * alpha * vt_slice) * mask

                        v = torch.cat([v[:, :s, :], v_slice, v[:, e:, :]], dim=1)

                    encoder_value = v.view_as(encoder_value)
                    encoder_value = torch.nan_to_num(encoder_value, nan=0.0, posinf=0.0, neginf=0.0)

                    # print throttling
                    global _FLUX_TARGET_PROJ_APPLY_COUNT, _FLUX_PROJ_PRINT_COUNT
                    if tag == "MIL_B":
                        if _FLUX_TARGET_PROJ_APPLY_COUNT < 5:
                            rms = encoder_value.float().pow(2).mean().sqrt().item()
                            print(f"[MIL_B] Applied target projection at dual block {block_index} | rms={rms:.6f}")
                            _FLUX_TARGET_PROJ_APPLY_COUNT += 1
                    else:
                        if _FLUX_PROJ_PRINT_COUNT < 10:
                            rms = encoder_value.float().pow(2).mean().sqrt().item()
                            print(
                                f"[MIL_C] Proj@dual block {block_index} | strength={proj_strength} | "
                                f"tokens=[{s}:{e}] | rms={rms:.6f}"
                            )
                            _FLUX_PROJ_PRINT_COUNT += 1

            # ===== dual-stream text-value zeroing =====
            if dual_zero_text_value:
                encoder_value = encoder_value * 0.0

            # concat context + image stream
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
        else:
            return hidden_states

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
        joint_attention_kwargs["text_seq_len"] = text_seq_len  # needed for single_zero_text_value

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
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, qk_norm: str = "rms_norm", eps: float = 1e-6
    ):
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
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        joint_attention_kwargs = joint_attention_kwargs or {}

        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        if len(attention_outputs) == 3:
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
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )

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

        # ---- ALWAYS work on a local copy (avoid mutating caller dict across calls) ----
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
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

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
            logger.warning("Passing `txt_ids` 3d torch.Tensor is deprecated. Please remove the batch dimension.")
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning("Passing `img_ids` 3d torch.Tensor is deprecated. Please remove the batch dimension.")
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        if is_torch_npu_available():
            freqs_cos, freqs_sin = self.pos_embed(ids.cpu())
            image_rotary_emb = (freqs_cos.npu(), freqs_sin.npu())
        else:
            image_rotary_emb = self.pos_embed(ids)

        # ---- handle ip-adapter embeds WITHOUT mutating external state ----
        if "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs["ip_hidden_states"] = ip_hidden_states

        # ---- dual blocks ----
        for index_block, block in enumerate(self.transformer_blocks):
            ja = joint_attention_kwargs.copy()
            ja["block_index"] = index_block  # REQUIRED for Milestone B/C in dual blocks

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    ja,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=ja,
                )

            if controlnet_block_samples is not None:
                interval_control = int(np.ceil(len(self.transformer_blocks) / len(controlnet_block_samples)))
                if controlnet_blocks_repeat:
                    hidden_states = hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        # ---- single blocks ----
        for index_block, block in enumerate(self.single_transformer_blocks):
            ja = joint_attention_kwargs.copy()
            ja["block_index"] = index_block  # ok for debug/record prints; text_seq_len set inside block

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    ja,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=ja,
                )

            if controlnet_single_block_samples is not None:
                interval_control = int(np.ceil(len(self.single_transformer_blocks) / len(controlnet_single_block_samples)))
                hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)