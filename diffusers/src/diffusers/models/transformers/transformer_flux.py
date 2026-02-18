# transformer_flux.py
import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

import torch  # type:ignore
import torch.nn as nn  # type:ignore
import torch.nn.functional as F  # type:ignore

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

# -----------------------------------------------------------------------------
# Globals for VT bank / concept erase debugging
# -----------------------------------------------------------------------------
_FLUX_TARGET_VT = None
_FLUX_TARGET_VT_READY = False
_FLUX_TARGET_VT_BANK: Dict[int, torch.Tensor] = {}
_FLUX_TARGET_VT_READY_SET = set()
_FLUX_SINGLE_VT_BANK: Dict[int, torch.Tensor] = {}
_FLUX_SINGLE_VT_READY_SET = set()
_FLUX_CONCEPT_C = None
_FLUX_PRINT_STRENGTH_EVERY = 32
_FLUX_PRINT_STRENGTH_COUNT = 0
_FLUX_SINGLE_VT_BANK_RETAIN: Dict[int, list[torch.Tensor]] = {}
_FLUX_SINGLE_VT_READY_SET_RETAIN = set()
_FLUX_TARGET_VT_BANK_RETAIN: Dict[int, list[torch.Tensor]] = {}
_FLUX_TARGET_VT_READY_SET_RETAIN = set()
logger = logging.get_logger(__name__)

def _flux_debug_print_strength(
    block_index: Optional[int],
    is_dual: bool,
    s: int,
    e: int,
    proj_strength: float,
    proj_strength_tokens: Optional[torch.Tensor],
    strength_tau: float,
    strength_gamma: float,
    g_slice: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
):
    """
    tqdm-safe debug printer:
      - Uses tqdm.write() to avoid breaking progress bars
      - Falls back to print() if tqdm isn't available
    """
    global _FLUX_PRINT_STRENGTH_COUNT

    _FLUX_PRINT_STRENGTH_COUNT += 1
    if (_FLUX_PRINT_STRENGTH_COUNT % _FLUX_PRINT_STRENGTH_EVERY) != 0:
        return

    # Lazy import so diffusers core doesn't require tqdm
    try:
        from tqdm.auto import tqdm as _tqdm #type:ignore
        _write = _tqdm.write
    except Exception:
        _write = print

    with torch.no_grad():
        # ---- g stats ----
        if g_slice is None and proj_strength_tokens is not None:
            g_all = proj_strength_tokens
            if g_all.ndim == 2:
                g_all = g_all[0]
            g_slice = g_all[s:e]

        if g_slice is None:
            mn = mx = mean = float(proj_strength)
            nz = int(e - s)
        else:
            g = g_slice.detach().float().flatten()
            mn = float(g.min().item())
            mx = float(g.max().item())
            mean = float(g.mean().item())
            nz = int((g > 1e-6).sum().item())

        # ---- alpha stats ----
        if alpha is None:
            a_min = a_mean = a_max = float("nan")
        else:
            a = alpha.detach().float().abs().flatten()
            a_min = float(a.min().item())
            a_mean = float(a.mean().item())
            a_max = float(a.max().item())

        stream = "DUAL" if is_dual else "SINGLE"
        b = -1 if block_index is None else int(block_index)
        tok_len = max(0, int(e - s))

        _write(
            f"[PROJ_STRENGTH] stream={stream} block={b} tok=[{s}:{e}) "
            f"tau={float(strength_tau):.4f} gamma={float(strength_gamma):.4f} "
            f"base={float(proj_strength):.4f} "
            f"g(min/mean/max)={mn:.4f}/{mean:.4f}/{mx:.4f} nz={nz}/{tok_len} "
            f"alpha_abs(min/mean/max)={a_min:.4f}/{a_mean:.4f}/{a_max:.4f}"
        )

def flux_set_concept_embed(c: torch.Tensor):
    global _FLUX_CONCEPT_C
    _FLUX_CONCEPT_C = c.detach().contiguous()

def flux_get_concept_embed():
    global _FLUX_CONCEPT_C
    return _FLUX_CONCEPT_C

def _make_lasttoken_duplicated_vt(
    vt: torch.Tensor,
    last_token_index: int = -1,
) -> torch.Tensor:
    """
    vt: [1, L, H, Dh]
    Returns vt2 where positions [1:] are filled with the last-token value vector,
    and vt2[:,0] is zeroed.
    """
    assert vt.ndim == 4 and vt.shape[0] == 1, f"Expected vt [1,L,H,Dh], got {tuple(vt.shape)}"
    L = vt.shape[1]
    idx = last_token_index
    if idx < 0: idx = L + idx
    idx = max(0, min(idx, L - 1))
    idx = 0
    v_last = vt[:, idx : idx + 1, :, :]  # [1,1,H,Dh]
    # print(f"idx:{idx}")
    vt2 = vt.clone()
    start_fill = 1
    if L > start_fill: vt2[:, start_fill:, :, :] = v_last.expand(1, L - start_fill, vt.shape[2], vt.shape[3])
    vt2[:, 0, :, :] = 0.0
    return vt2

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
    Attention-weighted detector built from phrase tokens [0,1], then duplicated across [fill_from:].
    vt, q, k: [1, L, H, Dh] in head space (i.e. after unflatten + RMSNorm for q/k)
    """
    assert vt.ndim == 4 and vt.shape[0] == 1, f"Expected vt [1,L,H,Dh], got {tuple(vt.shape)}"
    assert q.shape == vt.shape and k.shape == vt.shape, f"q/k must match vt shape. got q={tuple(q.shape)}, k={tuple(k.shape)}, vt={tuple(vt.shape)}"
    L, H, Dh = vt.shape[1], vt.shape[2], vt.shape[3]
    if L < 2:
        raise ValueError(f"Need at least 2 tokens to use phrase tokens [0,1]. Got L={L}")
    # Build attention A from q,k: [1,L,H,Dh] -> [1,H,L,Dh]
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 1, 3)
    logits = torch.matmul(qh, kh.transpose(-1, -2)) / (Dh ** 0.5)  # [1,H,L,L]
    A = F.softmax(logits, dim=-1)  # query->key
    # Token importance per head: [1,H,L]
    if attn_mode == "col":
        # importance(key=t): how much all queries attend TO token t
        imp = A.mean(dim=-2)  # mean over query dimension -> [1,H,L]
    elif attn_mode == "row":
        # importance(query=t): how token t attends to all keys
        imp = A.mean(dim=-1)  # mean over key dimension -> [1,H,L]
    else:
        raise ValueError("attn_mode must be 'col' or 'row'")
    # Restrict to phrase tokens [0,1] and normalize to weights per head: [1,H,2]
    imp01 = imp[:, :, 0:2]
    w = imp01 / imp01.sum(dim=-1, keepdim=True).clamp_min(eps)
    # Values for tokens [0,1]: vt01 [1,2,H,Dh] -> [1,H,2,Dh]
    vt01 = vt[:, 0:2, :, :].permute(0, 2, 1, 3)
    # Weighted sum over the 2 phrase tokens -> detector d: [1,H,1,Dh]
    d = (w.unsqueeze(-1) * vt01).sum(dim=-2, keepdim=True)
    if normalize_detector:
        d = d / torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(eps)
    # Back to token axis shape: [1,1,H,Dh]
    d = d.permute(0, 2, 1, 3)
    # Duplicate like before
    vt2 = vt.clone()
    if L > fill_from:
        vt2[:, fill_from:, :, :] = d.expand(1, L - fill_from, H, Dh)
    if zero_sot:
        vt2[:, 0, :, :] = 0.0
    return vt2 

def projection_retain(v_list: list[torch.Tensor],top_k = 3):
    """
    Given a list of VT tensors to retain, compute a single retain vector by averaging them and normalizing.
    """
    # each tensor is 1 x 512 x 24 x 128
    # convert each to 512 x 3072 first
    # perform singular value decomposition of each of these matrices to extract top basis vectors for each of them. 
    # we take top 3 basis vectors from each of the tensors and concatenate them into a single (3xN)x 3072 matrix, then we perform a final SVD on this matrix and extract top 3 singular vectors
    # we generate a projection matrix using these extracted 3 singular vectors. 
    if len(v_list) == 0:
        return None
    # print("length of v_list:",len(v_list))  # len of retain list
    v_temp = [v.squeeze(0).reshape(v.shape[1], -1).to(torch.float32).T for v in v_list]  # list of [3072,512]
    U_stack = None
    cnt = 0
    for v in v_temp:
        cnt+=1
        u, s, v_t = torch.linalg.svd(v, full_matrices=False)
        imp_vectors = u[:, :top_k]  # [3072,top_k]
        if U_stack is None:
            U_stack = imp_vectors
        else:
            U_stack = torch.cat([U_stack, imp_vectors], dim=1) # [3072, top_k * num_tensors]
    
    u, s, v_t = torch.linalg.svd(U_stack, full_matrices=False)
    v_retain = u[:, :top_k]  # [3072,top_k]
    proj_mat = v_retain @ v_retain.T  # [3072,3072]
    # print("cnt: ",cnt)  # len of retain list
    # assert(cnt==10)
    return proj_mat

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
        dual_zero_text_value: bool = False,
        single_zero_text_value: bool = False,
        text_seq_len: Optional[int] = None,
        record_target_vt: bool = False,
        record_retain_vt: bool = False,
        apply_target_proj: bool = False,
        target_block_index: int = 0,
        block_index: Optional[int] = None,
        proj_eps: float = 1e-8,
        target_block_indices: Optional[List[int]] = None,
        proj_strength: float = 1.0,
        proj_token_end: Optional[int] = None,
        target_single_block_indices: Optional[List[int]] = None,
        proj_strength_tokens: Optional[torch.Tensor] = None,  # adaptive per-token strength (optional)
        strength_tau: float = 0.2,
        strength_gamma: float = 1.0,
    ) -> torch.Tensor:
        # ---------------------------
        # QKV
        # ---------------------------
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # ---------------------------
        # SINGLE stream: encoder_hidden_states is None
        # ---------------------------
        if encoder_hidden_states is None and single_zero_text_value and (text_seq_len is not None):
            value[:, :text_seq_len] = 0.0

        if (encoder_hidden_states is None) and (text_seq_len is not None) and (block_index is not None):
            global _FLUX_SINGLE_VT_BANK, _FLUX_SINGLE_VT_READY_SET, _FLUX_SINGLE_VT_BANK_RETAIN, _FLUX_SINGLE_VT_READY_SET_RETAIN

            # Be robust if None
            target_single_block_indices = target_single_block_indices or []

            if (((record_target_vt and (block_index not in _FLUX_SINGLE_VT_READY_SET)) or (record_retain_vt and (block_index not in _FLUX_SINGLE_VT_READY_SET_RETAIN))) and (block_index in target_single_block_indices)):
                # vt_single = value[:, :text_seq_len].detach()
                # vt_single = vt_single[:1].contiguous()
                # vt_single = _make_lasttoken_duplicated_vt(vt_single, last_token_index=-1)
                vt_single = value[:, :text_seq_len].detach()[:1].contiguous()
                q_txt     = query[:, :text_seq_len].detach()[:1].contiguous()
                k_txt     = key[:, :text_seq_len].detach()[:1].contiguous()
                vt_single = _make_attn01_duplicated_vt(vt_single, q_txt, k_txt, attn_mode="col")
                if(record_target_vt): 
                    #it is assumed that retain is ALWAYS done first, so we can project vt_single to the orthogonal complement of the retain vector. editing here reduces the number of changes.
                    #TODO: for multiple concepts, create a list of retain vectors and create a projection matrix?? for now we just do a single retain vector for simplicity.
                    retain_signal = _FLUX_SINGLE_VT_BANK_RETAIN.get(block_index, None)
                    retain_proj = projection_retain(retain_signal) # debug print of projection retain strength stats
                    assert(retain_proj is not None)
                    # print(retain_signal)
                    # print("retain_signal shape: " ,retain_signal.shape) # 1 x 512 x 24 x 128
                    # assert(1==2) #forcing a break
                    retain_proj = retain_proj.to(device=vt_single.device, dtype=vt_single.dtype)
                    vt_all_head = vt_single.reshape(vt_single.shape[0], vt_single.shape[1], -1) # 1 x 512 x 3072
                    vt_all_head = vt_all_head @ (torch.eye(vt_all_head.shape[-1], device=vt_all_head.device, dtype=vt_all_head.dtype) - retain_proj) # project to orthogonal complement of retain space
                    vt_single = vt_all_head.reshape(vt_single.shape)
                    # vt_single = vt_single - (retain_signal * (vt_single * retain_signal).sum(dim=-1, keepdim=True) / (retain_signal * retain_signal).sum(dim=-1, keepdim=True).clamp_min(1e-8))
                    _FLUX_SINGLE_VT_BANK[block_index] = vt_single
                    _FLUX_SINGLE_VT_READY_SET.add(block_index)
                elif(record_retain_vt):
                    if(block_index not in _FLUX_SINGLE_VT_BANK_RETAIN):
                        _FLUX_SINGLE_VT_BANK_RETAIN[block_index] = []
                    _FLUX_SINGLE_VT_BANK_RETAIN[block_index].append(vt_single)
                    _FLUX_SINGLE_VT_READY_SET_RETAIN.add(block_index)

            do_apply_single = (
                apply_target_proj
                and (block_index in target_single_block_indices)
                and (block_index in _FLUX_SINGLE_VT_BANK)
                and (proj_token_end is not None)
            )

            if do_apply_single:
                vt = _FLUX_SINGLE_VT_BANK[block_index]
                if vt.device != value.device or vt.dtype != value.dtype:
                    vt = vt.to(device=value.device, dtype=value.dtype)

                v_txt = value[:, :text_seq_len].reshape(value.shape[0], text_seq_len, -1)
                vt_flat = vt.reshape(1, text_seq_len, -1)

                s,e = 0,int(proj_token_end)
                # print(f"s,e:{s},{e}")
                v_slice = v_txt[:, s:e, :]
                vt_slice = vt_flat[:, s:e, :]

                vt_norm2 = (vt_slice * vt_slice).sum(-1, keepdim=True)
                denom = vt_norm2.clamp_min(proj_eps)
                alpha = (v_slice * vt_slice).sum(-1, keepdim=True) / denom

                if proj_strength_tokens is None:
                    v_slice = v_slice - (proj_strength * alpha * vt_slice)
                    _flux_debug_print_strength(
                        block_index=block_index,
                        is_dual=False,
                        s=s,
                        e=e,
                        proj_strength=proj_strength,
                        proj_strength_tokens=None,
                        strength_tau=strength_tau,
                        strength_gamma=strength_gamma,
                        g_slice=None,
                        alpha=alpha,
                    )
                else:
                    g_all = proj_strength_tokens
                    if g_all.ndim == 2:
                        g_all = g_all[0]
                    g_slice = g_all[s:e].to(device=v_slice.device, dtype=v_slice.dtype).view(1, e - s, 1)
                    v_slice = v_slice - ((g_slice * alpha) * vt_slice)
                    _flux_debug_print_strength(
                        block_index=block_index,
                        is_dual=False,
                        s=s,
                        e=e,
                        proj_strength=proj_strength,
                        proj_strength_tokens=proj_strength_tokens,
                        strength_tau=strength_tau,
                        strength_gamma=strength_gamma,
                        g_slice=g_slice.view(-1),
                        alpha=alpha,
                    )

                v_txt = torch.cat([v_txt[:, :s, :], v_slice, v_txt[:, e:, :]], dim=1)
                value_txt_new = v_txt.view(value.shape[0], text_seq_len, value.shape[2], value.shape[3])
                value[:, :text_seq_len] = torch.nan_to_num(value_txt_new, nan=0.0, posinf=0.0, neginf=0.0)

        # ---------------------------
        # DUAL stream: encoder_hidden_states is not None (added kv proj)
        # ---------------------------
        if attn.added_kv_proj_dim is not None and encoder_hidden_states is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            global _FLUX_TARGET_VT_BANK, _FLUX_TARGET_VT_READY_SET, _FLUX_TARGET_VT_BANK_RETAIN, _FLUX_TARGET_VT_READY_SET_RETAIN

            # Be robust if None
            target_block_indices = target_block_indices or []

            if (((record_target_vt and (block_index not in _FLUX_TARGET_VT_READY_SET)) or (record_retain_vt and (block_index not in _FLUX_TARGET_VT_READY_SET_RETAIN))) and (block_index in target_block_indices)):
                # vt_dual = encoder_value.detach() #0 idx token
                # vt_dual = vt_dual[:1].contiguous()
                # vt_dual = _make_lasttoken_duplicated_vt(vt_dual, last_token_index=-1)
                
                vt_dual = encoder_value.detach()[:1].contiguous()
                q_txt   = encoder_query.detach()[:1].contiguous()
                k_txt   = encoder_key.detach()[:1].contiguous()
                vt_dual = _make_attn01_duplicated_vt(vt_dual, q_txt, k_txt, attn_mode="col")
                
                if record_retain_vt:
                    #it is assumed that retain is ALWAYS done first, so we can project vt_single to the orthogonal complement of the retain vector. editing here reduces the number of changes.
                    #TODO: for multiple concepts, create a list of retain vectors and create a projection matrix?? for now we just do a single retain vector for simplicity.
                    if(block_index not in _FLUX_TARGET_VT_BANK_RETAIN):
                        _FLUX_TARGET_VT_BANK_RETAIN[block_index] = []
                    _FLUX_TARGET_VT_BANK_RETAIN[block_index].append(vt_dual)
                    _FLUX_TARGET_VT_READY_SET_RETAIN.add(block_index)
                elif record_target_vt:
                    retain_signal = _FLUX_TARGET_VT_BANK_RETAIN.get(block_index, None)
                    retain_proj = projection_retain(retain_signal) # debug print of projection retain strength stats
                    assert(retain_proj is not None)
                    
                    retain_proj = retain_proj.to(device=vt_dual.device, dtype=vt_dual.dtype)
                    # v_erase_dash = v_erase - constant*v_retain
                    vt_dual_all_head = vt_dual.reshape(vt_dual.shape[0], vt_dual.shape[1], -1) # 1 x 512 x 3072
                    vt_dual_all_head = vt_dual_all_head @ (torch.eye(vt_dual_all_head.shape[-1], device=vt_dual_all_head.device, dtype=vt_dual_all_head.dtype) - retain_proj) # project to orthogonal complement of retain space
                    vt_dual = vt_dual_all_head.reshape(vt_dual.shape)
                    _FLUX_TARGET_VT_BANK[block_index] = vt_dual
                    _FLUX_TARGET_VT_READY_SET.add(block_index)

            do_apply_here = (
                apply_target_proj
                and (block_index is not None)
                and (block_index in target_block_indices)
                and (proj_token_end is not None)
            )

            if do_apply_here:
                vt_source = None
                if block_index in _FLUX_TARGET_VT_BANK:
                    vt_source = _FLUX_TARGET_VT_BANK[block_index]
                elif _FLUX_TARGET_VT_READY and (block_index == target_block_index) and (_FLUX_TARGET_VT is not None):
                    vt_source = _FLUX_TARGET_VT
                else:
                    vt_source = None

                if vt_source is not None:
                    vt = vt_source
                    if vt.device != encoder_value.device or vt.dtype != encoder_value.dtype:
                        vt = vt.to(device=encoder_value.device, dtype=encoder_value.dtype)

                    v = encoder_value.reshape(encoder_value.shape[0], encoder_value.shape[1], -1)
                    vt_flat = vt.reshape(1, vt.shape[1], -1)

                    s = 0
                    e = int(proj_token_end)
                    e = max(s, min(e, v.shape[1]))  # clamp safely

                    v_slice = v[:, s:e, :]
                    vt_slice = vt_flat[:, s:e, :]

                    vt_norm2 = (vt_slice * vt_slice).sum(-1, keepdim=True)
                    denom = vt_norm2.clamp_min(proj_eps)
                    alpha = (v_slice * vt_slice).sum(-1, keepdim=True) / denom

                    if proj_strength_tokens is None:
                         # v_prompt = vprompt - constant *(v_erase - constant*v_retain)
                         # v_prompt = v_prompt - v_erase + v_retain 
                        v_slice = v_slice - (proj_strength * alpha * vt_slice)
                        _flux_debug_print_strength(
                            block_index=block_index,
                            is_dual=True,
                            s=s,
                            e=e,
                            proj_strength=proj_strength,
                            proj_strength_tokens=None,
                            strength_tau=strength_tau,
                            strength_gamma=strength_gamma,
                            g_slice=None,
                            alpha=alpha,
                        )
                    else:
                        g_all = proj_strength_tokens
                        if g_all.ndim == 2:
                            g_all = g_all[0]
                        g_slice = g_all[s:e].to(device=v_slice.device, dtype=v_slice.dtype).view(1, e - s, 1)
                        v_slice = v_slice - ((g_slice * alpha) * vt_slice)
                        _flux_debug_print_strength(
                            block_index=block_index,
                            is_dual=True,
                            s=s,
                            e=e,
                            proj_strength=proj_strength,
                            proj_strength_tokens=proj_strength_tokens,
                            strength_tau=strength_tau,
                            strength_gamma=strength_gamma,
                            g_slice=g_slice.view(-1),
                            alpha=alpha,
                        )

                    v = torch.cat([v[:, :s, :], v_slice, v[:, e:, :]], dim=1)
                    encoder_value = v.view_as(encoder_value)
                    encoder_value = torch.nan_to_num(encoder_value, nan=0.0, posinf=0.0, neginf=0.0)

            if dual_zero_text_value:
                encoder_value = encoder_value * 0.0

            # concat (context + image)
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        # ---------------------------
        # rotary
        # ---------------------------
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # ---------------------------
        # attention
        # ---------------------------
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
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs
        else:
            # Defensive: should not happen
            attn_output, context_attn_output = attention_outputs[0], attention_outputs[1]
            ip_attn_output = None

        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        if len(attention_outputs) == 3 and ip_attn_output is not None:
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
