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

_FLUX_TARGET_VT = None
_FLUX_TARGET_VT_READY = False
_FLUX_TARGET_VT_BANK: Dict[int, List[torch.Tensor]] = {}
_FLUX_SINGLE_VT_BANK: Dict[int, List[List[torch.Tensor]]] = {}
_FLUX_TARGET_VT_READY_SET = set()
_FLUX_SINGLE_VT_READY_SET = set()
_FLUX_PRINT_STRENGTH_COUNT = 0
_FLUX_SINGLE_VT_BANK_RETAIN: Dict[int, List[List[torch.Tensor]]] = {}
_FLUX_SINGLE_VT_READY_SET_RETAIN = set()
_FLUX_TARGET_VT_BANK_RETAIN: Dict[int, List[torch.Tensor]] = {}
_FLUX_TARGET_VT_READY_SET_RETAIN = set()
_FLUX_TARGET_VT_BANK_ANCHOR: Dict[int, torch.Tensor] = {}
_FLUX_SINGLE_VT_BANK_ANCHOR: Dict[int, torch.Tensor] = {}
_FLUX_TARGET_VT_READY_SET_ANCHOR = set()
_FLUX_SINGLE_VT_READY_SET_ANCHOR = set()
_MAX_VT_PER_BLOCK = 8          
_VT_DEDUP_COS_THR = 0.98       
_FLUX_MAX_TARGET_VT_PER_BLOCK = 8
_FLUX_MAX_RETAIN_VT_PER_BLOCK = 16
_FLUX_CONCEPT_C_LIST: List[torch.Tensor] = []
logger = logging.get_logger(__name__)

def flux_reset_concept_embeds():
    global _FLUX_CONCEPT_C_LIST
    _FLUX_CONCEPT_C_LIST = []

def flux_add_concept_embed(c: torch.Tensor):
    global _FLUX_CONCEPT_C_LIST
    if c.ndim == 1: c = c.view(1, -1)
    c = c.detach().contiguous()
    c = c / (c.norm(dim=-1, keepdim=True) + 1e-8)
    _FLUX_CONCEPT_C_LIST.append(c)

def flux_get_concept_embeds():
    global _FLUX_CONCEPT_C_LIST
    return _FLUX_CONCEPT_C_LIST

def flux_reset_vt_banks(reset_retain: bool = True):
    global _FLUX_TARGET_VT, _FLUX_TARGET_VT_READY
    global _FLUX_TARGET_VT_BANK, _FLUX_SINGLE_VT_BANK
    global _FLUX_TARGET_VT_READY_SET, _FLUX_SINGLE_VT_READY_SET
    global _FLUX_TARGET_VT_BANK_RETAIN, _FLUX_SINGLE_VT_BANK_RETAIN
    global _FLUX_TARGET_VT_READY_SET_RETAIN, _FLUX_SINGLE_VT_READY_SET_RETAIN
    global _FLUX_PRINT_STRENGTH_COUNT
    global _FLUX_TARGET_VT_READY_SET_ANCHOR, _FLUX_SINGLE_VT_READY_SET_ANCHOR
    global _FLUX_TARGET_VT_BANK_ANCHOR, _FLUX_SINGLE_VT_BANK_ANCHOR
    _FLUX_TARGET_VT = None
    _FLUX_TARGET_VT_READY = False
    _FLUX_TARGET_VT_BANK.clear()
    _FLUX_SINGLE_VT_BANK.clear()
    _FLUX_TARGET_VT_READY_SET.clear()
    _FLUX_SINGLE_VT_READY_SET.clear()
    if reset_retain:
        _FLUX_TARGET_VT_BANK_ANCHOR.clear()
        _FLUX_SINGLE_VT_BANK_ANCHOR.clear()
        _FLUX_TARGET_VT_BANK_RETAIN.clear()
        _FLUX_SINGLE_VT_BANK_RETAIN.clear()
        _FLUX_TARGET_VT_READY_SET_RETAIN.clear()
        _FLUX_SINGLE_VT_READY_SET_RETAIN.clear()
        _FLUX_TARGET_VT_READY_SET_ANCHOR.clear()
        _FLUX_SINGLE_VT_READY_SET_ANCHOR.clear()
    _FLUX_PRINT_STRENGTH_COUNT = 0
    flux_reset_concept_embeds()

def _append_vt_capped(
    bank: Dict[int, List[List[torch.Tensor]]],
    block_index: int,
    vt_new: torch.Tensor,
    pooled_projections: Optional[torch.Tensor] = None,
    *,
    max_keep: int = _MAX_VT_PER_BLOCK,
):
    if block_index not in bank: bank[block_index] = []
    lst = bank[block_index]
    if len(lst) == 0:
        if(pooled_projections is None):
            lst.append([vt_new.detach()])
        else:
            lst.append([pooled_projections,vt_new.detach()])
        return
    with torch.no_grad():
        x = vt_new.reshape(-1).to(torch.float32)
        x = x / (x.norm() + 1e-8)
        best = -1.0
        for vt_old in lst:
            val_vec = vt_old[1] if pooled_projections is not None else vt_old[0]
            y = val_vec.reshape(-1).to(torch.float32)
            y = y / (y.norm() + 1e-8)
            sim = float((x * y).sum().item())
            if sim > best: best = sim
        if best >= _VT_DEDUP_COS_THR: return
    if(pooled_projections is None):
        lst.append([vt_new.detach()])
    else:
        lst.append([pooled_projections,vt_new.detach()])
    if max_keep is not None and max_keep > 0 and len(lst) > max_keep: del lst[0 : (len(lst) - max_keep)]

def _with_dedup_thr(thr: float, fn):
    global _VT_DEDUP_COS_THR
    old = _VT_DEDUP_COS_THR
    _VT_DEDUP_COS_THR = float(thr)
    try: return fn()
    finally: _VT_DEDUP_COS_THR = old

def norm_clip(vec):
    return vec/(vec.norm(dim=-1,keepdim=True)+1e-8)

def _project_out_span_clip(
    v_slice: torch.Tensor,          
    vt_list: List[torch.Tensor],    
    *,
    s: int,
    e: int,
    eps: float,
    strength: float,
    strength_tokens: Optional[torch.Tensor] = None, 
    anchor_vector: Optional[torch.Tensor] = None,
    anchor_strength: Optional[float] = 2.0,
    clip_embedding: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    #calculate beta using CLIP embeddings instead of t5, then edit the value vectors (defined by t5) using that beta.
    # print("Clip embedding: ",clip_embedding) #comes out same everytime
    if len(vt_list) == 0: return v_slice
    B, T, D = v_slice.shape # 1 x tokens_selected x 3072
    K = len(vt_list) #number of concept vectors that are significantly different (may not be equal to erased concepts)
    e = s + T
    V_tok = []
    vt_mat = None
    vt_mat_clip = None
    for vt in vt_list:
        # vt[0] has clip embeddings, while vt[1] has the t5 value vector
        vt_flat = vt[1].reshape(1, vt[1].shape[1], -1) # 1 x 512 x 3072
        if(vt_mat is None):
            vt_mat = vt_flat
        else:
            vt_mat = torch.cat([vt_mat, vt_flat], dim=0) # 
        if(vt_mat_clip is None):
            vt_mat_clip = norm_clip(vt[0])
        else: 
            vt_mat_clip = torch.cat([vt_mat_clip,norm_clip(vt[0])],dim=0)
        V_tok.append(vt_flat[0, s:e, :])  # tokens_selected x 3072        
    V_tok = torch.stack(V_tok, dim=1) #  tokens_selected x number of concept vectors that are significantly different x 3072          
    # print("V_tok shape: ",V_tok.shape) 
    # V32 = V_tok.to(torch.float32) # num of concepts x tokens_selected x 3072
    V32_clip = vt_mat_clip.to(device=v_slice.device, dtype=torch.float32).unsqueeze(0) # 1 x num of concepts x 768
    # print("V32_clip shape",V32_clip.shape)
    v32 = v_slice.to(torch.float32) #1 x tokens_selected x 3072
    # print("v32 shape",v32.shape)
    v32_clip = norm_clip(clip_embedding.to(device=v_slice.device, dtype=torch.float32)).unsqueeze(1) # 1 x 1 x 768
    # print("v32_clip shape",v32_clip.shape)
    # print("v_slice shape: ",v_slice.shape)# 1 x tokens_selected x 3072
    # print(G) #non invertible for clip in case of single concept erasure. 
    # print("G shape: ",G.shape)

    #G is non invertible for single concepts since all embeddings would be the same. 
    # G = torch.einsum("tkd,tld->tkl", V32_clip, V32_clip)
    # I = torch.eye(K, device=G.device, dtype=G.dtype).view(1, K, K)
    # G = G + eps * I
    # rhs = torch.einsum("tkd,btd->btk", V32_clip, v32_clip) # 1x4x768, 1x1x768
    # beta = torch.linalg.solve(G.unsqueeze(0).expand(B, -1, -1, -1), rhs.unsqueeze(-1)).squeeze(-1)

    beta = 75*torch.einsum("tkd,btd->btk", V32_clip, v32_clip) # 1x4x768, 1x1x768
    

    # print("beta shape: ",beta.shape)
    # temp1 = V32_clip.squeeze(0)
    # print((V32_clip.squeeze(0)@v32_clip.squeeze(0).T)[0][0])
    # removed = torch.einsum("tkd,btk->btd", V32, beta)
    # print("vt_mat shape: ",vt_mat.shape)
    vt_mat = projection_anchor(vt_mat,anchor_vector,anchor_strength = anchor_strength).to(torch.float32)
    # print("vt_mat shape: ",vt_mat.shape)
    V32_anchored = vt_mat.transpose(0,1) # 512 x erased_concepts x 3072
    beta_expanded = beta.expand(B,V32_anchored.shape[0], K)  # (1,512,K)
    # print("V32_anchored shape:",V32_anchored.shape)
    # print("V32_anchored shape:",V32_anchored.shape)
    # print("Beta expanded: ",beta_expanded.shape)
    # beta = torch.clamp(beta, min=0)
    
    removed = torch.einsum("tkd,btk->btd", V32_anchored, beta_expanded)
    removed = removed[:,s:e,:]
    # print("removed shape: ",removed.shape)
    if strength_tokens is None: v32 = v32 - float(strength) * removed
    else:
        g_all = strength_tokens
        if g_all.ndim == 2: g_all = g_all[0]
        g = g_all[s:e].to(device=v_slice.device, dtype=torch.float32).view(1, T, 1)
        v32 = v32 - (g * removed)
    return v32.to(dtype=v_slice.dtype)
    
def _project_out_span(
    v_slice: torch.Tensor,          
    vt_list: List[torch.Tensor],    
    *,
    s: int,
    e: int,
    eps: float,
    strength: float,
    strength_tokens: Optional[torch.Tensor] = None, 
    anchor_vector: Optional[torch.Tensor] = None,
    anchor_strength: Optional[float] = 2.0
) -> torch.Tensor:
    if len(vt_list) == 0: return v_slice
    B, T, D = v_slice.shape # 1 x tokens_selected x 3072
    K = len(vt_list) #number of concept vectors that are significantly different (may not be equal to erased concepts)
    # print("K: ",K)
    e = s + T
    V_tok = []
    vt_mat = None
    for vt in vt_list:
        vt_flat = vt[1].reshape(1, vt[1].shape[1], -1) # 1 x 512 x 3072
        if(vt_mat is None):
            vt_mat = vt_flat
        else:
            vt_mat = torch.cat([vt_mat, vt_flat], dim=0) # 
        V_tok.append(vt_flat[0, s:e, :])  # tokens_selected x 3072        
    V_tok = torch.stack(V_tok, dim=1) #  tokens_selected x number of concept vectors that are significantly different x 3072          
    # print("V_tok shape: ",V_tok.shape) 
    V32 = V_tok.to(torch.float32)
    v32 = v_slice.to(torch.float32) #1 x tokens_selected x 3072
    # print("v_slice shape: ",v_slice.shape)# 1 x tokens_selected x 3072
    G = torch.einsum("tkd,tld->tkl", V32, V32)
    I = torch.eye(K, device=G.device, dtype=G.dtype).view(1, K, K)
    G = G + eps * I
    rhs = torch.einsum("tkd,btd->btk", V32, v32)
    # print("rhs shape: ",rhs.shape)
    beta = torch.linalg.solve(G.unsqueeze(0).expand(B, -1, -1, -1), rhs.unsqueeze(-1)).squeeze(-1)
    # print("beta shape: ",beta.shape)
    # removed = torch.einsum("tkd,btk->btd", V32, beta)
    # print("vt_mat shape: ",vt_mat.shape)
    vt_mat = projection_anchor(vt_mat,anchor_vector,anchor_strength = anchor_strength).to(torch.float32)[:,s:e,:]
    # print("vt_mat shape: ",vt_mat.shape)
    V32_anchored = vt_mat.transpose(0,1)
    # print("V32_anchored shape:",V32_anchored.shape)
    removed = torch.einsum("tkd,btk->btd", V32_anchored, beta)
    # print("removed shape: ",removed.shape)
    if strength_tokens is None: v32 = v32 - float(strength) * removed
    else:
        g_all = strength_tokens
        if g_all.ndim == 2: g_all = g_all[0]
        g = g_all[s:e].to(device=v_slice.device, dtype=torch.float32).view(1, T, 1)
        v32 = v32 - (g * removed)
    return v32.to(dtype=v_slice.dtype)

def projection_anchor(value_vector,anchor_vector,anchor_strength=2.0):
    #in short, value vector is the "purified" erased vector and anchor vector represents the anchoring we need
    # we first project anchor vector on the orthogonal complement of the value vector. 
    # This ensures that the anchor has no component associated with the erased signal.
    # be careful.. anchor should not affect gating function calcualtion
    anc = anchor_vector.reshape(anchor_vector.shape[0], anchor_vector.shape[1], -1)
    value_vector = value_vector.reshape(value_vector.shape[0], value_vector.shape[1], -1)
    anc = anc.to(device=value_vector.device, dtype=value_vector.dtype)
    anc = anc - (value_vector*(anc*value_vector).sum(dim=-1, keepdim=True) / (value_vector*value_vector).sum(dim=-1, keepdim=True).clamp_min(1e-8)) #project anchor to orthogonal complement of erase vector.
    return value_vector - anchor_strength * anc # subtract so that it eventually is added when the prompt vector subtracts the edited value vector

def _make_attn01_duplicated_vt(
    vt: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    fill_from: int = 1,
    zero_sot: bool = True,
    attn_mode: str = "col",          
    normalize_detector: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    assert vt.ndim == 4 and vt.shape[0] == 1, f"Expected vt [1,L,H,Dh], got {tuple(vt.shape)}"
    assert q.shape == vt.shape and k.shape == vt.shape, f"q/k must match vt shape. got q={tuple(q.shape)}, k={tuple(k.shape)}, vt={tuple(vt.shape)}"
    L, H, Dh = vt.shape[1], vt.shape[2], vt.shape[3]
    if L < 2: raise ValueError(f"Need at least 2 tokens to use phrase tokens [0,1]. Got L={L}")
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 1, 3)
    logits = torch.matmul(qh, kh.transpose(-1, -2)) / (Dh ** 0.5)
    A = F.softmax(logits, dim=-1)
    if attn_mode == "col": imp = A.mean(dim=-2)
    elif attn_mode == "row": imp = A.mean(dim=-1)
    else: raise ValueError("attn_mode must be 'col' or 'row'")
    imp01 = imp[:, :, 0:2]
    w = imp01 / imp01.sum(dim=-1, keepdim=True).clamp_min(eps)
    vt01 = vt[:, 0:2, :, :].permute(0, 2, 1, 3)
    d = (w.unsqueeze(-1) * vt01).sum(dim=-2, keepdim=True)
    if normalize_detector: d = d / torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(eps)
    d = d.permute(0, 2, 1, 3)
    vt2 = vt.clone()
    if L > fill_from: vt2[:, fill_from:, :, :] = d.expand(1, L - fill_from, H, Dh)
    if zero_sot: vt2[:, 0, :, :] = 0.0
    return vt2 

def projection_retain(v_list: Optional[list[torch.Tensor]], top_k=3):
    if (v_list is None) or (len(v_list) == 0): return None
    v_temp = [v.squeeze(0).reshape(v.shape[1], -1).to(torch.float32).T for v in v_list]
    U_stack = None
    cnt = 0
    for v in v_temp:
        cnt+=1
        u, s, v_t = torch.linalg.svd(v, full_matrices=False)
        imp_vectors = u[:, :top_k]
        if U_stack is None: U_stack = imp_vectors
        else: U_stack = torch.cat([U_stack, imp_vectors], dim=1)
    u, s, v_t = torch.linalg.svd(U_stack, full_matrices=False)
    v_retain = u[:, :top_k]
    proj_mat = v_retain @ v_retain.T
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
    if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"): encoder_query, encoder_key, encoder_value = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)
    return query, key, value, encoder_query, encoder_key, encoder_value

def _get_qkv_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    if attn.fused_projections: return _get_fused_projections(attn, hidden_states, encoder_hidden_states)
    return _get_projections(attn, hidden_states, encoder_hidden_states)

class FluxAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"): raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")

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
        anchor_strength: Optional[float] = 2.0,
        target_block_index: int = 0,
        block_index: Optional[int] = None,
        proj_eps: float = 1e-8,
        target_block_indices: Optional[List[int]] = None,
        proj_strength: float = 1.0,
        proj_token_end: Optional[int] = None,
        target_single_block_indices: Optional[List[int]] = None,
        proj_strength_tokens: Optional[torch.Tensor] = None,
        strength_tau: float = 0.2,
        strength_gamma: float = 1.0,
        vt_dedup_cos_thr: Optional[float] = None,
        max_target_vt_per_block: Optional[int] = None,
        max_retain_vt_per_block: Optional[int] = None,
        pooled_projections: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        # print(pooled_projections.shape) # 1 x 768
        dedup_thr = float(_VT_DEDUP_COS_THR) if vt_dedup_cos_thr is None else float(vt_dedup_cos_thr)
        max_tgt  = int(_FLUX_MAX_TARGET_VT_PER_BLOCK) if max_target_vt_per_block is None else int(max_target_vt_per_block)
        max_ret  = int(_FLUX_MAX_RETAIN_VT_PER_BLOCK) if max_retain_vt_per_block is None else int(max_retain_vt_per_block)
        ####### SINGLE STREAM #########
        if encoder_hidden_states is None and single_zero_text_value and (text_seq_len is not None): value[:, :text_seq_len] = 0.0
        if (encoder_hidden_states is None) and (text_seq_len is not None) and (block_index is not None):
            global _FLUX_SINGLE_VT_BANK, _FLUX_SINGLE_VT_READY_SET, _FLUX_SINGLE_VT_BANK_RETAIN, _FLUX_SINGLE_VT_READY_SET_RETAIN, _FLUX_SINGLE_VT_READY_SET_ANCHOR, _FLUX_SINGLE_VT_BANK_ANCHOR
            target_single_block_indices = target_single_block_indices or []
            if ((record_target_vt or record_retain_vt or record_anchor_vt) and (block_index in target_single_block_indices)):
                vt_single = value[:, :text_seq_len].detach()[:1].contiguous()
                q_txt     = query[:, :text_seq_len].detach()[:1].contiguous()
                k_txt     = key[:, :text_seq_len].detach()[:1].contiguous()
                vt_single = _make_attn01_duplicated_vt(vt_single, q_txt, k_txt, attn_mode="col")
                if(record_target_vt): 
                    retain_signal = _FLUX_SINGLE_VT_BANK_RETAIN.get(block_index, None)
                    unwrapped_retain_signal = []
                    for rs in retain_signal:
                        unwrapped_retain_signal.append(rs[0])
                    retain_proj = projection_retain(unwrapped_retain_signal)
                    assert(retain_proj is not None)
                    retain_proj = retain_proj.to(device=vt_single.device, dtype=vt_single.dtype)
                    vt_all_head = vt_single.reshape(vt_single.shape[0], vt_single.shape[1], -1)
                    vt_all_head = vt_all_head @ (torch.eye(vt_all_head.shape[-1], device=vt_all_head.device, dtype=vt_all_head.dtype) - retain_proj) # project to orthogonal complement of retain space
                    vt_single = vt_all_head.reshape(vt_single.shape)
                    _with_dedup_thr(dedup_thr,lambda: _append_vt_capped(_FLUX_SINGLE_VT_BANK, block_index, vt_single, max_keep=max_tgt,pooled_projections=pooled_projections),) # dedup doesnt work?? almost adding for all timesteps
                    #need a more "global perspective" on the concept. for that, CLIP?? determine coeff using CLIP and apply them on Value tokens??
                    #for every rerased concept that is added... store their clip embedding. 
                    #for every prompt, check their clip embedding with the clip embeddings stored for the erased concepts.
                    #perform erasure using that similarity... scale each erased embedding with the clip score and perform erasure.
                    _FLUX_SINGLE_VT_READY_SET.add(block_index)
                    # print(f"For block idx {block_index} size of vt bank: ",len(_FLUX_SINGLE_VT_BANK[block_index]))
                elif record_retain_vt:
                    _with_dedup_thr(dedup_thr,lambda: _append_vt_capped(_FLUX_SINGLE_VT_BANK_RETAIN,block_index,vt_single,max_keep=max_ret,),)
                    _FLUX_SINGLE_VT_READY_SET_RETAIN.add(block_index)
                elif record_anchor_vt and (block_index not in _FLUX_SINGLE_VT_READY_SET_ANCHOR):
                    _FLUX_SINGLE_VT_BANK_ANCHOR[block_index] = vt_single
                    _FLUX_SINGLE_VT_READY_SET_ANCHOR.add(block_index)
            do_apply_single = (apply_target_proj and (block_index in target_single_block_indices)and (block_index in _FLUX_SINGLE_VT_BANK))
            if do_apply_single:
                vt_list = _FLUX_SINGLE_VT_BANK.get(block_index, [])
                anchor_vector = _FLUX_SINGLE_VT_BANK_ANCHOR.get(block_index,None)
                vt_list = [[vt[0].to(device=value.device, dtype=value.dtype),vt[1].to(device=value.device, dtype=value.dtype)] for vt in vt_list] #clip emb,t5 value vector
                v_txt = value[:, :text_seq_len].reshape(value.shape[0], text_seq_len, -1)
                s = 0
                e_req = proj_token_end
                e = text_seq_len if (e_req is None) else int(e_req)
                e = max(s, min(e, v_txt.shape[1]))
                v_slice = v_txt[:, s:e, :]
                # clip based gating needs fix... similarity isnt matching
                v_slice = _project_out_span_clip(v_slice,vt_list,s=s,e=e,eps=proj_eps,strength=proj_strength,strength_tokens=proj_strength_tokens,anchor_vector = anchor_vector,anchor_strength=anchor_strength,clip_embedding=pooled_projections)
                # v_slice = _project_out_span(v_slice,vt_list,s=s,e=e,eps=proj_eps,strength=proj_strength,strength_tokens=proj_strength_tokens,anchor_vector = anchor_vector,anchor_strength=anchor_strength) 
                v_txt = torch.cat([v_txt[:, :s, :], v_slice, v_txt[:, e:, :]], dim=1)
                value_txt_new = v_txt.view(value.shape[0], text_seq_len, value.shape[2], value.shape[3])
                value[:, :text_seq_len] = torch.nan_to_num(value_txt_new, nan=0.0, posinf=0.0, neginf=0.0)
        ####### DOUBLE STREAM #########
        if attn.added_kv_proj_dim is not None and encoder_hidden_states is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)
            global _FLUX_TARGET_VT_BANK, _FLUX_TARGET_VT_READY_SET, _FLUX_TARGET_VT_BANK_RETAIN, _FLUX_TARGET_VT_READY_SET_RETAIN, _FLUX_TARGET_VT_BANK_ANCHOR, _FLUX_TARGET_VT_READY_SET_ANCHOR
            target_block_indices = target_block_indices or []
            if ((record_target_vt or record_retain_vt or record_anchor_vt) and (block_index in target_block_indices)):
                vt_dual = encoder_value.detach()[:1].contiguous()
                q_txt   = encoder_query.detach()[:1].contiguous()
                k_txt   = encoder_key.detach()[:1].contiguous()
                vt_dual = _make_attn01_duplicated_vt(vt_dual, q_txt, k_txt, attn_mode="col")
                if record_retain_vt:
                    _with_dedup_thr(dedup_thr,lambda: _append_vt_capped(_FLUX_TARGET_VT_BANK_RETAIN,block_index,vt_dual,max_keep=max_ret,),)
                    _FLUX_TARGET_VT_READY_SET_RETAIN.add(block_index)
                elif record_target_vt:
                    retain_signal = _FLUX_TARGET_VT_BANK_RETAIN.get(block_index, None)
                    unwrapped_retain_signal = []
                    for rs in retain_signal:
                        unwrapped_retain_signal.append(rs[0])
                    retain_proj = projection_retain(unwrapped_retain_signal)
                    assert(retain_proj is not None)
                    retain_proj = retain_proj.to(device=vt_dual.device, dtype=vt_dual.dtype)
                    vt_dual_all_head = vt_dual.reshape(vt_dual.shape[0], vt_dual.shape[1], -1)
                    vt_dual_all_head = vt_dual_all_head @ (torch.eye(vt_dual_all_head.shape[-1], device=vt_dual_all_head.device, dtype=vt_dual_all_head.dtype) - retain_proj) # project to orthogonal complement of retain space
                    vt_dual = vt_dual_all_head.reshape(vt_dual.shape)
                    _with_dedup_thr(dedup_thr,lambda: _append_vt_capped(_FLUX_TARGET_VT_BANK, block_index, vt_dual, max_keep=max_tgt,pooled_projections=pooled_projections),)
                    _FLUX_TARGET_VT_READY_SET.add(block_index)
                elif record_anchor_vt and (block_index not in _FLUX_TARGET_VT_READY_SET_ANCHOR):
                    _FLUX_TARGET_VT_BANK_ANCHOR[block_index] = vt_dual
                    _FLUX_TARGET_VT_READY_SET_ANCHOR.add(block_index)
            do_apply_here = (apply_target_proj and (block_index is not None)and (block_index in target_block_indices))
            if do_apply_here:
                vt_list = _FLUX_TARGET_VT_BANK.get(block_index, [])
                anchor_vector = _FLUX_TARGET_VT_BANK_ANCHOR.get(block_index,None)
                vt_list = [[vt[0].to(device=encoder_value.device, dtype=encoder_value.dtype),vt[1].to(device=encoder_value.device, dtype=encoder_value.dtype)] for vt in vt_list]
                if len(vt_list) > 0:
                    v = encoder_value.reshape(encoder_value.shape[0], encoder_value.shape[1], -1)
                    s = 0
                    e_req = proj_token_end
                    e = v.shape[1] if (e_req is None) else int(e_req)
                    e = max(s, min(e, v.shape[1]))
                    v_slice = v[:, s:e, :]
                    v_slice = _project_out_span_clip(v_slice,vt_list,s=s,e=e,eps=proj_eps,strength=proj_strength,strength_tokens=proj_strength_tokens,anchor_vector = anchor_vector,anchor_strength=anchor_strength,clip_embedding=pooled_projections)
                    # v_slice = _project_out_span(v_slice,vt_list,s=s,e=e,eps=proj_eps,strength=proj_strength,strength_tokens=proj_strength_tokens,anchor_vector = anchor_vector,anchor_strength=anchor_strength)
                    v = torch.cat([v[:, :s, :], v_slice, v[:, e:, :]], dim=1)
                    encoder_value = v.view_as(encoder_value)
                    encoder_value = torch.nan_to_num(encoder_value, nan=0.0, posinf=0.0, neginf=0.0)
            if dual_zero_text_value: encoder_value = encoder_value * 0.0
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        hidden_states = dispatch_attention_fn(query,key,value,attn_mask=attention_mask,backend=self._attention_backend,parallel_config=self._parallel_config,)
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes([encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]],dim=1,)
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
        if not self.pre_only: self.to_out = torch.nn.ModuleList([torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias), torch.nn.Dropout(dropout)])
        if added_kv_proj_dim is not None:
            self.norm_added_q = torch.nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = torch.nn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.to_add_out = torch.nn.Linear(self.inner_dim, query_dim, bias=out_bias)
        if processor is None: processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        pooled_projections: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states"}
        unused_kwargs = [k for k in kwargs.keys() if (k not in attn_parameters and k not in quiet_attn_parameters)]
        if len(unused_kwargs) > 0: logger.warning(f"joint_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored.")
        kwargs = {k: v for k, v in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, pooled_projections=pooled_projections, **kwargs)

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
        pooled_projections: Optional[torch.Tensor] = None
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
            pooled_projections=pooled_projections,
            **joint_attention_kwargs,
        )
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states = gate.unsqueeze(1) * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16: hidden_states = hidden_states.clip(-65504, 65504)
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
        pooled_projections: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)
        joint_attention_kwargs = joint_attention_kwargs or {}
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            pooled_projections = pooled_projections,
            **joint_attention_kwargs,
        )
        if len(attention_outputs) == 2: attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3: attn_output, context_attn_output, ip_attn_output = attention_outputs
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
        if len(attention_outputs) == 3 and ip_attn_output is not None: hidden_states = hidden_states + ip_attn_output
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16: encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
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
        text_time_guidance_cls = (CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings)
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
        if USE_PEFT_BACKEND: scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs.get("scale", None) is not None: logger.warning("Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective.")
        hidden_states = self.x_embedder(hidden_states)
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None: guidance = guidance.to(hidden_states.dtype) * 1000
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        if txt_ids.ndim == 3: txt_ids = txt_ids[0]
        if img_ids.ndim == 3: img_ids = img_ids[0]
        ids = torch.cat((txt_ids, img_ids), dim=0)
        if is_torch_npu_available():
            freqs_cos, freqs_sin = self.pos_embed(ids.cpu())
            image_rotary_emb = (freqs_cos.npu(), freqs_sin.npu())
        else: image_rotary_emb = self.pos_embed(ids)
        for index_block, block in enumerate(self.transformer_blocks):
            ja = joint_attention_kwargs.copy()
            ja["block_index"] = index_block
            if torch.is_grad_enabled() and self.gradient_checkpointing: encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(block, hidden_states, encoder_hidden_states, temb, image_rotary_emb, ja)
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=ja,
                    pooled_projections = pooled_projections
                )
        for index_block, block in enumerate(self.single_transformer_blocks):
            ja = joint_attention_kwargs.copy()
            ja["block_index"] = index_block
            if torch.is_grad_enabled() and self.gradient_checkpointing: encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(block, hidden_states, encoder_hidden_states, temb, image_rotary_emb, ja)
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=ja,
                    pooled_projections = pooled_projections
                )
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        if USE_PEFT_BACKEND: unscale_lora_layers(self, lora_scale)
        if not return_dict: return (output,)
        return Transformer2DModelOutput(sample=output)
