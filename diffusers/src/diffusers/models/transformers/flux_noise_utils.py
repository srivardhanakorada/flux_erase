# flux_noise_utils.py
from typing import Any, Dict, List, Optional, Tuple

import torch  # type: ignore


def _as_dir(d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    d = d.to(torch.float32).contiguous()
    return d / (d.norm() + eps)


def _stack_dirs(dirs: List[torch.Tensor]) -> torch.Tensor:
    if len(dirs) == 0:
        raise ValueError("No dirs to stack")
    return torch.stack(dirs, dim=1)  # [d, n]


def build_subspace_from_dirs(
    dirs: List[torch.Tensor],
    *,
    rank: Optional[int] = None,
    energy: float = 0.9,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Build orthonormal basis U from dirs via SVD.
    If rank=None, choose minimal r so cumulative sigma^2 >= energy.
    """
    M = _stack_dirs(dirs)  # [d, n]
    u, s, _ = torch.linalg.svd(M, full_matrices=False)

    if rank is None:
        ss = s.pow(2)
        cum = torch.cumsum(ss, dim=0) / (ss.sum() + eps)
        r = int((cum < energy).sum().item()) + 1
    else:
        r = int(rank)

    r = max(1, min(r, u.shape[1]))
    U = u[:, :r].contiguous()
    U = U / (U.norm(dim=0, keepdim=True) + eps)
    return U


def subspace_similarity(U: torch.Tensor, V: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Sim(U,V) = (1/r) || U^T V ||_F^2, using r=min(rank(U), rank(V)).
    """
    r = min(U.shape[1], V.shape[1])
    if r <= 0:
        return 0.0
    UtV = U[:, :r].t() @ V[:, :r]
    sim = (UtV.pow(2).sum() / (r + eps)).item()
    return float(max(0.0, min(1.0, sim)))


def _split_half(dirs: List[torch.Tensor], seed: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if len(dirs) < 4:
        return dirs, dirs
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dirs), generator=g).tolist()
    mid = len(idx) // 2
    A = [dirs[i] for i in idx[:mid]]
    B = [dirs[i] for i in idx[mid:]]
    return A, B


def _noise_sim_from_dirs(
    dirs: List[torch.Tensor],
    *,
    rank: Optional[int],
    energy: float,
    seed: int,
) -> float:
    A, B = _split_half(dirs, seed=seed)
    if len(A) < 2 or len(B) < 2:
        return 0.0
    UA = build_subspace_from_dirs(A, rank=rank, energy=energy)
    UB = build_subspace_from_dirs(B, rank=rank, energy=energy)
    return subspace_similarity(UA, UB)


def _bootstrap_stats(
    dirs: List[torch.Tensor],
    *,
    n_boot: int,
    rank: Optional[int],
    energy: float,
    seed0: int,
) -> Tuple[float, float, int]:
    """
    Returns (mean_sim, std_sim, n_dirs)
    """
    n = len(dirs)
    if n < 4:
        return 0.0, 0.0, n
    sims = []
    for k in range(n_boot):
        sims.append(_noise_sim_from_dirs(dirs, rank=rank, energy=energy, seed=seed0 + 1009 * k))
    t = torch.tensor(sims)
    return float(t.mean().item()), float(t.std(unbiased=False).item()), n


def _collect_block_dirs(
    target_bank: Dict[int, Dict[str, List[torch.Tensor]]],
    concept: str,
    token_idx: int,
) -> Dict[int, List[torch.Tensor]]:
    out: Dict[int, List[torch.Tensor]] = {}
    for blk, cmap in target_bank.items():
        if concept not in cmap:
            continue
        out[blk] = [_as_dir(d) for d in cmap[concept]]
    return out


def _collect_stage_dirs(
    target_bank: Dict[int, Dict[str, List[torch.Tensor]]],
    concept: str,
    token_idx: int,
) -> List[torch.Tensor]:
    pooled: List[torch.Tensor] = []
    for blk_dirs in _collect_block_dirs(target_bank, concept, token_idx).values():
        pooled.extend(blk_dirs)
    return pooled


def build_noise_report(
    *,
    concept: str,
    target_bank_dual: Dict[int, Dict[str, List[torch.Tensor]]],
    target_bank_single: Dict[int, Dict[str, List[torch.Tensor]]],
    n_boot: int = 30,
    energy: float = 0.9,
    rank: Optional[int] = None,
    seed0: int = 0,
    token_idx: int = 1,
) -> Dict[str, Any]:
    dual_blocks_dirs = _collect_block_dirs(target_bank_dual, concept, token_idx)
    single_blocks_dirs = _collect_block_dirs(target_bank_single, concept, token_idx)

    dual_blocks = {
        str(blk): _bootstrap_stats(dirs, n_boot=n_boot, rank=rank, energy=energy, seed0=seed0)
        for blk, dirs in dual_blocks_dirs.items()
    }
    single_blocks = {
        str(blk): _bootstrap_stats(dirs, n_boot=n_boot, rank=rank, energy=energy, seed0=seed0)
        for blk, dirs in single_blocks_dirs.items()
    }

    dual_stage_dirs = _collect_stage_dirs(target_bank_dual, concept, token_idx)
    single_stage_dirs = _collect_stage_dirs(target_bank_single, concept, token_idx)

    dual_stage = _bootstrap_stats(dual_stage_dirs, n_boot=n_boot, rank=rank, energy=energy, seed0=seed0)
    single_stage = _bootstrap_stats(single_stage_dirs, n_boot=n_boot, rank=rank, energy=energy, seed0=seed0)

    return {
        "concept": concept,
        "n_boot": n_boot,
        "energy": energy,
        "rank": rank,
        "token_idx": token_idx,
        "dual_stage": dual_stage,       # (mean,std,n_dirs)
        "single_stage": single_stage,   # (mean,std,n_dirs)
        "dual_blocks": dual_blocks,     # blk -> (mean,std,n_dirs)
        "single_blocks": single_blocks,
    }