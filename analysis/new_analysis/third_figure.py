import os
import copy
import json
import math
from typing import Dict, List, Optional

import torch
import matplotlib.pyplot as plt
import numpy as np
from diffusers import FluxPipeline

import diffusers.models.transformers.transformer_flux as flux_tf


# ============================================================
# Config
# ============================================================

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
OUTDIR = "category_mechanism_figure_outputs"
os.makedirs(OUTDIR, exist_ok=True)

TARGETS: List[str] = [
    "Donald Trump",
]

RETAINS: List[str] = [
    "Melania Trump",
    "Hillary Clinton",
    "Barack Obama",
]

PERSON_BANK: List[str] = [
    "a portrait of a person",
    "a portrait of a man",
    "a portrait of a woman",
    "a middle-aged man",
    "a middle-aged woman",
]

RECORDING_TEMPLATES: List[str] = [
    "a photo of {}",
    "{} photographed with DSLR",
    "{}, studio portrait, sharp focus",
]

DUAL_BLOCKS = list(range(0, 19))
SINGLE_BLOCKS = list(range(0, 38))

REC_H, REC_W = 512, 512
STEPS = 4
GUIDANCE = 3.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

RETAIN_TOP_K = 6
PERSON_TOP_K = 6
DETECTOR_TOKEN_END = 2
PROJ_TOKEN_END = None

STRENGTH_TAU = 0.2
STRENGTH_GAMMA = 1.0
ANCHOR_STRENGTH = 1.5
USE_ANCHORS = False

EPS = 1e-8


# ============================================================
# Helpers
# ============================================================

def _maybe_clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


@torch.no_grad()
def run_one(
    pipe: FluxPipeline,
    prompt: str,
    *,
    record_target_vt: bool = False,
    record_retain_vt: bool = False,
    record_person_vt: bool = False,
    record_anchor_once: bool = False,
    apply_target_proj: bool = False,
    record_concept: Optional[str] = None,
    prompt_tag: Optional[str] = None,
    seed: int = 0,
):
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    ja = {
        "record_target_vt": record_target_vt,
        "record_retain_vt": record_retain_vt,
        "record_person_vt": record_person_vt,
        "record_anchor_once": record_anchor_once,
        "record_concept": record_concept,
        "apply_target_proj": apply_target_proj,
        "use_anchors": USE_ANCHORS,
        "target_block_indices": DUAL_BLOCKS,
        "target_single_block_indices": SINGLE_BLOCKS,
        "strength_tau": STRENGTH_TAU,
        "strength_gamma": STRENGTH_GAMMA,
        "anchor_strength": ANCHOR_STRENGTH,
        "proj_eps": EPS,
        "detector_token_end": DETECTOR_TOKEN_END,
        "proj_token_end": PROJ_TOKEN_END,
        "prompt_tag": prompt_tag,
    }

    _ = pipe(
        prompt=prompt,
        height=REC_H,
        width=REC_W,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        num_images_per_prompt=1,
        generator=generator,
        joint_attention_kwargs=ja,
        output_type="latent",
    )

    _maybe_clear_cache()


def record_target(pipe: FluxPipeline, concept: str, prompt: str, seed: int):
    run_one(
        pipe,
        prompt=prompt,
        record_target_vt=True,
        record_concept=concept,
        seed=seed,
    )


def record_retain(pipe: FluxPipeline, prompt: str, seed: int):
    run_one(
        pipe,
        prompt=prompt,
        record_retain_vt=True,
        seed=seed,
    )


def record_person(pipe: FluxPipeline, prompt: str, seed: int):
    run_one(
        pipe,
        prompt=prompt,
        record_person_vt=True,
        seed=seed,
    )


def _reset_all_state():
    flux_tf.flux_reset_vt_banks(reset_retain=True)
    for name in [
        "_FLUX_VRET_DUAL", "_FLUX_VRET_SINGLE",
        "_FLUX_VPERSON_DUAL", "_FLUX_VPERSON_SINGLE",
        "_FLUX_U_DUAL", "_FLUX_U_SINGLE",
        "_FLUX_A_DUAL", "_FLUX_A_SINGLE",
        "_FLUX_U_UNION_DUAL", "_FLUX_U_UNION_SINGLE",
        "_FLUX_A_UNION_DUAL", "_FLUX_A_UNION_SINGLE",
    ]:
        obj = getattr(flux_tf, name, None)
        if isinstance(obj, dict):
            obj.clear()


def _basis_from_raw_bank(bank: Dict[int, List[torch.Tensor]], top_k: int) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    for blk, vlist in bank.items():
        B = flux_tf._basis_from_vt_list(vlist, top_k=top_k, eps=EPS)
        if B is not None and B.numel() > 0 and B.shape[1] > 0:
            out[blk] = B
    return out


def _normalized_subspace_overlap(U: Optional[torch.Tensor], V: Optional[torch.Tensor]) -> float:
    if U is None or V is None:
        return float("nan")
    if U.numel() == 0 or V.numel() == 0:
        return float("nan")
    if U.shape[1] == 0 or V.shape[1] == 0:
        return float("nan")

    U = U.to(torch.float32)
    V = V.to(torch.float32)
    M = U.t() @ V
    denom = math.sqrt(float(min(U.shape[1], V.shape[1])))
    if denom == 0:
        return float("nan")
    return float(torch.linalg.norm(M, ord="fro").item() / denom)


def _nanmean(x: List[float]) -> float:
    arr = np.array(x, dtype=np.float32)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _collect_dirs_from_target_bank(
    target_bank: Dict[int, Dict[str, List[torch.Tensor]]]
) -> Dict[int, List[torch.Tensor]]:
    out: Dict[int, List[torch.Tensor]] = {}
    for blk, concept_map in target_bank.items():
        dirs = []
        for _, vt_list in concept_map.items():
            for vt in vt_list:
                d = flux_tf._vt_to_dir(vt, eps=EPS)
                if d.norm() > EPS:
                    d = d / (d.norm() + EPS)
                    dirs.append(d)
        if len(dirs) > 0:
            out[blk] = dirs
    return out


def _orth_from_dirs(dirs: List[torch.Tensor]) -> Optional[torch.Tensor]:
    if dirs is None or len(dirs) == 0:
        return None
    M = torch.stack(dirs, dim=1)
    Q = flux_tf._orth_columns(M, eps=EPS)
    if Q is None or Q.numel() == 0 or Q.shape[1] == 0:
        return None
    return Q


def _retain_free_dirs(
    target_dirs: Dict[int, List[torch.Tensor]],
    Vret_bank: Dict[int, torch.Tensor],
    retain_lambda: float,
) -> Dict[int, List[torch.Tensor]]:
    out: Dict[int, List[torch.Tensor]] = {}
    for blk, dirs in target_dirs.items():
        Vret = Vret_bank.get(blk, None)
        new_dirs = []
        for d in dirs:
            d2 = d.clone()
            if Vret is not None and Vret.numel() > 0 and Vret.shape[1] > 0:
                proj = Vret @ (Vret.t() @ d2)
                d2 = d2 - retain_lambda * proj
            if d2.norm() > EPS:
                d2 = d2 / (d2.norm() + EPS)
                new_dirs.append(d2)
        if len(new_dirs) > 0:
            out[blk] = new_dirs
    return out


def _category_residual_dirs(
    free_dirs: Dict[int, List[torch.Tensor]],
    Vcat_bank: Dict[int, torch.Tensor],
) -> Dict[int, List[torch.Tensor]]:
    out: Dict[int, List[torch.Tensor]] = {}
    for blk, dirs in free_dirs.items():
        Vcat = Vcat_bank.get(blk, None)
        new_dirs = []
        for d in dirs:
            d2 = d.clone()
            if Vcat is not None and Vcat.numel() > 0 and Vcat.shape[1] > 0:
                proj = Vcat @ (Vcat.t() @ d2)
                d2 = d2 - proj
            if d2.norm() > EPS:
                d2 = d2 / (d2.norm() + EPS)
                new_dirs.append(d2)
        if len(new_dirs) > 0:
            out[blk] = new_dirs
    return out


def _basis_overlap_series_from_dir_dict(
    dir_dict: Dict[int, List[torch.Tensor]],
    Vcat_bank: Dict[int, torch.Tensor],
    block_indices: List[int],
) -> List[float]:
    vals = []
    for blk in block_indices:
        B = _orth_from_dirs(dir_dict.get(blk, []))
        Vcat = Vcat_bank.get(blk, None)
        vals.append(_normalized_subspace_overlap(B, Vcat))
    return vals


def _category_energy_fraction_series(
    free_dirs: Dict[int, List[torch.Tensor]],
    Vcat_bank: Dict[int, torch.Tensor],
    block_indices: List[int],
) -> List[float]:
    vals = []
    for blk in block_indices:
        dirs = free_dirs.get(blk, [])
        Vcat = Vcat_bank.get(blk, None)

        if len(dirs) == 0 or Vcat is None or Vcat.numel() == 0 or Vcat.shape[1] == 0:
            vals.append(float("nan"))
            continue

        fracs = []
        for d in dirs:
            proj = Vcat @ (Vcat.t() @ d)
            frac = float(proj.norm().item() / (d.norm().item() + EPS))
            fracs.append(frac)

        vals.append(float(np.mean(fracs)))
    return vals


# ============================================================
# Main analysis runner
# ============================================================

def run_analysis(pipe: FluxPipeline) -> Dict[str, object]:
    _reset_all_state()
    _maybe_clear_cache()

    for i, retain in enumerate(RETAINS):
        record_retain(pipe, retain, seed=1000 + i)

    for i, p in enumerate(PERSON_BANK):
        record_person(pipe, p, seed=2000 + i)

    for i, target in enumerate(TARGETS):
        for j, template in enumerate(RECORDING_TEMPLATES):
            prompt = template.format(target)
            record_target(pipe, target, prompt, seed=3000 + 100 * i + j)

    raw_person_dual = copy.deepcopy(flux_tf._FLUX_PERSON_VT_BANK_DUAL)
    raw_person_single = copy.deepcopy(flux_tf._FLUX_PERSON_VT_BANK_SINGLE)
    raw_target_dual = copy.deepcopy(flux_tf._FLUX_TARGET_VT_BANK_DUAL)
    raw_target_single = copy.deepcopy(flux_tf._FLUX_TARGET_VT_BANK_SINGLE)

    flux_tf.flux_finalize_cora_bases(
        retain_top_k=RETAIN_TOP_K,
        person_top_k=PERSON_TOP_K,
        eps=EPS,
    )
    _maybe_clear_cache()

    Vret_dual = copy.deepcopy(flux_tf._FLUX_VRET_DUAL)
    Vret_single = copy.deepcopy(flux_tf._FLUX_VRET_SINGLE)

    Vcat_dual = _basis_from_raw_bank(raw_person_dual, top_k=PERSON_TOP_K)
    Vcat_single = _basis_from_raw_bank(raw_person_single, top_k=PERSON_TOP_K)

    raw_dirs_dual = _collect_dirs_from_target_bank(raw_target_dual)
    raw_dirs_single = _collect_dirs_from_target_bank(raw_target_single)

    retain_lambda = getattr(flux_tf, "_FLUX_RETAIN_LAMBDA", 1.5)

    free_dirs_dual = _retain_free_dirs(raw_dirs_dual, Vret_dual, retain_lambda=retain_lambda)
    free_dirs_single = _retain_free_dirs(raw_dirs_single, Vret_single, retain_lambda=retain_lambda)

    res_dirs_dual = _category_residual_dirs(free_dirs_dual, Vcat_dual)
    res_dirs_single = _category_residual_dirs(free_dirs_single, Vcat_single)

    dual_raw = _basis_overlap_series_from_dir_dict(raw_dirs_dual, Vcat_dual, DUAL_BLOCKS)
    dual_free = _basis_overlap_series_from_dir_dict(free_dirs_dual, Vcat_dual, DUAL_BLOCKS)
    dual_res = _basis_overlap_series_from_dir_dict(res_dirs_dual, Vcat_dual, DUAL_BLOCKS)

    single_raw = _basis_overlap_series_from_dir_dict(raw_dirs_single, Vcat_single, SINGLE_BLOCKS)
    single_free = _basis_overlap_series_from_dir_dict(free_dirs_single, Vcat_single, SINGLE_BLOCKS)
    single_res = _basis_overlap_series_from_dir_dict(res_dirs_single, Vcat_single, SINGLE_BLOCKS)

    dual_energy = _category_energy_fraction_series(free_dirs_dual, Vcat_dual, DUAL_BLOCKS)
    single_energy = _category_energy_fraction_series(free_dirs_single, Vcat_single, SINGLE_BLOCKS)

    return {
        "dual": {
            "raw": dual_raw,
            "retain_free": dual_free,
            "category_residual": dual_res,
            "category_energy_fraction": dual_energy,
            "raw_mean": _nanmean(dual_raw),
            "retain_free_mean": _nanmean(dual_free),
            "category_residual_mean": _nanmean(dual_res),
            "category_energy_fraction_mean": _nanmean(dual_energy),
        },
        "single": {
            "raw": single_raw,
            "retain_free": single_free,
            "category_residual": single_res,
            "category_energy_fraction": single_energy,
            "raw_mean": _nanmean(single_raw),
            "retain_free_mean": _nanmean(single_free),
            "category_residual_mean": _nanmean(single_res),
            "category_energy_fraction_mean": _nanmean(single_energy),
        },
    }


# ============================================================
# Plotting
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 20,
    "axes.linewidth": 1.2,
})

def _style_axis(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.tick_params(axis="both", which="major", width=1.2, length=4)
    ax.tick_params(axis="both", which="minor", width=1.0, length=2)


def _plot_heatline(
    ax,
    values,
    vmin,
    vmax,
    cmap,
):
    arr = np.array(values, dtype=np.float32)[None, :]
    x = np.arange(len(values))

    im = ax.imshow(
        arr,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_yticks([])
    ax.set_xticks(x[::2])
    ax.set_xticklabels([str(i) for i in x[::2]], fontsize=13)
    _style_axis(ax)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        values,
        linewidth=4.6,
        color="white",
        alpha=0.98,
        solid_capstyle="round",
        zorder=5,
    )
    ax2.plot(
        x,
        values,
        linewidth=2.3,
        color="black",
        alpha=0.95,
        solid_capstyle="round",
        zorder=6,
    )

    ax2.set_ylim(vmin, vmax)
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)

    return im


def _add_column_headers(fig, axes, left_text="Dual blocks", right_text="Single blocks", y_pad=0.025):
    left_pos = axes[0, 0].get_position()
    right_pos = axes[0, 1].get_position()

    fig.text(
        (left_pos.x0 + left_pos.x1) / 2,
        left_pos.y1 + y_pad,
        left_text,
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="semibold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
        zorder=20,
    )
    fig.text(
        (right_pos.x0 + right_pos.x1) / 2,
        right_pos.y1 + y_pad,
        right_text,
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="semibold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
        zorder=20,
    )


def make_overlap_figure(results: Dict[str, object], outdir: str):
    dual_raw = np.array(results["dual"]["raw"], dtype=np.float32)
    dual_free = np.array(results["dual"]["retain_free"], dtype=np.float32)
    dual_res = np.array(results["dual"]["category_residual"], dtype=np.float32)

    single_raw = np.array(results["single"]["raw"], dtype=np.float32)
    single_free = np.array(results["single"]["retain_free"], dtype=np.float32)
    single_res = np.array(results["single"]["category_residual"], dtype=np.float32)

    all_vals = np.concatenate([dual_raw, dual_free, dual_res, single_raw, single_free, single_res])
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))

    fig, axes = plt.subplots(
        3, 2,
        figsize=(16, 9.2),
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(h_pad=0.08, w_pad=0.04, hspace=0.06, wspace=0.06)

    cmap_overlap = "viridis"

    im = _plot_heatline(axes[0, 0], dual_raw, vmin=vmin, vmax=vmax, cmap=cmap_overlap)
    _plot_heatline(axes[0, 1], single_raw, vmin=vmin, vmax=vmax, cmap=cmap_overlap)

    _plot_heatline(axes[1, 0], dual_free, vmin=vmin, vmax=vmax, cmap=cmap_overlap)
    _plot_heatline(axes[1, 1], single_free, vmin=vmin, vmax=vmax, cmap=cmap_overlap)

    _plot_heatline(axes[2, 0], dual_res, vmin=vmin, vmax=vmax, cmap=cmap_overlap)
    _plot_heatline(axes[2, 1], single_res, vmin=vmin, vmax=vmax, cmap=cmap_overlap)

    axes[0, 0].set_ylabel("Raw", fontsize=16)
    axes[1, 0].set_ylabel("Retain-free", fontsize=16)
    axes[2, 0].set_ylabel("Residual", fontsize=16)

    fig.suptitle(
        "How category subtraction purifies target directions",
        fontsize=21,
        y=1.06,
        fontweight="semibold",
    )

    _add_column_headers(fig, axes, left_text="Dual blocks", right_text="Single blocks", y_pad=0.025)

    cbar = fig.colorbar(im, ax=axes[:, :], fraction=0.025, pad=0.02)
    cbar.set_label("Normalized overlap with category basis", fontsize=15)
    cbar.ax.tick_params(labelsize=13, width=1.1, length=4)

    png_path = os.path.join(outdir, "target_overlap_stages.png")
    pdf_path = os.path.join(outdir, "target_overlap_stages.pdf")
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def make_energy_figure(results: Dict[str, object], outdir: str):
    dual_energy = np.array(results["dual"]["category_energy_fraction"], dtype=np.float32)
    single_energy = np.array(results["single"]["category_energy_fraction"], dtype=np.float32)

    vals = np.concatenate([dual_energy, single_energy])
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))

    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, 4.1),
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(h_pad=0.08, w_pad=0.04, hspace=0.06, wspace=0.06)

    cmap_energy = "magma"

    im = _plot_heatline(axes[0], dual_energy, vmin=vmin, vmax=vmax, cmap=cmap_energy)
    _plot_heatline(axes[1], single_energy, vmin=vmin, vmax=vmax, cmap=cmap_energy)

    fig.suptitle(
        "Amount of category structure removed by the carving step",
        fontsize=21,
        y=1.08,
        fontweight="semibold",
    )

    pos0 = axes[0].get_position()
    pos1 = axes[1].get_position()

    fig.text(
        (pos0.x0 + pos0.x1) / 2,
        pos0.y1 + 0.025,
        "Dual blocks",
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="semibold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
        zorder=20,
    )
    fig.text(
        (pos1.x0 + pos1.x1) / 2,
        pos1.y1 + 0.025,
        "Single blocks",
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="semibold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
        zorder=20,
    )

    cbar = fig.colorbar(im, ax=axes[:], fraction=0.03, pad=0.02)
    cbar.set_label(
        r"$\|\mathrm{Proj}_{V_{\mathrm{cat}}}(x_{\mathrm{free}})\| / \|x_{\mathrm{free}}\|$",
        fontsize=15
    )
    cbar.ax.tick_params(labelsize=13, width=1.1, length=4)

    png_path = os.path.join(outdir, "category_energy_fraction.png")
    pdf_path = os.path.join(outdir, "category_energy_fraction.pdf")
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path

# ============================================================
# Main
# ============================================================

def main():
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
    ).to(DEVICE)

    results = run_analysis(pipe)
    save_json(results, os.path.join(OUTDIR, "category_mechanism_stats.json"))

    overlap_png, overlap_pdf = make_overlap_figure(results, OUTDIR)
    energy_png, energy_pdf = make_energy_figure(results, OUTDIR)

    print("\nSaved:")
    print(os.path.join(OUTDIR, "category_mechanism_stats.json"))
    print(overlap_png)
    print(overlap_pdf)
    print(energy_png)
    print(energy_pdf)


if __name__ == "__main__":
    main()