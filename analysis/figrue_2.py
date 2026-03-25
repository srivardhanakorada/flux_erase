# figure2_heatmap_exact_values.py

import os
import csv
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from diffusers import FluxPipeline
from diffusers.models.transformers.transformer_flux import (  # type: ignore
    flux_reset_vt_banks,
    flux_finalize_cora_bases,
    flux_reset_target_info_stats,
    flux_get_target_info_stats,
)

MODEL_ID = "black-forest-labs/FLUX.1-schnell"

DUAL_BLOCKS = list(range(0, 19))
SINGLE_BLOCKS = list(range(0, 38))

REC_H, REC_W = 768, 768
GEN_H, GEN_W = 768, 768
STEPS = 4
GUIDANCE = 3.5
N_IMAGES_PER_PROMPT = 1

STRENGTH_TAU = 0.1
STRENGTH_GAMMA = 1.75
ANCHOR_STRENGTH = 1.0
USE_ANCHORS = False

START_SEED = 0
END_SEED = 4
SEEDS = [i for i in range(START_SEED, END_SEED + 1)]

FIGURE2_METRIC = "proj_norm_mean"
# alternatives:
# "coeff_norm_mean"
# "max_coeff_mean"
# "relative_proj_mean"

OUTDIR = "results_new/ours/paper_target_amount/figure2_exact_heatmaps"
os.makedirs(OUTDIR, exist_ok=True)


@dataclass
class ExperimentConfig:
    name: str
    plot_title: str
    targets: List[str]
    retains: List[str]
    generic_bank: List[str]
    prompt_templates: List[str]
    recording_templates: List[str]
    retain_top_k: int = 4
    generic_top_k: int = 2
    anchor: str = "a generic anchor"


CELEB_CONFIG = ExperimentConfig(
    name="identities",
    plot_title="Celebrities",
    targets=[
        "Donald Trump",
    ],
    retains=[
        "Hillary Clinton",
        "Melania Trump",
        "Barack Obama",
    ],
    generic_bank=[
        "a portrait of a person",
        "a portrait of a man",
        "a portrait of a woman",
        "a middle-aged man",
        "a middle-aged woman",
    ],
    prompt_templates=[
        "a photo of {}",
    ],
    recording_templates=[
        "a photo of {}",
        "{}, studio portrait, sharp focus",
    ],
    retain_top_k=4,
    generic_top_k=2,
    anchor="a portrait of a person",
)

STYLE_CONFIG = ExperimentConfig(
    name="styles",
    plot_title="Styles",
    targets=[
        "Van Gogh",
    ],
    retains=[
        "Pablo Picasso",
        "Claude Monet",
        "Leonardo da Vinci",
    ],
    generic_bank=[
        "a painting",
        "an artwork",
        "an oil painting",
        "a watercolor painting",
        "a canvas painting",
    ],
    prompt_templates=[
        "a painting in the style of {}",
    ],
    recording_templates=[
        "a painting in the style of {}",
        "an artwork in the style of {}",
    ],
    retain_top_k=4,
    generic_top_k=2,
    anchor="a painting",
)

OBJECT_CONFIG = ExperimentConfig(
    name="objects",
    plot_title="Objects",
    targets=[
        "goldfish",
    ],
    retains=[
        "tench",
        "stingray",
        "hammerhead",
    ],
    generic_bank=[
        "a photo of a fish",
        "a photo of an animal",
        "a studio photo of an aquatic animal",
        "a realistic image of a fish",
        "a close-up photo of an animal",
    ],
    prompt_templates=[
        "a photo of {}",
    ],
    recording_templates=[
        "a photo of {}",
        "a close-up photo of {}",
    ],
    retain_top_k=4,
    generic_top_k=2,
    anchor="a photo of a fish",
)


def _make_prompt(x: str, prompt_template: str) -> str:
    return prompt_template.format(x)


def _maybe_clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    measure_target_info: bool = False,
    measure_label: str = "target",
    record_concept: Optional[str] = None,
    seed: int = 0,
    record_mode: bool = False,
):
    g = torch.Generator(device=pipe.device).manual_seed(seed)
    height = REC_H if record_mode else GEN_H
    width = REC_W if record_mode else GEN_W

    ja = {
        "record_target_vt": record_target_vt,
        "record_retain_vt": record_retain_vt,
        "record_person_vt": record_person_vt,
        "record_anchor_once": record_anchor_once,
        "record_concept": record_concept,
        "apply_target_proj": apply_target_proj,
        "measure_target_info": measure_target_info,
        "measure_label": measure_label,
        "use_anchors": USE_ANCHORS,
        "target_block_indices": DUAL_BLOCKS,
        "target_single_block_indices": SINGLE_BLOCKS,
        "strength_tau": STRENGTH_TAU,
        "strength_gamma": STRENGTH_GAMMA,
        "anchor_strength": ANCHOR_STRENGTH,
        "proj_eps": 1e-8,
        "debug_tokens": False,
    }

    _ = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        num_images_per_prompt=N_IMAGES_PER_PROMPT,
        generator=g,
        joint_attention_kwargs=ja,
        output_type="latent",
    )
    _maybe_clear_cache()


def run_target_measurement_set(
    pipe: FluxPipeline,
    items: List[str],
    templates: List[str],
):
    for item in items:
        for prompt_template in templates:
            p = _make_prompt(item, prompt_template)
            for s in SEEDS:
                run_one(
                    pipe,
                    p,
                    apply_target_proj=False,
                    measure_target_info=True,
                    measure_label=item,
                    seed=s,
                    record_mode=False,
                )


def save_raw_measurement_csv(raw_stats: Dict[str, Any], save_dir: str, name: str):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{name}_raw_measurements.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "kind",
                "label",
                "block_index",
                "measurement_index",
                "coeff_norm_mean",
                "proj_norm_mean",
                "relative_proj_mean",
                "max_coeff_mean",
            ]
        )

        for kind in ["dual", "single"]:
            for label, block_map in raw_stats.get(kind, {}).items():
                for blk, entries in block_map.items():
                    for idx, stats in enumerate(entries):
                        writer.writerow(
                            [
                                kind,
                                label,
                                blk,
                                idx,
                                stats["coeff_norm_mean"],
                                stats["proj_norm_mean"],
                                stats["relative_proj_mean"],
                                stats["max_coeff_mean"],
                            ]
                        )

    print(f"Saved raw CSV: {csv_path}")


def set_paper_plot_style():
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": 15,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _bold_axis_text(ax):
    ax.title.set_fontweight("bold")
    # ax.xaxis.label.set_fontweight("bold")
    # ax.yaxis.label.set_fontweight("bold")
    # for tick in ax.get_xticklabels():
    #     tick.set_fontweight("bold")
    # for tick in ax.get_yticklabels():
    #     tick.set_fontweight("bold")


def run_experiment(
    pipe: FluxPipeline,
    cfg: ExperimentConfig,
) -> Dict[str, Any]:
    print(f"\n================ Running {cfg.name} ================\n")

    flux_reset_vt_banks(reset_retain=True)
    _maybe_clear_cache()

    for i, rp in enumerate(cfg.retains):
        run_one(
            pipe,
            prompt=rp,
            record_retain_vt=True,
            seed=1000 + i,
            record_mode=True,
        )

    for i, gp in enumerate(cfg.generic_bank):
        run_one(
            pipe,
            prompt=gp,
            record_person_vt=True,
            seed=2000 + i,
            record_mode=True,
        )

    for i, t in enumerate(cfg.targets):
        for j, pt in enumerate(cfg.recording_templates):
            prompt = pt.format(t)
            run_one(
                pipe,
                prompt=prompt,
                record_target_vt=True,
                record_concept=t,
                seed=3000 + 100 * i + j,
                record_mode=True,
            )

    if USE_ANCHORS:
        run_one(
            pipe,
            prompt=cfg.anchor,
            record_anchor_once=True,
            seed=4000,
            record_mode=True,
        )

    flux_finalize_cora_bases(
        retain_top_k=cfg.retain_top_k,
        person_top_k=cfg.generic_top_k,
    )
    _maybe_clear_cache()

    flux_reset_target_info_stats()
    run_target_measurement_set(pipe, cfg.targets, cfg.prompt_templates)

    raw_stats = flux_get_target_info_stats()
    save_raw_measurement_csv(raw_stats, OUTDIR, cfg.name)
    return raw_stats


def build_instance_heatmap_matrix(
    raw_stats: Dict[str, Any],
    kind: str,
    label: str,
    metric: str,
    all_blocks: List[int],
) -> np.ndarray:
    block_map = raw_stats.get(kind, {}).get(label, {})
    if len(block_map) == 0:
        return np.zeros((1, len(all_blocks)), dtype=np.float32)

    max_rows = max(len(entries) for entries in block_map.values())
    mat = np.zeros((max_rows, len(all_blocks)), dtype=np.float32)

    for j, blk in enumerate(all_blocks):
        entries = block_map.get(blk, [])
        for i, stats in enumerate(entries):
            mat[i, j] = float(stats[metric])

    return mat


def metric_to_cbar_label(metric: str) -> str:
    return {
        "proj_norm_mean": "Target projection norm",
        "coeff_norm_mean": "Target coefficient norm",
        "max_coeff_mean": "Max target coefficient",
        "relative_proj_mean": "Target-aligned fraction",
    }.get(metric, metric)


def plot_heatmap_panel(
    ax,
    raw_stats: Dict[str, Any],
    label: str,
    kind: str,
    blocks: List[int],
    metric: str,
    title: str,
    show_ylabel: bool,
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
):
    mat = build_instance_heatmap_matrix(raw_stats, kind, label, metric, blocks)

    yticklabels = [f"M{i+1}" for i in range(mat.shape[0])] if show_ylabel else False

    hm = sns.heatmap(
        mat,
        ax=ax,
        cmap=cmap,
        cbar=False,
        xticklabels=blocks,
        yticklabels=yticklabels,
        linewidths=0.3,
        linecolor="white",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title, pad=8, fontweight="bold")
    ax.set_xlabel("Block index", fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("Measurement", fontweight="bold")
    else:
        ax.set_ylabel("")
        ax.set_yticks([])

    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)
    _bold_axis_text(ax)

    return hm.collections[0]


def compute_global_minmax(
    raw_groups: List[Tuple[Dict[str, Any], str]],
    kind: str,
    metric: str,
    blocks: List[int],
) -> Tuple[float, float]:
    vals = []
    for raw_stats, label in raw_groups:
        mat = build_instance_heatmap_matrix(raw_stats, kind, label, metric, blocks)
        vals.append(mat.reshape(-1))
    all_vals = np.concatenate(vals, axis=0)
    return float(np.min(all_vals)), float(np.max(all_vals))


def plot_combined_figure2_heatmaps_only(
    celeb_raw: Dict[str, Any],
    celeb_label: str,
    object_raw: Dict[str, Any],
    object_label: str,
    style_raw: Dict[str, Any],
    style_label: str,
    save_dir: str,
    metric: str = "proj_norm_mean",
    kind: str = "single",
):
    set_paper_plot_style()
    os.makedirs(save_dir, exist_ok=True)

    if kind == "dual":
        blocks = DUAL_BLOCKS
        row_tag = "dual"
        figsize = (14.5, 4.3)
        suptitle_text = "Dual Blocks"
    elif kind == "single":
        blocks = SINGLE_BLOCKS
        row_tag = "single"
        figsize = (16.5, 4.3)
        suptitle_text = "Single Blocks"
    else:
        raise ValueError("kind must be 'dual' or 'single'")

    domains = [
        ("Celebrities", celeb_raw, celeb_label),
        ("Objects", object_raw, object_label),
        ("Styles", style_raw, style_label),
    ]

    vmin, vmax = compute_global_minmax(
        raw_groups=[(celeb_raw, celeb_label), (object_raw, object_label), (style_raw, style_label)],
        kind=row_tag,
        metric=metric,
        blocks=blocks,
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    cmap = "mako"

    mappable = None
    for col, (title, raw_stats, label) in enumerate(domains):
        mappable = plot_heatmap_panel(
            ax=axes[col],
            raw_stats=raw_stats,
            label=label,
            kind=row_tag,
            blocks=blocks,
            metric=metric,
            title=title,
            show_ylabel=(col == 0),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    cbar = fig.colorbar(
        mappable,
        ax=axes,
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label(metric_to_cbar_label(metric), fontweight="bold")
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")

    fig.suptitle(suptitle_text, y=1.02, fontweight="bold")

    png_path = os.path.join(save_dir, f"figure2_heatmaps_only_{kind}_{metric}.png")
    pdf_path = os.path.join(save_dir, f"figure2_heatmaps_only_{kind}_{metric}.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved Figure 2 PNG: {png_path}")
    print(f"Saved Figure 2 PDF: {pdf_path}")


def plot_combined_figure2_heatmaps_with_box_summary(
    celeb_raw: Dict[str, Any],
    celeb_label: str,
    object_raw: Dict[str, Any],
    object_label: str,
    style_raw: Dict[str, Any],
    style_label: str,
    save_dir: str,
    metric: str = "proj_norm_mean",
    kind: str = "single",
):
    set_paper_plot_style()
    os.makedirs(save_dir, exist_ok=True)

    if kind == "dual":
        blocks = DUAL_BLOCKS
        row_tag = "dual"
        figsize = (14.5, 6.8)
        suptitle_text = "Dual Blocks"
    elif kind == "single":
        blocks = SINGLE_BLOCKS
        row_tag = "single"
        figsize = (16.5, 6.8)
        suptitle_text = "Single Blocks"
    else:
        raise ValueError("kind must be 'dual' or 'single'")

    domains = [
        ("Celebrities", celeb_raw, celeb_label),
        ("Objects", object_raw, object_label),
        ("Styles", style_raw, style_label),
    ]

    vmin, vmax = compute_global_minmax(
        raw_groups=[(celeb_raw, celeb_label), (object_raw, object_label), (style_raw, style_label)],
        kind=row_tag,
        metric=metric,
        blocks=blocks,
    )

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.9, 1.0], hspace=0.38, wspace=0.22)

    cmap = "mako"
    mappable = None

    for col, (title, raw_stats, label) in enumerate(domains):
        mat = build_instance_heatmap_matrix(raw_stats, row_tag, label, metric, blocks)

        ax_h = fig.add_subplot(gs[0, col])
        hm = sns.heatmap(
            mat,
            ax=ax_h,
            cmap=cmap,
            cbar=False,
            xticklabels=blocks,
            yticklabels=[f"M{i+1}" for i in range(mat.shape[0])] if col == 0 else False,
            linewidths=0.3,
            linecolor="white",
            vmin=vmin,
            vmax=vmax,
        )
        mappable = hm.collections[0]
        ax_h.set_title(title, pad=8, fontweight="bold")
        ax_h.set_xlabel("Block index", fontweight="bold")
        ax_h.set_ylabel("Measurement" if col == 0 else "", fontweight="bold")
        if col != 0:
            ax_h.set_yticks([])
        ax_h.tick_params(axis="x", rotation=90)
        ax_h.tick_params(axis="y", rotation=0)
        _bold_axis_text(ax_h)

        ax_b = fig.add_subplot(gs[1, col])
        block_series = [mat[:, j] for j in range(mat.shape[1])]

        ax_b.boxplot(
            block_series,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"linewidth": 1.6, "color": "black"},
            boxprops={"facecolor": "#8ecae6", "edgecolor": "#457b9d", "linewidth": 1.0},
            whiskerprops={"linewidth": 1.0, "color": "#457b9d"},
            capprops={"linewidth": 1.0, "color": "#457b9d"},
        )
        ax_b.set_xlabel("Block index", fontweight="bold")
        ax_b.set_ylabel(metric_to_cbar_label(metric) if col == 0 else "", fontweight="bold")
        ax_b.set_xticks(range(1, len(blocks) + 1))
        ax_b.set_xticklabels(blocks, rotation=90)
        _bold_axis_text(ax_b)
        sns.despine(ax=ax_b)

    cbar = fig.colorbar(
        mappable,
        ax=fig.axes,
        fraction=0.018,
        pad=0.01,
    )
    cbar.set_label(metric_to_cbar_label(metric), fontweight="bold")
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")

    fig.suptitle(suptitle_text, y=0.995, fontweight="bold")

    png_path = os.path.join(save_dir, f"figure2_heatmaps_boxsummary_{kind}_{metric}.png")
    pdf_path = os.path.join(save_dir, f"figure2_heatmaps_boxsummary_{kind}_{metric}.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved Figure 2 PNG: {png_path}")
    print(f"Saved Figure 2 PDF: {pdf_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    ).to(device)

    celeb_raw = run_experiment(pipe, CELEB_CONFIG)
    object_raw = run_experiment(pipe, OBJECT_CONFIG)
    style_raw = run_experiment(pipe, STYLE_CONFIG)

    plot_combined_figure2_heatmaps_only(
        celeb_raw=celeb_raw,
        celeb_label=CELEB_CONFIG.targets[0],
        object_raw=object_raw,
        object_label=OBJECT_CONFIG.targets[0],
        style_raw=style_raw,
        style_label=STYLE_CONFIG.targets[0],
        save_dir=OUTDIR,
        metric=FIGURE2_METRIC,
        kind="single",
    )

    plot_combined_figure2_heatmaps_only(
        celeb_raw=celeb_raw,
        celeb_label=CELEB_CONFIG.targets[0],
        object_raw=object_raw,
        object_label=OBJECT_CONFIG.targets[0],
        style_raw=style_raw,
        style_label=STYLE_CONFIG.targets[0],
        save_dir=OUTDIR,
        metric=FIGURE2_METRIC,
        kind="dual",
    )

    plot_combined_figure2_heatmaps_with_box_summary(
        celeb_raw=celeb_raw,
        celeb_label=CELEB_CONFIG.targets[0],
        object_raw=object_raw,
        object_label=OBJECT_CONFIG.targets[0],
        style_raw=style_raw,
        style_label=STYLE_CONFIG.targets[0],
        save_dir=OUTDIR,
        metric=FIGURE2_METRIC,
        kind="single",
    )

    plot_combined_figure2_heatmaps_with_box_summary(
        celeb_raw=celeb_raw,
        celeb_label=CELEB_CONFIG.targets[0],
        object_raw=object_raw,
        object_label=OBJECT_CONFIG.targets[0],
        style_raw=style_raw,
        style_label=STYLE_CONFIG.targets[0],
        save_dir=OUTDIR,
        metric=FIGURE2_METRIC,
        kind="dual",
    )


if __name__ == "__main__":
    main()