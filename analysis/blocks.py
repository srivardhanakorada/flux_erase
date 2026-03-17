import os
import csv
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
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
ANCHOR = "a generic anchor"

PERSON_TOP_K = 2
RETAIN_TOP_K = 4

START_SEED = 0
END_SEED = 4
SEEDS = [i for i in range(START_SEED, END_SEED + 1)]


@dataclass
class ExperimentConfig:
    name: str
    outdir: str
    targets: List[str]
    retains: List[str]
    generic_bank: List[str]
    prompt_templates: List[str]
    recording_templates: List[str]
    retain_top_k: int = 4
    generic_top_k: int = 2
    anchor: str = "a generic anchor"


# ------------------------------------------------------------
# Example configs
# ------------------------------------------------------------

CELEB_CONFIG = ExperimentConfig(
    name="identities",
    outdir="results_new/ours/paper_target_amount/celebs",
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
    name="artistic styles",
    outdir="results_new/ours/paper_target_amount/styles",
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
    outdir="results_new/ours/paper_target_amount/objects",
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
    output_type = "latent"

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
        output_type=output_type,
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
                    measure_label="target",
                    seed=s,
                    record_mode=False,
                )
        print(f"Measured target :: {item}")


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if len(xs) > 0 else 0.0


def _summarize_target_only(raw_stats: Dict[str, Any]) -> Dict[str, Dict[int, Dict[str, float]]]:
    out: Dict[str, Dict[int, Dict[str, float]]] = {
        "dual": {},
        "single": {},
    }

    for kind in ["dual", "single"]:
        label_stats = raw_stats.get(kind, {}).get("target", {})
        for blk, entries in label_stats.items():
            if len(entries) == 0:
                continue

            out[kind][blk] = {
                "coeff_norm_mean": _mean([x["coeff_norm_mean"] for x in entries]),
                "proj_norm_mean": _mean([x["proj_norm_mean"] for x in entries]),
                "relative_proj_mean": _mean([x["relative_proj_mean"] for x in entries]),
                "max_coeff_mean": _mean([x["max_coeff_mean"] for x in entries]),
            }

    return out


def save_target_csv(summary: Dict[str, Dict[int, Dict[str, float]]], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "target_amount_by_block.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "kind",
                "block_index",
                "coeff_norm_mean",
                "proj_norm_mean",
                "relative_proj_mean",
                "max_coeff_mean",
            ]
        )

        for kind in ["dual", "single"]:
            for blk in sorted(summary[kind].keys()):
                stats = summary[kind][blk]
                writer.writerow(
                    [
                        kind,
                        blk,
                        stats["coeff_norm_mean"],
                        stats["proj_norm_mean"],
                        stats["relative_proj_mean"],
                        stats["max_coeff_mean"],
                    ]
                )

    print(f"Saved CSV: {csv_path}")


def set_paper_plot_style():
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(axis="both", which="both", width=1.0, length=4)
    ax.grid(False)


def _plot_single_curve(
    ax,
    blocks: List[int],
    vals: List[float],
    xlabel: str,
    ylabel: str,
    title: str,
):
    ax.plot(
        blocks,
        vals,
        linewidth=2.2,
        marker="o",
        markersize=4.8,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    _style_axis(ax)


def save_paper_plots(
    summary: Dict[str, Dict[int, Dict[str, float]]],
    save_dir: str,
    experiment_name: str,
    metric: str = "relative_proj_mean",
):
    os.makedirs(save_dir, exist_ok=True)
    set_paper_plot_style()

    metric_to_ylabel = {
        "relative_proj_mean": "Target-aligned fraction",
        "proj_norm_mean": "Target projection norm",
        "coeff_norm_mean": "Target coefficient norm",
        "max_coeff_mean": "Max target coefficient",
    }
    ylabel = metric_to_ylabel.get(metric, metric)

    # Separate dual
    dual_blocks = sorted(summary["dual"].keys())
    if len(dual_blocks) > 0:
        dual_vals = [summary["dual"][b][metric] for b in dual_blocks]
        fig, ax = plt.subplots(figsize=(5.8, 3.8))
        _plot_single_curve(
            ax=ax,
            blocks=dual_blocks,
            vals=dual_vals,
            xlabel="Dual block index",
            ylabel=ylabel,
            title="Dual blocks",
        )
        png_path = os.path.join(save_dir, f"{experiment_name}_dual_{metric}_paper.png")
        pdf_path = os.path.join(save_dir, f"{experiment_name}_dual_{metric}_paper.pdf")
        fig.savefig(png_path, dpi=300)
        fig.savefig(pdf_path)
        plt.close(fig)
        print(f"Saved plot: {png_path}")
        print(f"Saved plot: {pdf_path}")

    # Separate single
    single_blocks = sorted(summary["single"].keys())
    if len(single_blocks) > 0:
        single_vals = [summary["single"][b][metric] for b in single_blocks]
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        _plot_single_curve(
            ax=ax,
            blocks=single_blocks,
            vals=single_vals,
            xlabel="Single block index",
            ylabel=ylabel,
            title="Single blocks",
        )
        png_path = os.path.join(save_dir, f"{experiment_name}_single_{metric}_paper.png")
        pdf_path = os.path.join(save_dir, f"{experiment_name}_single_{metric}_paper.pdf")
        fig.savefig(png_path, dpi=300)
        fig.savefig(pdf_path)
        plt.close(fig)
        print(f"Saved plot: {png_path}")
        print(f"Saved plot: {pdf_path}")

    # Combined figure for paper
    if len(dual_blocks) > 0 and len(single_blocks) > 0:
        dual_vals = [summary["dual"][b][metric] for b in dual_blocks]
        single_vals = [summary["single"][b][metric] for b in single_blocks]

        fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))
        _plot_single_curve(
            ax=axes[0],
            blocks=dual_blocks,
            vals=dual_vals,
            xlabel="Dual block index",
            ylabel=ylabel,
            title="Dual blocks",
        )
        _plot_single_curve(
            ax=axes[1],
            blocks=single_blocks,
            vals=single_vals,
            xlabel="Single block index",
            ylabel=ylabel,
            title="Single blocks",
        )
        fig.suptitle(f"{experiment_name.capitalize()}", y=1.02)

        png_path = os.path.join(save_dir, f"{experiment_name}_{metric}_combined_paper.png")
        pdf_path = os.path.join(save_dir, f"{experiment_name}_{metric}_combined_paper.pdf")
        fig.savefig(png_path, dpi=300)
        fig.savefig(pdf_path)
        plt.close(fig)
        print(f"Saved plot: {png_path}")
        print(f"Saved plot: {pdf_path}")


def print_top_target_blocks(summary: Dict[str, Dict[int, Dict[str, float]]], metric: str = "relative_proj_mean", top_k: int = 15):
    rows = []
    for kind in ["dual", "single"]:
        for blk, stats in summary[kind].items():
            rows.append((stats[metric], kind, blk))

    rows.sort(reverse=True, key=lambda x: x[0])

    print(f"\nTop blocks by {metric}:")
    for i, (val, kind, blk) in enumerate(rows[:top_k], start=1):
        print(f"{i:02d}. {kind:6s} block={blk:2d} | {metric}={val:.6f}")


def run_experiment(pipe: FluxPipeline, cfg: ExperimentConfig):
    stats_dir = os.path.join(cfg.outdir, "block_target_amount")
    os.makedirs(cfg.outdir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    global ANCHOR
    ANCHOR = cfg.anchor

    flux_reset_vt_banks(reset_retain=True)
    _maybe_clear_cache()

    # Retain bank
    for i, rp in enumerate(cfg.retains):
        run_one(
            pipe,
            prompt=rp,
            record_retain_vt=True,
            seed=1000 + i,
            record_mode=True,
        )

    # Generic bank
    for i, gp in enumerate(cfg.generic_bank):
        run_one(
            pipe,
            prompt=gp,
            record_person_vt=True,
            seed=2000 + i,
            record_mode=True,
        )

    # Target bank
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
    summary = _summarize_target_only(raw_stats)

    save_target_csv(summary, stats_dir)
    save_paper_plots(
        summary=summary,
        save_dir=stats_dir,
        experiment_name=cfg.name,
        metric="relative_proj_mean",
    )
    save_paper_plots(
        summary=summary,
        save_dir=stats_dir,
        experiment_name=cfg.name,
        metric="proj_norm_mean",
    )
    print_top_target_blocks(summary, metric="relative_proj_mean", top_k=15)

    print(f"\nDone. Saved to: {stats_dir}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    ).to(device)

    # Choose whichever experiments you want to run
    experiments = [
        CELEB_CONFIG,
        STYLE_CONFIG,
        OBJECT_CONFIG,
    ]

    for cfg in experiments:
        print(f"\n================ Running {cfg.name} ================\n")
        run_experiment(pipe, cfg)


if __name__ == "__main__":
    main()