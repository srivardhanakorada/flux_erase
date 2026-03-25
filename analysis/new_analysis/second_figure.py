# analyze_flux_block_localization_small.py

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from diffusers import FluxPipeline
from diffusers.models.transformers.transformer_flux import (
    flux_reset_vt_banks,
    flux_finalize_cora_bases,
    flux_get_concept_localization_report,
)

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
OUTDIR = "analysis_flux_figs_small"
os.makedirs(OUTDIR, exist_ok=True)

DUAL_BLOCKS = list(range(19))
SINGLE_BLOCKS = list(range(38))

# smaller recording settings
REC_H, REC_W = 512, 512
STEPS = 2
GUIDANCE = 3.5
N_IMAGES_PER_PROMPT = 1

STRENGTH_TAU = 0.1
STRENGTH_GAMMA = 1.5
ANCHOR_STRENGTH = 1.0
USE_ANCHORS = False
RETAIN_TOP_K = 2
PERSON_TOP_K = 1

FIG2_METRIC = "res_proj_norm"
FIG3_METRIC = "res_proj_norm"

# very small domain configs
DOMAIN_CONFIGS: Dict[str, Dict[str, List[str]]] = {
    "celebrities": {
        "targets": [
            "Donald Trump",
            "Taylor Swift",
        ],
        "retain_bank": [
            "Hillary Clinton",
            "Barack Obama",
        ],
        "person_bank": [
            "a portrait of a person",
            "a portrait of a woman",
        ],
        "recording_templates": [
            "a photo of {}",
        ],
    },
    "objects": {
        "targets": [
            "Dog",
            "Car",
        ],
        "retain_bank": [
            "Chair",
            "Apple",
        ],
        "person_bank": [
            "an object",
            "an animal",
        ],
        "recording_templates": [
            "a photo of {}",
        ],
    },
    "styles": {
        "targets": [
            "Picasso",
            "Van Gogh",
        ],
        "retain_bank": [
            "oil painting",
            "watercolor painting",
        ],
        "person_bank": [
            "a painting",
            "an artwork",
        ],
        "recording_templates": [
            "a painting in the style of {}",
        ],
    },
}


def _maybe_clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _make_prompt(concept: str, template: str) -> str:
    return template.format(concept)


@torch.no_grad()
def run_record_only(
    pipe: FluxPipeline,
    prompt: str,
    *,
    record_target_vt: bool = False,
    record_retain_vt: bool = False,
    record_person_vt: bool = False,
    record_anchor_once: bool = False,
    record_concept: Optional[str] = None,
    seed: int = 0,
):
    g = torch.Generator(device=pipe.device).manual_seed(seed)

    ja = {
        "record_target_vt": record_target_vt,
        "record_retain_vt": record_retain_vt,
        "record_person_vt": record_person_vt,
        "record_anchor_once": record_anchor_once,
        "record_concept": record_concept,
        "apply_target_proj": False,
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
        height=REC_H,
        width=REC_W,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        num_images_per_prompt=N_IMAGES_PER_PROMPT,
        generator=g,
        joint_attention_kwargs=ja,
        output_type="latent",
    )
    _maybe_clear_cache()


def record_domain(
    pipe: FluxPipeline,
    domain_name: str,
    cfg: Dict[str, List[str]],
) -> pd.DataFrame:
    print(f"\n=== Recording domain: {domain_name} ===")

    flux_reset_vt_banks(reset_retain=True)
    _maybe_clear_cache()

    retain_bank = cfg["retain_bank"]
    person_bank = cfg["person_bank"]
    targets = cfg["targets"]
    recording_templates = cfg["recording_templates"]

    for i, concept in enumerate(retain_bank):
        run_record_only(
            pipe,
            prompt=concept,
            record_retain_vt=True,
            seed=1000 + i,
        )

    for i, prompt in enumerate(person_bank):
        run_record_only(
            pipe,
            prompt=prompt,
            record_person_vt=True,
            seed=2000 + i,
        )

    for i, target in enumerate(targets):
        for j, template in enumerate(recording_templates):
            run_record_only(
                pipe,
                prompt=_make_prompt(target, template),
                record_target_vt=True,
                record_concept=target,
                seed=3000 + 100 * i + j,
            )

    flux_finalize_cora_bases(
        retain_top_k=RETAIN_TOP_K,
        person_top_k=PERSON_TOP_K,
    )

    rows = flux_get_concept_localization_report()
    df = pd.DataFrame(rows)
    df["domain"] = domain_name

    csv_path = os.path.join(OUTDIR, f"localization_report_{domain_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    return df


def aggregate_block_curves(
    df: pd.DataFrame,
    metric: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tmp = (
        df.groupby(["kind", "concept", "block_index"], as_index=False)[metric]
        .mean()
    )

    dual = tmp[tmp["kind"] == "dual"].copy()
    single = tmp[tmp["kind"] == "single"].copy()

    dual_summary = (
        dual.groupby("block_index", as_index=False)[metric]
        .mean()
        .sort_values("block_index")
    )
    single_summary = (
        single.groupby("block_index", as_index=False)[metric]
        .mean()
        .sort_values("block_index")
    )
    return dual_summary, single_summary


def compute_cumulative_topk(
    df: pd.DataFrame,
    metric: str,
    kind: str,
) -> np.ndarray:
    sub = df[df["kind"] == kind].copy()
    tmp = (
        sub.groupby(["concept", "block_index"], as_index=False)[metric]
        .mean()
    )

    curves = []
    for _, cdf in tmp.groupby("concept"):
        vals = cdf.sort_values("block_index")[metric].to_numpy(dtype=float)
        vals = np.maximum(vals, 0.0)
        total = vals.sum()
        if total <= 1e-12:
            continue
        vals_sorted = np.sort(vals)[::-1]
        curve = np.cumsum(vals_sorted) / total
        curves.append(curve)

    if len(curves) == 0:
        return np.array([])

    max_len = max(len(c) for c in curves)
    padded = np.stack(
        [np.pad(c, (0, max_len - len(c)), mode="edge") for c in curves],
        axis=0,
    )
    return padded.mean(axis=0)


def plot_figure2(
    domain_to_df: Dict[str, pd.DataFrame],
    metric: str,
    out_png: str,
):
    domains = ["celebrities", "objects", "styles"]
    titles = {
        "celebrities": "Celebrities",
        "objects": "Objects",
        "styles": "Styles",
    }

    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=False)
    for col, domain in enumerate(domains):
        df = domain_to_df[domain]
        dual_summary, single_summary = aggregate_block_curves(df, metric)

        ax = axes[0, col]
        ax.plot(dual_summary["block_index"], dual_summary[metric], marker="o", markersize=2, linewidth=1.6)
        ax.set_title(titles[domain], fontsize=10)
        ax.set_xlabel("Block index", fontsize=8)
        if col == 0:
            ax.set_ylabel("Projection norm", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(alpha=0.25)

        ax = axes[1, col]
        ax.plot(single_summary["block_index"], single_summary[metric], marker="o", markersize=2, linewidth=1.6)
        ax.set_xlabel("Block index", fontsize=8)
        if col == 0:
            ax.set_ylabel("Projection norm", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(alpha=0.25)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 2: {out_png}")


def plot_figure3(
    domain_to_df: Dict[str, pd.DataFrame],
    metric: str,
    out_png: str,
):
    domains = ["celebrities", "objects", "styles"]
    titles = {
        "celebrities": "Celebrities",
        "objects": "Objects",
        "styles": "Styles",
    }

    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharey=True)

    for col, domain in enumerate(domains):
        df = domain_to_df[domain]

        dual_curve = compute_cumulative_topk(df, metric, kind="dual")
        single_curve = compute_cumulative_topk(df, metric, kind="single")

        ax = axes[0, col]
        if dual_curve.size > 0:
            k = np.arange(1, len(dual_curve) + 1)
            ax.plot(k, dual_curve, marker="o", markersize=2, linewidth=1.6)
        ax.set_title(titles[domain], fontsize=10)
        ax.set_xlabel("Top-k blocks", fontsize=8)
        if col == 0:
            ax.set_ylabel("Cumulative coverage", fontsize=8)
        ax.set_ylim(0.0, 1.02)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(alpha=0.25)

        ax = axes[1, col]
        if single_curve.size > 0:
            k = np.arange(1, len(single_curve) + 1)
            ax.plot(k, single_curve, marker="o", markersize=2, linewidth=1.6)
        ax.set_xlabel("Top-k blocks", fontsize=8)
        if col == 0:
            ax.set_ylabel("Cumulative coverage", fontsize=8)
        ax.set_ylim(0.0, 1.02)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(alpha=0.25)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 3: {out_png}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading pipeline on {device} with dtype={dtype} ...")
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    ).to(device)

    # optional memory helpers
    if device == "cuda":
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass
        try:
            pipe.vae.enable_slicing()
        except Exception:
            pass
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass

    domain_to_df: Dict[str, pd.DataFrame] = {}

    for domain_name, cfg in DOMAIN_CONFIGS.items():
        df = record_domain(pipe, domain_name, cfg)
        if len(df) == 0:
            raise RuntimeError(f"No rows were returned for domain={domain_name}.")
        domain_to_df[domain_name] = df

    fig2_path = os.path.join(OUTDIR, "figure2_blockwise_projection_strength.png")
    fig3_path = os.path.join(OUTDIR, "figure3_cumulative_topk_coverage.png")

    plot_figure2(domain_to_df, FIG2_METRIC, fig2_path)
    plot_figure3(domain_to_df, FIG3_METRIC, fig3_path)

    print("\nDone.")
    print(fig2_path)
    print(fig3_path)


if __name__ == "__main__":
    main()