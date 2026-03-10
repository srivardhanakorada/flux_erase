import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
from PIL import Image  # type: ignore

from diffusers import FluxPipeline
from diffusers.models.transformers.transformer_flux import (  # type: ignore
    flux_finalize_cora_bases,
    flux_get_diag_stats,
    flux_reset_diag_stats,
    flux_reset_vt_banks,
)

import diffusers.models.transformers.transformer_flux as tf

print("USING TRANSFORMER FILE:", tf.__file__)
print("FluxAttnProcessor args:", tf.inspect.signature(tf.FluxAttnProcessor.__call__))

MODEL_ID = "black-forest-labs/FLUX.1-schnell"

# ============================================================
# Prompts
# ============================================================
PROMPT_TEMPLATES = [
    "a photo of {}",
]

RECORDING_TEMPLATES = [
    "a photo of {}",
    "{} photographed with DSLR",
    "{}, studio portrait, sharp focus",
    "professional portrait of {}",
    "close-up photo of {}",
]

TARGETS: List[str] = [
    "Donald Trump",
]

RETAINS: List[str] = [
    "Melania Trump",
    "Hillary Clinton",
    "Barack Obama",
]

NON_TARGETS: List[str] = [
    "Bill Clinton",
    "Joe Biden",
    "President of the United States of America",
    "Husband of Melania Trump",
]

PROBE_PROMPTS: List[str] = [
    "Donald Trump",
    "a photo of Donald Trump",
    "portrait of Donald Trump",
    "President of the United States of America",
    "Husband of Melania Trump",
    "Melania Trump",
    "Barack Obama",
    "Hillary Clinton",
    "Joe Biden",
    "Bill Clinton",
]

# ============================================================
# Block selection
# ============================================================
DUAL_BLOCKS = list(range(0, 19))
SINGLE_BLOCKS = list(range(0, 38))

# Optional custom overrides. Leave empty to use transformer defaults.
DUAL_BLOCK_EDIT_SCALE_MAP: Dict[int, float] = {}
SINGLE_BLOCK_EDIT_SCALE_MAP: Dict[int, float] = {
    # 34: 0.25,
    # 35: 0.25,
    # 36: 0.25,
    # 37: 0.0,
}

USE_DEFAULT_GENERALIZATION_PROFILE = False

# ============================================================
# Hyperparameters
# ============================================================
OUTDIR = "temp_generase_better_generalization"
PLOT_DIR = os.path.join(OUTDIR, "plots")

STRENGTH_TAU = 0.08
ANCHOR_STRENGTH = 0.75
USE_ANCHORS = True
ANCHOR = "a portrait of a person"

REC_H, REC_W = 512, 512
GEN_H, GEN_W = 512, 512

STEPS = 4
GUIDANCE = 3.5
N_IMAGES_PER_PROMPT = 1

START_SEED = 0
END_SEED = 0
SEEDS = [i for i in range(START_SEED, END_SEED + 1)]

# finalize params for better target subspace construction
MAX_RANK_PER_CONCEPT = 4 
PCA_MIN_RATIO = 0.05

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# Utils
# ============================================================
def _save(img: Image.Image, path: str):
    img.save(path)


def _sanitize(s: str, max_len: int = 120) -> str:
    s = s.strip().replace(" ", "_")
    return "".join(c for c in s if c.isalnum() or c in ("_", "-"))[:max_len]


def _make_prompt(x: str, prompt_template: str) -> str:
    return prompt_template.format(x)


def _maybe_clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _safe_mean(vals: List[float]) -> float:
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))


# ============================================================
# Core runner
# ============================================================
@torch.no_grad()
def run_one(
    pipe: FluxPipeline,
    prompt: str,
    *,
    record_target_vt: bool = False,
    record_retain_vt: bool = False,
    record_anchor_vt: bool = False,
    apply_target_proj: bool = False,
    probe_target_score: bool = False,
    record_diag_stats: bool = False,
    diag_stat_concept: Optional[str] = None,
    record_concept: Optional[str] = None,
    seed: int = 0,
    record_mode: bool = False,
):
    g = torch.Generator(device=pipe.device).manual_seed(seed)

    height = REC_H if record_mode else GEN_H
    width = REC_W if record_mode else GEN_W
    output_type = "latent" if record_mode else "pil"

    ja = {
        "record_target_vt": record_target_vt,
        "record_retain_vt": record_retain_vt,
        "record_anchor_vt": record_anchor_vt,
        "record_concept": record_concept,
        "apply_target_proj": apply_target_proj,
        "probe_target_score": probe_target_score,
        "record_diag_stats": record_diag_stats,
        "diag_stat_concept": diag_stat_concept,
        "use_anchors": USE_ANCHORS,
        "use_default_generalization_profile": USE_DEFAULT_GENERALIZATION_PROFILE,
        "dual_block_edit_scale_map": DUAL_BLOCK_EDIT_SCALE_MAP,
        "single_block_edit_scale_map": SINGLE_BLOCK_EDIT_SCALE_MAP,
        "target_block_indices": DUAL_BLOCKS,
        "target_single_block_indices": SINGLE_BLOCKS,
        "strength_tau": STRENGTH_TAU,
        "anchor_strength": ANCHOR_STRENGTH,
        "proj_eps": 1e-8,
        "debug_tokens": False,
    }

    out = pipe(
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

    if record_mode:
        _maybe_clear_cache()
        return None

    return out.images[0]


# ============================================================
# Bank recording
# ============================================================
def record_retain_bank(pipe: FluxPipeline):
    for i, rp in enumerate(RETAINS):
        run_one(
            pipe,
            prompt=rp,
            record_retain_vt=True,
            record_concept=rp,
            seed=1000 + i,
            record_mode=True,
        )


def record_target_bank(pipe: FluxPipeline):
    for i, t in enumerate(TARGETS):
        for j, pt in enumerate(RECORDING_TEMPLATES):
            run_one(
                pipe,
                prompt=pt.format(t),
                record_target_vt=True,
                record_concept=t,
                seed=3000 + 100 * i + j,
                record_mode=True,
            )


def record_anchor_bank(pipe: FluxPipeline):
    if not USE_ANCHORS:
        return

    for i, t in enumerate(TARGETS):
        run_one(
            pipe,
            prompt=ANCHOR,
            record_anchor_vt=True,
            record_concept=t,
            seed=4000 + i,
            record_mode=True,
        )


# ============================================================
# Diagnostics probing
# ============================================================
def probe_diags(pipe: FluxPipeline):
    flux_reset_diag_stats()

    for i, p in enumerate(PROBE_PROMPTS):
        run_one(
            pipe,
            prompt=p,
            apply_target_proj=False,
            probe_target_score=True,
            record_diag_stats=True,
            diag_stat_concept=p,
            seed=9000 + i,
            record_mode=False,
        )


# ============================================================
# Image generation
# ============================================================
def generate_images(pipe: FluxPipeline, items: List[str], templates: List[str], split_name: str):
    for item in items:
        before_path = os.path.join(OUTDIR, split_name, item, "before")
        after_path = os.path.join(OUTDIR, split_name, item, "after")
        os.makedirs(before_path, exist_ok=True)
        os.makedirs(after_path, exist_ok=True)

        for prompt_template in templates:
            p = _make_prompt(item, prompt_template)
            for s in SEEDS:
                file_name = f"{_sanitize(f'{p}_{s}')}.png"

                base_img = run_one(pipe, p, apply_target_proj=False, seed=s, record_mode=False)
                edit_img = run_one(pipe, p, apply_target_proj=True, seed=s, record_mode=False)

                _save(base_img, os.path.join(before_path, file_name))
                _save(edit_img, os.path.join(after_path, file_name))

        print(f"{split_name} :: {item} DONE!")


# ============================================================
# Plot helpers
# ============================================================
def _mean_scalar_curve(blk_map):
    xs = sorted(blk_map.keys())
    ys = [_safe_mean(blk_map[b]) for b in xs]
    return xs, ys


def _mean_token_matrix(blk_map):
    xs = sorted(blk_map.keys())
    if len(xs) == 0:
        return xs, np.zeros((0, 0), dtype=np.float32)

    max_tok = 0
    for b in xs:
        for arr in blk_map[b]:
            max_tok = max(max_tok, len(arr))

    M = np.full((len(xs), max_tok), np.nan, dtype=np.float32)
    for i, b in enumerate(xs):
        rows = blk_map[b]
        if len(rows) == 0:
            continue

        tmp = []
        for arr in rows:
            a = np.full((max_tok,), np.nan, dtype=np.float32)
            a[: len(arr)] = np.array(arr, dtype=np.float32)
            tmp.append(a)

        M[i] = np.nanmean(np.stack(tmp, axis=0), axis=0)

    return xs, M


def plot_diag_curves():
    stats = flux_get_diag_stats()

    for stream_name in ["dual", "single"]:
        for diag_name, bank in stats["scalar"][stream_name].items():
            if len(bank) == 0:
                continue

            plt.figure(figsize=(10, 6))
            for concept, blk_map in bank.items():
                if len(blk_map) == 0:
                    continue
                xs, ys = _mean_scalar_curve(blk_map)
                plt.plot(xs, ys, marker="o", label=concept)

            plt.xlabel("Block index")
            plt.ylabel(diag_name)
            plt.title(f"{stream_name}_{diag_name} (tau={STRENGTH_TAU})")
            plt.grid(True, alpha=0.25)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, f"{stream_name}_{diag_name}.png"), dpi=180)
            plt.close()

    for stream_name in ["dual", "single"]:
        for diag_name, bank in stats["token"][stream_name].items():
            if len(bank) == 0:
                continue

            for concept, blk_map in bank.items():
                xs, M = _mean_token_matrix(blk_map)
                if M.size == 0:
                    continue

                plt.figure(figsize=(10, 5))
                plt.imshow(M, aspect="auto", interpolation="nearest")
                plt.colorbar(label=diag_name)
                plt.xlabel("Token index")
                plt.ylabel("Block row")
                plt.title(f"{stream_name}_{diag_name} :: {concept}")
                plt.yticks(range(len(xs)), xs)
                plt.tight_layout()
                fname = f"{stream_name}_{diag_name}_{_sanitize(concept)}.png"
                plt.savefig(os.path.join(PLOT_DIR, fname), dpi=180)
                plt.close()


def save_diag_summary_txt():
    stats = flux_get_diag_stats()
    outpath = os.path.join(PLOT_DIR, "diag_summary.txt")

    with open(outpath, "w", encoding="utf-8") as f:
        for stream_name in ["dual", "single"]:
            for diag_name, bank in stats["scalar"][stream_name].items():
                f.write(f"\n===== {stream_name}_{diag_name} =====\n")
                for concept, blk_map in bank.items():
                    xs, ys = _mean_scalar_curve(blk_map)
                    f.write(f"\n{concept}\n")
                    for x, y in zip(xs, ys):
                        f.write(f"  block {x:02d} : {y:.6f}\n")


# ============================================================
# Main
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    ).to(device)

    flux_reset_vt_banks(reset_retain=True)
    _maybe_clear_cache()

    # 1) Record banks
    record_retain_bank(pipe)
    record_target_bank(pipe)
    record_anchor_bank(pipe)

    # 2) Finalize improved target subspaces
    flux_finalize_cora_bases(
        max_rank_per_concept=MAX_RANK_PER_CONCEPT,
        pca_min_ratio=PCA_MIN_RATIO,
    )
    _maybe_clear_cache()

    # 3) Probe diagnostics without editing
    probe_diags(pipe)
    plot_diag_curves()
    save_diag_summary_txt()
    _maybe_clear_cache()

    # 4) Generate comparisons
    generate_images(pipe, TARGETS, PROMPT_TEMPLATES, split_name="targets")
    generate_images(pipe, RETAINS, PROMPT_TEMPLATES, split_name="retains")
    generate_images(pipe, NON_TARGETS, PROMPT_TEMPLATES, split_name="non_targets")

    print(f"Done. Results saved to: {OUTDIR}")
    print(f"Plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()