# run_noise_test.py
import os
import json
from typing import List, Optional, Dict, Any

import torch  # type: ignore
from diffusers import FluxPipeline  # type: ignore

# IMPORTANT: make sure your diffusers points to the modified transformer_flux.py
from diffusers.models.transformers.transformer_flux import (  # type: ignore
    flux_reset_vt_banks,
    flux_get_noise_report,
)

# -------------------------
# Config (edit as needed)
# -------------------------
MODEL_ID = "black-forest-labs/FLUX.1-schnell"

TARGETS: List[str] = ["Donald Trump"]
RETAINS: List[str] = ["Melania Trump", "Hillary Clinton", "Barack Obama"]  # optional for later
ANCHOR = "A generic person"  # not used for noise, kept for consistency

RECORDING_TEMPLATES = [
    "a photo of {}",
    "{} photographed with DSLR",
    "Photo of {} in natural light",
]

# We will record from all blocks for noise analysis:
DUAL_BLOCKS = list(range(0, 19))
SINGLE_BLOCKS = list(range(0, 38))

# Recording seeds
START_SEED = 0
END_SEED = 24
SEEDS = list(range(START_SEED, END_SEED + 1))

# Flux generation settings (noise recording does not need high steps)
H, W = 768, 768
STEPS = 4
GUIDANCE = 3.5
N_IMAGES_PER_PROMPT = 1

OUTDIR = "results/noise_test"
os.makedirs(OUTDIR, exist_ok=True)


@torch.no_grad()
def run_one(
    pipe: FluxPipeline,
    prompt: str,
    *,
    seed: int,
    record_target_vt: bool = False,
    record_retain_vt: bool = False,
    record_concept: Optional[str] = None,
) -> None:
    """
    We don't need to save images for noise tests. Just run pipeline to record VT.
    """
    g = torch.Generator(device=pipe.device).manual_seed(seed)

    ja = {
        # recording flags
        "record_target_vt": record_target_vt,
        "record_retain_vt": record_retain_vt,
        "record_concept": record_concept,
        # which blocks to record from
        "target_block_indices": DUAL_BLOCKS,
        "target_single_block_indices": SINGLE_BLOCKS,
        # token pooling cutoff (passed from pipeline normally; but safe to leave None)
        # "detector_token_end": None,
        # dedup behavior
        "vt_dedup_cos_thr": 0.995,
        "max_target_vt_per_block": 512,  # keep more for noise experiments
        "max_retain_vt_per_block": 512,
    }

    _ = pipe(
        prompt=prompt,
        height=H,
        width=W,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        num_images_per_prompt=N_IMAGES_PER_PROMPT,
        generator=g,
        joint_attention_kwargs=ja,
        output_type="latent",   # faster; we don't need decoded images
    )


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)

    # Reset banks
    flux_reset_vt_banks(reset_retain=True)

    # (Optional) record retains too (not required for noise, but good to keep for later)
    for i, r in enumerate(RETAINS):
        run_one(pipe, prompt=r, seed=1000 + i, record_retain_vt=True)

    # Record target concepts across templates and seeds
    for t in TARGETS:
        for pt in RECORDING_TEMPLATES:
            prompt = pt.format(t)
            for s in SEEDS:
                run_one(pipe, prompt=prompt, seed=s, record_target_vt=True, record_concept=t)

    # Produce report
    report: Dict[str, Any] = flux_get_noise_report(
        concept=TARGETS[0],
        n_boot=30,
        energy=0.90,     # choose rank by 90% explained sigma^2
        rank=None,       # or set an int like 16 for fixed rank
        seed0=123,
        token_idx=1,     # must match how you extract the dir
    )

    out_path = os.path.join(OUTDIR, "noise_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n===== NOISE REPORT =====")
    print("Dual stage stability (mean,std,n_dirs):", report["dual_stage"])
    print("Single stage stability (mean,std,n_dirs):", report["single_stage"])
    print("Saved:", out_path)

    # Print worst blocks (lowest mean similarity)
    dual_blocks_sorted = sorted(report["dual_blocks"].items(), key=lambda kv: kv[1][0])
    single_blocks_sorted = sorted(report["single_blocks"].items(), key=lambda kv: kv[1][0])
    print("\nWorst dual blocks (blk -> (mean,std,n)):", dual_blocks_sorted[:5])
    print("Worst single blocks (blk -> (mean,std,n)):", single_blocks_sorted[:5])


if __name__ == "__main__":
    main()