import os
from typing import List, Optional

import torch  # type: ignore
from PIL import Image  # type: ignore
from diffusers import FluxPipeline
from diffusers.models.transformers.transformer_flux import (  # type: ignore
    flux_reset_vt_banks,
    flux_finalize_cora_bases,
)

MODEL_ID = "black-forest-labs/FLUX.1-schnell"

# ------------------------------------------------------------
# Erasure setup
# ------------------------------------------------------------
TARGETS: List[str] = [
    "Barack Obama",
]

# Use the "other guys" as preserved retain concepts
RETAINS: List[str] = [
    "Bill Clinton",
    "Joe Biden",
    "Donald Trump",
]

PERSON_BANK = [
    "a portrait of a person",
    "a portrait of a man",
    "a portrait of a woman",
    "a middle-aged man",
    "a middle-aged woman",
]

RECORDING_TEMPLATES = [
    "a photo of {}",
    "{}, studio portrait, sharp focus",
]

# ------------------------------------------------------------
# Localization prompts
# ------------------------------------------------------------
LOCALIZATION_PROMPTS = [
    "a group photo of Barack Obama and Bill Clinton",
    "a group photo of Barack Obama, Bill Clinton, and Joe Biden",
    "a group photo of Barack Obama, Bill Clinton, Joe Biden, and Donald Trump",
]

# ------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------
DUAL_BLOCKS = list(range(0, 19))
SINGLE_BLOCKS = list(range(0, 38))

OUTDIR = "display/localization"

STRENGTH_TAU = 0.1
STRENGTH_GAMMA = 1.5
ANCHOR_STRENGTH = 1.0
USE_ANCHORS = False
ANCHOR = "a portrait of a person"

PERSON_TOP_K = 2
RETAIN_TOP_K = 4

REC_H, REC_W = 768, 768
GEN_H, GEN_W = 768, 768

STEPS = 4
GUIDANCE = 3.5
N_IMAGES_PER_PROMPT = 1

START_SEED = 0
END_SEED = 9
SEEDS = [i for i in range(START_SEED, END_SEED + 1)]

os.makedirs(OUTDIR, exist_ok=True)


def _save(img: Image.Image, path: str):
    img.save(path)


def _sanitize(s: str, max_len: int = 180) -> str:
    s = s.strip().replace(" ", "_")
    return "".join(c for c in s if c.isalnum() or c in ("_", "-"))[:max_len]


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


def generate_localization_images(pipe: FluxPipeline, prompts: List[str]):
    for idx, prompt in enumerate(prompts, start=1):
        case_name = f"case_{idx}"
        before_path = os.path.join(OUTDIR, case_name, "before")
        after_path = os.path.join(OUTDIR, case_name, "after")
        os.makedirs(before_path, exist_ok=True)
        os.makedirs(after_path, exist_ok=True)

        for seed in SEEDS:
            file_name = f"{_sanitize(prompt)}_{seed}.png"

            base_img = run_one(
                pipe,
                prompt,
                apply_target_proj=False,
                seed=seed,
                record_mode=False,
            )
            edit_img = run_one(
                pipe,
                prompt,
                apply_target_proj=True,
                seed=seed,
                record_mode=False,
            )

            _save(base_img, os.path.join(before_path, file_name))
            _save(edit_img, os.path.join(after_path, file_name))

        with open(os.path.join(OUTDIR, case_name, "prompt.txt"), "w") as f:
            f.write(prompt + "\n")

        print(f"{case_name} DONE :: {prompt}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    ).to(device)

    # Reset banks
    flux_reset_vt_banks(reset_retain=True)
    _maybe_clear_cache()

    # Record retains
    for i, rp in enumerate(RETAINS):
        run_one(
            pipe,
            prompt=rp,
            record_retain_vt=True,
            seed=1000 + i,
            record_mode=True,
        )

    # Record generic person bank
    for i, pp in enumerate(PERSON_BANK):
        run_one(
            pipe,
            prompt=pp,
            record_person_vt=True,
            seed=2000 + i,
            record_mode=True,
        )

    # Record target concept with a couple of templates
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

    if USE_ANCHORS:
        run_one(
            pipe,
            prompt=ANCHOR,
            record_anchor_once=True,
            seed=4000,
            record_mode=True,
        )

    # Finalize bases
    flux_finalize_cora_bases(
        retain_top_k=RETAIN_TOP_K,
        person_top_k=PERSON_TOP_K,
    )
    _maybe_clear_cache()

    # Generate localization test prompts only
    generate_localization_images(pipe, LOCALIZATION_PROMPTS)

    print(f"Done. Results saved to: {OUTDIR}")


if __name__ == "__main__":
    main()