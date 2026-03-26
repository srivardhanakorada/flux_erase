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

PROMPT_TEMPLATES = [
    "a photo of {}",
]

RECORDING_TEMPLATES = [
    "a photo of {}",
    "{}, studio portrait, sharp focus",
]

TARGETS: List[str] = [
    "Donald Trump",
    "Hugh Jackman",
    "Michael Jackson",
    "Brad Pitt",
    "Bill Clinton",
    "Cristiano Ronaldo",
    "Novak Djokovic",
    "Rihanna",
    "Vladimir Putin",
    "Lewis Hamilton",
    "Kevin Durrant",
    "Stephen Curry",
    "George Bush",
    "Steve Jobs",
    "Bernie Sanders",
]

RETAINS = [
    "Hillary Clinton",
    "Taylor Swift",
    "Kanye West",
    "Snoop Dogg",
    "Barack Obama",
    "Lionel Messi",
    "Serena Williams",
    "Beyonce",
    "Kim Jong Un",
    "Queen Elizabeth",
    "Lebron James",
    "Kobe Bryant",
    "Ronald Reagan",
    "Justin Bieber",
    "Marilyn Monroe",
]

PERSON_BANK = [
    "a portrait of a person",
    "a portrait of a man",
    "a portrait of a woman",
    "a middle-aged man",
    "a middle-aged woman",
]

DUAL_BLOCKS = list(range(0, 19))
SINGLE_BLOCKS = list(range(0, 38))

SCALABILITY_SIZES = [1,2,5,10,15]

OUTDIR = "results/scalability/multi_celeb_clean"

STRENGTH_TAU = 0.1
STRENGTH_GAMMA = 1.5
ANCHOR_STRENGTH = 1.0
USE_ANCHORS = False
ANCHOR = "a portrait of a person"

PERSON_TOP_K = 8
RETAIN_TOP_K = 8

REC_H, REC_W = 768, 768
GEN_H, GEN_W = 768, 768

STEPS = 4
GUIDANCE = 3.5
N_IMAGES_PER_PROMPT = 1

START_SEED = 0
END_SEED = 99
SEEDS = [i for i in range(START_SEED, END_SEED + 1)]

os.makedirs(OUTDIR, exist_ok=True)


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


def generate_images(
    pipe: FluxPipeline,
    items: List[str],
    templates: List[str],
    split_name: str,
    run_outdir: str,
):
    for item in items:
        after_path = os.path.join(run_outdir, split_name, item, "after")
        os.makedirs(after_path, exist_ok=True)
        for prompt_template in templates:
            p = _make_prompt(item, prompt_template)
            for s in SEEDS:
                file_name = f"{_sanitize(f'{p}_{s}')}.png"
                edit_img = run_one(
                    pipe,
                    p,
                    apply_target_proj=True,
                    seed=s,
                    record_mode=False,
                )
                _save(edit_img, os.path.join(after_path, file_name))
        print(f"{split_name} :: {item} DONE!")


def run_scalability_case(pipe: FluxPipeline, size: int):
    targets = TARGETS[:size]
    retains = RETAINS[:size]

    run_outdir = os.path.join(OUTDIR, f"multi_celeb_{size}")
    os.makedirs(run_outdir, exist_ok=True)

    print("=" * 80)
    print(f"Running scalability case: k = {size}")
    print(f"Targets ({len(targets)}): {targets}")
    print(f"Retains ({len(retains)}): {retains}")
    print("=" * 80)

    flux_reset_vt_banks(reset_retain=True)
    _maybe_clear_cache()

    for i, rp in enumerate(retains):
        run_one(
            pipe,
            prompt=rp,
            record_retain_vt=True,
            seed=1000 + i,
            record_mode=True,
        )

    for i, pp in enumerate(PERSON_BANK):
        run_one(
            pipe,
            prompt=pp,
            record_person_vt=True,
            seed=2000 + i,
            record_mode=True,
        )

    for i, t in enumerate(targets):
        for j, pt in enumerate(RECORDING_TEMPLATES):
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
            prompt=ANCHOR,
            record_anchor_once=True,
            seed=4000,
            record_mode=True,
        )

    flux_finalize_cora_bases(
        retain_top_k=RETAIN_TOP_K,
        person_top_k=PERSON_TOP_K,
    )
    _maybe_clear_cache()

    generate_images(
        pipe,
        targets,
        PROMPT_TEMPLATES,
        split_name="targets",
        run_outdir=run_outdir,
    )
    generate_images(
        pipe,
        retains,
        PROMPT_TEMPLATES,
        split_name="retains",
        run_outdir=run_outdir,
    )
    with open(os.path.join(run_outdir, "concepts_used.txt"), "w") as f:
        f.write(f"size = {size}\n\n")
        f.write("TARGETS\n")
        for x in targets:
            f.write(f"{x}\n")
        f.write("\nRETAINS\n")
        for x in retains:
            f.write(f"{x}\n")
    print(f"Done for k={size}. Results saved to: {run_outdir}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    ).to(device)

    for size in SCALABILITY_SIZES:
        run_scalability_case(pipe, size)

    print(f"All scalability runs finished. Results saved under: {OUTDIR}")


if __name__ == "__main__":
    main()
