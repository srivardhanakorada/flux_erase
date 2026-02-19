# color_word_eraser_driver.py
# Erase the *word-concept* "red" (record VT from prompt="red") with configurable vt_phrase_end
# and then probe: "red car", "red apple", "red tomato", plus controls.
#
# Assumptions:
#  - Your patched transformer supports joint_attention_kwargs["vt_phrase_end"] (int)
#    used ONLY during record_target_vt / record_retain_vt stages.
#  - You already have proj_token_end logic in pipeline_flux.py (EOS-based) or you can ignore it.
#  - Batch size 1.

import os
import math
import torch  # type: ignore
from PIL import Image  # type: ignore
from diffusers import FluxPipeline  # type: ignore

# -----------------------------
# Config
# -----------------------------
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
OUTDIR = "color_red_eraser_out"
os.makedirs(OUTDIR, exist_ok=True)

# Sampling
N_SAMPLES = 5
BASE_SEED = 0
STEPS = 4
GUIDANCE = 3.5
H, W = 768, 768

# Projection / blocks
DUAL_BLOCKS = list(range(0, 19))
SINGLE_BLOCKS = list(range(0, 38))

PROJ_STRENGTH = 6.0      # start modest for "red"; increase if too weak
PROJ_EPS = 1e-8
VT_DEDUP_COS_THR = 0.98
MAX_TARGET_VT_PER_BLOCK = 8
MAX_RETAIN_VT_PER_BLOCK = 16

# Adaptive token gating knobs (can keep, even if you think it's not needed)
STRENGTH_TAU = 0.0
STRENGTH_GAMMA = 1.0

SAVE_RECORD_IMAGES = False

# -----------------------------
# Experiment definition
# -----------------------------
# We are "erasing" the concept RED, but preserving other colors (retain) so the model
# doesn't just lose "colorfulness".
TARGET_CONCEPTS = ["red"]  # record target VT using only this phrase
RETAIN_CONCEPTS = ["blue", "green", "yellow", "black", "white"]  # record retain VT too

def span_end_for(concept: str) -> int:
    # As requested: span_end = number of words
    # (If concept is "dark red", span_end=2)
    return max(1, len(concept.strip().split()))

# Probes: baseline vs erased.
PROBES = [
    ("probe_red",   "a photo of a red car, studio lighting"),
    ("probe_red",   "a photo of a red apple on a table, studio lighting"),
    ("probe_red",   "a photo of a red tomato on a table, studio lighting"),
    ("probe_red",   "a photo of a shiny red sports car, high detail"),
    ("probe_red",   "a photo of a red shirt on a hanger, studio lighting"),

    # Controls: non-red
    ("control",     "a photo of a blue car, studio lighting"),
    ("control",     "a photo of a green apple on a table, studio lighting"),
    ("control",     "a photo of a yellow tomato on a table, studio lighting"),  # silly but good stress
    ("control",     "a photo of a black car, studio lighting"),
]

# Optional: show what happens when user asks for "car" without red
PROBES += [
    ("control", "a photo of a car, studio lighting"),
    ("control", "a photo of an apple on a table, studio lighting"),
    ("control", "a photo of a tomato on a table, studio lighting"),
]

# -----------------------------
# Helpers
# -----------------------------
def make_generator(seed: int, device: torch.device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g

def save_img(img: Image.Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

def save_grid(imgs, path: str, cols: int):
    assert len(imgs) > 0
    w, h = imgs[0].size
    rows = math.ceil(len(imgs) / cols)
    grid = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
    for i, im in enumerate(imgs):
        r = i // cols
        c = i % cols
        grid.paste(im, (c * w, r * h))
    save_img(grid, path)

def safe_name(s: str) -> str:
    return (
        s.replace(" ", "_")
         .replace("/", "_")
         .replace(":", "_")
         .replace(",", "")
         .replace("__", "_")
    )

def ja_kwargs_common():
    return dict(
        target_block_indices=DUAL_BLOCKS,
        target_single_block_indices=SINGLE_BLOCKS,
        proj_strength=float(PROJ_STRENGTH),
        proj_eps=float(PROJ_EPS),
        strength_tau=float(STRENGTH_TAU),
        strength_gamma=float(STRENGTH_GAMMA),
        vt_dedup_cos_thr=float(VT_DEDUP_COS_THR),
        max_target_vt_per_block=int(MAX_TARGET_VT_PER_BLOCK),
        max_retain_vt_per_block=int(MAX_RETAIN_VT_PER_BLOCK),
    )

@torch.no_grad()
def run_one(pipe: FluxPipeline, prompt: str, *, ja: dict, seed: int):
    device = pipe._execution_device
    g = make_generator(seed, device=device)
    out = pipe(
        prompt=prompt,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        height=H,
        width=W,
        num_images_per_prompt=1,
        generator=g,
        joint_attention_kwargs=ja,
        disable_clip=False,
    )
    return out.images[0]

# -----------------------------
# Main
# -----------------------------
def main():
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    # Reset banks (your transformer file already exposes this)
    from diffusers.models.transformers.transformer_flux import flux_reset_vt_banks  # type: ignore
    flux_reset_vt_banks(reset_retain=True)

    # ---------------------------------------------------
    # 1) Record RETAIN concepts (colors to keep)
    # ---------------------------------------------------
    print(f"[1/3] Recording RETAIN VT for: {RETAIN_CONCEPTS}")
    for i, concept in enumerate(RETAIN_CONCEPTS):
        ja = ja_kwargs_common()
        ja.update(
            record_retain_vt=True,
            record_target_vt=False,
            apply_target_proj=False,
            vt_phrase_end=span_end_for(concept),  # <-- requested
        )
        seed = BASE_SEED + 1000 + i
        img = run_one(pipe, concept, ja=ja, seed=seed)  # record using only the phrase
        if SAVE_RECORD_IMAGES:
            save_img(img, os.path.join(OUTDIR, "record_retain", f"{i:02d}_{safe_name(concept)}.png"))

    # ---------------------------------------------------
    # 2) Record TARGET concepts (what to erase)
    # ---------------------------------------------------
    print(f"[2/3] Recording TARGET VT for: {TARGET_CONCEPTS}")
    for i, concept in enumerate(TARGET_CONCEPTS):
        ja = ja_kwargs_common()
        ja.update(
            record_retain_vt=False,
            record_target_vt=True,
            apply_target_proj=False,
            vt_phrase_end=span_end_for(concept),  # <-- requested
        )
        seed = BASE_SEED + 2000 + i
        img = run_one(pipe, concept, ja=ja, seed=seed)  # record using only the phrase
        if SAVE_RECORD_IMAGES:
            save_img(img, os.path.join(OUTDIR, "record_target", f"{i:02d}_{safe_name(concept)}.png"))

    # ---------------------------------------------------
    # 3) Probe prompts: baseline vs erased
    # ---------------------------------------------------
    print("[3/3] Probing prompts (baseline vs erased)...")

    for idx, (kind, prompt) in enumerate(PROBES):
        tag = f"{idx:02d}__{safe_name(kind)}__{safe_name(prompt)[:80]}"

        # baseline
        base_imgs = []
        for n in range(N_SAMPLES):
            ja = ja_kwargs_common()
            ja.update(
                record_retain_vt=False,
                record_target_vt=False,
                apply_target_proj=False,
            )
            seed = BASE_SEED + 3000 + 100 * idx + n
            base_imgs.append(run_one(pipe, prompt, ja=ja, seed=seed))

        # erased
        erased_imgs = []
        for n in range(N_SAMPLES):
            ja = ja_kwargs_common()
            ja.update(
                record_retain_vt=False,
                record_target_vt=False,
                apply_target_proj=True,
            )
            seed = BASE_SEED + 4000 + 100 * idx + n
            erased_imgs.append(run_one(pipe, prompt, ja=ja, seed=seed))

        # Save grids + individuals
        save_grid(base_imgs, os.path.join(OUTDIR, "grids", kind, f"{tag}__baseline.png"), cols=min(N_SAMPLES, 5))
        save_grid(erased_imgs, os.path.join(OUTDIR, "grids", kind, f"{tag}__erased.png"), cols=min(N_SAMPLES, 5))

        for n, im in enumerate(base_imgs):
            save_img(im, os.path.join(OUTDIR, "samples", kind, tag, f"baseline_{n:02d}.png"))
        for n, im in enumerate(erased_imgs):
            save_img(im, os.path.join(OUTDIR, "samples", kind, tag, f"erased_{n:02d}.png"))

        print(f"  ✓ {kind}: {prompt}")

    print(f"\nDone. Results in: {OUTDIR}")
    print("Check OUTDIR/grids/probe_red/*__baseline.png vs *__erased.png")
    print("If 'red' persists, increase PROJ_STRENGTH (e.g., 4→6). If images collapse, reduce PROJ_STRENGTH.")

if __name__ == "__main__":
    main()