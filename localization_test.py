import os
import math
import torch  # type: ignore
from PIL import Image  # type: ignore
from diffusers import FluxPipeline  # type: ignore

MODEL_ID = "black-forest-labs/FLUX.1-schnell"

TARGETS_AND_RETAINS = {
    "Donald Trump": ["Melania Trump", "Barack Obama", "Hillary Clinton"],
    "Christiano Ronaldo": ["Lionel Messi", "Zlatan Ibrahimović", "Sergio Ramos"],
    "Michael Jackson": ["Taylor Swift", "Ed Sheeran", "Justin Bieber"],
}
TARGETS = list(TARGETS_AND_RETAINS.keys())
RETAINS_COMBINED = sorted({r for rs in TARGETS_AND_RETAINS.values() for r in rs})

OUTDIR = "three_people_localization_cmp"
os.makedirs(OUTDIR, exist_ok=True)

N_SAMPLES = 5
BASE_SEED = 0
STEPS = 4
GUIDANCE = 3.5
H, W = 768, 768

DUAL_BLOCKS = list(range(0, 19))
SINGLE_BLOCKS = list(range(0, 38))

PROJ_STRENGTH = 1.5
PROJ_EPS = 1e-8
VT_DEDUP_COS_THR = 0.98
MAX_TARGET_VT_PER_BLOCK = 8
MAX_RETAIN_VT_PER_BLOCK = 16

# adaptive gating hyperparams (only used in adaptive condition)
STRENGTH_TAU = 0.0
STRENGTH_GAMMA = 2.0

SAVE_RECORD_IMAGES = False

THREE_PERSON_PROMPTS = [
    ("trump_obama__studio", "a photo of Donald Trump, Bill Clinton and Barack Obama standing together, studio lighting"),
    ("trump_melania__event", "a photo of Donald Trump, Bill Clinton and Melania Trump at a formal event, red carpet, flash photography"),
    ("ronaldo_messi__stadium", "a photo of Christiano Ronaldo, Bill Clinton and Lionel Messi on a football field, stadium lights, crowd in background"),
    ("ronaldo_ramos__training", "a photo of Christiano Ronaldo, Bill Clinton and Sergio Ramos during training, sports photography"),
    ("mj_tswift__stage", "a photo of Michael Jackson, Bill Clinton and Taylor Swift performing on stage, concert lighting"),
    ("mj_bieber__street", "a photo of Michael Jackson, Bill Clinton and Justin Bieber walking on a city street, candid photo"),
]

CONTROL_PROMPTS = [
    ("retain_only__messi_ramos", "a photo of Lionel Messi, Bill Clinton and Sergio Ramos standing together, studio lighting"),
    ("retain_only__swift_bieber", "a photo of Taylor Swift, Bill Clinton and Justin Bieber standing together, studio lighting"),
    ("retain_only__obama_clinton", "a photo of Barack Obama, Bill Clinton and Hillary Clinton standing together, studio lighting"),
]


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

def ja_kwargs_common():
    return dict(
        target_block_indices=DUAL_BLOCKS,
        target_single_block_indices=SINGLE_BLOCKS,
        proj_strength=PROJ_STRENGTH,
        proj_eps=PROJ_EPS,
        vt_dedup_cos_thr=VT_DEDUP_COS_THR,
        max_target_vt_per_block=MAX_TARGET_VT_PER_BLOCK,
        max_retain_vt_per_block=MAX_RETAIN_VT_PER_BLOCK,
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


def snapshot_concept_embeds():
    # returns list of tensors (cloned to be safe)
    from diffusers.models.transformers.transformer_flux import flux_get_concept_embeds  # type: ignore
    C = flux_get_concept_embeds() or []
    # clone so we can restore even if something mutates
    return [c.detach().clone() for c in C]

def restore_concept_embeds(C_snap):
    from diffusers.models.transformers.transformer_flux import flux_reset_concept_embeds, flux_add_concept_embed  # type: ignore
    flux_reset_concept_embeds()
    for c in C_snap:
        flux_add_concept_embed(c)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=False)

    from diffusers.models.transformers.transformer_flux import flux_reset_vt_banks  # type: ignore
    flux_reset_vt_banks(reset_retain=True)

    # -------------------------------------------------
    # 1) Record retain banks (combined retains)
    # -------------------------------------------------
    print(f"[1/3] Recording retain VT banks for {len(RETAINS_COMBINED)} retains...")
    for i, retain in enumerate(RETAINS_COMBINED):
        ja = ja_kwargs_common()
        ja.update(record_retain_vt=True, record_target_vt=False, apply_target_proj=False)
        img = run_one(pipe, retain, ja=ja, seed=BASE_SEED + 1000 + i)
        if SAVE_RECORD_IMAGES:
            save_img(img, os.path.join(OUTDIR, "record_retain", f"{i:02d}_{retain}.png"))

    # -------------------------------------------------
    # 2) Record target banks (also APPENDS concept embeds)
    # -------------------------------------------------
    print(f"[2/3] Recording target VT banks for {len(TARGETS)} targets...")
    for i, target in enumerate(TARGETS):
        ja = ja_kwargs_common()
        ja.update(record_retain_vt=False, record_target_vt=True, apply_target_proj=False)
        img = run_one(pipe, target, ja=ja, seed=BASE_SEED + 2000 + i)
        if SAVE_RECORD_IMAGES:
            save_img(img, os.path.join(OUTDIR, "record_target", f"{i:02d}_{target}.png"))

    # snapshot concept embeds once (used for adaptive gating)
    C_snap = snapshot_concept_embeds()

    # -------------------------------------------------
    # 3) Probes: baseline vs adaptive vs uniform
    #    Uniform = same VT banks, but concept-embeds cleared so proj_strength_tokens is NOT created
    # -------------------------------------------------
    print("[3/3] Probing three-person prompts: baseline vs adaptive vs uniform...")

    ALL = [("three_person",) + x for x in THREE_PERSON_PROMPTS] + [("control",) + x for x in CONTROL_PROMPTS]

    for kind, tag, prompt in ALL:
        tag_seed_base = 100 * (abs(hash(tag)) % 10000)

        # ----- baseline (no projection)
        baseline = []
        for n in range(N_SAMPLES):
            ja = ja_kwargs_common()
            ja.update(record_retain_vt=False, record_target_vt=False, apply_target_proj=False)
            img = run_one(pipe, prompt, ja=ja, seed=BASE_SEED + 3000 + tag_seed_base + n)
            baseline.append(img)

        # ----- adaptive (projection + token gating computed from concept embeds)
        restore_concept_embeds(C_snap)
        adaptive = []
        for n in range(N_SAMPLES):
            ja = ja_kwargs_common()
            ja.update(
                record_retain_vt=False,
                record_target_vt=False,
                apply_target_proj=True,
                strength_tau=float(STRENGTH_TAU),
                strength_gamma=float(STRENGTH_GAMMA),
            )
            img = run_one(pipe, prompt, ja=ja, seed=BASE_SEED + 4000 + tag_seed_base + n)
            adaptive.append(img)

        # ----- uniform (projection ON but concept embeds cleared => no proj_strength_tokens)
        from diffusers.models.transformers.transformer_flux import flux_reset_concept_embeds  # type: ignore
        flux_reset_concept_embeds()
        uniform = []
        for n in range(N_SAMPLES):
            ja = ja_kwargs_common()
            ja.update(
                record_retain_vt=False,
                record_target_vt=False,
                apply_target_proj=True,
                # NOTE: do NOT pass strength_tau/gamma here
            )
            img = run_one(pipe, prompt, ja=ja, seed=BASE_SEED + 5000 + tag_seed_base + n)
            uniform.append(img)

        # save grids
        base_dir = os.path.join(OUTDIR, "grids", kind, tag)
        save_grid(baseline, os.path.join(base_dir, "A_baseline.png"), cols=min(N_SAMPLES, 5))
        save_grid(adaptive, os.path.join(base_dir, "B_adaptive.png"), cols=min(N_SAMPLES, 5))
        save_grid(uniform,  os.path.join(base_dir, "C_uniform.png"),  cols=min(N_SAMPLES, 5))

        print(f"DONE {kind}: {tag}")

    print(f"\nDone. Results in: {OUTDIR}")
    print("Open: grids/<kind>/<tag>/{A_baseline,B_adaptive,C_uniform}.png")


if __name__ == "__main__":
    main()