# multi_concept_runner.py
import os
import math
import torch  # type: ignore
from PIL import Image  # type: ignore
from diffusers import FluxPipeline  # type: ignore

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
# TARGETS_AND_RETAINS = {
#     "Donald Trump": ["Melania Trump", "Barack Obama", "Hillary Clinton"],
#     "Christiano Ronaldo": ["Lionel Messi", "Zlatan Ibrahimović", "Sergio Ramos"],
#     "Michael Jackson": ["Taylor Swift", "Ed Sheeran", "Justin Bieber"],
# }
# TARGETS_AND_RETAINS = {
#     "Donald Trump": ["Melania Trump", "Barack Obama", "Hillary Clinton"],
#     "Michael Jackson": ['Ed Sheeran','Taylor Swift','Justin Bieber'],
#     "Christiano Ronaldo": ["Lionel Messi", "Zlatan Ibrahimović", "Sergio Ramos"]
# }

TARGETS_AND_RETAINS = {
    "Dog": ['Sheep']
}

TARGETS = list(TARGETS_AND_RETAINS.keys())
RETAINS_COMBINED = sorted({r for rs in TARGETS_AND_RETAINS.values() for r in rs})
OUTDIR = "temp"
os.makedirs(OUTDIR, exist_ok=True)
N_SAMPLES = 5
BASE_SEED = 0
STEPS = 4
GUIDANCE = 3.5
H, W = 768, 768
DUAL_BLOCKS = list(range(0, 19))
SINGLE_BLOCKS = list(range(0, 38))
PROJ_STRENGTH = 6.0
ANCHOR = "cat"
PROJ_EPS = 1e-8
VT_DEDUP_COS_THR = 0.98
MAX_TARGET_VT_PER_BLOCK = 8
MAX_RETAIN_VT_PER_BLOCK = 16
STRENGTH_TAU = 0.0
STRENGTH_GAMMA = 3.0
SAVE_RECORD_IMAGES = False

def prompt_for_person(name: str) -> str: return f"a photo of {name}"

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
        strength_tau=STRENGTH_TAU,
        strength_gamma=STRENGTH_GAMMA,
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

def main():
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    from diffusers.models.transformers.transformer_flux import flux_reset_vt_banks #type:ignore
    flux_reset_vt_banks(reset_retain=True)
    print(f"[1/4] Recording anchor VT banks for anchor concept {ANCHOR}")
    ja = ja_kwargs_common()
    ja.update(
        record_anchor_vt = True,
        record_target_vt = False,
        record_retain_vt = False,
        apply_target_proj = False,
    )
    p = ANCHOR
    seed = BASE_SEED + 1000
    img = run_one(pipe,p,ja=ja, seed = seed)
    if(SAVE_RECORD_IMAGES):
        save_img(img, os.path.join(OUTDIR, "record_anchor", f"{i:02d}_{ANCHOR}.png"))
    print(f"[2/4] Recording retain VT banks for {len(RETAINS_COMBINED)} retain concepts...")
    for i, retain in enumerate(RETAINS_COMBINED):
        ja = ja_kwargs_common()
        ja.update(
            record_retain_vt=True,
            record_target_vt=False,
            apply_target_proj=False,
        )
        p = retain
        seed = BASE_SEED + 1000 + i
        img = run_one(pipe, p, ja=ja, seed=seed)
        if SAVE_RECORD_IMAGES:
            save_img(img, os.path.join(OUTDIR, "record_retain", f"{i:02d}_{retain}.png"))
    print(f"[3/4] Recording target VT banks for {len(TARGETS)} targets...")
    for i, target in enumerate(TARGETS):
        ja = ja_kwargs_common()
        ja.update(
            record_retain_vt=False,
            record_target_vt=True,
            apply_target_proj=False,
        )
        p = target
        seed = BASE_SEED + 2000 + i
        img = run_one(pipe, p, ja=ja, seed=seed)
        if SAVE_RECORD_IMAGES:
            save_img(img, os.path.join(OUTDIR, "record_target", f"{i:02d}_{target}.png"))
    print("[4/4] Applying multi-concept eraser and sampling images...")
    RETAINS_COMBINED.append("Doggo")
    RETAINS_COMBINED.append("Mans best friend")
    RETAINS_COMBINED.append("Hound")
    RETAINS_COMBINED.append("Canine")
    PROBES = [("target", t) for t in TARGETS] + [("retain", r) for r in RETAINS_COMBINED]
    for kind, concept in PROBES:
        print(f"Erasing {kind}: {concept}")
        p = prompt_for_person(concept)
        base_imgs = []
        for n in range(N_SAMPLES):
            ja = ja_kwargs_common()
            ja.update(
                record_retain_vt=False,
                record_target_vt=False,
                apply_target_proj=False,
            )
            img = run_one(pipe, p, ja=ja, seed=BASE_SEED + 3000 + 100 * hash(concept) % 10000 + n)
            base_imgs.append(img)
        erased_imgs = []
        for n in range(N_SAMPLES):
            ja = ja_kwargs_common()
            ja.update(
                record_retain_vt=False,
                record_target_vt=False,
                apply_target_proj=True,
            )
            img = run_one(pipe, p, ja=ja, seed=BASE_SEED + 4000 + 100 * hash(concept) % 10000 + n)
            erased_imgs.append(img)
        safe_name = concept.replace("/", "_")
        save_grid(base_imgs, os.path.join(OUTDIR, "grids", f"{kind}__{safe_name}__baseline.png"), cols=min(N_SAMPLES, 5))
        save_grid(erased_imgs, os.path.join(OUTDIR, "grids", f"{kind}__{safe_name}__erased.png"), cols=min(N_SAMPLES, 5))
        for n, im in enumerate(base_imgs):
            save_img(im, os.path.join(OUTDIR, "samples", kind, safe_name, f"baseline_{n:02d}.png"))
        for n, im in enumerate(erased_imgs):
            save_img(im, os.path.join(OUTDIR, "samples", kind, safe_name, f"erased_{n:02d}.png"))
    print(f"\nDone. Results in: {OUTDIR}")

if __name__ == "__main__": main()