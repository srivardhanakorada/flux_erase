# driver_flux_cora_union_anchor.py
import os
from typing import List, Optional
import torch #type:ignore
from PIL import Image #type:ignore
from diffusers import FluxPipeline
from diffusers.models.transformers.transformer_flux import flux_reset_vt_banks, flux_finalize_cora_bases #type:ignore

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
PROMPT_TEMPLATES = [
    "a photo of {}",
    "{}",
    "high quality photo of {}"
]
TARGETS: List[str] = [
    "Donald Trump",
    "Hugh Jackman",
    "Michael Jackson"
]
RETAIN_PROMPTS: List[str] = [
    "Hillary Clinton",
    "Barack Obama",
    "Melania Trump",
    "Angelina Jolie",
    "Brad Pitt",
    "Tom Cruise",
    "Ed Sheeran",
    "Taylor Swift",
    "Justin Bieber"
]
NON_TARGETS: List[str] = [
    "President of the United States of America",
    "King of Pop",
    "Bill Clinton",
    "Dog",
    "Lemon"
]
ANCHOR_PROMPT = "a generic person"
OUTDIR = "multiple_celeb"
os.makedirs(OUTDIR, exist_ok=True)
DUAL_BLOCKS = list(range(0, 19))
SINGLE_BLOCKS = list(range(0, 38))
STRENGTH_TAU = 0.1
STRENGTH_GAMMA = 2.5
ANCHOR_STRENGTH = 1.0

USE_ANCHORS = True
H, W = 768, 768
STEPS = 4
GUIDANCE = 3.5
N_IMAGES_PER_PROMPT = 1
BASE_SEED = 0
SEEDS = [BASE_SEED + i for i in range(5)]


def _save(img: Image.Image, path: str): img.save(path)
def _sanitize(s: str, max_len: int = 80) -> str:
    s = s.strip().replace(" ", "_")
    return "".join(c for c in s if c.isalnum() or c in ("_", "-"))[:max_len]
def _make_grid(rows: List[List[Image.Image]], pad: int = 6, bg=(0, 0, 0)) -> Image.Image:
    assert len(rows) > 0
    n_rows = len(rows)
    n_cols = len(rows[0])
    for r in rows: assert len(r) == n_cols, "All rows must have same number of columns"
    cell_w, cell_h = rows[0][0].size
    grid_w = n_cols * cell_w + (n_cols - 1) * pad
    grid_h = n_rows * cell_h + (n_rows - 1) * pad
    canvas = Image.new("RGB", (grid_w, grid_h), bg)
    for ri in range(n_rows):
        for ci in range(n_cols):
            x = ci * (cell_w + pad)
            y = ri * (cell_h + pad)
            canvas.paste(rows[ri][ci].convert("RGB"), (x, y))
    return canvas
@torch.no_grad()
def _run_one(
    pipe: FluxPipeline,
    prompt: str,
    *,
    record_target_vt: bool = False,
    record_retain_vt: bool = False,
    record_anchor_once: bool = False,
    apply_target_proj: bool = False,
    record_concept: Optional[str] = None,
    seed: int = 0,
) -> Image.Image:
    g = torch.Generator(device=pipe.device).manual_seed(seed)
    ja = {
        "record_target_vt": record_target_vt,
        "record_retain_vt": record_retain_vt,
        "record_anchor_once": record_anchor_once,
        "record_concept": record_concept,  # required only for record_target_vt
        "apply_target_proj": apply_target_proj,
        "use_anchors": USE_ANCHORS,
        "target_block_indices": DUAL_BLOCKS,
        "target_single_block_indices": SINGLE_BLOCKS,
        "strength_tau": STRENGTH_TAU,
        "strength_gamma": STRENGTH_GAMMA,
        "anchor_strength": ANCHOR_STRENGTH,
        "proj_eps": 1e-8,
        "debug_tokens" : True,
    }
    out = pipe(
        prompt=prompt,
        height=H,
        width=W,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        num_images_per_prompt=N_IMAGES_PER_PROMPT,
        generator=g,
        joint_attention_kwargs=ja,
    )
    return out.images[0]
def _photo_prompt(x: str) -> str: return f"a photo of {x}"
def _eval_bucket_grids(pipe: FluxPipeline, bucket_name: str, items: List[str]):
    for item in items:
        p = _photo_prompt(item)
        base_imgs: List[Image.Image] = []
        edit_imgs: List[Image.Image] = []
        for s in SEEDS:
            base_imgs.append(_run_one(pipe, p, apply_target_proj=False, seed=s))
            edit_imgs.append(_run_one(pipe, p, apply_target_proj=True, seed=s))
        grid = _make_grid([base_imgs, edit_imgs], pad=6, bg=(0, 0, 0))
        out_path = os.path.join(OUTDIR, f"{bucket_name}_{_sanitize(item)}.png")
        _save(grid, out_path)
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    flux_reset_vt_banks(reset_retain=True)
    for i, rp in enumerate(RETAIN_PROMPTS):
        _run_one(
            pipe,
            prompt=rp,
            record_retain_vt=True,
            seed=1000 + i,
        )
    for i, t in enumerate(TARGETS):
        for pt in PROMPT_TEMPLATES:
            prompt = pt.format(t)
            _run_one(
                pipe,
                prompt=prompt,
                record_target_vt=True,
                record_concept=t,
                seed=2000 + i,
            )
    _run_one(
        pipe,
        prompt=ANCHOR_PROMPT,
        record_anchor_once=True,
        seed=3000,
    )
    flux_finalize_cora_bases(retain_top_k=3, eps=1e-8)
    _eval_bucket_grids(pipe, "target", TARGETS)
    _eval_bucket_grids(pipe, "retain", RETAIN_PROMPTS)
    _eval_bucket_grids(pipe, "nontarget", NON_TARGETS)
    print(f"Done. Grids saved to: {OUTDIR}")
    print(f"Seeds used: {SEEDS}")
if __name__ == "__main__": main()
