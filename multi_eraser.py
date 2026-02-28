import os
import json
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import FluxPipeline
from diffusers.models.transformers.transformer_flux import (
    flux_reset_vt_banks,
    flux_finalize_cora_bases,
)

# -----------------------------
# Config
# -----------------------------
MODEL_ID = "black-forest-labs/FLUX.1-schnell"  # or dev
DEVICE = "cuda"
DTYPE = torch.bfloat16

H, W = 768, 768
STEPS = 4
GUIDANCE = 3.5

BLOCKS_DUAL = list(range(0, 19))
BLOCKS_SINGLE = list(range(0, 38))

PROJ_EPS = 1e-8
VT_DEDUP_COS_THR = 0.98
MAX_TARGET_VT_PER_BLOCK = 8
MAX_RETAIN_VT_PER_BLOCK = 16
MAX_ANCHOR_VT_PER_BLOCK = 8

RETAIN_TOP_K = 8
N_RECORD_RETAIN = 8
N_RECORD_TARGET = 4
N_RECORD_ANCHOR = 4

STRENGTH_TAU = 0.15
STRENGTH_GAMMA = 2.5
ANCHOR_STRENGTH = 2.0   # won’t affect union until patch, but fine to set

N_GEN = 5
BASE_SEED = 0

OUTDIR = "temp"
OUT_SINGLES = os.path.join(OUTDIR, "singles")
OUT_GRIDS = os.path.join(OUTDIR, "grids")
os.makedirs(OUT_SINGLES, exist_ok=True)
os.makedirs(OUT_GRIDS, exist_ok=True)

# -----------------------------
# Concepts (multi-target)
# -----------------------------
TARGET_CONCEPTS = [
    "Donald Trump",
    "Christiano Ronaldo", 
    "Michael Jackson",
]

# Retains = union across all experiments
RETAIN_CONCEPTS = [
    "Melania Trump", "Barack Obama", "Hillary Clinton",
    "Lionel Messi", "Zlatan Ibrahimović", "Sergio Ramos",
    "Taylor Swift", "Ed Sheeran", "Justin Bieber",
]

# Non-targets shared
NONTARGET_CONCEPTS = [
    "Eiffel Tower",
    "Golden Retriever",
    "Bill Clinton",
]

# Per-target anchor concept (phrase-only, target-free)
ANCHOR_CONCEPT_PER_TARGET: Dict[str, str] = {
    "Donald Trump": "man",
    "Christiano Ronaldo": "man",
    "Michael Jackson": "man",
}

# ============================================================
# Utils
# ============================================================
def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _safe_filename(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-", "."):
            keep.append(ch)
        elif ch == " ":
            keep.append("_")
    return "".join(keep)[:180]

def gen_prompt(concept: str) -> str:
    return f"A photo of {concept}"

def base_kwargs() -> Dict:
    return dict(
        target_block_indices=BLOCKS_DUAL,
        target_single_block_indices=BLOCKS_SINGLE,
        anchor_strength=ANCHOR_STRENGTH,
        proj_eps=PROJ_EPS,
        strength_tau=STRENGTH_TAU,
        strength_gamma=STRENGTH_GAMMA,
        vt_dedup_cos_thr=VT_DEDUP_COS_THR,
        max_target_vt_per_block=MAX_TARGET_VT_PER_BLOCK,
        max_retain_vt_per_block=MAX_RETAIN_VT_PER_BLOCK,
        max_anchor_vt_per_block=MAX_ANCHOR_VT_PER_BLOCK,
        apply_target_proj=False,
        dual_zero_text_value=False,
        single_zero_text_value=False,
    )

def run_pipe(
    pipe: FluxPipeline,
    prompt: str,
    *,
    joint_attention_kwargs: Dict,
    seed: int,
    out_path: Optional[str] = None,
) -> Image.Image:
    g = torch.Generator(device=DEVICE)
    g.manual_seed(int(seed))
    img = pipe(
        prompt=prompt,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        height=H,
        width=W,
        generator=g,
        joint_attention_kwargs=joint_attention_kwargs,
        output_type="pil",
        return_dict=True,
        finalize_cora_bases=False,
    ).images[0]
    if out_path is not None:
        img.save(out_path)
    return img

def _get_font(size: int = 22) -> ImageFont.ImageFont:
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()

def grid_2x5(
    base_imgs: List[Image.Image],
    apply_imgs: List[Image.Image],
    *,
    tile_size: Tuple[int, int] = (512, 512),
    pad: int = 12,
    title: Optional[str] = None,
) -> Image.Image:
    assert len(base_imgs) == 5 and len(apply_imgs) == 5, "Need exactly 5 base and 5 apply images."
    cols = 5
    rows = 2
    tw, th = tile_size

    title_h = 0
    if title:
        title_h = 54

    out_w = cols * tw + (cols + 1) * pad
    out_h = rows * th + (rows + 1) * pad + title_h

    canvas = Image.new("RGB", (out_w, out_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    y0 = pad
    if title:
        font = _get_font(26)
        draw.text((pad, y0), title, fill=(0, 0, 0), font=font)
        y0 += title_h

    def paste_row(imgs: List[Image.Image], row_idx: int):
        for c in range(cols):
            x = pad + c * (tw + pad)
            y = y0 + pad + row_idx * (th + pad)
            im = imgs[c].convert("RGB").resize((tw, th), Image.BICUBIC)
            canvas.paste(im, (x, y))

    paste_row(base_imgs, 0)
    paste_row(apply_imgs, 1)
    return canvas


# ============================================================
# Recording + Finalize (ONCE)
# ============================================================
def record_retain(pipe: FluxPipeline):
    print("\n[A] Recording RETAIN vt banks (phrase-only prompts)...")
    for concept in RETAIN_CONCEPTS:
        p = concept
        for j in range(N_RECORD_RETAIN):
            ja = base_kwargs()
            ja.update(
                record_retain_vt=True,
                record_target_vt=False,
                record_anchor_vt=False,
                record_concept=f"RETAIN::{concept}",
            )
            run_pipe(pipe, p, joint_attention_kwargs=ja, seed=BASE_SEED + 1000 + j)
        print(f"  retain recorded: {concept}")

def record_targets_and_anchors(pipe: FluxPipeline):
    print("\n[B] Recording TARGETS + per-target ANCHORS (phrase-only)...")
    for target in TARGET_CONCEPTS:
        # target phrase-only
        for j in range(N_RECORD_TARGET):
            ja = base_kwargs()
            ja.update(
                record_retain_vt=False,
                record_target_vt=True,
                record_anchor_vt=False,
                record_concept=target,
            )
            run_pipe(pipe, target, joint_attention_kwargs=ja, seed=BASE_SEED + 2000 + j)
        print(f"  target recorded: {target}")

        # anchor phrase-only (target-free), stored under record_concept=target
        #anchor for each target??
        anchor = ANCHOR_CONCEPT_PER_TARGET[target]
        for j in range(N_RECORD_ANCHOR):
            ja = base_kwargs()
            ja.update(
                record_retain_vt=False,
                record_target_vt=False,
                record_anchor_vt=True,
                record_concept=target,
            )
            run_pipe(pipe, anchor, joint_attention_kwargs=ja, seed=BASE_SEED + 3000 + j)
        print(f"  anchor recorded: {target} -> {anchor}")

def finalize_cora():
    print("\n[C] Finalizing CORA bases (builds UNION target subspace)...")
    flux_finalize_cora_bases(retain_top_k=int(RETAIN_TOP_K))
    print("  finalize done.")


# ============================================================
# Generation (UNION APPLY)
# ============================================================
def gen_one_concept_grid(
    pipe: FluxPipeline,
    *,
    category: str,
    concept: str,
    prompt: str,
    seed_offset: int,
) -> dict:
    cat_dir = os.path.join(
        OUT_SINGLES,
        _safe_filename(category),
        _safe_filename(concept),
    )
    os.makedirs(cat_dir, exist_ok=True)

    grid_dir = os.path.join(OUT_GRIDS, _safe_filename(category))
    os.makedirs(grid_dir, exist_ok=True)

    seeds = [BASE_SEED + seed_offset + i for i in range(N_GEN)]
    base_imgs: List[Image.Image] = []
    apply_imgs: List[Image.Image] = []
    base_paths: List[str] = []
    apply_paths: List[str] = []

    for i, sd in enumerate(seeds):
        # BASE
        jb = base_kwargs()
        jb.update(
            apply_target_proj=False,
            record_retain_vt=False,
            record_target_vt=False,
            record_anchor_vt=False,
        )
        fn_base = os.path.join(cat_dir, f"s{i}__seed{sd}__BASE.png")
        img_b = run_pipe(pipe, prompt, joint_attention_kwargs=jb, seed=sd, out_path=fn_base)
        base_imgs.append(img_b)
        base_paths.append(fn_base)

        # APPLY (UNION erase): active_concept=None -> uses U_UNION_*
        ja = base_kwargs()
        ja.update(
            apply_target_proj=True,
            active_concept=None,     # <-- IMPORTANT: union multi-target erase
            use_anchors=True,       # <-- IMPORTANT: anchors not supported in union mode as implemented
            record_retain_vt=False,
            record_target_vt=False,
            record_anchor_vt=False,
        )
        fn_apply = os.path.join(cat_dir, f"s{i}__seed{sd}__APPLY.png")
        img_a = run_pipe(pipe, prompt, joint_attention_kwargs=ja, seed=sd, out_path=fn_apply)
        apply_imgs.append(img_a)
        apply_paths.append(fn_apply)

    title = f"{category} | concept={concept} | ERASE=UNION({len(TARGET_CONCEPTS)} targets)"
    grid_img = grid_2x5(base_imgs, apply_imgs, tile_size=(512, 512), title=title)

    grid_path = os.path.join(grid_dir, f"grid__{_safe_filename(concept)}.png")
    grid_img.save(grid_path)

    return dict(
        category=category,
        concept=concept,
        prompt=prompt,
        seeds=seeds,
        base_paths=base_paths,
        apply_paths=apply_paths,
        grid_path=grid_path,
        apply_mode="union",
        targets_union=TARGET_CONCEPTS,
    )


def main():
    seed_everything(BASE_SEED)

    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)

    # start clean once
    flux_reset_vt_banks(reset_retain=True)

    # record once
    record_retain(pipe)
    record_targets_and_anchors(pipe)
    finalize_cora()

    print("\n[D] Generating per-concept grids (2x5: BASE top, APPLY bottom) with UNION erase...")
    manifest: List[dict] = []

    # deterministic offsets by category
    off_targets = 100_000
    off_anchors = 200_000
    off_retains = 300_000
    off_nontargets = 400_000

    # 1) TARGETS
    for i, concept in enumerate(TARGET_CONCEPTS):
        manifest.append(
            gen_one_concept_grid(
                pipe,
                category="1_TARGETS",
                concept=concept,
                prompt=gen_prompt(concept),
                seed_offset=off_targets + i * 100,
            )
        )
        print(f"    saved grid: 1_TARGETS | {concept}")

    # 2) ANCHORS (note: generated prompts are normal "A photo of {anchor}")
    # (Anchors are still useful to visually see what union erase does to "man", etc.)
    for i, t in enumerate(TARGET_CONCEPTS):
        anchor = ANCHOR_CONCEPT_PER_TARGET[t]
        manifest.append(
            gen_one_concept_grid(
                pipe,
                category="2_ANCHORS",
                concept=f"ANCHOR_for_{t}__{anchor}",
                prompt=gen_prompt(anchor),
                seed_offset=off_anchors + i * 100,
            )
        )
        print(f"    saved grid: 2_ANCHORS | {t} -> {anchor}")

    # 3) RETAINS
    for i, concept in enumerate(RETAIN_CONCEPTS):
        manifest.append(
            gen_one_concept_grid(
                pipe,
                category="3_RETAINS",
                concept=concept,
                prompt=gen_prompt(concept),
                seed_offset=off_retains + i * 100,
            )
        )
        print(f"    saved grid: 3_RETAINS | {concept}")

    # 4) NON-TARGETS
    for i, concept in enumerate(NONTARGET_CONCEPTS):
        manifest.append(
            gen_one_concept_grid(
                pipe,
                category="4_NONTARGETS",
                concept=concept,
                prompt=gen_prompt(concept),
                seed_offset=off_nontargets + i * 100,
            )
        )
        print(f"    saved grid: 4_NONTARGETS | {concept}")

    manifest_path = os.path.join(OUTDIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nDone.")
    print(f"Manifest: {manifest_path}")
    print(f"Singles:  {OUT_SINGLES}")
    print(f"Grids:    {OUT_GRIDS}")


if __name__ == "__main__":
    main()