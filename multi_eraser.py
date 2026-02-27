# cora_driver_flux_3cats.py
# ============================================================
# CORA-style FLUX driver (record minimal, generate normal) with
# THREE generation categories + base vs modified grids:
#   1) Targets
#   2) Preserved list (Retained)
#   3) Non-targets (neither erased nor retained)
#
# Recording (MINIMAL):
#   retain recording prompt  = "<retain_concept>"
#   target recording prompt  = "<target_concept>"
#   anchor recording prompt  = "<anchor_prompt>"  (target-free, per target)
#
# Generation (NORMAL user prompts):
#   you provide prompts like: "a photo of Donald Trump ..."
#
# Outputs:
#   OUTDIR/
#     singles/   -> individual images (base/apply)
#     grids/     -> per-category grids + an overall grid
#     manifest.json
# ============================================================

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
MODEL_ID = "black-forest-labs/FLUX.1-schnell"   # or dev
DEVICE = "cuda"
DTYPE = torch.bfloat16

SEED = 0

H, W = 768, 768
STEPS = 8
GUIDANCE = 3.5

# Which blocks to record/apply
BLOCKS_DUAL = list(range(0, 19))
BLOCKS_SINGLE = list(range(0, 38))

# CORA / projection knobs (must match transformer_flux.py kwargs)
ANCHOR_STRENGTH = 2.5
PROJ_EPS = 1e-8
STRENGTH_TAU = 0.10
STRENGTH_GAMMA = 4.0

VT_DEDUP_COS_THR = 0.98
MAX_TARGET_VT_PER_BLOCK = 8
MAX_RETAIN_VT_PER_BLOCK = 16
RETAIN_TOP_K = 3

# Recording repeat count
N_RECORD_RETAIN = 8
N_RECORD_TARGET = 8
N_RECORD_ANCHOR = 4

# Generation per prompt: how many samples to draw (each sample => 1 base + 1 apply)
N_SAMPLES_PER_PROMPT = 4

# Output
OUTDIR = "cora_flux_out_3cats"
OUT_SINGLES = os.path.join(OUTDIR, "singles")
OUT_GRIDS = os.path.join(OUTDIR, "grids")
os.makedirs(OUT_SINGLES, exist_ok=True)
os.makedirs(OUT_GRIDS, exist_ok=True)

# -----------------------------
# Concepts
# -----------------------------
TARGET_CONCEPTS = [
    "Donald Trump",
]

RETAIN_CONCEPTS = [
    "Barack Obama",
    "Hillary Clinton",
    "Melania Trump",
]

# Non-targets: not in target or retain
NONTARGET_CONCEPTS = [
    "Chris Hemsworth",
    "Lion",
    "Eiffel Tower",
    "Golden Retriever",
    "A red car",
]

# Per-target anchor prompts (MUST be target-free).
# Keep it stable + generic; per-target separate anchor as in CORA.
ANCHOR_PROMPT_PER_TARGET: Dict[str, str] = {
    "Donald Trump": "man",
}

# -----------------------------
# Generation prompts (NORMAL prompts)
# Provide user-like prompts for each category.
# You can add more prompts freely.
# -----------------------------
GEN_PROMPTS_TARGETS: Dict[str, List[str]] = {
    "Donald Trump": [
        "a photo of Donald Trump",
        "a portrait photo of Donald Trump, realistic, high detail",
        "Donald Trump speaking at a podium, photojournalism, realistic",
    ],
}

GEN_PROMPTS_RETAIN: Dict[str, List[str]] = {
    "Barack Obama": [
        "a portrait photo of Barack Obama, realistic",
        "Barack Obama giving a speech, realistic, shallow depth of field",
    ],
    "Hillary Clinton": [
        "a portrait photo of Hillary Clinton, realistic",
        "Hillary Clinton at a press conference, realistic photo",
    ],
    "Melania Trump": [
        "a portrait photo of Melania Trump, realistic",
        "Melania Trump outdoors, candid photo, realistic",
    ],
}

GEN_PROMPTS_NONTARGETS: Dict[str, List[str]] = {
    "Chris Hemsworth": [
        "a portrait photo of Chris Hemsworth, realistic",
    ],
    "Lion": [
        "a lion in the savannah, wildlife photography, realistic",
    ],
    "Eiffel Tower": [
        "the Eiffel Tower at sunset, realistic photo",
    ],
    "Golden Retriever": [
        "a golden retriever sitting on grass, realistic photo",
    ],
    "A red car": [
        "a red car on the road, realistic photo",
    ],
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

def base_kwargs() -> Dict:
    # kwargs understood by your modified transformer_flux.py
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
        # toggles
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
        finalize_cora_bases=False,  # finalize explicitly after recording
    ).images[0]

    if out_path is not None:
        img.save(out_path)
    return img

def _get_font(size: int = 18) -> ImageFont.ImageFont:
    # Try common fonts; fall back to default
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

def make_grid(
    tiles: List[Tuple[Image.Image, str]],
    *,
    cols: int,
    tile_size: Tuple[int, int] = (512, 512),
    pad: int = 10,
    header_h: int = 56,
    bg: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
    title: Optional[str] = None,
) -> Image.Image:
    """
    tiles: list of (image, caption)
    """
    if len(tiles) == 0:
        # empty fallback
        img = Image.new("RGB", (tile_size[0], tile_size[1]), bg)
        return img

    rows = (len(tiles) + cols - 1) // cols
    Wt, Ht = tile_size
    out_w = cols * Wt + (cols + 1) * pad
    out_h = rows * (Ht + header_h) + (rows + 1) * pad
    if title is not None:
        out_h += header_h + pad

    canvas = Image.new("RGB", (out_w, out_h), bg)
    draw = ImageDraw.Draw(canvas)
    font = _get_font(18)
    title_font = _get_font(24)

    y0 = pad
    if title is not None:
        draw.text((pad, y0), title, fill=text_color, font=title_font)
        y0 += header_h + pad

    for idx, (im, caption) in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        x = pad + c * (Wt + pad)
        y = y0 + pad + r * (Ht + header_h + pad)

        # caption
        draw.text((x, y), caption, fill=text_color, font=font)

        # image area
        y_img = y + header_h
        im2 = im.convert("RGB").resize((Wt, Ht), Image.BICUBIC)
        canvas.paste(im2, (x, y_img))

    return canvas

def paired_grid(
    pairs: List[Tuple[Image.Image, Image.Image, str]],
    *,
    cols: int = 2,
    tile_size: Tuple[int, int] = (512, 512),
    title: Optional[str] = None,
) -> Image.Image:
    """
    Each pair => two tiles: (BASE, APPLY) with captions.
    """
    tiles: List[Tuple[Image.Image, str]] = []
    for base_im, apply_im, caption in pairs:
        tiles.append((base_im, f"{caption} | BASE"))
        tiles.append((apply_im, f"{caption} | APPLY"))
    return make_grid(tiles, cols=cols, tile_size=tile_size, title=title)

# ============================================================
# Recording + Finalize
# ============================================================
def record_retain(pipe: FluxPipeline):
    print("\n[A] Recording retain vt banks (phrase-only prompts)...")
    for concept in RETAIN_CONCEPTS:
        p = concept  # phrase-only
        for j in range(N_RECORD_RETAIN):
            ja = base_kwargs()
            ja.update(
                record_retain_vt=True,
                record_target_vt=False,
                record_anchor_vt=False,
                record_concept=f"RETAIN::{concept}",
            )
            run_pipe(pipe, p, joint_attention_kwargs=ja, seed=SEED + 1000 + j)
        print(f"  retain recorded: {concept}")

def record_target_and_anchor(pipe: FluxPipeline):
    print("\n[B] Recording target + anchor banks (phrase-only + anchor prompt)...")
    for concept in TARGET_CONCEPTS:
        # target: phrase-only
        p_target = concept
        for j in range(N_RECORD_TARGET):
            ja = base_kwargs()
            ja.update(
                record_retain_vt=False,
                record_target_vt=True,
                record_anchor_vt=False,
                record_concept=concept,
            )
            run_pipe(pipe, p_target, joint_attention_kwargs=ja, seed=SEED + 2000 + j)
        print(f"  target recorded: {concept}")

        # anchor: target-free
        p_anchor = ANCHOR_PROMPT_PER_TARGET[concept]
        for j in range(N_RECORD_ANCHOR):
            ja = base_kwargs()
            ja.update(
                record_retain_vt=False,
                record_target_vt=False,
                record_anchor_vt=True,
                record_concept=concept,
            )
            run_pipe(pipe, p_anchor, joint_attention_kwargs=ja, seed=SEED + 3000 + j)
        print(f"  anchor recorded: {concept}")

def finalize_cora():
    print("\n[C] Finalizing CORA bases...")
    flux_finalize_cora_bases(retain_top_k=RETAIN_TOP_K)
    print("  finalize done.")

# ============================================================
# Generation
# ============================================================
def generate_category(
    pipe: FluxPipeline,
    *,
    category_name: str,
    prompts_by_subject: Dict[str, List[str]],
    active_target: str,
    base_seed: int,
) -> Tuple[List[dict], List[Tuple[Image.Image, Image.Image, str]]]:
    """
    Returns:
      manifest_entries: list of dicts
      pair_tiles: list of (base_img, apply_img, caption)
    """
    manifest_entries: List[dict] = []
    pair_tiles: List[Tuple[Image.Image, Image.Image, str]] = []

    cat_dir = os.path.join(OUT_SINGLES, _safe_filename(category_name))
    os.makedirs(cat_dir, exist_ok=True)

    idx_global = 0
    for subject, prompts in prompts_by_subject.items():
        for p_i, prompt in enumerate(prompts):
            for s_i in range(N_SAMPLES_PER_PROMPT):
                # --- baseline ---
                jb = base_kwargs()
                jb.update(
                    apply_target_proj=False,
                    record_retain_vt=False,
                    record_target_vt=False,
                    record_anchor_vt=False,
                )
                seed_b = base_seed + 10000 + idx_global * 13 + s_i
                fn_base = os.path.join(
                    cat_dir,
                    f"{_safe_filename(subject)}__p{p_i}__s{s_i}__BASE.png",
                )
                img_base = run_pipe(pipe, prompt, joint_attention_kwargs=jb, seed=seed_b, out_path=fn_base)

                # --- apply ---
                ja = base_kwargs()
                ja.update(
                    apply_target_proj=True,
                    active_concept=active_target,  # erase this target everywhere
                    use_anchors=True,
                    record_retain_vt=False,
                    record_target_vt=False,
                    record_anchor_vt=False,
                )
                seed_a = base_seed + 20000 + idx_global * 13 + s_i
                fn_apply = os.path.join(
                    cat_dir,
                    f"{_safe_filename(subject)}__p{p_i}__s{s_i}__APPLY.png",
                )
                img_apply = run_pipe(pipe, prompt, joint_attention_kwargs=ja, seed=seed_a, out_path=fn_apply)

                caption = f"{category_name} | {subject} | {prompt[:60]}"
                pair_tiles.append((img_base, img_apply, caption))

                manifest_entries.append(
                    dict(
                        category=category_name,
                        subject=subject,
                        prompt=prompt,
                        sample_index=int(s_i),
                        base_path=fn_base,
                        apply_path=fn_apply,
                        active_target=active_target,
                        seed_base=int(seed_b),
                        seed_apply=int(seed_a),
                    )
                )
                idx_global += 1

    return manifest_entries, pair_tiles

# ============================================================
# Main
# ============================================================
def main():
    seed_everything(SEED)

    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)

    # Always start clean
    flux_reset_vt_banks(reset_retain=True)

    # (A) retain record
    record_retain(pipe)

    # (B) target + anchors record
    record_target_and_anchor(pipe)

    # (C) finalize CORA
    finalize_cora()

    # (D) Generate 3 categories for EACH target (you can loop multiple targets)
    print("\n[D] Generating base vs apply for 3 categories...")
    all_manifest: List[dict] = []

    # grids: store per category per target
    for t_i, target in enumerate(TARGET_CONCEPTS):
        print(f"\n  == Active erase target: {target} ==")

        # Targets category (generate prompts for target)
        m1, pairs1 = generate_category(
            pipe,
            category_name="1_TARGETS",
            prompts_by_subject=GEN_PROMPTS_TARGETS,
            active_target=target,
            base_seed=SEED + 400000 + t_i * 10000,
        )
        all_manifest.extend(m1)
        grid1 = paired_grid(
            pairs1,
            cols=2,
            tile_size=(512, 512),
            title=f"TARGETS | active erase = {target}",
        )
        grid1_path = os.path.join(OUT_GRIDS, f"grid_targets__erase_{_safe_filename(target)}.png")
        grid1.save(grid1_path)
        print(f"    saved grid: {grid1_path}")

        # Retained category
        m2, pairs2 = generate_category(
            pipe,
            category_name="2_RETAINED",
            prompts_by_subject=GEN_PROMPTS_RETAIN,
            active_target=target,
            base_seed=SEED + 500000 + t_i * 10000,
        )
        all_manifest.extend(m2)
        grid2 = paired_grid(
            pairs2,
            cols=2,
            tile_size=(512, 512),
            title=f"RETAINED | active erase = {target}",
        )
        grid2_path = os.path.join(OUT_GRIDS, f"grid_retained__erase_{_safe_filename(target)}.png")
        grid2.save(grid2_path)
        print(f"    saved grid: {grid2_path}")

        # Non-target category
        m3, pairs3 = generate_category(
            pipe,
            category_name="3_NONTARGETS",
            prompts_by_subject=GEN_PROMPTS_NONTARGETS,
            active_target=target,
            base_seed=SEED + 600000 + t_i * 10000,
        )
        all_manifest.extend(m3)
        grid3 = paired_grid(
            pairs3,
            cols=2,
            tile_size=(512, 512),
            title=f"NON-TARGETS | active erase = {target}",
        )
        grid3_path = os.path.join(OUT_GRIDS, f"grid_nontargets__erase_{_safe_filename(target)}.png")
        grid3.save(grid3_path)
        print(f"    saved grid: {grid3_path}")

        # Optional: one big combined grid (first N pairs from each category)
        N_SHOW = 8  # keep grids readable
        combined_pairs = pairs1[:N_SHOW] + pairs2[:N_SHOW] + pairs3[:N_SHOW]
        grid_all = paired_grid(
            combined_pairs,
            cols=2,
            tile_size=(512, 512),
            title=f"ALL CATEGORIES (subset) | active erase = {target}",
        )
        grid_all_path = os.path.join(OUT_GRIDS, f"grid_all_subset__erase_{_safe_filename(target)}.png")
        grid_all.save(grid_all_path)
        print(f"    saved grid: {grid_all_path}")

    # Save manifest
    manifest_path = os.path.join(OUTDIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(all_manifest, f, indent=2)
    print(f"\nDone. Manifest: {manifest_path}")
    print(f"Singles:  {OUT_SINGLES}")
    print(f"Grids:    {OUT_GRIDS}")


if __name__ == "__main__":
    main()