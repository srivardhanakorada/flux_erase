import os
import json
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from diffusers import FluxPipeline

# Import from your patched transformer file
from diffusers.models.transformers.transformer_flux import (
    flux_reset_vt_banks,
    flux_finalize_cora_bases,
    flux_set_edit_logging,
    flux_reset_edit_logs,
    flux_get_edit_logs,
)

# ============================================================
# Config
# ============================================================

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
DEVICE = "cuda"
DTYPE = torch.bfloat16

OUTDIR = "figure_current_method_cols134"
os.makedirs(OUTDIR, exist_ok=True)

H, W = 768, 768
STEPS = 4
GUIDANCE = 3.5
BASE_SEED = 0

DUAL_BLOCKS = list(range(0, 19))
SINGLE_BLOCKS = list(range(0, 38))

# ------------------------------------------------------------
# Replace these with your actual concepts/prompts
# ------------------------------------------------------------

TARGET_RECORDS = [
    "Donald Trump",
]

RETAIN_RECORDS = [
    "Barack Obama",
    "Hillary Clinton",
    "Melania Trump",
]

PERSON_RECORDS = [
    "person",
    "man",
    "woman",
    "portrait",
    "face",
]

ANCHOR_RECORDS = [
    "person",
]

# Row 1
TARGET_PROMPTS = [
    "A studio portrait photograph of Donald Trump",
    "A cinematic outdoor portrait of Donald Trump",
]

# Row 2
PRESERVE_PROMPTS = [
    "A studio portrait photograph of Barack Obama",
    "A cinematic outdoor portrait of Hillary Clinton",
]

# Row 3
NONTARGET_PROMPTS = [
    "A studio portrait photograph of a man",
    "A cinematic outdoor portrait of a woman",
]

EDIT_KWARGS = dict(
    apply_target_proj=True,
    target_block_indices=DUAL_BLOCKS,
    target_single_block_indices=SINGLE_BLOCKS,
    strength_tau=0.02,
    strength_gamma=1.25,
    anchor_strength=2.5,
    proj_eps=1e-8,
    use_anchors=True,
    person_weight=0.35,
    gate_sharpness=16.0,
    use_soft_gate=True,
    proj_token_end=128,
    detector_token_end=2,
)

FINALIZE_KWARGS = dict(
    retain_top_k=4,
    person_top_k=6,
    person_remove_scale=0.5,
    eps=1e-8,
)


# ============================================================
# Helpers
# ============================================================

def make_generator(seed: int):
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    return g


def save_image(img: Image.Image, path: str):
    img.save(path)


def add_caption(img: Image.Image, text: str, height: int = 40) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + height), "white")
    out.paste(img, (0, height))
    draw = ImageDraw.Draw(out)
    draw.text((10, 10), text, fill="black")
    return out


def pad_to_same_height(images: List[Image.Image]) -> List[Image.Image]:
    h = max(im.height for im in images)
    out = []
    for im in images:
        if im.height == h:
            out.append(im)
            continue
        canvas = Image.new("RGB", (im.width, h), "white")
        canvas.paste(im, (0, 0))
        out.append(canvas)
    return out


def merge_horiz(images: List[Image.Image], outpath: str):
    images = pad_to_same_height(images)
    total_w = sum(im.width for im in images)
    h = max(im.height for im in images)
    canvas = Image.new("RGB", (total_w, h), "white")
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.width
    canvas.save(outpath)


def stack_dual_single_heatmaps(dual: np.ndarray, single: np.ndarray) -> np.ndarray:
    width = max(dual.shape[1], single.shape[1])
    dual2 = np.pad(dual, ((0, 0), (0, width - dual.shape[1])))
    single2 = np.pad(single, ((0, 0), (0, width - single.shape[1])))
    sep = np.zeros((1, width), dtype=np.float32)
    return np.concatenate([dual2, sep, single2], axis=0)


def aggregate_removed_heatmap(logs: Dict[str, Dict[int, List[torch.Tensor]]]) -> Dict[str, np.ndarray]:
    def build(pre_key: str, post_key: str) -> np.ndarray:
        pre_logs = logs[pre_key]
        post_logs = logs[post_key]
        blocks = sorted(set(pre_logs.keys()) & set(post_logs.keys()))

        if len(blocks) == 0:
            return np.zeros((1, 1), dtype=np.float32)

        rows = []
        max_t = 1

        for blk in blocks:
            cur = []
            n = min(len(pre_logs[blk]), len(post_logs[blk]))
            for i in range(n):
                pre = pre_logs[blk][i].float()   # [B,T,D]
                post = post_logs[blk][i].float() # [B,T,D]
                delta = pre - post
                score = delta.norm(dim=-1).mean(dim=0).cpu().numpy()  # [T]
                cur.append(score)

            if len(cur) == 0:
                cur_mean = np.zeros((1,), dtype=np.float32)
            else:
                max_local = max(x.shape[0] for x in cur)
                cur = [np.pad(x, (0, max_local - x.shape[0])) for x in cur]
                cur_mean = np.mean(cur, axis=0)

            max_t = max(max_t, cur_mean.shape[0])
            rows.append(cur_mean)

        rows = [np.pad(r, (0, max_t - r.shape[0])) for r in rows]
        return np.stack(rows, axis=0)

    dual = build("dual_pre", "dual_post")
    single = build("single_pre", "single_post")
    return {"dual": dual, "single": single}


def save_heatmap(mat: np.ndarray, outpath: str, title: str):
    plt.figure(figsize=(9, 4))
    plt.imshow(mat, aspect="auto", interpolation="nearest", vmin=0, vmax=400)
    plt.colorbar(label="||pre - post||")
    plt.xlabel("Token index")
    plt.ylabel("Block index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


# ============================================================
# Record + finalize bases
# ============================================================

@torch.no_grad()
def record_and_finalize(pipe: FluxPipeline):
    flux_reset_vt_banks(reset_retain=True)

    common = dict(
        target_block_indices=DUAL_BLOCKS,
        target_single_block_indices=SINGLE_BLOCKS,
        detector_token_end=2,
    )

    # Retain
    for phrase in RETAIN_RECORDS:
        _ = pipe(
            prompt=phrase,
            height=H,
            width=W,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=make_generator(BASE_SEED),
            joint_attention_kwargs=dict(
                record_retain_vt=True,
                **common,
            ),
            output_type="pil",
        )

    # Person/category
    for phrase in PERSON_RECORDS:
        _ = pipe(
            prompt=phrase,
            height=H,
            width=W,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=make_generator(BASE_SEED),
            joint_attention_kwargs=dict(
                record_person_vt=True,
                **common,
            ),
            output_type="pil",
        )

    # Target
    for phrase in TARGET_RECORDS:
        _ = pipe(
            prompt=phrase,
            height=H,
            width=W,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=make_generator(BASE_SEED),
            joint_attention_kwargs=dict(
                record_target_vt=True,
                record_concept=phrase,
                **common,
            ),
            output_type="pil",
        )

    # Anchor once
    for phrase in ANCHOR_RECORDS:
        _ = pipe(
            prompt=phrase,
            height=H,
            width=W,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=make_generator(BASE_SEED),
            joint_attention_kwargs=dict(
                record_anchor_once=True,
                **common,
            ),
            output_type="pil",
        )

    flux_finalize_cora_bases(**FINALIZE_KWARGS)


# ============================================================
# Run one prompt: Col1, Col3, Col4
# ============================================================

@torch.no_grad()
def run_prompt(pipe: FluxPipeline, prompt: str, seed: int, row_tag: str, idx: int):
    prefix = f"{row_tag}_{idx:02d}"

    # -------------------------
    # Col 1: base image
    # -------------------------
    base = pipe(
        prompt=prompt,
        height=H,
        width=W,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=make_generator(seed),
    ).images[0]

    base_path = os.path.join(OUTDIR, f"{prefix}_col1_base.png")
    save_image(base, base_path)

    # -------------------------
    # Col 3 + Col 4
    # -------------------------
    flux_reset_edit_logs()
    flux_set_edit_logging(True)

    edited = pipe(
        prompt=prompt,
        height=H,
        width=W,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=make_generator(seed),
        joint_attention_kwargs=EDIT_KWARGS,
    ).images[0]

    flux_set_edit_logging(False)

    edited_path = os.path.join(OUTDIR, f"{prefix}_col3_current.png")
    save_image(edited, edited_path)

    logs = flux_get_edit_logs()
    mats = aggregate_removed_heatmap(logs)
    full_mat = stack_dual_single_heatmaps(mats["dual"], mats["single"])

    heatmap_path = os.path.join(OUTDIR, f"{prefix}_col4_removed_heatmap.png")
    save_heatmap(full_mat, heatmap_path, title=f"{row_tag}: removed-component norm")

    # Optional triptych for easy viewing now: col1 | blank col2 | col3 | col4
    blank_col2 = Image.new("RGB", base.size, "white")
    blank_col2 = add_caption(blank_col2, "Old method (later)")
    base_cap = add_caption(base, "Col 1: Base")
    edited_cap = add_caption(edited, "Col 3: Current method")
    heat_cap = add_caption(Image.open(heatmap_path).convert("RGB"), "Col 4: Removed heatmap")

    merge_horiz(
        [base_cap, blank_col2, edited_cap, heat_cap],
        os.path.join(OUTDIR, f"{prefix}_grid_preview.png")
    )

    meta = {
        "prompt": prompt,
        "seed": seed,
        "row_tag": row_tag,
        "base_path": base_path,
        "edited_path": edited_path,
        "heatmap_path": heatmap_path,
    }
    with open(os.path.join(OUTDIR, f"{prefix}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ============================================================
# Run all rows
# ============================================================

def run_prompt_group(pipe: FluxPipeline, prompts: List[str], row_tag: str, seed_offset: int):
    for i, prompt in enumerate(prompts):
        run_prompt(pipe, prompt, BASE_SEED + seed_offset + i, row_tag=row_tag, idx=i)


def main():
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
    ).to(DEVICE)

    record_and_finalize(pipe)

    # Row 1
    run_prompt_group(pipe, TARGET_PROMPTS, row_tag="row1_target", seed_offset=0)

    # Row 2
    run_prompt_group(pipe, PRESERVE_PROMPTS, row_tag="row2_preserve", seed_offset=100)

    # Row 3
    run_prompt_group(pipe, NONTARGET_PROMPTS, row_tag="row3_nontarget", seed_offset=200)

    print(f"Saved outputs to: {OUTDIR}")


if __name__ == "__main__":
    main()