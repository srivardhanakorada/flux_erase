import os
import csv
from typing import List, Dict, Any, Optional

import torch  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from diffusers import FluxPipeline
from diffusers.models.transformers.transformer_flux import (  # type: ignore
    flux_reset_vt_banks,
    flux_finalize_cora_bases,
    flux_get_concept_similarity_report_coip,
)

MODEL_ID = "black-forest-labs/FLUX.1-schnell"

# ============================================================
# Concepts
# ============================================================

TARGET = "Dog"

PARAPHRASE_LIKE = [
    "Doggo",
    "Canine",
    # "Golden Retriever",
]

RELATED_ANIMALS = [
    "Lion",
    # "Fox",
    "Wolf",
]

FRUITS = [
    "Apple",
    "Taylor Swift",
]

CELEBRITIES = [
    ###
]

OTHERS = PARAPHRASE_LIKE + RELATED_ANIMALS + FRUITS + CELEBRITIES
ALL_CONCEPTS = [TARGET] + OTHERS

GROUP_MAP: Dict[str, str] = {}
for x in PARAPHRASE_LIKE:
    GROUP_MAP[x] = "paraphrase_like"
for x in RELATED_ANIMALS:
    GROUP_MAP[x] = "related_animals"
for x in FRUITS:
    GROUP_MAP[x] = "fruits"
for x in CELEBRITIES:
    GROUP_MAP[x] = "celebrities"

# ============================================================
# Recording setup
# ============================================================

# Keep your same style: generic bank + retain bank + target bank
RECORDING_TEMPLATES = [
    "a photo of {}",
    "{} on a plain background",
]

# retain prompts: concepts you do NOT want erased, to stabilize the free space
RETAINS = [
    "Cat",
    "Horse",
    "Elephant",
    "Car",
]

# generic animal/category bank
PERSON_BANK = [
    "an animal",
    "a mammal",
    "a pet animal",
    # "a four-legged animal",
    # "a domestic animal",
]

# Optional anchor, normally off
USE_ANCHORS = False
ANCHOR = "an animal"

DUAL_BLOCKS = list(range(0, 19))
SINGLE_BLOCKS = list(range(0, 38))

OUTDIR = "results_new/concept_similarity_dog_coip"
os.makedirs(OUTDIR, exist_ok=True)

STRENGTH_TAU = 0.10
STRENGTH_GAMMA = 1.75
ANCHOR_STRENGTH = 1.0

PERSON_TOP_K = 2
RETAIN_TOP_K = 4

REC_H, REC_W = 768, 768
STEPS = 4
GUIDANCE = 3.5
N_IMAGES_PER_PROMPT = 1

# use a few seeds per concept to stabilize recorded prototypes
RECORD_SEEDS = [0]


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
    record_mode: bool = True,
):
    g = torch.Generator(device=pipe.device).manual_seed(seed)

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

    _ = pipe(
        prompt=prompt,
        height=REC_H,
        width=REC_W,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        num_images_per_prompt=N_IMAGES_PER_PROMPT,
        generator=g,
        joint_attention_kwargs=ja,
        output_type="latent",
    )

    _maybe_clear_cache()


def save_csv(report: Dict[str, Dict[str, Any]], out_csv: str, report_name: str):
    rows = []

    for branch in ["dual", "single"]:
        per_block = report[branch]["per_block"]
        global_scores = report[branch]["global"]

        for blk, score_map in per_block.items():
            for concept, score in score_map.items():
                rows.append({
                    "report": report_name,
                    "branch": branch,
                    "block_index": blk,
                    "concept": concept,
                    "group": GROUP_MAP.get(concept, "other"),
                    "similarity_to_target": score,
                    "is_global": 0,
                })

        for concept, score in global_scores.items():
            rows.append({
                "report": report_name,
                "branch": branch,
                "block_index": -1,
                "concept": concept,
                "group": GROUP_MAP.get(concept, "other"),
                "similarity_to_target": score,
                "is_global": 1,
            })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "report",
                "branch",
                "block_index",
                "concept",
                "group",
                "similarity_to_target",
                "is_global",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_combined_csv(
    raw_report: Dict[str, Dict[str, Any]],
    cleaned_report: Dict[str, Dict[str, Any]],
    out_csv: str,
):
    rows = []

    for report_name, report in [("raw", raw_report), ("cleaned", cleaned_report)]:
        for branch in ["dual", "single"]:
            per_block = report[branch]["per_block"]
            global_scores = report[branch]["global"]

            for blk, score_map in per_block.items():
                for concept, score in score_map.items():
                    rows.append({
                        "report": report_name,
                        "branch": branch,
                        "block_index": blk,
                        "concept": concept,
                        "group": GROUP_MAP.get(concept, "other"),
                        "similarity_to_target": score,
                        "is_global": 0,
                    })

            for concept, score in global_scores.items():
                rows.append({
                    "report": report_name,
                    "branch": branch,
                    "block_index": -1,
                    "concept": concept,
                    "group": GROUP_MAP.get(concept, "other"),
                    "similarity_to_target": score,
                    "is_global": 1,
                })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "report",
                "branch",
                "block_index",
                "concept",
                "group",
                "similarity_to_target",
                "is_global",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_branch(report: Dict[str, Dict[str, Any]], branch_name: str, out_png: str, title_prefix: str):
    per_block = report[branch_name]["per_block"]
    global_scores = report[branch_name]["global"]

    blocks = sorted(per_block.keys())
    if len(blocks) == 0:
        print(f"[WARN] No blocks found for branch={branch_name}")
        return

    plt.figure(figsize=(14, 7))

    for concept in OTHERS:
        xs, ys = [], []
        for blk in blocks:
            if concept in per_block.get(blk, {}):
                xs.append(blk)
                ys.append(per_block[blk][concept])

        if len(xs) > 0:
            gtxt = ""
            if concept in global_scores:
                gtxt = f" ({global_scores[concept]:.2f})"
            plt.plot(xs, ys, marker="o", linewidth=1.5, label=f"{concept}{gtxt}")

    plt.ylim(0.0, 1.0)
    plt.xlabel("Block Index")
    plt.ylabel(f"Cosine similarity to target: {TARGET}")
    plt.title(f"{title_prefix} | {branch_name.capitalize()} blocks | target={TARGET}")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_group_means(report: Dict[str, Dict[str, Any]], branch_name: str, out_png: str, title_prefix: str):
    per_block = report[branch_name]["per_block"]
    blocks = sorted(per_block.keys())
    if len(blocks) == 0:
        print(f"[WARN] No blocks found for branch={branch_name}")
        return

    groups = {
        "paraphrases": PARAPHRASE_LIKE,
        "related": RELATED_ANIMALS,
        "unrelated": FRUITS,
        # "celebrities": CELEBRITIES,
    }

    plt.figure(figsize=(12, 6))

    for group_name, concepts in groups.items():
        xs, ys = [], []
        for blk in blocks:
            vals = []
            for c in concepts:
                if c in per_block.get(blk, {}):
                    vals.append(per_block[blk][c])
            if len(vals) > 0:
                xs.append(blk)
                ys.append(sum(vals) / len(vals))
        if len(xs) > 0:
            plt.plot(xs, ys, marker="o", linewidth=2.5, label=group_name)

    plt.ylim(0.0, 1.0)
    plt.xlabel("Block Index")
    plt.ylabel(f"Mean cosine similarity to target: {TARGET}")
    plt.title(f"{title_prefix} | {branch_name.capitalize()} blocks | group means")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def print_summary(report: Dict[str, Dict[str, Any]], name: str):
    print(f"\n===== GLOBAL COSINE SUMMARY ({name.upper()}) =====")
    for branch_name in ["dual", "single"]:
        print(f"\n[{branch_name.upper()}]")
        global_scores = report[branch_name]["global"]

        for group_name, items in [
            ("PARAPHRASE_LIKE", PARAPHRASE_LIKE),
            ("RELATED_ANIMALS", RELATED_ANIMALS),
            ("FRUITS", FRUITS),
            ("CELEBRITIES", CELEBRITIES),
        ]:
            print(f"\n  {group_name}")
            for concept in items:
                val = global_scores.get(concept, None)
                if val is None:
                    print(f"    {concept:20s} : N/A")
                else:
                    print(f"    {concept:20s} : {val:.4f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    ).to(device)

    flux_reset_vt_banks(reset_retain=True)
    _maybe_clear_cache()

    # ------------------------------------------------------------
    # 1) Record retain bank
    # ------------------------------------------------------------
    for i, rp in enumerate(RETAINS):
        for j, seed in enumerate(RECORD_SEEDS):
            run_one(
                pipe,
                prompt=rp,
                record_retain_vt=True,
                seed=1000 + 100 * i + j + seed,
            )

    # ------------------------------------------------------------
    # 2) Record generic animal/category bank
    # ------------------------------------------------------------
    for i, pp in enumerate(PERSON_BANK):
        for j, seed in enumerate(RECORD_SEEDS):
            run_one(
                pipe,
                prompt=pp,
                record_person_vt=True,
                seed=2000 + 100 * i + j + seed,
            )

    # ------------------------------------------------------------
    # 3) Record target and all comparison concepts into target bank
    # ------------------------------------------------------------
    for i, concept in enumerate(ALL_CONCEPTS):
        for j, template in enumerate(RECORDING_TEMPLATES):
            prompt = template.format(concept)
            for k, seed in enumerate(RECORD_SEEDS):
                print(f"[REC] {concept} | {prompt} | seed={seed}")
                run_one(
                    pipe,
                    prompt=prompt,
                    record_target_vt=True,
                    record_concept=concept,
                    seed=3000 + 1000 * i + 100 * j + k + seed,
                )

    # ------------------------------------------------------------
    # 4) Optional anchor
    # ------------------------------------------------------------
    if USE_ANCHORS:
        for i, seed in enumerate(RECORD_SEEDS):
            run_one(
                pipe,
                prompt=ANCHOR,
                record_anchor_once=True,
                seed=4000 + i + seed,
            )

    # ------------------------------------------------------------
    # 5) Finalize bases
    # ------------------------------------------------------------
    flux_finalize_cora_bases(
        retain_top_k=RETAIN_TOP_K,
        person_top_k=PERSON_TOP_K,
    )
    _maybe_clear_cache()

    # ------------------------------------------------------------
    # 6) Raw vs cleaned reports
    # ------------------------------------------------------------
    raw_report = flux_get_concept_similarity_report_coip(
        target_concept=TARGET,
        other_concepts=OTHERS,
        cleaned=False,
    )

    cleaned_report = flux_get_concept_similarity_report_coip(
        target_concept=TARGET,
        other_concepts=OTHERS,
        cleaned=True,
    )

    print_summary(raw_report, "raw")
    print_summary(cleaned_report, "cleaned")

    # ------------------------------------------------------------
    # 7) Save CSVs
    # ------------------------------------------------------------
    raw_csv = os.path.join(OUTDIR, "dog_similarity_raw.csv")
    cleaned_csv = os.path.join(OUTDIR, "dog_similarity_cleaned.csv")
    combined_csv = os.path.join(OUTDIR, "dog_similarity_combined.csv")

    save_csv(raw_report, raw_csv, "raw")
    save_csv(cleaned_report, cleaned_csv, "cleaned")
    save_combined_csv(raw_report, cleaned_report, combined_csv)

    # ------------------------------------------------------------
    # 8) Save plots
    # ------------------------------------------------------------
    plot_branch(raw_report, "dual", os.path.join(OUTDIR, "dog_raw_dual_all.png"), "Raw")
    plot_branch(raw_report, "single", os.path.join(OUTDIR, "dog_raw_single_all.png"), "Raw")
    plot_group_means(raw_report, "dual", os.path.join(OUTDIR, "dog_raw_dual_group_means.png"), "Raw")
    plot_group_means(raw_report, "single", os.path.join(OUTDIR, "dog_raw_single_group_means.png"), "Raw")

    plot_branch(cleaned_report, "dual", os.path.join(OUTDIR, "dog_cleaned_dual_all.png"), "Cleaned")
    plot_branch(cleaned_report, "single", os.path.join(OUTDIR, "dog_cleaned_single_all.png"), "Cleaned")
    plot_group_means(cleaned_report, "dual", os.path.join(OUTDIR, "dog_cleaned_dual_group_means.png"), "Cleaned")
    plot_group_means(cleaned_report, "single", os.path.join(OUTDIR, "dog_cleaned_single_group_means.png"), "Cleaned")

    print(f"\nDone. Results saved to: {OUTDIR}")


if __name__ == "__main__":
    main()