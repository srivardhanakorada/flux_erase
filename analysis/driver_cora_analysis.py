import os
import json
from typing import Dict, List, Optional

import torch  # type: ignore
from diffusers import FluxPipeline  # type: ignore
from diffusers.models.transformers.transformer_flux import (  # type: ignore
    flux_reset_vt_banks,
    flux_set_analysis_logging,
    flux_reset_analysis_logs,
    flux_get_analysis_logs,
    flux_finalize_cora_bases,
    flux_collect_basis_overlap,
)

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
OUTDIR = "cora_analysis_outputs"
os.makedirs(OUTDIR, exist_ok=True)

# ----------------------------
# Concepts / prompt groups
# ----------------------------
TARGETS: List[str] = [
    "Donald Trump",
]

RETAINS: List[str] = [
    "Melania Trump",
    "Hillary Clinton",
    "Barack Obama",
]

NON_TARGETS: List[str] = [
    "Bill Clinton",
    "Joe Biden",
    "President of the United States of America",
    "Husband of Melania Trump",
]

# analysis-only person bank, only for overlap / rP comparison
PERSON_BANK: List[str] = [
    "a portrait of a person",
    "a portrait of a man",
    "a portrait of a woman",
    "a middle-aged man",
    "a middle-aged woman",
]

RECORDING_TEMPLATES: List[str] = [
    "a photo of {}",
    "{} photographed with DSLR",
    "{}, studio portrait, sharp focus",
]

PROMPT_GROUPS: Dict[str, List[str]] = {
    "target": [
        "a portrait of Donald Trump",
        "Donald Trump, studio portrait",
        "Donald Trump speaking at a podium",
    ],
    "retain": [
        "a portrait of Melania Trump",
        "a portrait of Hillary Clinton",
        "a portrait of Barack Obama",
    ],
    "person": [
        "a portrait of Joe Biden",
        "a portrait of Bill Clinton",
        "a portrait of a generic man",
        "a portrait of a generic woman",
        "a professional businessperson portrait",
    ],
    "nonperson": [
        "a red sports car on a road",
        "a wooden chair in a living room",
        "a tiger in the forest",
        "a mountain landscape at sunrise",
    ],
}

# ----------------------------
# Model / inference config
# ----------------------------
DUAL_BLOCKS = list(range(0, 19))
SINGLE_BLOCKS = list(range(0, 38))

REC_H, REC_W = 512, 512
GEN_H, GEN_W = 512, 512
STEPS = 4
GUIDANCE = 3.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# ----------------------------
# CORA settings
# ----------------------------
USE_ANCHORS = False
ANCHOR = "a portrait of a person"
ANCHOR_STRENGTH = 1.5

STRENGTH_TAU = 0.2
STRENGTH_GAMMA = 1.0

RETAIN_TOP_K = 6
PERSON_TOP_K = 6  # analysis-only
DETECTOR_TOKEN_END = 2
PROJ_TOKEN_END = None


def _maybe_clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


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
    prompt_tag: Optional[str] = None,
    seed: int = 0,
    record_mode: bool = False,
):
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    height = REC_H if record_mode else GEN_H
    width = REC_W if record_mode else GEN_W
    output_type = "latent" if record_mode else "pil"

    ja = {
        "record_target_vt": record_target_vt,
        "record_retain_vt": record_retain_vt,
        "record_person_vt": record_person_vt,   # analysis-only
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
        "detector_token_end": DETECTOR_TOKEN_END,
        "proj_token_end": PROJ_TOKEN_END,
        "prompt_tag": prompt_tag,
    }

    out = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        num_images_per_prompt=1,
        generator=generator,
        joint_attention_kwargs=ja,
        output_type=output_type,
    )

    if record_mode:
        _maybe_clear_cache()
        return None
    return out


def record_target(pipe: FluxPipeline, concept: str, prompt: str, seed: int):
    run_one(
        pipe,
        prompt=prompt,
        record_target_vt=True,
        record_concept=concept,
        seed=seed,
        record_mode=True,
    )


def record_retain(pipe: FluxPipeline, prompt: str, seed: int):
    run_one(
        pipe,
        prompt=prompt,
        record_retain_vt=True,
        seed=seed,
        record_mode=True,
    )


def record_person(pipe: FluxPipeline, prompt: str, seed: int):
    run_one(
        pipe,
        prompt=prompt,
        record_person_vt=True,
        seed=seed,
        record_mode=True,
    )


def record_anchor_once(pipe: FluxPipeline, prompt: str, seed: int):
    run_one(
        pipe,
        prompt=prompt,
        record_anchor_once=True,
        seed=seed,
        record_mode=True,
    )


def run_gate_probe(pipe: FluxPipeline, prompt: str, prompt_tag: str, seed: int):
    run_one(
        pipe,
        prompt=prompt,
        apply_target_proj=True,
        prompt_tag=prompt_tag,
        seed=seed,
        record_mode=False,
    )


def main():
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
    ).to(DEVICE)

    flux_reset_vt_banks(reset_retain=True)
    flux_set_analysis_logging(False)
    flux_reset_analysis_logs()
    _maybe_clear_cache()

    # ----------------------------------------
    # 1) Record retain banks
    # ----------------------------------------
    for i, retain in enumerate(RETAINS):
        record_retain(pipe, retain, seed=1000 + i)

    # ----------------------------------------
    # 2) Record person/category banks (analysis-only)
    # ----------------------------------------
    for i, person_prompt in enumerate(PERSON_BANK):
        record_person(pipe, person_prompt, seed=2000 + i)

    # ----------------------------------------
    # 3) Record target banks
    # ----------------------------------------
    for i, target in enumerate(TARGETS):
        for j, template in enumerate(RECORDING_TEMPLATES):
            prompt = template.format(target)
            record_target(pipe, target, prompt, seed=3000 + 100 * i + j)

    # ----------------------------------------
    # 4) Optional anchor-once bank
    # ----------------------------------------
    if USE_ANCHORS:
        record_anchor_once(pipe, ANCHOR, seed=4000)

    # ----------------------------------------
    # 5) Finalize CORA bases
    # ----------------------------------------
    flux_finalize_cora_bases(
        retain_top_k=RETAIN_TOP_K,
        person_top_k=PERSON_TOP_K,
    )
    _maybe_clear_cache()

    # ----------------------------------------
    # 6) Save basis overlap stats
    # ----------------------------------------
    basis_overlap = flux_collect_basis_overlap()
    save_json(basis_overlap, os.path.join(OUTDIR, "basis_overlap_cora.json"))

    # ----------------------------------------
    # 7) Run gate probes and save analysis logs
    # ----------------------------------------
    flux_reset_analysis_logs()
    flux_set_analysis_logging(True)

    seed_base = 5000
    tags = list(PROMPT_GROUPS.keys())
    for tag_idx, (tag, prompts) in enumerate(PROMPT_GROUPS.items()):
        for i, prompt in enumerate(prompts):
            run_gate_probe(pipe, prompt, prompt_tag=tag, seed=seed_base + 100 * tag_idx + i)

    flux_set_analysis_logging(False)

    analysis_logs = flux_get_analysis_logs()
    save_json(analysis_logs, os.path.join(OUTDIR, "gate_stats_cora.json"))

    print(f"Saved basis overlap to: {os.path.join(OUTDIR, 'basis_overlap_cora.json')}")
    print(f"Saved gate stats to:   {os.path.join(OUTDIR, 'gate_stats_cora.json')}")
    print(f"Done. Results saved to: {OUTDIR}")


if __name__ == "__main__":
    main()