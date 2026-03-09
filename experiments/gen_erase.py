import os
from typing import List, Optional

import torch  # type: ignore
from PIL import Image  # type: ignore
from diffusers import FluxPipeline
from diffusers.models.transformers.transformer_flux import (  # type: ignore
    flux_reset_vt_banks,
    flux_finalize_cora_bases,
)

import diffusers.models.transformers.transformer_flux as tf

print("USING TRANSFORMER FILE:", tf.__file__)
print("FluxAttnProcessor args:", tf.inspect.signature(tf.FluxAttnProcessor.__call__))

MODEL_ID = "black-forest-labs/FLUX.1-schnell"

# Prompts used for final generation/evaluation
PROMPT_TEMPLATES = [
    "a photo of {}",
]

# Prompts used while recording target/retain concepts
RECORDING_TEMPLATES = [
    "a photo of {}",
    # "{} photographed with DSLR",
    # "{}, studio portrait, sharp focus",
]

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

# NOTE:
# In the exact transformer you pasted, PERSON_BANK is NOT used because
# FluxAttnProcessor does not support record_person_vt.
# So keep this empty unless you later extend the transformer.
PERSON_BANK: List[str] = []

DUAL_BLOCKS = list(range(0, 19))
SINGLE_BLOCKS = list(range(0, 38))

OUTDIR = "temp_generase"

# Main erasure control in your exact current setup
STRENGTH_TAU = 0.10

# Anchor replacement strength (used only if USE_ANCHORS=True)
ANCHOR_STRENGTH = 1.5
USE_ANCHORS = True
ANCHOR = "a portrait of a person"

# These are not used by the exact token-wise GenErase finalize you showed,
# but we keep them for compatibility if your local finalize signature changes later.
PERSON_TOP_K = 2
RETAIN_TOP_K = 4

REC_H, REC_W = 256, 256
GEN_H, GEN_W = 512, 512

STEPS = 4
GUIDANCE = 3.5
N_IMAGES_PER_PROMPT = 1

START_SEED = 0
END_SEED = 0
SEEDS = list(range(START_SEED, END_SEED + 1))

os.makedirs(OUTDIR, exist_ok=True)


def _save(img: Image.Image, path: str):
    img.save(path)


def _sanitize(s: str, max_len: int = 160) -> str:
    s = s.strip().replace(" ", "_")
    return "".join(c for c in s if c.isalnum() or c in ("_", "-"))[:max_len]


def _make_prompt(x: str, prompt_template: str) -> str:
    return prompt_template.format(x)


def _maybe_clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def run_one(
    pipe: FluxPipeline,
    prompt: str,
    *,
    device: torch.device,
    record_target_vt: bool = False,
    record_retain_vt: bool = False,
    record_anchor_vt: bool = False,
    apply_target_proj: bool = False,
    record_concept: Optional[str] = None,
    seed: int = 0,
    record_mode: bool = False,
):
    g = torch.Generator(device=device).manual_seed(seed)

    height = REC_H if record_mode else GEN_H
    width = REC_W if record_mode else GEN_W
    output_type = "latent" if record_mode else "pil"

    ja = {
        "record_target_vt": record_target_vt,
        "record_retain_vt": record_retain_vt,
        "record_anchor_vt": record_anchor_vt,
        "record_concept": record_concept,
        "apply_target_proj": apply_target_proj,
        "use_anchors": USE_ANCHORS,
        "target_block_indices": DUAL_BLOCKS,
        "target_single_block_indices": SINGLE_BLOCKS,
        "strength_tau": STRENGTH_TAU,
        "anchor_strength": ANCHOR_STRENGTH,
        "proj_eps": 1e-8,
        "max_target_vt_per_block": 8,
        "max_retain_vt_per_block": 8,
        "max_anchor_vt_per_block": 4,
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


def record_retain_bank(pipe: FluxPipeline, device: torch.device):
    """
    Record retains using the SAME template diversity as targets.
    This is important for better preservation.
    """
    for i, retain in enumerate(RETAINS):
        for j, template in enumerate(RECORDING_TEMPLATES):
            prompt = template.format(retain)
            run_one(
                pipe,
                prompt=prompt,
                device=device,
                record_retain_vt=True,
                record_concept=retain,
                seed=1000 + 100 * i + j,
                record_mode=True,
            )


def record_target_bank(pipe: FluxPipeline, device: torch.device):
    for i, target in enumerate(TARGETS):
        for j, template in enumerate(RECORDING_TEMPLATES):
            prompt = template.format(target)
            run_one(
                pipe,
                prompt=prompt,
                device=device,
                record_target_vt=True,
                record_concept=target,
                seed=3000 + 100 * i + j,
                record_mode=True,
            )


def record_anchor_bank(pipe: FluxPipeline, device: torch.device):
    if not USE_ANCHORS:
        return

    # One anchor recording per target concept label
    for i, target in enumerate(TARGETS):
        run_one(
            pipe,
            prompt=ANCHOR,
            device=device,
            record_anchor_vt=True,
            record_concept=target,
            seed=4000 + i,
            record_mode=True,
        )


def generate_images(pipe: FluxPipeline, device: torch.device, items: List[str], templates: List[str], split_name: str):
    for item in items:
        before_path = os.path.join(OUTDIR, split_name, item, "before")
        after_path = os.path.join(OUTDIR, split_name, item, "after")
        os.makedirs(before_path, exist_ok=True)
        os.makedirs(after_path, exist_ok=True)

        for prompt_template in templates:
            prompt = _make_prompt(item, prompt_template)
            for s in SEEDS:
                file_name = f"{_sanitize(f'{prompt}_{s}')}.png"

                base_img = run_one(
                    pipe,
                    prompt,
                    device=device,
                    apply_target_proj=False,
                    seed=s,
                    record_mode=False,
                )
                edit_img = run_one(
                    pipe,
                    prompt,
                    device=device,
                    apply_target_proj=True,
                    seed=s,
                    record_mode=False,
                )

                _save(base_img, os.path.join(before_path, file_name))
                _save(edit_img, os.path.join(after_path, file_name))

        print(f"{split_name} :: {item} DONE!")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    ).to(device)

    flux_reset_vt_banks(reset_retain=True)
    _maybe_clear_cache()

    # 1) Record retain bank
    record_retain_bank(pipe, device)
    _maybe_clear_cache()

    # 2) PERSON_BANK is not used in the exact transformer you pasted.
    #    So we intentionally skip it here.

    # 3) Record target bank
    record_target_bank(pipe, device)
    _maybe_clear_cache()

    # 4) Record anchor bank
    record_anchor_bank(pipe, device)
    _maybe_clear_cache()

    # 5) Finalize token-wise bases/projectors
    try:
        # For older/custom variants
        flux_finalize_cora_bases(
            retain_top_k=RETAIN_TOP_K,
            person_top_k=PERSON_TOP_K,
        )
    except TypeError:
        # For the exact GenErase-style transformer you pasted
        flux_finalize_cora_bases()

    _maybe_clear_cache()

    # 6) Generate
    generate_images(pipe, device, TARGETS, PROMPT_TEMPLATES, split_name="targets")
    generate_images(pipe, device, RETAINS, PROMPT_TEMPLATES, split_name="retains")
    generate_images(pipe, device, NON_TARGETS, PROMPT_TEMPLATES, split_name="non_targets")

    print(f"Done. Results saved to: {OUTDIR}")


if __name__ == "__main__":
    main()