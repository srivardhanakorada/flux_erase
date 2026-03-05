####################################################################
# single_celeb.py
# Target : Donald Trump
# Retain : Melania Trump, Hillary Clinton, Barack Obama
# Anchor : A generic person
# Seeds : 0 - 24 for every concept
# Num of Steps : 4
# Guidance : 3.5
# Image Dimensions : 768 * 768
# Prompt Templates : 
####  "a photo of {}"
####  "a high-quality portrait photo of {}"
####  "{}, studio portrait, sharp focus"
####  "{}, professional headshot"
####  "close-up photo of {}"
####  "cinematic portrait of {}"
####  "{} photographed in natural light"
####  "detailed facial photo of {}"
####  "{} photographed with DSLR"
####  "realistic photo of {}"
####################################################################

import os
from typing import List, Optional
import torch #type:ignore
from PIL import Image #type:ignore
from diffusers import FluxPipeline
from diffusers.models.transformers.transformer_flux import flux_reset_vt_banks, flux_finalize_cora_bases #type:ignore

### config
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
PROMPT_TEMPLATES = [
    "a photo of {}",
    "a high-quality portrait photo of {}",
    "{}, studio portrait, sharp focus",
    "{}, professional headshot",
    "close-up photo of {}",
    "cinematic portrait of {}",
    "{} photographed in natural light",
    "detailed facial photo of {}",
    "{} photographed with DSLR",
    "realistic photo of {}",
]
RECORDING_TEMPLATES = [
    "a photo of {}",
    "{} photographed with DSLR",
    "Photo of {} in natural light"
]
TARGETS: List[str] = [
    "Donald Trump",
]
RETAINS: List[str] = [
    "Melania Trump",
    "Hillary Clinton",
    "Barack Obama"
]
ANCHOR = "A generic person"
OUTDIR = "results/single_celeb"
DUAL_BLOCKS = list(range(0, 19))
SINGLE_BLOCKS = list(range(0, 38))
STRENGTH_TAU = 0.1
STRENGTH_GAMMA = 2.0
ANCHOR_STRENGTH = 1.0
USE_ANCHORS = True
H, W = 768, 768
STEPS = 4
GUIDANCE = 3.5
N_IMAGES_PER_PROMPT = 1
START_SEED = 0
END_SEED = 24
SEEDS = [i for i in range(START_SEED,END_SEED+1)]
os.makedirs(OUTDIR, exist_ok=True)
###

### util functions
def _save(img: Image.Image, path: str): img.save(path)
def _sanitize(s: str, max_len: int = 80) -> str:
    s = s.strip().replace(" ", "_")
    return "".join(c for c in s if c.isalnum() or c in ("_", "-"))[:max_len]
def _make_prompt(x: str, prompt_template: str) -> str: return prompt_template.format(x)
###

### main functions
@torch.no_grad()
def run_one(
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
        "record_concept": record_concept,  
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
def generate_images(pipe: FluxPipeline, items: List[str], templates: List[str]):
    for item in items:
        before_path = f"{OUTDIR}/{item}/before"
        after_path = f"{OUTDIR}/{item}/after"
        os.makedirs(before_path,exist_ok=True)
        os.makedirs(after_path,exist_ok=True)
        for prompt_template in templates:
            p = _make_prompt(item,prompt_template)
            for s in SEEDS:
                file_name = f"{_sanitize(f'{p}_{s}')}.png"
                base_img = run_one(pipe, p, apply_target_proj=False, seed=s)
                edit_img = run_one(pipe, p, apply_target_proj=True, seed=s)
                _save(base_img,os.path.join(before_path,file_name))
                _save(edit_img,os.path.join(after_path,file_name))
        print(f"{item} DONE!")
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = FluxPipeline.from_pretrained(MODEL_ID,torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,).to(device)
    flux_reset_vt_banks(reset_retain=True)
    for i, rp in enumerate(RETAINS): run_one(pipe,prompt=rp,record_retain_vt=True,seed=1000 + i,)
    for i, t in enumerate(TARGETS):
        for pt in RECORDING_TEMPLATES:
            prompt = pt.format(t)
            run_one(pipe,prompt=prompt,record_target_vt=True,record_concept=t,seed=2000 + i,)
    run_one(pipe,prompt=ANCHOR,record_anchor_once=True,seed=3000,)
    flux_finalize_cora_bases()
    generate_images(pipe, TARGETS,PROMPT_TEMPLATES)
    generate_images(pipe, RETAINS,PROMPT_TEMPLATES)
    print(f"Done. Grids saved to: {OUTDIR}")
if __name__ == "__main__": main()
###
