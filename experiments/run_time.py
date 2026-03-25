import time
from typing import List, Tuple

import torch  # type: ignore
from diffusers import FluxPipeline  # type: ignore
from diffusers.models.transformers.transformer_flux import (  # type: ignore
    flux_reset_vt_banks,
    flux_finalize_cora_bases,
)

# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

TARGETS: List[str] = ["Donald Trump"]
RETAINS: List[str] = ["Hillary Clinton", "Melania Trump", "Barack Obama"]

PERSON_BANK = [
    "a portrait of a person",
    "a portrait of a man",
    "a portrait of a woman",
    "a middle-aged man",
    "a middle-aged woman",
]

DUAL_BLOCKS = list(range(0, 19))
SINGLE_BLOCKS = list(range(0, 38))

STRENGTH_TAU = 0.1
STRENGTH_GAMMA = 1.5
ANCHOR_STRENGTH = 1.0
USE_ANCHORS = False
ANCHOR = "a portrait of a person"
PERSON_TOP_K = 2
RETAIN_TOP_K = 4

REC_H, REC_W = 768, 768
GEN_H, GEN_W = 768, 768
STEPS = 4
GUIDANCE = 3.5
BASE_SEED = 0

# Runtime table says 100 image generations per concept
N_IMAGES_PER_CONCEPT = 100

PROMPT_TEMPLATES = [
    "a photo of {}",
]

RECORDING_TEMPLATES = [
    "a photo of {}",
    "{}, studio portrait, sharp focus",
]

# ============================================================
# HELPERS
# ============================================================

def sync_if_needed():
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def make_prompts(concept: str, n: int, templates: List[str]) -> List[str]:
    return [templates[i % len(templates)].format(concept) for i in range(n)]


def timed_generate_one(
    pipe: FluxPipeline,
    prompt: str,
    *,
    apply_target_proj: bool,
    seed: int,
) -> float:
    g = torch.Generator(device=pipe.device).manual_seed(seed)

    joint_attention_kwargs = {
        "record_target_vt": False,
        "record_retain_vt": False,
        "record_person_vt": False,
        "record_anchor_once": False,
        "record_concept": None,
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

    sync_if_needed()
    t0 = time.perf_counter()

    _ = pipe(
        prompt=prompt,
        height=GEN_H,
        width=GEN_W,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        num_images_per_prompt=1,
        generator=g,
        joint_attention_kwargs=joint_attention_kwargs,
        output_type="pil",
    ).images[0]

    sync_if_needed()
    t1 = time.perf_counter()
    return t1 - t0


def warmup_pipeline(pipe: FluxPipeline):
    g = torch.Generator(device=pipe.device).manual_seed(12345)
    _ = pipe(
        prompt="a portrait photo",
        height=GEN_H,
        width=GEN_W,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        num_images_per_prompt=1,
        generator=g,
        output_type="pil",
    ).images[0]
    sync_if_needed()


# ============================================================
# PIPELINE LOADING
# ============================================================

def load_base_pipeline() -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    return pipe


# ============================================================
# RECORDING / SETUP
# ============================================================

@torch.no_grad()
def run_one_record(
    pipe: FluxPipeline,
    prompt: str,
    *,
    record_target_vt: bool = False,
    record_retain_vt: bool = False,
    record_person_vt: bool = False,
    record_anchor_once: bool = False,
    record_concept: str | None = None,
    seed: int = 0,
):
    g = torch.Generator(device=pipe.device).manual_seed(seed)

    joint_attention_kwargs = {
        "record_target_vt": record_target_vt,
        "record_retain_vt": record_retain_vt,
        "record_person_vt": record_person_vt,
        "record_anchor_once": record_anchor_once,
        "record_concept": record_concept,
        "apply_target_proj": False,
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
        num_images_per_prompt=1,
        generator=g,
        joint_attention_kwargs=joint_attention_kwargs,
        output_type="latent",
    )


def setup_ours_intervention(
    pipe: FluxPipeline,
    targets: List[str],
    retains: List[str],
) -> None:
    # Matches your working script API
    flux_reset_vt_banks(reset_retain=True)

    # retain banks
    for i, rp in enumerate(retains):
        run_one_record(
            pipe,
            prompt=rp,
            record_retain_vt=True,
            seed=1000 + i,
        )

    # generic person bank
    for i, pp in enumerate(PERSON_BANK):
        run_one_record(
            pipe,
            prompt=pp,
            record_person_vt=True,
            seed=2000 + i,
        )

    # target banks
    for i, t in enumerate(targets):
        for j, pt in enumerate(RECORDING_TEMPLATES):
            run_one_record(
                pipe,
                prompt=pt.format(t),
                record_target_vt=True,
                record_concept=t,
                seed=3000 + 100 * i + j,
            )

    if USE_ANCHORS:
        run_one_record(
            pipe,
            prompt=ANCHOR,
            record_anchor_once=True,
            seed=4000,
        )

    # IMPORTANT: no positional transformer argument here
    flux_finalize_cora_bases(
        retain_top_k=RETAIN_TOP_K,
        person_top_k=PERSON_TOP_K,
    )

    sync_if_needed()


# ============================================================
# BENCHMARK
# ============================================================

def benchmark_pipe(
    pipe: FluxPipeline,
    concepts: List[str],
    *,
    apply_target_proj: bool,
) -> Tuple[float, List[float]]:
    times: List[float] = []
    seed = BASE_SEED

    for concept in concepts:
        prompts = make_prompts(concept, N_IMAGES_PER_CONCEPT, PROMPT_TEMPLATES)
        for prompt in prompts:
            dt = timed_generate_one(
                pipe,
                prompt,
                apply_target_proj=apply_target_proj,
                seed=seed,
            )
            times.append(dt)
            seed += 1

    return mean(times), times


# ============================================================
# MAIN
# ============================================================

def main():
    concepts = TARGETS + RETAINS

    # print("==================================================")
    # print("Measuring BASE FLUX inference runtime")
    # print("==================================================")
    # base_pipe = load_base_pipeline()
    # warmup_pipeline(base_pipe)
    # base_mean_sec, _ = benchmark_pipe(
    #     base_pipe,
    #     concepts,
    #     apply_target_proj=False,
    # )
    # print(f"Base mean inference cost: {base_mean_sec:.4f} s / image")

    print("\n==================================================")
    print("Measuring OURS setup + inference runtime")
    print("==================================================")
    ours_pipe = load_base_pipeline()
    warmup_pipeline(ours_pipe)

    sync_if_needed()
    t0 = time.perf_counter()
    setup_ours_intervention(ours_pipe, TARGETS, RETAINS)
    sync_if_needed()
    t1 = time.perf_counter()
    setup_sec = t1 - t0

    ours_mean_sec, _ = benchmark_pipe(
        ours_pipe,
        concepts,
        apply_target_proj=True,
    )
    base_mean_sec = 3.6256
    overhead_pct = 100.0 * (ours_mean_sec - base_mean_sec) / base_mean_sec

    print("\n================ FINAL NUMBERS ================")
    print(f"Setup Cost     : {setup_sec:.2f} s")
    print(f"Inference Cost : {ours_mean_sec:.2f} s / image")
    print(f"Base Cost      : {base_mean_sec:.2f} s / image")
    print(f"Overhead       : {overhead_pct:.2f}%")
    print("===============================================")

    latex_row = (
        f"Ours & {setup_sec:.2f} s & "
        f"{ours_mean_sec:.2f} s / image & "
        f"{overhead_pct:.2f}\\% \\\\"
    )

    print("\nPaste this into your table:")
    print(latex_row)


if __name__ == "__main__":
    main()