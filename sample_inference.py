import os
import torch  # type:ignore
from diffusers import FluxPipeline
from PIL import Image  # type:ignore

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
TARGETS = [
    "Donald Trump", "Barack Obama", "Taylor Swift",
    "Pen", "Chair", "Car", "Tiger", "Cat", "Dog"
]
TARGETS = {'Fruit', 'Ball', 'Car', 'Airplane',
'Tower', 'Building', 'Celebrity',
'Shoes', 'Cat', 'Dog'}
PERSON_TARGETS = {"Donald Trump", "Barack Obama", "Taylor Swift"}
N_SAMPLES = 5
BASE_SEED = 0
STEPS = 4
GUIDANCE = 3.5
H, W = 768, 768
OUTDIR = "object_gen"
os.makedirs(OUTDIR, exist_ok=True)
DUAL_BLOCKS = list(range(0, 19))
SINGLE_BLOCKS = list(range(0, 38))
PROJ_STRENGTH = 6.0
PROJ_TOKEN_END = 128
STRENGTH_TAU = 0.1
STRENGTH_GAMMA = 4.0

def target_prompt(concept: str) -> str: return concept

def gen_prompt(concept: str) -> str: return f"a photo of {concept}"

def retain_prompt_for(concept: str) -> str:
    if concept in PERSON_TARGETS: return "a photo of Hillary Clinton" # was Chris Hemsworth initially, changing it since the model doesn't always produce the right image
    return "a photo of Lion"

def safe_name(s: str) -> str: return s.replace(" ", "_").replace("/", "_").replace(":", "_")

def reset_vt_banks_and_concept(pipe: FluxPipeline):
    try:
        proc = pipe.transformer.transformer_blocks[0].attn.processor
        mod_name = proc.__class__.__module__
        mod = __import__(mod_name, fromlist=["*"])
    except Exception as e:
        print("[WARN] Could not reset vt banks automatically:", e)
        return None
    reset_items = {
        "_FLUX_TARGET_VT": None,
        "_FLUX_TARGET_VT_READY": False,
        "_FLUX_TARGET_VT_BANK": {},
        "_FLUX_TARGET_VT_READY_SET": set(),
        "_FLUX_SINGLE_VT_BANK": {},
        "_FLUX_SINGLE_VT_READY_SET": set(),
        "_FLUX_CONCEPT_C": None,
        "_FLUX_PRINT_STRENGTH_COUNT": 0,
    }
    for k, v in reset_items.items():
        if hasattr(mod, k): setattr(mod, k, v)
    print(f"[OK] Reset vt banks + concept in module: {mod_name}")
    return mod

def run(pipe, prompt, seed, kwargs):
    device = pipe._execution_device
    gen = torch.Generator(device=device).manual_seed(int(seed))
    out = pipe(
        prompt=prompt,
        height=H,
        width=W,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=gen,
        disable_clip=False,
        joint_attention_kwargs=kwargs,
    )
    return out.images[0]

def base_kwargs():
    return dict(
        proj_token_end=int(PROJ_TOKEN_END),
        proj_strength=float(PROJ_STRENGTH),
        target_block_indices=DUAL_BLOCKS,
        target_single_block_indices=SINGLE_BLOCKS,
    )

def uniform_kwargs(): return base_kwargs()

def adaptive_kwargs_loose():
    kw = base_kwargs()
    kw["strength_tau"] = float(STRENGTH_TAU)
    kw["strength_gamma"] = float(STRENGTH_GAMMA)
    return kw

def make_grid(rows, pad=10):
    R = len(rows)
    C = len(rows[0])
    w, h = rows[0][0].size
    grid_w = C * w + (C + 1) * pad
    grid_h = R * h + (R + 1) * pad
    grid = Image.new("RGB", (grid_w, grid_h), (18, 18, 18))
    y = pad
    for r in range(R):
        x = pad
        for c in range(C):
            grid.paste(rows[r][c], (x, y))
            x += w + pad
        y += h + pad
    return grid

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    for concept in TARGETS:
        concept_dir = os.path.join(OUTDIR, safe_name(concept))
        os.makedirs(concept_dir, exist_ok=True)
        mod = reset_vt_banks_and_concept(pipe)
        kw = adaptive_kwargs_loose()
        kw.update(record_target_vt=True, apply_target_proj=False)
        _ = run(pipe, target_prompt(concept), BASE_SEED, kw)
        if mod is not None:
            tb = getattr(mod, "_FLUX_TARGET_VT_BANK", {})
            sb = getattr(mod, "_FLUX_SINGLE_VT_BANK", {})
            print(f"[BANKS] target={concept} dual_ready={len(tb)} single_ready={len(sb)}")
        rows_target = []
        for i in range(N_SAMPLES):
            seed = BASE_SEED + i
            prompt = gen_prompt(concept)
            kw = adaptive_kwargs_loose()
            kw.update(record_target_vt=False, apply_target_proj=False)
            img_base = run(pipe, prompt, seed, kw)
            kw = uniform_kwargs()
            kw.update(record_target_vt=False, apply_target_proj=True)
            img_uniform = run(pipe, prompt, seed, kw)
            kw = adaptive_kwargs_loose()
            kw.update(record_target_vt=False, apply_target_proj=True)
            img_adapt = run(pipe, prompt, seed, kw)
            rows_target.append([img_base, img_uniform, img_adapt])
        grid_target = make_grid(rows_target, pad=10)
        out_target = os.path.join(concept_dir, f"TARGET_grid_5x3_seed{BASE_SEED}.png")
        grid_target.save(out_target)
        print("Saved:", out_target)
        print("Legend cols: [baseline | uniform-erase | adaptive-erase(tau=-0.2,gamma=2.0)]")
        retain_prompt = retain_prompt_for(concept)
        rows_retain = []
        for i in range(N_SAMPLES):
            seed = BASE_SEED + i
            prompt = retain_prompt
            kw = adaptive_kwargs_loose()
            kw.update(record_target_vt=False, apply_target_proj=False)
            img_base = run(pipe, prompt, seed, kw)
            kw = uniform_kwargs()
            kw.update(record_target_vt=False, apply_target_proj=True)
            img_uniform = run(pipe, prompt, seed, kw)
            kw = adaptive_kwargs_loose()
            kw.update(record_target_vt=False, apply_target_proj=True)
            img_adapt = run(pipe, prompt, seed, kw)
            rows_retain.append([img_base, img_uniform, img_adapt])
        grid_retain = make_grid(rows_retain, pad=10)
        out_retain = os.path.join(concept_dir, f"RETAIN_grid_5x3_seed{BASE_SEED}.png")
        grid_retain.save(out_retain)
        print("Saved:", out_retain)
        print(f"Retain prompt used: {retain_prompt}")