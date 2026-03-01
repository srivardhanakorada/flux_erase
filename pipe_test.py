import os
import torch
from diffusers import FluxPipeline

MODEL_ID = "black-forest-labs/FLUX.1-schnell"

def get_pipeline():
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16
    ).to("cuda")
    return pipe

@torch.no_grad()
def run_and_save(
    pipe: FluxPipeline,
    prompt: str,
    *,
    seed: int = 0,
    output_dir: str = "outputs"
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=0.0, 
        num_images_per_prompt=5,
        generator=g,
    )
    for i, img in enumerate(out.images):
        file_path = os.path.join(output_dir, f"result_{i}.png")
        img.save(file_path)
        print(f"Saved: {file_path}")

pipeline = get_pipeline()
run_and_save(pipeline, "Hugh Jackman")