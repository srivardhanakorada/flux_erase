import torch
import clip
from diffusers import FluxPipeline

import torch.nn.functional as F
MODEL_ID = "black-forest-labs/FLUX.1-schnell"

pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = pipe.to(device)
pipe.set_progress_bar_config(disable=False)

text1 = "dog"
text2 = "a photo of a lion"

def norm_clip(vec):
    return vec/(vec.norm(dim=-1,keepdim=True)+1e-8)


emb = pipe._get_clip_prompt_embeds([text1,text2])
# print(emb[0].norm(dim=-1,keepdim=True))
# print(emb[1].norm(dim=-1,keepdim=True))
emb = [norm_clip(emb[0]),norm_clip(emb[1])]
# print(emb[0].norm(dim=-1,keepdim=True))
# print(emb[1].norm(dim=-1,keepdim=True))
similarity = (emb[0] @ emb[1]).item()
print(f"Similarity with Flux based encoding: {similarity:.4f}")

# model, preprocess = clip.load("ViT-B/32", device="cuda")

# tokens = clip.tokenize([text1, text2]).to("cuda")
# with torch.no_grad():
#     emb = model.encode_text(tokens)
#     # print(emb[0].norm(dim=-1,keepdim=True))
#     # print(emb[1].norm(dim=-1,keepdim=True))
#     emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize
#     # print(emb[0].norm(dim=-1,keepdim=True))
#     # print(emb[1].norm(dim=-1,keepdim=True))    

# similarity = (emb[0] @ emb[1]).item()
# print(f"Similarity from clip method: {similarity:.4f}")