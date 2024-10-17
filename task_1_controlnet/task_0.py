import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

# prompt = "pale golden rod circle with old lace background"
# image = pipe(prompt).images[0]
# image.save("pale_golden_circle.png")
prompt = "sea green circle with a light cyan background"
image = pipe(prompt).images[0]
image.save("2.png")
prompt = "deep sky blue circle with a light yellow background"
image = pipe(prompt).images[0]
image.save("3.png")
prompt = "rosy brown circle with a misty rose background"
image = pipe(prompt).images[0]
image.save("4.png")
prompt = "forest green circle with an antique brown background"
image = pipe(prompt).images[0]
image.save("5.png")
