import torch
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

pipe = WanImageToVideoPipeline.from_pretrained(
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers",  # Single model, not MoE
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

image = load_image("./astronaut.jpg")
output = pipe(
    image=image,
    prompt="Cyberpunk city, neon rain, cinematic lighting",
    num_frames=81,
    height=480,
    width=832,
).frames[0]

export_to_video(output, "output.mp4", fps=16)