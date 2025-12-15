import torch
from diffusers import WanImageToVideoPipeline, WanTransformer3DModel, AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image
from transformers import BitsAndBytesConfig

# 1. Setup 4-bit Quantization Config
# This config is specific to the "BitsAndBytes" library, which compresses the model.
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
print("1. Loading VAE (Float32 for quality)...")
vae = AutoencoderKLWan.from_pretrained(
    base_model_id, 
    subfolder="vae", 
    torch_dtype=torch.float32
)

print("2. Loading Transformer (4-bit Quantized)...")
# We load the "transformer" component specifically.
## This bypasses the Pipeline error and applies 4-bit compression to the heaviest part of the model.
transformer = WanTransformer3DModel.from_pretrained(
    base_model_id,
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16
)
print("3. Assembling Pipeline...")
# We pass our pre-loaded, quantized transformer into the pipeline.
pipe = WanImageToVideoPipeline.from_pretrained(
    base_model_id,
    transformer=transformer,
    vae=vae,
    torch_dtype=torch.bfloat16
)

# 4. Load LoRA Adapters
# We use a try/except block to handle potential file name changes in the repo.
print("4. Loading LoRAs...")
try:
    # LightX2V Acceleration (Reduces steps from 50 -> 4)
    # Repo: https://huggingface.co/lightx2v/Wan2.1-Distill-Loras
    pipe.load_lora_weights(
        "lightx2v/Wan2.1-Distill-Loras", 
        weight_name="Wan2.1_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
        adapter_name="acceleration"
    )
    
    # Style LoRA (Example: Flat Color)
    pipe.load_lora_weights(
        "motimalu/wan-flat-color-v2", 
        adapter_name="style"
    )
    
    # Fuse: 1.0 for speed, 0.8 for style
    pipe.set_adapters(["acceleration", "style"], adapter_weights=[1.0, 0.8])
    print("   -> LoRAs loaded successfully.")

except Exception as e:
    print(f"   -> Warning: LoRA load failed ({e}). Proceeding with standard model (slow).")

# 5. Configure Scheduler for Distillation
# The distilled model requires specific scheduler settings (Shift=5.0)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.config.shift = 5.0 

# 6. Memory Optimization (CRITICAL for 16GB VRAM)
# This keeps the T5 Text Encoder in CPU RAM and only puts the Transformer on GPU
pipe.enable_model_cpu_offload()
# pipe.enable_vae_tiling() # Optional: Enable if you hit OOM during the final video decode

# 7. Define Inputs
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k, flat color style"
negative_prompt = "bright colors, 3d render, realistic, photo, noise, grainy"
img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
image = load_image(img_url)

print("8. Generating Video (4 steps)...")

# 9. Run Inference
output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_inference_steps=4,  # MUST be 4 for this LoRA
        guidance_scale=1.0,     # MUST be 1.0 for distilled models
        num_frames=81,          # 81 frames = ~5 seconds
        height=480,
        width=832,
    ).frames[0]  # Extract first batch of frames

# 10. Save Video
output_filename = "wan_distilled.mp4"
export_to_video(output, output_filename, fps=16)
print(f"Done! Saved to {output_filename}")