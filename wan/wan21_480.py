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
print("\n3. Assembling Pipeline...")
# We pass our pre-loaded, quantized transformer into the pipeline.
pipe = WanImageToVideoPipeline.from_pretrained(
    base_model_id,
    transformer=transformer,
    vae=vae,
    torch_dtype=torch.bfloat16
)
print("The warning about CLIPVisionModelWithProjection vs CLIPVisionModel is a known diffusers issue that doesn't affect functionality. It's just a strict type check in the pipeline. You can safely ignore it or suppress it with a warning filter.")

# 4. Load LoRA Adapters
# We use a try/except block to handle potential file name changes in the repo.
print("4. Loading LoRAs...")
try:
    # Style LoRA
    try:
        pipe.load_lora_weights(
            #"motimalu/wan-flat-color-v2",
            "NSFW-API/NSFW-Wan-14b-Cumshot-Facials",
            weight_name = "nsfw_wan_14b_cumshot_facials.safetensors", 
            adapter_name="style"
        )
    except Exception as e:
        print(f"   -> Warning: NSFW-API/NSFW-Wan-14b-Cumshot-FacialsStyle LoRA load failed ({e}). Proceeding without style LoRA.")
    print(" NSFW-API LoRA Loaded.")

    # LightX2V Acceleration (Reduces steps from 50 -> 4)
    # Repo: https://huggingface.co/lightx2v/Wan2.1-Distill-Loras
    try:
        pipe.load_lora_weights(
            "lightx2v/Wan2.1-Distill-Loras", 
            weight_name="wan2.1_i2v_lora_rank64_lightx2v_4step.safetensors",
            adapter_name="acceleration"
        )
    except Exception as e:
        print(f"   -> Warning: lightx2v/Wan2.1-Distill-Loras Acceleration LoRA load failed ({e}). Proceeding without acceleration LoRA.")


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
"""prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k, flat color style"
negative_prompt = "bright colors, 3d render, realistic, photo, noise, grainy"
img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
image = load_image(img_url)"""

prompt = "facial closeup, detailed skin texture, realistic lighting, photorealistic"
negative_prompt = "lowres, bad anatomy, blurry, deformed, ugly, disfigured, poorly drawn, mutation, mutated, extra limbs, cloned face, big head, low quality, jpeg artifacts, ugly, disgusting"
image_path = r"C:\Users\jared\Documents\code\local_jarvis\xserver\aphrodite\apollo\aphrodite_in_apollo-p1_a6_f1_c1.png"
image = load_image(image_path)

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

import torch
import os
import io
from PIL import Image
from diffusers import WanImageToVideoPipeline, WanTransformer3DModel, AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image
from transformers import BitsAndBytesConfig

class Wan21:
    def __init__(self, base_model_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", lora_dict=None):
        """
        Initializes the Wan2.1 Model with 4-bit quantization and optional LoRAs.
        
        :param base_model_id: HuggingFace model ID.
        :param lora_dict: Dictionary of LoRAs to load. 
                          Format: { "adapter_name": {"repo": str, "file": str, "weight": float} }
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model_id = base_model_id
        
        print(f"[Wan21] Initializing model: {base_model_id}")
        
        # 1. Setup Quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # 2. Load VAE
        print("[Wan21] Loading VAE...")
        self.vae = AutoencoderKLWan.from_pretrained(
            base_model_id, 
            subfolder="vae", 
            torch_dtype=torch.float32
        )

        # 3. Load Transformer (Quantized)
        print("[Wan21] Loading Transformer (4-bit)...")
        self.transformer = WanTransformer3DModel.from_pretrained(
            base_model_id,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16
        )


        # 4. Assemble Pipeline
        print("[Wan21] Assembling Pipeline...")
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            base_model_id,
            transformer=self.transformer,
            vae=self.vae,
            torch_dtype=torch.bfloat16
        )

        print("The warning about CLIPVisionModelWithProjection vs CLIPVisionModel is a known diffusers issue that doesn't affect functionality. It's just a strict type check in the pipeline. You can safely ignore it or suppress it with a warning filter.")


        # 5. Load LoRAs if provided
        self.active_adapters = []
        self.adapter_weights = []
        
        if lora_dict:
            self._load_loras(lora_dict)

        # 6. Configure Scheduler (Optimized for Distillation/LightX2V)
        # Note: If not using the distillation LoRA, you might want to remove the shift=5.0
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler.config.shift = 5.0 

        # 7. Memory Optimization
        self.pipe.enable_model_cpu_offload()
        # self.pipe.enable_vae_tiling() # Uncomment if VRAM issues during decode
        
        print("[Wan21] Model Ready.")

    def _load_loras(self, lora_dict):
        print(f"[Wan21] Loading {len(lora_dict)} LoRAs...")
        try:
            for name, details in lora_dict.items():
                print(f"  -> Loading adapter: {name}")
                self.pipe.load_lora_weights(
                    details["repo"],
                    weight_name=details["file"],
                    adapter_name=name
                )
                self.active_adapters.append(name)
                self.adapter_weights.append(details["weight"])

            if self.active_adapters:
                self.pipe.set_adapters(self.active_adapters, adapter_weights=self.adapter_weights)
                print("  -> LoRAs fused successfully.")
                
        except Exception as e:
            print(f"  -> [Error] LoRA load failed: {e}. Proceeding with base model.")

    def generate(self, prompt, negative_prompt, image_input, output_filename="output.mp4"):
        """
        Generates a video from text and image.
        
        :param prompt: Text prompt.
        :param negative_prompt: Negative text prompt.
        :param image_input: Can be a URL string, local path string, or BytesIO/Blob.
        :return: Path to the saved video file.
        """
        print(f"\n[Wan21] Generating for: '{prompt[:50]}...'")
        
        # Handle Image Input
        source_image = None
        if isinstance(image_input, str):
            # Handles URLs and local paths
            source_image = load_image(image_input)
        elif isinstance(image_input, (bytes, bytearray, io.BytesIO)):
            # Handles Blobs/Bytes
            if isinstance(image_input, (bytes, bytearray)):
                image_input = io.BytesIO(image_input)
            source_image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            source_image = image_input
        else:
            raise ValueError("Unsupported image format. Provide URL, Path, Bytes, or PIL Image.")

        # Run Inference
        # Defaults set for the LightX2V Distilled workflow (4 steps, guidance 1.0)
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=source_image,
            num_inference_steps=4,      # Optimized for Distilled LoRA
            guidance_scale=1.0,         # Optimized for Distilled LoRA
            num_frames=81,              # ~5 seconds
            height=480,
            width=832,
        ).frames[0]

        export_to_video(output, output_filename, fps=16)
        print(f"[Wan21] Saved to {output_filename}")
        return output_filename

# ==========================================
# Main Execution Loop
# ==========================================
if __name__ == "__main__":
    print("\n=== Wan2.1 Interactive Video Generation ===\n")
    # Define LoRAs to load
    my_loras = {
        "acceleration": {
            "repo": "lightx2v/Wan2.1-Distill-Loras",
            "file": "wan2.1_i2v_lora_rank64_lightx2v_4step.safetensors",
            "weight": 1.0
        },
        "style": {
            "repo": "NSFW-API/NSFW-Wan-14b-Cumshot-Facials",
            "file": "nsfw_wan_14b_cumshot_facials.safetensors", # Some repos don't need a specific weight file name if it's the default
            "weight": 0.8
        }
    }

    # Initialize the generator (This loads the model only ONCE)
    print("--- System Startup ---")
    try:
        generator = Wan21(lora_dict=my_loras)
    except Exception as e:
        print(f"Initialization Failed: {e}")
        exit()

    print("\n--- Interactive Mode ---")
    print("Press Ctrl+C to exit.")

    counter = 1

    print("\n\n\nPROMPT: facial closeup, detailed skin texture, realistic lighting, photorealistic")
    print("\nIMAGE PATH: C:\\Users\\jared\\Documents\\code\\local_jarvis\\xserver\\aphrodite\\apollo\\aphrodite_in_apollo-p1_a6_f1_c1.png")

    while True:
        try:
            print(f"\n[Job #{counter}]")
            
            # 1. Get Prompt
            user_prompt = input("Enter Prompt: ").strip()
            if not user_prompt:
                print("Skipping empty prompt.")
                continue
                
            # 2. Get Negative Prompt (Optional)
            user_neg = input("Enter Negative Prompt (or press Enter for default): ").strip()
            if not user_neg:
                user_neg = "lowres, bad anatomy, blurry, deformed, ugly, disfigured, poorly drawn, mutation, mutated, extra limbs, cloned face, big head, low quality, jpeg artifacts, ugly, disgusting, hands"
            
            # 3. Get Image Source
            user_img = input("Enter Image URL or Local Path: ").strip()
            # Default for testing if empty
            if not user_img:
                user_img = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
                print(f"Using default astronaut image: {user_img}")

            # 4. Generate
            filename = f"wan_gen_{counter}.mp4"
            generator.generate(
                prompt=user_prompt,
                negative_prompt=user_neg,
                image_input=user_img,
                output_filename=filename
            )
            
            counter += 1
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred during generation: {e}")