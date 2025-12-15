import os
import torch
import gc
from huggingface_hub import hf_hub_download
from diffusers import WanImageToVideoPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import load_image, export_to_video
from transformers import CLIPVisionModel, CLIPImageProcessor
import numpy as np

# --- Configuration ---
MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"#https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers
LORA_REPO = "lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v" #https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v/tree/main
LORA_FILENAME = "loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"#https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v/tree/main/loras

class Wan14B_Workstation:
    def __init__(self, use_quantization=True):
        self.model_id = MODEL_ID
        self.pipeline = None
        self.use_quantization = use_quantization
        print(f">> Initializing Wan 2.2 14B Workstation ({'4-bit Quantized' if use_quantization else 'bfloat16'})...")
        self._load_pipeline()
    
    def _load_pipeline(self):
        """
        Load the Wan2.2 pipeline with optional 4-bit quantization using PipelineQuantizationConfig.
        Properly distributes model across both GPUs.
        """
        print(">> Loading Wan 2.2 I2V Pipeline...")
        
        # Load Image Encoder and Processor explicitly
        print(">> Loading Image Encoder (CLIP-ViT-H-14)...")
        # Loads image_encoder and image_processor from Wan2.1
        image_encoder = CLIPVisionModel.from_pretrained(
            "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
            subfolder="image_encoder",
            torch_dtype=torch.float32
        )

        image_processor = CLIPImageProcessor.from_pretrained(
            "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
            subfolder="image_processor"
        )

        
        if self.use_quantization:
            # Create PipelineQuantizationConfig for diffusers pipelines
            # This is the CORRECT way for diffusers models (not BitsAndBytesConfig)
            pipeline_quant_config = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                    "bnb_4bit_use_double_quant": True,
                },
                # Quantize both MoE transformers (high-noise and low-noise experts)
                components_to_quantize=["transformer", "transformer_2"]
            )
            
            print(">> Loading with 4-bit quantization (PipelineQuantizationConfig)...")
            self.pipeline = WanImageToVideoPipeline.from_pretrained(
                self.model_id,
                image_encoder=image_encoder,
                image_processor=image_processor,
                torch_dtype=torch.bfloat16,
                quantization_config=pipeline_quant_config,
                #device_map="balanced"  # Distribute across both GPUs
            )
        else:
            # Load without quantization using balanced distribution
            print(">> Loading with bfloat16 precision across GPUs...")
            self.pipeline = WanImageToVideoPipeline.from_pretrained(
                self.model_id,
                image_encoder=image_encoder,
                image_processor=image_processor,
                torch_dtype=torch.bfloat16,
                #device_map="balanced"
            )
        
        # Enable VAE tiling for memory efficiency
        #self.pipeline.enable_vae_tiling()
        
        # Try to enable memory-efficient attention
        try:
            self.pipeline.enable_xformers_memory_efficient_attention()
            print(">> ✓ Enabled xformers memory efficient attention")
        except Exception as e:
            print(f">> xformers not available (this is OK): {e}")

        print(f">> Downloading and loading LoRA...")
        lora_path = hf_hub_download(
            repo_id=LORA_REPO,
            filename=LORA_FILENAME
        )
        
        # Load LoRA weights
        self.pipeline.load_lora_weights(lora_path, adapter_name="lightx2v")
        self.pipeline.set_adapters(["lightx2v"], adapter_weights=[1.0])
        print(">> ✓ LoRA loaded successfully")
        
        # Set Scheduler
        self.pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipeline.scheduler.config
        )
        
        print(">> ✓ Pipeline loaded successfully!")
        self._print_device_map()

    def _print_device_map(self):
        """Print which devices components are loaded on"""
        print("\n" + "="*60)
        print("DEVICE ALLOCATION")
        print("="*60)
        
        if hasattr(self.pipeline, 'hf_device_map'):
            print("\nDetailed component mapping:")
            device_counts = {}
            for name, device in self.pipeline.hf_device_map.items():
                device_str = str(device)
                device_counts[device_str] = device_counts.get(device_str, 0) + 1
                # Only print first few and last few to avoid spam
                if len(self.pipeline.hf_device_map) > 20:
                    continue  # Skip detailed printing for large models
                print(f"  {name}: {device}")
            
            print("\nDevice summary:")
            for device, count in sorted(device_counts.items()):
                print(f"  {device}: {count} components")
        
        # Check specific major components
        print("\nMajor components:")
        components = {
            'transformer': 'High-noise Expert (MoE)',
            'transformer_2': 'Low-noise Expert (MoE)', 
            'text_encoder': 'Text Encoder',
            'vae': 'VAE'
        }
        
        for comp_name, description in components.items():
            if hasattr(self.pipeline, comp_name):
                comp = getattr(self.pipeline, comp_name)
                if comp is None:
                    continue
                if hasattr(comp, 'device'):
                    print(f"  {comp_name} ({description}): {comp.device}")
                elif hasattr(comp, 'hf_device_map'):
                    devices = set(comp.hf_device_map.values())
                    print(f"  {comp_name} ({description}): distributed across {devices}")
        print("="*60)

    def generate(self, image_path, prompt, output_path="output.mp4", seed=None,
                 height=480, width=854, num_frames=81, num_inference_steps=6, guidance_scale=1.0):
        """
        Generate video from image using the Wan2.2 pipeline.
        """
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        # For multi-GPU setups, use CPU generator to avoid conflicts
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        # Load and prepare image
        image = load_image(image_path)
        
        # Calculate optimal dimensions (following official example)
        max_area = height * width
        aspect_ratio = image.height / image.width
        mod_value = self.pipeline.vae_scale_factor_spatial * self.pipeline.transformer.config.patch_size[1]
        
        calculated_height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        calculated_width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        
        image = image.resize((calculated_width, calculated_height))
        
        print(f"\n{'='*60}")
        print("GENERATION SETTINGS")
        print(f"{'='*60}")
        print(f"Seed: {seed}")
        print(f"Resolution: {calculated_width}x{calculated_height}")
        print(f"Frames: {num_frames}")
        print(f"Inference Steps: {num_inference_steps}")
        print(f"Guidance Scale: {guidance_scale}")
        print(f"{'='*60}\n")
        
        # Print GPU memory before generation
        self.print_gpu_memory("BEFORE GENERATION")
        
        print(">> Generating video...")
        with torch.no_grad():
            video_frames = self.pipeline(
                image=image,
                prompt=prompt,
                negative_prompt="low quality, distortion, morphing, jitter, watermark, static, blurry details",
                height=calculated_height,
                width=calculated_width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).frames[0]

        export_to_video(video_frames, output_path, fps=16)
        print(f"\n✅ Video saved to {output_path}")
        
        # Print GPU memory after generation
        self.print_gpu_memory("AFTER GENERATION")
        
        self._clear_memory()
        return output_path

    def _clear_memory(self):
        """Clear GPU memory cache"""
        gc.collect()
        torch.cuda.empty_cache()
        
    def print_gpu_memory(self, label="GPU MEMORY STATUS"):
        """Print current GPU memory usage for all available GPUs"""
        print(f"\n{'='*60}")
        print(label)
        print(f"{'='*60}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Allocated: {allocated:>6.2f} GB / {total:.2f} GB ({allocated/total*100:>5.1f}%)")
                print(f"  Reserved:  {reserved:>6.2f} GB / {total:.2f} GB ({reserved/total*100:>5.1f}%)")
                print(f"  Free:      {total-allocated:>6.2f} GB")
        else:
            print("⚠ CUDA not available!")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    # System info
    print("="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  CUDA Capability: {props.major}.{props.minor}")
    print("="*60 + "\n")
    
    # Initialize workstation
    # Set use_quantization=True for 4-bit, False for bfloat16
    USE_4BIT = True  # ← Change this to False to disable quantization
    
    bot = Wan14B_Workstation(use_quantization=USE_4BIT)
    
    # Check memory after loading
    bot.print_gpu_memory("AFTER MODEL LOADING")
    
    # Generate video
    img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
    
    bot.generate(
        image_path=img_url,
        prompt="Cinematic, huge red mars mountains background, astronaut walking forward, camera pan right",
        output_path="mars_wan_output.mp4",
        seed=42,
        height=480,
        width=854,
        num_frames=81,
        num_inference_steps=6,
        guidance_scale=1.0
    )
    
    # Final cleanup
    print("\n>> Cleaning up memory...")
    gc.collect()
    torch.cuda.empty_cache()
    print("✅ Done!\n")