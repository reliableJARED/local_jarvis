"""
Wan 2.1 I2V - EMERGENCY WORKAROUND FOR SHARD LOADING CRASH (FIXED)

This version uses device_map="auto" with explicit memory limits to work around
the checkpoint shard loading crash at 86% (shard 12/14).

FIX: Removed conflicting enable_model_cpu_offload() call that caused meta tensor error.
"""

import torch
from diffusers import WanImageToVideoPipeline, WanTransformer3DModel, AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image
from safetensors.torch import load_file, save_file
from huggingface_hub import hf_hub_download
import gc
import os
import psutil
import time
from pathlib import Path
import tempfile

# Set memory management - critical for 16GB VRAM
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

def get_memory_stats():
    """Get current memory usage statistics"""
    stats = {}
    ram = psutil.virtual_memory()
    stats['ram_total_gb'] = ram.total / (1024**3)
    stats['ram_available_gb'] = ram.available / (1024**3)
    stats['ram_used_gb'] = ram.used / (1024**3)
    stats['ram_percent'] = ram.percent
    
    if torch.cuda.is_available():
        stats['vram_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
        stats['vram_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
        stats['vram_total_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return stats

def print_memory_stats(label=""):
    """Print formatted memory statistics"""
    stats = get_memory_stats()
    print(f"\n{'='*60}")
    print(f"Memory: {label}")
    print(f"{'='*60}")
    print(f"RAM:  {stats['ram_used_gb']:.2f}GB / {stats['ram_total_gb']:.2f}GB ({stats['ram_percent']:.1f}%)")
    print(f"      Available: {stats['ram_available_gb']:.2f}GB")
    if 'vram_allocated_gb' in stats:
        print(f"VRAM: {stats['vram_allocated_gb']:.2f}GB / {stats['vram_total_gb']:.2f}GB")
    print(f"{'='*60}\n")

def clear_memory():
    """Aggressively clear GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def main():
    print("=" * 70)
    print("Wan 2.1 I2V - EMERGENCY WORKAROUND (FIXED)")
    print("Using device_map='auto' to bypass shard loading crash")
    print("=" * 70)
    
    print_memory_stats("Initial")
    
    # Configuration
    BASE_MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    IMAGE_PATH = r"C:\Users\jared\Documents\code\local_jarvis\xserver\aphrodite\apollo\aphrodite_in_apollo-p1_a6_f1_c1.png"
    OUTPUT_FILE = "wan_output.mp4"
    
    NUM_FRAMES = 49
    HEIGHT = 480
    WIDTH = 832
    
    PROMPT = "facial closeup, detailed skin texture, realistic lighting, photorealistic"
    NEGATIVE_PROMPT = "lowres, bad anatomy, blurry, deformed, ugly, static"
    
    # =========================================================================
    # CRITICAL WORKAROUND: Use device_map="auto" with memory constraints
    # =========================================================================
    
    print("\n" + "="*70)
    print("STEP 1: Loading VAE (normal loading)")
    print("="*70)
    print("VAE doesn't support device_map, using standard loading")
    
    start_time = time.time()
    
    try:
        # VAE doesn't support device_map, load normally
        vae = AutoencoderKLWan.from_pretrained(
            BASE_MODEL_ID, 
            subfolder="vae", 
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        elapsed = time.time() - start_time
        print(f"✓ VAE loaded in {elapsed:.1f}s")
        print_memory_stats("After VAE")
        
    except Exception as e:
        print(f"✗ VAE failed: {e}")
        import traceback
        traceback.print_exc()
        return

    clear_memory()
    
    print("\n" + "="*70)
    print("STEP 2: Loading Transformer with device_map='auto' (CRITICAL WORKAROUND)")
    print("="*70)
    print("This is where the crash happens at shard 12/14")
    print("Using device_map='auto' to bypass problematic loading path...")
    print("This may take 5-10 minutes...")
    
    start_time = time.time()
    
    try:
        # WORKAROUND: device_map="auto" loads differently than regular loading
        # It should bypass the problematic shard loading path
        max_memory = {
            0: "14GB",  # GPU 0 - conservative limit
            "cpu": "40GB"  # Allow plenty of RAM headroom
        }
        
        transformer = WanTransformer3DModel.from_pretrained(
            BASE_MODEL_ID,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            device_map="auto",  # CRITICAL: Different loading path
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            offload_state_dict=True,  # Offload to disk if needed
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ Transformer loaded in {elapsed:.1f}s (~{elapsed/60:.1f} min)")
        print_memory_stats("After Transformer")
        
        # Check device map
        if hasattr(transformer, 'hf_device_map'):
            print(f"Device map: {transformer.hf_device_map}")
        
    except Exception as e:
        print(f"\n✗ Transformer failed!")
        print(f"Error: {e}")
        
        stats = get_memory_stats()
        print(f"\nMemory at failure:")
        print(f"  RAM: {stats['ram_used_gb']:.2f}GB / {stats['ram_total_gb']:.2f}GB")
        print(f"  Available: {stats['ram_available_gb']:.2f}GB")
        
        print(f"\nTROUBLESHOOTING:")
        print(f"1. The model shards may be corrupted - try re-downloading:")
        print(f"   rm -rf ~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-I2V-14B-480P-Diffusers")
        print(f"2. Try downloading model manually first:")
        print(f"   huggingface-cli download {BASE_MODEL_ID}")
        print(f"3. Use sequential offload as last resort (very slow)")
        
        import traceback
        traceback.print_exc()
        return

    clear_memory()
    
    print("\n" + "="*70)
    print("STEP 3: Assembling Pipeline")
    print("="*70)
    
    try:
        # Build pipeline from components
        # NOTE: device_map="auto" and enable_model_cpu_offload() are MUTUALLY EXCLUSIVE
        # The transformer already has intelligent offloading via device_map
        # Do NOT call enable_model_cpu_offload() - it will cause meta tensor errors
        pipe = WanImageToVideoPipeline.from_pretrained(
            BASE_MODEL_ID,
            transformer=transformer,
            vae=vae,
            torch_dtype=torch.bfloat16
        )
        
        print(f"✓ Pipeline assembled")
        print("Using device_map offloading (already configured in transformer)")
        
        # Move VAE to GPU for encoding/decoding
        pipe.vae.to("cuda")
        print("✓ VAE moved to GPU")
        
        print_memory_stats("After Pipeline Setup")
        
    except Exception as e:
        print(f"✗ Pipeline setup failed: {e}")
        import traceback
        traceback.print_exc()
        return

    del transformer
    del vae
    clear_memory()
    
    # =========================================================================
    # LoRA Loading (simplified - skip for now to test)
    # =========================================================================
    
    print("\n" + "="*70)
    print("STEP 4: Skipping LoRAs (test run)")
    print("="*70)
    print("Using base model only to test if generation works")
    
    # =========================================================================
    # Scheduler Configuration
    # =========================================================================
    
    print("\n" + "="*70)
    print("STEP 5: Configuring Scheduler")
    print("="*70)
    
    try:
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config,
            flow_shift=5.0
        )
        print("✓ Scheduler configured")
    except Exception as e:
        print(f"✗ Scheduler error: {e}")
    
    clear_memory()
    print_memory_stats("Before Generation")
    
    # =========================================================================
    # Image Loading
    # =========================================================================
    
    print("\n" + "="*70)
    print("STEP 6: Loading Input Image")
    print("="*70)
    
    if not os.path.exists(IMAGE_PATH):
        print(f"✗ Image not found: {IMAGE_PATH}")
        return
    
    image = load_image(IMAGE_PATH)
    print(f"✓ Image loaded: {image.size}")
    
    # =========================================================================
    # Generation
    # =========================================================================
    
    num_steps = 30  # No acceleration LoRA
    guidance_scale = 5.0
    
    print("\n" + "="*70)
    print("STEP 7: Generating Video (TEST RUN)")
    print("="*70)
    print(f"Steps: {num_steps}")
    print(f"Frames: {NUM_FRAMES}")
    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print(f"Estimated time: 10-15 minutes")
    
    start_time = time.time()
    
    try:
        with torch.inference_mode():
            output = pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                image=image,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                num_frames=NUM_FRAMES,
                height=HEIGHT,
                width=WIDTH,
            ).frames[0]
        
        elapsed = time.time() - start_time
        
        export_to_video(output, OUTPUT_FILE, fps=16)
        
        print(f"\n{'='*70}")
        print("✓ SUCCESS!")
        print(f"{'='*70}")
        print(f"Saved to: {OUTPUT_FILE}")
        print(f"Time: {elapsed:.1f}s (~{elapsed/60:.1f} minutes)")
        print(f"{'='*70}")
        
        print_memory_stats("After Generation")
        
    except torch.cuda.OutOfMemoryError:
        print(f"\n✗ CUDA Out of Memory!")
        print_memory_stats("At OOM")
        print("\nReduce NUM_FRAMES to 33 or resolution to 384x640")
        
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        print_memory_stats("At Error")
    
    finally:
        clear_memory()
        print("\n" + "="*70)
        print("Complete")
        print("="*70)


if __name__ == "__main__":
    main()