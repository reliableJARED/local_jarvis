#!/usr/bin/env python3
"""
Text-to-Image Generation Demo
Uses diffusers library to generate images from text prompts
Updated to use LUSTIFY-SDXL-NSFW-checkpoint-v2-0-INPAINTING model
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def install_dependencies():
    """Install required packages automatically"""
    required_packages = [
        "torch",
        "diffusers", 
        "pillow",
        "transformers",
        "accelerate"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package if package != "pillow" else "PIL")
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages. Installing automatically...")
        print(f"Installing: {', '.join(missing_packages)}")
        
        try:
            # Install missing packages
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade"
            ] + missing_packages)
            print("✓ Dependencies installed successfully!")
            print("Restarting script with installed dependencies...")
            
            # Restart the script
            os.execv(sys.executable, [sys.executable] + sys.argv)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("Please install manually:")
            print(f"pip install {' '.join(missing_packages)}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Unexpected error during installation: {e}")
            print("Please install manually:")
            print(f"pip install {' '.join(missing_packages)}")
            sys.exit(1)

# Install dependencies first
install_dependencies()

# Now import the packages (they should be available)
import torch
from diffusers import DiffusionPipeline
from PIL import Image

def check_dependencies():
    """Verify all dependencies are properly installed"""
    try:
        import torch
        import diffusers
        from PIL import Image
        import transformers
        import accelerate
        print("✓ All dependencies verified")
    except ImportError as e:
        print(f"❌ Dependency check failed: {e}")
        print("Please run: pip install torch diffusers pillow transformers accelerate")
        sys.exit(1)

def setup_cache_dir():
    """Setup cache directory for model storage"""
    cache_dir = Path.home() / ".cache" / "text_to_image_demo"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def load_model(model_name="andro-flock/LUSTIFY-SDXL-NSFW-checkpoint-v2-0-INPAINTING", cache_dir=None, use_inpainting=False):
    """
    Load the diffusion model, downloading if necessary
    This model supports both text-to-image and inpainting capabilities
    """
    print(f"Loading model: {model_name}")
    print(f"Mode: {'Inpainting' if use_inpainting else 'Text-to-Image'}")
    
    try:
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Configure model loading parameters
        kwargs = {
            "cache_dir": cache_dir,
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "use_safetensors": True,  # Use safer tensor format when available
        }
        
        # Load the appropriate pipeline based on use case
        print("Downloading/loading model (this may take a few minutes on first run)...")
        if use_inpainting:
            from diffusers import AutoPipelineForInpainting
            pipe = AutoPipelineForInpainting.from_pretrained(model_name, **kwargs)
        else:
            # For text-to-image generation
            pipe = DiffusionPipeline.from_pretrained(model_name, **kwargs)
        
        pipe = pipe.to(device)
        
        # Enable memory optimizations
        if hasattr(pipe, 'enable_attention_slicing'):
            pipe.enable_attention_slicing()
        
        # Enable model CPU offload for memory efficiency (if accelerate is properly configured)
        try:
            if hasattr(pipe, 'enable_model_cpu_offload'):
                pipe.enable_model_cpu_offload()
                print("✓ Model CPU offload enabled")
        except Exception as e:
            print(f"⚠️  Model CPU offload not available: {e}")
            print("This is fine - the model will use regular GPU/CPU memory management")
        
        print("Model loaded successfully!")
        return pipe
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have sufficient disk space and internet connection.")
        print("If using PyTorch < 2.0, consider upgrading for better performance.")
        sys.exit(1)

def generate_image(pipe, prompt, output_path="test.png", mask_image=None, init_image=None, **kwargs):
    """
    Generate image from text prompt (text-to-image or inpainting)
    """
    is_inpainting = mask_image is not None and init_image is not None
    mode = "inpainting" if is_inpainting else "text-to-image"
    print(f"Generating image for prompt: '{prompt}' (mode: {mode})")
    
    # Determine device type for parameter adjustments
    device = next(pipe.parameters()).device if hasattr(pipe, 'parameters') else torch.device('cpu')
    is_cpu = device.type == 'cpu'
    
    # Recommended parameters from model card, adjusted for device capabilities
    generation_kwargs = {
        "num_inference_steps": 20 if is_cpu else 30,  # Fewer steps on CPU for speed
        "guidance_scale": 5.5,      # Model recommends CFG 4-7
        "height": 512 if is_cpu else 1024,  # Smaller resolution on CPU to avoid memory issues
        "width": 512 if is_cpu else 1024,   # Smaller resolution on CPU
    }
    
    # Add inpainting-specific parameters if applicable
    if is_inpainting:
        generation_kwargs.update({
            "image": init_image,
            "mask_image": mask_image,
            "strength": 0.99,  # High strength for good inpainting results
        })
    
    # Update with any provided kwargs, but filter out None values
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    generation_kwargs.update(filtered_kwargs)
    
    # Remove any None values from final kwargs
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
    
    if is_cpu:
        print("⚠️  Running on CPU - using reduced resolution and steps for better performance")
        print(f"Resolution: {generation_kwargs['width']}x{generation_kwargs['height']}")
    
    try:
        # Generate image
        print("Generating... (this may take a few minutes depending on your machine)")
        with torch.no_grad():
            result = pipe(prompt, **generation_kwargs)
            image = result.images[0]
        
        # Save image
        image.save(output_path)
        print(f"Image saved to: {output_path}")
        
        return image
        
    except Exception as e:
        print(f"Error generating image: {e}")
        print(f"Generation parameters used: {generation_kwargs}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Text-to-Image Generation and Inpainting Demo with LUSTIFY-SDXL")
    parser.add_argument("--prompt", "-p", 
                       default="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                       help="Text prompt for image generation")
    parser.add_argument("--output", "-o", 
                       default="test.png",
                       help="Output image filename")
    parser.add_argument("--model", "-m",
                       default="TheImposterImposters/LUSTIFY-v2.0",
                       help="Model name/path (use TheImposterImposters/LUSTIFY-v2.0 for text-to-image)")
    parser.add_argument("--inpaint_model", type=str,
                       default="andro-flock/LUSTIFY-SDXL-NSFW-checkpoint-v2-0-INPAINTING", 
                       help="Inpainting model to use when --inpaint is enabled")
    parser.add_argument("--steps", type=int, default=30,
                       help="Number of inference steps (recommended: 30)")
    parser.add_argument("--guidance", type=float, default=5.5,
                       help="Guidance scale (recommended: 4-7)")
    parser.add_argument("--width", type=int, default=1024,
                       help="Image width (SDXL works best at 1024x1024)")
    parser.add_argument("--height", type=int, default=1024,
                       help="Image height (SDXL works best at 1024x1024)")
    
    # Inpainting specific arguments
    parser.add_argument("--inpaint", action="store_true",
                       help="Use inpainting mode (requires --init_image and --mask_image)")
    parser.add_argument("--init_image", type=str,
                       help="Path to initial image for inpainting")
    parser.add_argument("--mask_image", type=str,
                       help="Path to mask image for inpainting (white=inpaint, black=keep)")
    parser.add_argument("--strength", type=float, default=0.99,
                       help="Inpainting strength (0.0-1.0, higher=more change)")
    
    args = parser.parse_args()
    
    # Validate inpainting arguments
    if args.inpaint and (not args.init_image or not args.mask_image):
        print("❌ Error: Inpainting mode requires both --init_image and --mask_image")
        sys.exit(1)
    
    # Check dependencies (should pass now after auto-install)
    check_dependencies()
    
    # Setup cache directory
    cache_dir = setup_cache_dir()
    print(f"Using cache directory: {cache_dir}")
    
    # Load model - use appropriate model based on mode
    model_to_use = args.inpaint_model if args.inpaint else args.model
    pipe = load_model(model_to_use, cache_dir, use_inpainting=args.inpaint)
    
    # Prepare inpainting images if needed
    init_image = None
    mask_image = None
    if args.inpaint:
        from PIL import Image
        try:
            print(f"Loading init image: {args.init_image}")
            init_image = Image.open(args.init_image).convert("RGB").resize((args.width, args.height))
            print(f"Loading mask image: {args.mask_image}")
            mask_image = Image.open(args.mask_image).convert("RGB").resize((args.width, args.height))
        except Exception as e:
            print(f"❌ Error loading inpainting images: {e}")
            sys.exit(1)
    
    # Generate image
    image = generate_image(
        pipe, 
        args.prompt, 
        args.output,
        mask_image=mask_image,
        init_image=init_image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        strength=args.strength if args.inpaint else None
    )
    
    if image:
        print("Generation complete!")
        print(f"Image size: {image.size}")
        print(f"Output file: {os.path.abspath(args.output)}")
        if args.inpaint:
            print("Inpainting completed successfully!")
    else:
        print("Generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()