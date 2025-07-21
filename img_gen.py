#!/usr/bin/env python3
"""
Text-to-Image Generation Demo
Uses diffusers library to generate images from text prompts
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

def load_model(model_name="UnfilteredAI/NSFW-gen-v2", cache_dir=None):
    """
    Load the diffusion model, downloading if necessary
    """
    print(f"Loading model: {model_name}")
    
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
        
        # Load the pipeline
        print("Downloading/loading model (this may take a few minutes on first run)...")
        pipe = DiffusionPipeline.from_pretrained(model_name, **kwargs)
        pipe = pipe.to(device)
        
        # Enable attention slicing for memory efficiency (still useful)
        if hasattr(pipe, 'enable_attention_slicing'):
            pipe.enable_attention_slicing()
        
        # Note: enable_memory_efficient_attention is deprecated in favor of PyTorch 2.0's SDPA
        # PyTorch 2.0+ automatically uses optimized attention (SDPA) when available
        
        print("Model loaded successfully!")
        return pipe
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have sufficient disk space and internet connection.")
        print("If using PyTorch < 2.0, consider upgrading for better performance.")
        sys.exit(1)

def generate_image(pipe, prompt, output_path="test.png", **kwargs):
    """
    Generate image from text prompt
    """
    print(f"Generating image for prompt: '{prompt}'")
    
    # Default generation parameters
    generation_kwargs = {
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "height": 512,
        "width": 512,
    }
    
    # Update with any provided kwargs
    generation_kwargs.update(kwargs)
    
    try:
        # Generate image
        print("Generating... (this may take a minute)")
        with torch.no_grad():
            # Call the pipeline directly with the prompt
            result = pipe(prompt, **generation_kwargs)
            image = result.images[0]
        
        # Save image
        image.save(output_path)
        print(f"Image saved to: {output_path}")
        
        return image
        
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Text-to-Image Generation Demo")
    parser.add_argument("--prompt", "-p", 
                       default="3D Photorealistic. A monkey eating a banana",
                       help="Text prompt for image generation")
    parser.add_argument("--output", "-o", 
                       default="test.png",
                       help="Output image filename")
    parser.add_argument("--model", "-m",
                       default="UnfilteredAI/NSFW-gen-v2",
                       help="Model name/path")
    parser.add_argument("--steps", type=int, default=20,
                       help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5,
                       help="Guidance scale")
    parser.add_argument("--width", type=int, default=512,
                       help="Image width")
    parser.add_argument("--height", type=int, default=512,
                       help="Image height")
    
    args = parser.parse_args()
    
    # Check dependencies (should pass now after auto-install)
    check_dependencies()
    
    # Setup cache directory
    cache_dir = setup_cache_dir()
    print(f"Using cache directory: {cache_dir}")
    
    # Load model
    pipe = load_model(args.model, cache_dir)
    
    # Generate image
    image = generate_image(
        pipe, 
        args.prompt, 
        args.output,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height
    )
    
    if image:
        print("Generation complete!")
        print(f"Image size: {image.size}")
        print(f"Output file: {os.path.abspath(args.output)}")
    else:
        print("Generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()