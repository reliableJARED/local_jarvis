#!/usr/bin/env python3
"""
Text-to-Image and Text-to-Video Generation Class
Uses diffusers library to generate images and videos from text prompts
Updated to include AnimateDiff for video generation with LUSTIFY-SDXL model
"""

import os
import sys
import subprocess
from pathlib import Path
import torch
from PIL import Image


class ImageVideoGenerator:
    """
    A class for generating images and videos using diffusion models.
    Supports text-to-image, image-to-image, inpainting, and text-to-video capabilities.
    """
    
    def __init__(self, 
                 model_name="TheImposterImposters/LUSTIFY-v2.0",
                 inpaint_model="andro-flock/LUSTIFY-SDXL-NSFW-checkpoint-v2-0-INPAINTING",
                 cache_dir=None,
                 use_mps=True):
        """
        Initialize the ImageVideoGenerator with model loading and dependency checking.
        
        Args:
            model_name (str): Main model for text-to-image and image-to-image
            inpaint_model (str): Model for inpainting tasks
            cache_dir (str): Cache directory for model storage
            use_mps (bool): Use Metal Performance Shaders on Mac (if available)
        """
        self.model_name = model_name
        self.inpaint_model = inpaint_model
        self.cache_dir = cache_dir or self._setup_cache_dir()
        
        # Enhanced device configuration for Mac
        self.device = self._get_best_device(use_mps)
        self.is_cpu = self.device == "cpu"
        self.is_mps = self.device == "mps"
        
        # Install dependencies and check
        self._install_dependencies()
        self._check_dependencies()
        
        # Initialize pipelines (will be loaded on demand)
        self.text2img_pipe = None
        self.img2img_pipe = None
        self.inpaint_pipe = None
        self.video_pipe = None  # New: AnimateDiff pipeline
        
        print(f"✅ ImageVideoGenerator initialized")
        print(f"Device: {self.device}")
        if self.is_mps:
            print("🚀 Using Metal Performance Shaders for acceleration!")
        print(f"Cache directory: {self.cache_dir}")
    
    def _get_best_device(self, use_mps=True):
        """Determine the best available device for Mac optimization"""
        if torch.cuda.is_available():
            return "cuda"
        elif use_mps and torch.backends.mps.is_available():
            # MPS (Metal Performance Shaders) for Mac acceleration
            return "mps"
        else:
            return "cpu"
    
    def _install_dependencies(self):
        """Install required packages automatically including AnimateDiff dependencies"""
        required_packages = [
            "torch",
            "diffusers>=0.28.0",  # Updated for AnimateDiff SDXL support
            "pillow",
            "transformers",
            "accelerate",
            "imageio",  # For video export
            "imageio-ffmpeg"  # For MP4 support
        ]
        
        missing_packages = []
        
        for package in required_packages:
            package_name = package.split(">=")[0]  # Handle version requirements
            try:
                if package_name == "pillow":
                    __import__("PIL")
                elif package_name == "imageio-ffmpeg":
                    import imageio_ffmpeg
                else:
                    __import__(package_name)
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
    
    def _check_dependencies(self):
        """Verify all dependencies are properly installed"""
        try:
            import torch
            import diffusers
            from PIL import Image
            import transformers
            import accelerate
            import imageio
            print("✓ All dependencies verified")
            
            # Check diffusers version for AnimateDiff support
            from packaging import version
            if version.parse(diffusers.__version__) < version.parse("0.28.0"):
                print("⚠️  Warning: diffusers version is below 0.28.0")
                print("AnimateDiff SDXL requires diffusers>=0.28.0")
                print("Run: pip install --upgrade diffusers>=0.28.0")
            else:
                print(f"✓ diffusers {diffusers.__version__} supports AnimateDiff SDXL")
                
        except ImportError as e:
            print(f"❌ Dependency check failed: {e}")
            print("Please run: pip install torch diffusers>=0.28.0 pillow transformers accelerate imageio imageio-ffmpeg")
            sys.exit(1)
    
    def _setup_cache_dir(self):
        """Setup cache directory for model storage"""
        cache_dir = Path.home() / ".cache" / "lustify_video_demo"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def _load_pipeline(self, pipeline_type="text-to-image"):
        """
        Load the appropriate diffusion pipeline
        
        Args:
            pipeline_type (str): "text-to-image", "image-to-image", "inpainting", or "text-to-video"
        
        Returns:
            Pipeline object
        """
        from diffusers import (DiffusionPipeline, AutoPipelineForInpainting, 
                              AutoPipelineForImage2Image, AnimateDiffSDXLPipeline, 
                              DDIMScheduler)
        from diffusers.models import MotionAdapter
        
        # Determine which model to use
        if pipeline_type == "inpainting":
            model_name = self.inpaint_model
        else:
            model_name = self.model_name
        
        print(f"Loading {pipeline_type} pipeline: {model_name}")
        
        try:
            # Configure model loading parameters with Mac optimizations
            kwargs = {
                "cache_dir": self.cache_dir,
                "torch_dtype": torch.float32,  # MPS works better with float32
                "use_safetensors": True,
            }
            
            # Use float16 only for CUDA, float32 for MPS and CPU
            if self.device == "cuda":
                kwargs["torch_dtype"] = torch.float16
            
            # Load the appropriate pipeline
            print("Downloading/loading model (this may take a few minutes on first run)...")
            
            if pipeline_type == "text-to-video":
                # Special handling for AnimateDiff
                print("🎬 Loading AnimateDiff motion adapter...")
                adapter = MotionAdapter.from_pretrained(
                    "guoyww/animatediff-motion-adapter-sdxl-beta",
                    torch_dtype=kwargs["torch_dtype"]
                )
                
                # Configure scheduler for AnimateDiff
                scheduler = DDIMScheduler.from_pretrained(
                    model_name,
                    subfolder="scheduler",
                    clip_sample=False,
                    timestep_spacing="linspace",
                    beta_schedule="linear",
                    steps_offset=1,
                )
                
                pipe = AnimateDiffSDXLPipeline.from_pretrained(
                    model_name,
                    motion_adapter=adapter,
                    scheduler=scheduler,
                    **kwargs
                )
                
            elif pipeline_type == "inpainting":
                pipe = AutoPipelineForInpainting.from_pretrained(model_name, **kwargs)
            elif pipeline_type == "image-to-image":
                pipe = AutoPipelineForImage2Image.from_pretrained(model_name, **kwargs)
            else:
                pipe = DiffusionPipeline.from_pretrained(model_name, **kwargs)
            
            pipe = pipe.to(self.device)
            
            # Mac-specific optimizations
            if self.is_mps:
                # MPS optimizations
                print("🔧 Applying MPS optimizations...")
                if hasattr(pipe, 'enable_attention_slicing'):
                    pipe.enable_attention_slicing(1)  # More aggressive slicing for MPS
                
                # Enable VAE optimizations for video generation
                if pipeline_type == "text-to-video":
                    if hasattr(pipe, 'enable_vae_slicing'):
                        pipe.enable_vae_slicing()
                    if hasattr(pipe, 'enable_vae_tiling'):
                        pipe.enable_vae_tiling()
                    print("✓ VAE optimizations enabled for video generation")
                
                # Enable sequential CPU offload for better memory management on Mac
                try:
                    if hasattr(pipe, 'enable_sequential_cpu_offload'):
                        pipe.enable_sequential_cpu_offload()
                        print("✓ Sequential CPU offload enabled for MPS")
                except Exception as e:
                    print(f"⚠️  Sequential CPU offload not available: {e}")
            
            elif self.is_cpu:
                # CPU optimizations
                if hasattr(pipe, 'enable_attention_slicing'):
                    pipe.enable_attention_slicing(1)
                
                # Set number of threads for CPU inference
                torch.set_num_threads(torch.get_num_threads())
                print(f"🔧 Using {torch.get_num_threads()} CPU threads")
            
            else:
                # GPU optimizations
                if hasattr(pipe, 'enable_attention_slicing'):
                    pipe.enable_attention_slicing()
                
                # Video generation optimizations
                if pipeline_type == "text-to-video":
                    if hasattr(pipe, 'enable_vae_slicing'):
                        pipe.enable_vae_slicing()
                    if hasattr(pipe, 'enable_vae_tiling'):
                        pipe.enable_vae_tiling()
                
                # Enable model CPU offload for memory efficiency
                try:
                    if hasattr(pipe, 'enable_model_cpu_offload'):
                        pipe.enable_model_cpu_offload()
                        print("✓ Model CPU offload enabled")
                except Exception as e:
                    print(f"⚠️  Model CPU offload not available: {e}")
            
            print(f"✅ {pipeline_type.title()} pipeline loaded successfully!")
            return pipe
            
        except Exception as e:
            print(f"❌ Error loading {pipeline_type} pipeline: {e}")
            if pipeline_type == "text-to-video":
                print("💡 Make sure you have diffusers>=0.28.0 for AnimateDiff SDXL support")
                print("💡 Run: pip install --upgrade diffusers>=0.28.0")
            print("Make sure you have sufficient disk space and internet connection.")
            sys.exit(1)
    
    def _get_pipeline(self, pipeline_type):
        """Get or load the requested pipeline"""
        if pipeline_type == "text-to-image":
            if self.text2img_pipe is None:
                self.text2img_pipe = self._load_pipeline("text-to-image")
            return self.text2img_pipe
        elif pipeline_type == "image-to-image":
            if self.img2img_pipe is None:
                self.img2img_pipe = self._load_pipeline("image-to-image")
            return self.img2img_pipe
        elif pipeline_type == "inpainting":
            if self.inpaint_pipe is None:
                self.inpaint_pipe = self._load_pipeline("inpainting")
            return self.inpaint_pipe
        elif pipeline_type == "text-to-video":
            if self.video_pipe is None:
                self.video_pipe = self._load_pipeline("text-to-video")
            return self.video_pipe
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
    
    def text_to_image(self, 
                     prompt, 
                     output_path="output.png",
                     num_inference_steps=30,  # Model docs recommend 30 steps
                     guidance_scale=5.5,     # Within the 4-7 range from docs
                     width=1024,
                     height=1024,
                     use_enhanced_prompting=True,  # Use LUSTIFY-specific tags
                     **kwargs):
        """
        Generate image from text prompt
        
        Args:
            prompt (str): Text description of desired image
            output_path (str): Path to save generated image
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): Guidance scale for generation
            width (int): Image width
            height (int): Image height
            use_enhanced_prompting (bool): Add LUSTIFY-specific style tags
            **kwargs: Additional generation parameters
        
        Returns:
            PIL.Image: Generated image
        """
        print(f"🎨 Generating image from text: '{prompt}'")
        
        # Enhance prompt for LUSTIFY model if requested
        if use_enhanced_prompting:
            enhanced_prompt = self._enhance_prompt_for_lustify(prompt)
            print(f"🎨 Enhanced prompt: '{enhanced_prompt}'")
            prompt = enhanced_prompt
        
        # Get pipeline
        pipe = self._get_pipeline("text-to-image")
        
        # Adjust parameters for device capabilities while respecting SDXL's optimal resolution
        if self.is_cpu:
            # Even on CPU, try to maintain closer to SDXL's native resolution
            width = min(width, 768)  # Compromise between speed and quality
            height = min(height, 768)
            num_inference_steps = min(num_inference_steps, 20)  # Reduce steps for CPU
            print(f"⚠️  Running on CPU - using optimized settings: {width}x{height}, {num_inference_steps} steps")
        elif self.is_mps:
            # MPS can handle full SDXL resolution better, but reduce steps slightly
            # Keep 1024x1024 as SDXL works best at this resolution
            num_inference_steps = min(num_inference_steps, 25)
            print(f"🚀 Running on MPS - SDXL optimized: {width}x{height}, {num_inference_steps} steps")
        
        # Add LUSTIFY-specific negative prompt for better quality
        negative_prompt = kwargs.pop('negative_prompt', None)
        if not negative_prompt:
            negative_prompt = "blurry, low quality, distorted, deformed, extra limbs, bad anatomy"
        
        # Generation parameters
        generation_kwargs = {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "negative_prompt": negative_prompt,
        }
        generation_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
        
        try:
            print("Generating... (this may take a minute)")
            with torch.no_grad():
                # Mac-specific inference optimizations
                if self.is_mps:
                    # MPS sometimes has issues with certain operations
                    with torch.autocast(device_type='cpu', enabled=False):
                        result = pipe(prompt, **generation_kwargs)
                else:
                    result = pipe(prompt, **generation_kwargs)
                image = result.images[0]
            
            # Save image
            image.save(output_path)
            print(f"✅ Image saved to: {output_path}")
            return image
            
        except Exception as e:
            print(f"❌ Error generating image: {e}")
            print(f"💡 Try lowering guidance scale or reducing resolution")
            return None
    
    def text_to_video(self,
                     prompt,
                     output_path="output.gif",
                     num_frames=16,          # SDXL motion module optimal frame count
                     num_inference_steps=25, # Reduced for video generation
                     guidance_scale=8.0,     # Slightly higher for video
                     width=1024,
                     height=1024,
                     fps=8,                  # Output frame rate
                     use_enhanced_prompting=True,
                     export_format="gif",    # "gif" or "mp4"
                     **kwargs):
        """
        Generate video from text prompt using AnimateDiff
        
        Args:
            prompt (str): Text description of desired video
            output_path (str): Path to save generated video
            num_frames (int): Number of frames to generate (16 is optimal for SDXL)
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): Guidance scale for generation
            width (int): Video width (1024 recommended for SDXL)
            height (int): Video height (1024 recommended for SDXL)
            fps (int): Frames per second for output video
            use_enhanced_prompting (bool): Add LUSTIFY-specific style tags
            export_format (str): "gif" or "mp4"
            **kwargs: Additional generation parameters
        
        Returns:
            list: List of PIL Images representing video frames
        """
        print(f"🎬 Generating video from text: '{prompt}'")
        print(f"📊 Video specs: {width}x{height}, {num_frames} frames @ {fps}fps")
        
        # Enhance prompt for LUSTIFY model if requested
        if use_enhanced_prompting:
            enhanced_prompt = self._enhance_prompt_for_lustify(prompt)
            # Add motion-specific enhancements
            enhanced_prompt = self._enhance_prompt_for_video(enhanced_prompt)
            print(f"🎨 Enhanced prompt: '{enhanced_prompt}'")
            prompt = enhanced_prompt
        
        # Get video pipeline
        pipe = self._get_pipeline("text-to-video")
        
        # Adjust parameters for device capabilities
        if self.is_cpu:
            # Reduce everything for CPU
            width = min(width, 512)
            height = min(height, 512)
            num_frames = min(num_frames, 8)
            num_inference_steps = min(num_inference_steps, 15)
            print(f"⚠️  Running on CPU - using reduced settings: {width}x{height}, {num_frames} frames, {num_inference_steps} steps")
        elif self.is_mps:
            # MPS can handle better settings but still optimize
            num_frames = min(num_frames, 16)  # Keep at 16 for optimal results
            num_inference_steps = min(num_inference_steps, 20)
            print(f"🚀 Running on MPS - optimized settings: {width}x{height}, {num_frames} frames, {num_inference_steps} steps")
        
        # Add video-specific negative prompt
        negative_prompt = kwargs.pop('negative_prompt', None)
        if not negative_prompt:
            negative_prompt = "blurry, low quality, distorted, deformed, static image, no movement, frozen"
        
        # Generation parameters
        generation_kwargs = {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "negative_prompt": negative_prompt,
        }
        generation_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
        
        try:
            print("Generating video... (this may take several minutes)")
            with torch.no_grad():
                if self.is_mps:
                    with torch.autocast(device_type='cpu', enabled=False):
                        result = pipe(prompt, **generation_kwargs)
                else:
                    result = pipe(prompt, **generation_kwargs)
                frames = result.frames[0]
            
            # Export video
            self._export_video(frames, output_path, fps, export_format)
            print(f"✅ Video saved to: {output_path}")
            return frames
            
        except Exception as e:
            print(f"❌ Error generating video: {e}")
            print(f"💡 Try reducing num_frames or resolution for your device")
            print(f"💡 Current settings: {width}x{height}, {num_frames} frames")
            return None
    
    def _export_video(self, frames, output_path, fps, format_type):
        """Export frames as video file"""
        try:
            if format_type.lower() == "mp4":
                # Export as MP4
                import imageio
                imageio.mimsave(output_path, frames, fps=fps, quality=8)
            else:
                # Export as GIF (default)
                from diffusers.utils import export_to_gif
                if not output_path.endswith('.gif'):
                    output_path = output_path.replace('.mp4', '.gif')
                export_to_gif(frames, output_path, fps=fps)
        except Exception as e:
            print(f"❌ Error exporting video: {e}")
            print("💡 Falling back to basic GIF export...")
            # Fallback: manual GIF creation
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=1000//fps,
                loop=0
            )
    
    def _enhance_prompt_for_lustify(self, prompt):
        """
        Enhance prompts with LUSTIFY-specific tags for better results
        Based on model documentation and community findings
        """
        # LUSTIFY responds well to photography-style prompting
        photo_enhancers = [
            "photograph",
            "shot on Canon EOS 5D", 
            "cinematic lighting",
            "professional photography"
        ]
        
        # Check if prompt already has photography terms
        prompt_lower = prompt.lower()
        has_photo_terms = any(term in prompt_lower for term in [
            "shot on", "photograph", "photo", "camera", "lighting", 
            "shot with", "taken with"
        ])
        
        if not has_photo_terms:
            # Add basic photography enhancement
            enhanced = f"photograph, {prompt}, shot on Canon EOS 5D, professional photography"
        else:
            # Prompt already has photo terms, just clean it up
            enhanced = prompt
            
        return enhanced
    
    def _enhance_prompt_for_video(self, prompt):
        """Add video-specific enhancements to prompts"""
        video_enhancers = [
            "smooth motion",
            "cinematic movement", 
            "flowing",
            "dynamic"
        ]
        
        # Check if prompt already has motion terms
        prompt_lower = prompt.lower()
        has_motion_terms = any(term in prompt_lower for term in [
            "moving", "motion", "flowing", "dancing", "walking", "running",
            "swaying", "floating", "flying", "swimming", "dynamic"
        ])
        
        if not has_motion_terms:
            # Add subtle motion enhancement
            enhanced = f"{prompt}, smooth motion, cinematic movement"
        else:
            enhanced = prompt
            
        return enhanced
    
    # Keep all your existing methods (image_to_image, inpaint, etc.)
    def image_to_image(self, 
                      prompt, 
                      input_image,
                      output_path="output.png",
                      strength=0.20,           # MUCH lower for LUSTIFY consistency
                      num_inference_steps=30,  # Model docs recommend 30 steps  
                      guidance_scale=5.5,     # Within the 4-7 range from docs
                      use_enhanced_prompting=True,  # Use LUSTIFY-specific tags
                      **kwargs):
        """
        Transform an existing image based on text prompt
        (Implementation same as original - keeping for compatibility)
        """
        # [Keep your existing image_to_image implementation]
        pass
    
    def inpaint(self, 
               prompt,
               input_image,
               mask_image,
               output_path="output.png",
               strength=0.99,
               num_inference_steps=30,  # Model docs recommend 30 steps
               guidance_scale=5.5,     # Within the 4-7 range from docs
               **kwargs):
        """
        Inpaint parts of an image based on a mask and text prompt
        (Implementation same as original - keeping for compatibility)
        """
        # [Keep your existing inpaint implementation]
        pass


# Example usage
if __name__ == "__main__":
    # Initialize the generator
    generator = ImageVideoGenerator()
    
    # Example 1: Text-to-image generation (your existing workflow)
    image1 = generator.text_to_image(
        prompt="photograph, photo of monkey eating a banana, 8k",
        output_path="step1_text2img.png"
    )
    
    # Example 2: NEW - Text-to-video generation
    if image1:
        print("\n🎬 Now generating video...")
        video_frames = generator.text_to_video(
            prompt="photograph, photo of monkey eating a banana, monkey moving naturally, 8k",
            output_path="step2_text2video.gif",
            num_frames=16,
            fps=8,
            export_format="gif"
        )
        
        if video_frames:
            print("🎉 Image and video generation complete!")
            print("Generated files:")
            print(f"  - Image: step1_text2img.png")
            print(f"  - Video: step2_text2video.gif")
            
            # Optional: Also create MP4
            print("\n🎬 Creating MP4 version...")
            generator._export_video(video_frames, "step2_text2video.mp4", 8, "mp4")
            print(f"  - Video MP4: step2_text2video.mp4")