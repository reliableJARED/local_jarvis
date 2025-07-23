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
                    # Enable model CPU offload for video generation to save memory
                    if hasattr(pipe, 'enable_model_cpu_offload'):
                        try:
                            pipe.enable_model_cpu_offload()
                            print("✓ Model CPU offload enabled for video generation")
                        except Exception as e:
                            print(f"⚠️  Model CPU offload not available: {e}")
                    print("✓ VAE optimizations enabled for video generation")
                
                # Try sequential CPU offload for non-video pipelines
                if pipeline_type != "text-to-video":
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
                     export_format="mp4",    # Changed default to MP4
                     auto_fallback=True,     # Automatically reduce settings on memory error
                     **kwargs):
        """
        Generate video from text prompt using AnimateDiff with memory-aware fallback
        
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
            auto_fallback (bool): Automatically try reduced settings on memory error
            **kwargs: Additional generation parameters
        
        Returns:
            list: List of PIL Images representing video frames
        """
        print(f"🎬 Generating video from text: '{prompt}'")
        
        # Enhance prompt for LUSTIFY model if requested
        if use_enhanced_prompting:
            enhanced_prompt = self._enhance_prompt_for_lustify(prompt)
            # Add motion-specific enhancements
            enhanced_prompt = self._enhance_prompt_for_video(enhanced_prompt)
            print(f"🎨 Enhanced prompt: '{enhanced_prompt}'")
            prompt = enhanced_prompt
        
        # Progressive fallback configurations for memory management
        configs = self._get_video_configs(width, height, num_frames, num_inference_steps)
        
        for i, config in enumerate(configs):
            try:
                print(f"\n📊 Attempt {i+1}/{len(configs)}: {config['width']}x{config['height']}, {config['frames']} frames, {config['steps']} steps")
                
                # Get video pipeline
                pipe = self._get_pipeline("text-to-video")
                
                # Add video-specific negative prompt
                negative_prompt = kwargs.pop('negative_prompt', None)
                if not negative_prompt:
                    negative_prompt = "blurry, low quality, distorted, deformed, static image, no movement, frozen"
                
                # Generation parameters
                generation_kwargs = {
                    "num_inference_steps": config['steps'],
                    "guidance_scale": guidance_scale,
                    "height": config['height'],
                    "width": config['width'],
                    "num_frames": config['frames'],
                    "negative_prompt": negative_prompt,
                }
                generation_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
                
                print("Generating video... (this may take several minutes)")
                
                # Clear cache before generation
                if self.is_mps:
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
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
                print(f"🎯 Final settings used: {config['width']}x{config['height']}, {config['frames']} frames")
                return frames
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Attempt {i+1} failed: {error_msg}")
                
                # Check if it's a memory error
                is_memory_error = any(term in error_msg.lower() for term in [
                    'out of memory', 'memory', 'oom', 'cuda out of memory', 'mps backend out of memory'
                ])
                
                if is_memory_error:
                    print(f"🧠 Memory error detected, trying reduced settings...")
                    # Clear cache
                    if self.is_mps:
                        torch.mps.empty_cache()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    if not auto_fallback or i == len(configs) - 1:
                        print(f"💡 All configurations failed. Try manually setting lower values:")
                        print(f"💡 Example: width=512, height=512, num_frames=8")
                        return None
                    else:
                        print(f"🔄 Trying next configuration...")
                        continue
                else:
                    # Non-memory error, don't continue
                    print(f"❌ Non-memory error occurred: {error_msg}")
                    return None
        
        print("❌ All fallback configurations failed")
        return None
    
    def _get_video_configs(self, width, height, num_frames, num_inference_steps):
        """
        Generate progressive fallback configurations for video generation
        Starting from requested settings and progressively reducing memory usage
        """
        configs = []
        
        if self.is_cpu:
            # CPU configurations - very conservative
            configs = [
                {"width": 512, "height": 512, "frames": 8, "steps": 15},
                {"width": 448, "height": 448, "frames": 8, "steps": 12},
                {"width": 384, "height": 384, "frames": 6, "steps": 10},
            ]
        elif self.is_mps:
            # MPS configurations - progressive fallback for Mac
            configs = [
                # Start with user request (but capped for sanity)
                {"width": min(width, 1024), "height": min(height, 1024), "frames": min(num_frames, 16), "steps": min(num_inference_steps, 20)},
                # First fallback: reduce resolution but keep frames
                {"width": 768, "height": 768, "frames": min(num_frames, 16), "steps": 18},
                # Second fallback: reduce frames
                {"width": 768, "height": 768, "frames": 12, "steps": 16},
                # Third fallback: further reduce resolution
                {"width": 640, "height": 640, "frames": 12, "steps": 15},
                # Fourth fallback: minimal frames
                {"width": 640, "height": 640, "frames": 8, "steps": 15},
                # Last resort: very conservative
                {"width": 512, "height": 512, "frames": 8, "steps": 12},
            ]
        else:
            # GPU configurations
            configs = [
                # Start with user request
                {"width": width, "height": height, "frames": num_frames, "steps": num_inference_steps},
                # Progressive fallbacks
                {"width": min(width, 1024), "height": min(height, 1024), "frames": min(num_frames, 16), "steps": min(num_inference_steps, 25)},
                {"width": 768, "height": 768, "frames": 16, "steps": 20},
                {"width": 768, "height": 768, "frames": 12, "steps": 18},
                {"width": 640, "height": 640, "frames": 12, "steps": 15},
                {"width": 512, "height": 512, "frames": 8, "steps": 12},
            ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_configs = []
        for config in configs:
            config_tuple = (config['width'], config['height'], config['frames'], config['steps'])
            if config_tuple not in seen:
                seen.add(config_tuple)
                unique_configs.append(config)
        
        return unique_configs
    
    def text_to_video_quick(self, 
                           prompt, 
                           output_path="quick_video.mp4",  # Changed default to MP4
                           export_format="mp4",            # Changed default to MP4
                           use_enhanced_prompting=True):
        """
        Quick video generation with Mac-optimized settings
        Perfect for testing and fast iterations
        
        Args:
            prompt (str): Text description of desired video
            output_path (str): Path to save generated video
            export_format (str): "mp4" or "gif" 
            use_enhanced_prompting (bool): Add LUSTIFY-specific style tags
            
        Returns:
            list: List of PIL Images representing video frames
        """
        print("🚀 Quick video generation (Mac-optimized settings)")
        
        # Conservative settings for Mac
        if self.is_mps:
            return self.text_to_video(
                prompt=prompt,
                output_path=output_path,
                width=640,
                height=640,
                num_frames=8,
                num_inference_steps=15,
                guidance_scale=7.0,
                fps=8,
                export_format=export_format,
                use_enhanced_prompting=use_enhanced_prompting,
                auto_fallback=True
            )
        else:
            # CPU or other devices
            return self.text_to_video(
                prompt=prompt,
                output_path=output_path,
                width=512,
                height=512,
                num_frames=6,
                num_inference_steps=12,
                guidance_scale=6.0,
                fps=6,
                export_format=export_format,
                use_enhanced_prompting=use_enhanced_prompting,
                auto_fallback=True
            )
    
    def image_to_video(self,
                      prompt,
                      input_image,
                      output_path="image_to_video_output.mp4",
                      num_frames=16,
                      num_inference_steps=25,
                      guidance_scale=8.0,
                      fps=8,
                      export_format="mp4",
                      use_enhanced_prompting=True,
                      auto_fallback=True,
                      **kwargs):
        """
        Generate video from an input image using AnimateDiff
        Animates a static image based on text prompt
        
        Args:
            prompt (str): Text description of desired animation/movement
            input_image (str or PIL.Image): Path to input image or PIL Image object
            output_path (str): Path to save generated video
            num_frames (int): Number of frames to generate
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): Guidance scale for generation
            fps (int): Frames per second for output video
            export_format (str): "mp4" or "gif"
            use_enhanced_prompting (bool): Add LUSTIFY-specific style tags
            auto_fallback (bool): Automatically try reduced settings on memory error
            **kwargs: Additional generation parameters
        
        Returns:
            list: List of PIL Images representing video frames
        """
        print(f"🎬 Generating video from image with prompt: '{prompt}'")
        
        # Load and prepare input image
        if isinstance(input_image, str):
            try:
                input_image = Image.open(input_image).convert("RGB")
                print(f"📷 Loaded input image: {input_image.size}")
            except Exception as e:
                print(f"❌ Error loading input image: {e}")
                return None
        
        # Get image dimensions
        original_width, original_height = input_image.size
        
        # Enhance prompt for video generation
        if use_enhanced_prompting:
            enhanced_prompt = self._enhance_prompt_for_lustify(prompt)
            enhanced_prompt = self._enhance_prompt_for_video(enhanced_prompt)
            print(f"🎨 Enhanced prompt: '{enhanced_prompt}'")
            prompt = enhanced_prompt
        
        # Progressive fallback configurations for image-to-video
        configs = self._get_image_to_video_configs(original_width, original_height, num_frames, num_inference_steps)
        
        for i, config in enumerate(configs):
            try:
                print(f"\n📊 Attempt {i+1}/{len(configs)}: {config['width']}x{config['height']}, {config['frames']} frames, {config['steps']} steps")
                
                # Resize input image to match config
                if config['width'] != original_width or config['height'] != original_height:
                    resized_image = input_image.resize((config['width'], config['height']))
                    print(f"🔄 Resized image: {original_width}x{original_height} → {config['width']}x{config['height']}")
                else:
                    resized_image = input_image
                
                # Get video pipeline
                pipe = self._get_pipeline("text-to-video")
                
                # Add video-specific negative prompt
                negative_prompt = kwargs.pop('negative_prompt', None)
                if not negative_prompt:
                    negative_prompt = "blurry, low quality, distorted, deformed, static image, no movement, frozen"
                
                # Generation parameters for image-to-video
                generation_kwargs = {
                    "prompt": prompt,
                    "image": resized_image,  # Input image for conditioning
                    "num_inference_steps": config['steps'],
                    "guidance_scale": guidance_scale,
                    "height": config['height'],
                    "width": config['width'],
                    "num_frames": config['frames'],
                    "negative_prompt": negative_prompt,
                }
                generation_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
                
                print("Generating video from image... (this may take several minutes)")
                
                # Clear cache before generation
                if self.is_mps:
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                with torch.no_grad():
                    if self.is_mps:
                        with torch.autocast(device_type='cpu', enabled=False):
                            result = pipe(**generation_kwargs)
                    else:
                        result = pipe(**generation_kwargs)
                    frames = result.frames[0]
                
                # Export video
                self._export_video(frames, output_path, fps, export_format)
                print(f"✅ Video saved to: {output_path}")
                print(f"🎯 Final settings used: {config['width']}x{config['height']}, {config['frames']} frames")
                return frames
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Attempt {i+1} failed: {error_msg}")
                
                # Check if it's a memory error
                is_memory_error = any(term in error_msg.lower() for term in [
                    'out of memory', 'memory', 'oom', 'cuda out of memory', 'mps backend out of memory'
                ])
                
                if is_memory_error:
                    print(f"🧠 Memory error detected, trying reduced settings...")
                    # Clear cache
                    if self.is_mps:
                        torch.mps.empty_cache()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    if not auto_fallback or i == len(configs) - 1:
                        print(f"💡 All configurations failed. Try manually setting lower values")
                        return None
                    else:
                        print(f"🔄 Trying next configuration...")
                        continue
                else:
                    # Non-memory error, don't continue
                    print(f"❌ Non-memory error occurred: {error_msg}")
                    return None
        
        print("❌ All fallback configurations failed")
        return None
    
    def _get_image_to_video_configs(self, original_width, original_height, num_frames, num_inference_steps):
        """
        Generate progressive fallback configurations for image-to-video generation
        Takes into account original image dimensions
        """
        configs = []
        
        # Calculate aspect ratio
        aspect_ratio = original_width / original_height
        
        if self.is_cpu:
            # CPU configurations - very conservative
            configs = [
                {"width": 512, "height": 512, "frames": 8, "steps": 15},
                {"width": 448, "height": 448, "frames": 8, "steps": 12},
                {"width": 384, "height": 384, "frames": 6, "steps": 10},
            ]
        elif self.is_mps:
            # MPS configurations - try to preserve aspect ratio when possible
            configs = []
            
            # Try original size first if reasonable
            if original_width <= 1024 and original_height <= 1024:
                configs.append({
                    "width": original_width, 
                    "height": original_height, 
                    "frames": min(num_frames, 16), 
                    "steps": min(num_inference_steps, 20)
                })
            
            # Add progressive fallbacks with aspect ratio preservation
            target_sizes = [(768, 768), (640, 640), (512, 512)]
            
            for target_w, target_h in target_sizes:
                # Try to preserve aspect ratio
                if aspect_ratio > 1:  # Landscape
                    width = target_w
                    height = int(target_w / aspect_ratio)
                    # Ensure height is divisible by 8 (requirement for diffusion models)
                    height = (height // 8) * 8
                    if height < 256:
                        height = 256
                else:  # Portrait or square
                    height = target_h
                    width = int(target_h * aspect_ratio)
                    # Ensure width is divisible by 8
                    width = (width // 8) * 8
                    if width < 256:
                        width = 256
                
                configs.append({
                    "width": width,
                    "height": height,
                    "frames": min(num_frames, 12),
                    "steps": min(num_inference_steps, 18)
                })
            
            # Add safe fallback
            configs.append({"width": 512, "height": 512, "frames": 8, "steps": 12})
            
        else:
            # GPU configurations
            configs = [
                # Try original dimensions if reasonable
                {"width": min(original_width, 1024), "height": min(original_height, 1024), "frames": num_frames, "steps": num_inference_steps},
                # Progressive fallbacks
                {"width": 768, "height": 768, "frames": 16, "steps": 20},
                {"width": 640, "height": 640, "frames": 12, "steps": 18},
                {"width": 512, "height": 512, "frames": 8, "steps": 12},
            ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_configs = []
        for config in configs:
            config_tuple = (config['width'], config['height'], config['frames'], config['steps'])
            if config_tuple not in seen:
                seen.add(config_tuple)
                unique_configs.append(config)
        
        return unique_configs
    
    def image_to_video_quick(self,
                            prompt,
                            input_image,
                            output_path="quick_img2vid.mp4",
                            export_format="mp4",
                            use_enhanced_prompting=True):
        """
        Quick image-to-video generation with Mac-optimized settings
        Perfect for testing and fast iterations
        
        Args:
            prompt (str): Text description of desired animation
            input_image (str or PIL.Image): Path to input image or PIL Image object
            output_path (str): Path to save generated video
            export_format (str): "mp4" or "gif"
            use_enhanced_prompting (bool): Add LUSTIFY-specific style tags
            
        Returns:
            list: List of PIL Images representing video frames
        """
        print("🚀 Quick image-to-video generation (Mac-optimized settings)")
        
        # Conservative settings for Mac
        if self.is_mps:
            return self.image_to_video(
                prompt=prompt,
                input_image=input_image,
                output_path=output_path,
                num_frames=8,
                num_inference_steps=15,
                guidance_scale=7.0,
                fps=8,
                export_format=export_format,
                use_enhanced_prompting=use_enhanced_prompting,
                auto_fallback=True
            )
        else:
            # CPU or other devices
            return self.image_to_video(
                prompt=prompt,
                input_image=input_image,
                output_path=output_path,
                num_frames=6,
                num_inference_steps=12,
                guidance_scale=6.0,
                fps=6,
                export_format=export_format,
                use_enhanced_prompting=use_enhanced_prompting,
                auto_fallback=True
            )
    
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
        
        Args:
            prompt (str): Text description of desired changes
            input_image (str or PIL.Image): Path to input image or PIL Image object
            output_path (str): Path to save generated image
            strength (float): How much to transform the image (0.0-1.0)
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): Guidance scale for generation
            use_enhanced_prompting (bool): Add LUSTIFY-specific style tags
            **kwargs: Additional generation parameters
        
        Returns:
            PIL.Image: Generated image
        """
        print(f"🖼️ Transforming image with prompt: '{prompt}'")
        
        # Load and prepare input image
        if isinstance(input_image, str):
            try:
                input_image = Image.open(input_image).convert("RGB")
                print(f"📷 Loaded input image: {input_image.size}")
            except Exception as e:
                print(f"❌ Error loading input image: {e}")
                return None
        
        # Enhance prompt for LUSTIFY model if requested
        if use_enhanced_prompting:
            enhanced_prompt = self._enhance_prompt_for_lustify(prompt)
            print(f"🎨 Enhanced prompt: '{enhanced_prompt}'")
            prompt = enhanced_prompt
        
        # Get pipeline
        pipe = self._get_pipeline("image-to-image")
        
        # Add LUSTIFY-specific negative prompt for better quality
        negative_prompt = kwargs.pop('negative_prompt', None)
        if not negative_prompt:
            negative_prompt = "blurry, low quality, distorted, deformed, extra limbs, bad anatomy"
        
        # Generation parameters
        generation_kwargs = {
            "prompt": prompt,
            "image": input_image,
            "strength": strength,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt,
        }
        generation_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
        
        try:
            print("Transforming image... (this may take a minute)")
            with torch.no_grad():
                # Mac-specific inference optimizations
                if self.is_mps:
                    # MPS sometimes has issues with certain operations
                    with torch.autocast(device_type='cpu', enabled=False):
                        result = pipe(**generation_kwargs)
                else:
                    result = pipe(**generation_kwargs)
                image = result.images[0]
            
            # Save image
            image.save(output_path)
            print(f"✅ Transformed image saved to: {output_path}")
            return image
            
        except Exception as e:
            print(f"❌ Error transforming image: {e}")
            print(f"💡 Try lowering strength or guidance scale")
            return None
    
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
        
        Args:
            prompt (str): Text description of what to paint in masked area
            input_image (str or PIL.Image): Path to input image or PIL Image object
            mask_image (str or PIL.Image): Path to mask image or PIL Image object (white = inpaint)
            output_path (str): Path to save generated image
            strength (float): How much to change the masked area (0.0-1.0)
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): Guidance scale for generation
            **kwargs: Additional generation parameters
        
        Returns:
            PIL.Image: Generated image with inpainted areas
        """
        print(f"🎨 Inpainting image with prompt: '{prompt}'")
        
        # Load and prepare input image
        if isinstance(input_image, str):
            try:
                input_image = Image.open(input_image).convert("RGB")
                print(f"📷 Loaded input image: {input_image.size}")
            except Exception as e:
                print(f"❌ Error loading input image: {e}")
                return None
        
        # Load and prepare mask image
        if isinstance(mask_image, str):
            try:
                mask_image = Image.open(mask_image).convert("L")  # Convert to grayscale
                print(f"🎭 Loaded mask image: {mask_image.size}")
            except Exception as e:
                print(f"❌ Error loading mask image: {e}")
                return None
        
        # Get pipeline
        pipe = self._get_pipeline("inpainting")
        
        # Add LUSTIFY-specific negative prompt for better quality
        negative_prompt = kwargs.pop('negative_prompt', None)
        if not negative_prompt:
            negative_prompt = "blurry, low quality, distorted, deformed, extra limbs, bad anatomy"
        
        # Generation parameters
        generation_kwargs = {
            "prompt": prompt,
            "image": input_image,
            "mask_image": mask_image,
            "strength": strength,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt,
        }
        generation_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
        
        try:
            print("Inpainting... (this may take a minute)")
            with torch.no_grad():
                # Mac-specific inference optimizations
                if self.is_mps:
                    # MPS sometimes has issues with certain operations
                    with torch.autocast(device_type='cpu', enabled=False):
                        result = pipe(**generation_kwargs)
                else:
                    result = pipe(**generation_kwargs)
                image = result.images[0]
            
            # Save image
            image.save(output_path)
            print(f"✅ Inpainted image saved to: {output_path}")
            return image
            
        except Exception as e:
            print(f"❌ Error inpainting image: {e}")
            print(f"💡 Try lowering strength or guidance scale")
            return None


# Example usage
if __name__ == "__main__":
    # Initialize the generator
    generator = ImageVideoGenerator()
    
    # Example 1: Text-to-image generation (your existing workflow)
    image1 = generator.text_to_image(
        prompt="photograph, photo of monkey hanging from a tree, 8k",
        output_path="step1_text2img.png"
    )
    
    # Example 2: Text-to-video generation
    if image1:
        """print("\n🎬 Trying text-to-video generation...")
        text_video = generator.text_to_video_quick(
            prompt="photograph, photo of monkey eating a banana, monkey moving naturally, 8k",
            output_path="step2_text2video.mp4"
        )"""
        if image1:
        #if text_video:
            print("✅ Text-to-video generation successful!")
            
            # Example 3: Image-to-video generation (animate the generated image)
            print("\n🎬 Now trying image-to-video generation...")
            img_video = generator.image_to_video_quick(
                prompt="monkey moving naturally, eating banana, subtle head movements, chewing motion",
                input_image="step1_text2img.png",  # Use the generated image
                output_path="step3_img2video.mp4"
            )

            #You control all the settings
            video = generator.image_to_video(
                prompt="gentle movement",
                input_image="step1_text2img.png",
                width=768,           # Your choice
                height=768,          # Your choice  
                num_frames=12,       # Your choice
                num_inference_steps=18,
                auto_fallback=True   # Falls back if memory fails
            )
                        
            if img_video:
                print("🎉 All generation types complete!")
                print("Generated files:")
                print(f"  - Original Image: step1_text2img.png")
                print(f"  - Text-to-Video: step2_text2video.mp4")
                print(f"  - Image-to-Video: step3_img2video.mp4")
                print("\n💡 Compare the videos:")
                print("  - Text-to-video: Creates entirely new video based on prompt")
                print("  - Image-to-video: Animates your existing image with motion")
            else:
                print("🎉 Text-to-video generation complete!")
                print("Generated files:")
                print(f"  - Original Image: step1_text2img.png")
                print(f"  - Text-to-Video: step2_text2video.mp4")
                print("💡 Image-to-video failed - try with smaller image or different prompt")
        else:
            print("❌ Text-to-video generation failed.")
            print("💡 Try with even lower settings or check your system resources")