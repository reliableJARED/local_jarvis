#!/usr/bin/env python3
"""
Text-to-Image Generation Class
Uses diffusers library to generate images from text prompts
Updated to use LUSTIFY-SDXL-NSFW-checkpoint-v2-0-INPAINTING model
"""

import os
import sys
import subprocess
from pathlib import Path
import torch
from PIL import Image
import uuid
import time
from diffusers import DiffusionPipeline, AutoPipelineForInpainting, AutoPipelineForImage2Image
from lustify_xwork import ImageGenerator

gpu_count = torch.cuda.device_count()
print(f"Number of GPUs available: {gpu_count}")

#when running in terminal use this to set the GPU
#$env:CUDA_VISIBLE_DEVICES="0"
#python lustify_xwork.py


class ImageGenerator:
    """
    A class for generating images using diffusion models.
    Supports text-to-image, image-to-image, and inpainting capabilities.
    """
    
    def __init__(self, 
                 model_name="TheImposterImposters/LUSTIFY-v2.0",
                 inpaint_model="andro-flock/LUSTIFY-SDXL-NSFW-checkpoint-v2-0-INPAINTING",
                 cache_dir=None,
                 use_mps=True,
                 use_cuda=None):
        """
        Initialize the ImageGenerator with model loading and dependency checking.
        
        Args:
            model_name (str): Main model for text-to-image and image-to-image
            inpaint_model (str): Model for inpainting tasks
            cache_dir (str): Cache directory for model storage
            use_mps (bool): Use Metal Performance Shaders on Mac (if available)
            use_cuda (bool): Use CUDA if available (None=auto-detect)
        """
        self.model_name = model_name
        self.inpaint_model = inpaint_model
        self.cache_dir = cache_dir or self._setup_cache_dir()
        
        # Enhanced device configuration for Mac and CUDA
        self.device = self._get_best_device(use_mps, use_cuda)
        self.is_cpu = self.device == "cpu"
        self.is_mps = self.device == "mps"
        self.is_cuda = self.device == "cuda"
        
        # Install dependencies and check
        self._install_dependencies()
        self._check_dependencies()
        
        # Initialize pipelines (will be loaded on demand)
        self.text2img_pipe = None
        self.img2img_pipe = None
        self.inpaint_pipe = None
        
        print(f"✅ ImageGenerator initialized")
        print(f"Device: {self.device}")
        if self.is_cuda:
            print("🚀 Using CUDA GPU acceleration!")
        elif self.is_mps:
            print("🚀 Using Metal Performance Shaders for acceleration!")
        print(f"Cache directory: {self.cache_dir}")
    
    def _get_best_device(self, use_mps=True, use_cuda=None):
        """Determine the best available device with CUDA priority"""
        # Auto-detect CUDA if not explicitly specified
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        
        if use_cuda and torch.cuda.is_available():
            return "cuda"
        elif use_mps and torch.backends.mps.is_available():
            # MPS (Metal Performance Shaders) for Mac acceleration
            return "mps"
        else:
            return "cpu"
    
    def _install_dependencies(self):
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
            print("✓ All dependencies verified")
        except ImportError as e:
            print(f"❌ Dependency check failed: {e}")
            print("Please run: pip install torch diffusers pillow transformers accelerate")
            sys.exit(1)
    
    def _setup_cache_dir(self):
        """Setup cache directory for model storage"""
        cache_dir = Path.home() / ".cache" / "text_to_image_demo"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def _load_pipeline(self, pipeline_type="text-to-image"):
        """
        Load the appropriate diffusion pipeline
        
        Args:
            pipeline_type (str): "text-to-image", "image-to-image", or "inpainting"
        
        Returns:
            Pipeline object
        """
        

        # Determine which model to use
        if pipeline_type == "inpainting":
            model_name = self.inpaint_model
        else:
            model_name = self.model_name
        
        print(f"Loading {pipeline_type} pipeline: {model_name}")
        
        try:
            # Configure model loading parameters with device-specific optimizations
            kwargs = {
                "cache_dir": self.cache_dir,
                "use_safetensors": True,
            }
            
            # Set torch_dtype based on device
            if self.is_cuda:
                kwargs["torch_dtype"] = torch.float32  # CUDA works well with float16
            else:
                kwargs["torch_dtype"] = torch.float32  # MPS and CPU work better with float32
            
            # Load the appropriate pipeline
            print("Downloading/loading model (this may take a few minutes on first run)...")
            if pipeline_type == "inpainting":
                pipe = AutoPipelineForInpainting.from_pretrained(model_name, **kwargs)
            elif pipeline_type == "image-to-image":
                pipe = AutoPipelineForImage2Image.from_pretrained(model_name, **kwargs)
            else:
                pipe = DiffusionPipeline.from_pretrained(model_name, **kwargs)
            
            pipe = pipe.to(self.device)
            
            # Configure scheduler per documentation recommendations (DPM++ 2M SDE with Karras)
            try:
                from diffusers import DPMSolverMultistepScheduler
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config,
                    use_karras_sigmas=True,
                    algorithm_type="sde-dpmsolver++"  # For DPM++ SDE variant
                )
                print("✓ Configured DPM++ 2M SDE scheduler with Karras sigmas")
            except Exception as e:
                print(f"⚠️  Could not configure DPM++ scheduler, using default: {e}")
            
            # Device-specific optimizations
            if self.is_cuda:
                # CUDA optimizations
                print("🔧 Applying CUDA optimizations...")
                if hasattr(pipe, 'enable_attention_slicing'):
                    pipe.enable_attention_slicing()
                
                # Enable model CPU offload for memory efficiency
                try:
                    if hasattr(pipe, 'enable_model_cpu_offload'):
                        pipe.enable_model_cpu_offload()
                        print("✓ Model CPU offload enabled for CUDA")
                except Exception as e:
                    print(f"⚠️  Model CPU offload not available: {e}")
                
                # Enable memory efficient attention if available
                try:
                    if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
                        pipe.enable_xformers_memory_efficient_attention()
                        print("✓ XFormers memory efficient attention enabled")
                except Exception as e:
                    print(f"⚠️  XFormers not available: {e}")
            
            elif self.is_mps:
                # MPS optimizations
                print("🔧 Applying MPS optimizations...")
                if hasattr(pipe, 'enable_attention_slicing'):
                    pipe.enable_attention_slicing(1)  # More aggressive slicing for MPS
                
                # Enable sequential CPU offload for better memory management on Mac
                try:
                    if hasattr(pipe, 'enable_sequential_cpu_offload'):
                        pipe.enable_sequential_cpu_offload()
                        print("✓ Sequential CPU offload enabled for MPS")
                except Exception as e:
                    print(f"⚠️  Sequential CPU offload not available: {e}")
            
            else:
                # CPU optimizations
                if hasattr(pipe, 'enable_attention_slicing'):
                    pipe.enable_attention_slicing(1)
                
                # Set number of threads for CPU inference
                torch.set_num_threads(torch.get_num_threads())
                print(f"🔧 Using {torch.get_num_threads()} CPU threads")
            
            print(f"✅ {pipeline_type.title()} pipeline loaded successfully!")
            return pipe
            
        except Exception as e:
            print(f"❌ Error loading {pipeline_type} pipeline: {e}")
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
        elif self.is_cuda:
            # CUDA can handle full resolution and steps efficiently
            print(f"🚀 Running on CUDA - Full SDXL resolution: {width}x{height}, {num_inference_steps} steps")
        
        # Add LUSTIFY-specific negative prompt for better quality
        negative_prompt = kwargs.pop('negative_prompt', None)
        if not negative_prompt:
            negative_prompt = "blurry, low quality, distorted, deformed, extra limbs, bad anatomy, hands"
        
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
                # Device-specific inference optimizations
                if self.is_mps:
                    # MPS sometimes has issues with certain operations
                    with torch.autocast(device_type='cpu', enabled=False):
                        result = pipe(prompt, **generation_kwargs)
                elif self.is_cuda:
                    # Use autocast for CUDA to improve performance
                    with torch.autocast(device_type='cuda', dtype=torch.float32):
                        result = pipe(prompt, **generation_kwargs)
                else:
                    # CPU inference
                    result = pipe(prompt, **generation_kwargs)
                image = result.images[0]
            
            # Save image
            image.save(output_path)
            print(f"✅ Image saved to: {output_path}")
            return image
            
        except Exception as e:
            print(f"❌ Error generating image: {e}")
            print(f"💡 Try lowering strength (current: strength) for more consistency")
            print(f"💡 Or enhance your prompt with camera/lighting details")
            return None
    
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
            prompt (str): Text description of desired transformation
            input_image (str or PIL.Image): Path to input image or PIL Image object
            output_path (str): Path to save generated image
            strength (float): Transformation strength (0.0-1.0, LOWER=more consistency)
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): Guidance scale for generation
            use_enhanced_prompting (bool): Add LUSTIFY-specific style tags
            **kwargs: Additional generation parameters
        
        Returns:
            PIL.Image: Generated image
        """
        print(f"🖼️  Transforming image with prompt: '{prompt}'")
        print(f"💡 Using strength: {strength} (lower=more consistent with original)")
        
        # Enhance prompt for LUSTIFY model if requested
        if use_enhanced_prompting:
            enhanced_prompt = self._enhance_prompt_for_lustify(prompt)
            print(f"🎨 Enhanced prompt: '{enhanced_prompt}'")
            prompt = enhanced_prompt
        
        # Get pipeline
        pipe = self._get_pipeline("image-to-image")
        
        # Load and prepare input image
        if isinstance(input_image, str):
            try:
                input_image = Image.open(input_image).convert("RGB")
            except Exception as e:
                print(f"❌ Error loading input image: {e}")
                return None
        
        # Get image dimensions for consistency
        width, height = input_image.size
        
        # Adjust parameters for device capabilities while respecting SDXL's optimal resolution
        if self.is_cpu:
            # Resize if needed, but try to maintain aspect ratio
            if width > 768 or height > 768:
                # Scale down proportionally
                scale = min(768/width, 768/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                input_image = input_image.resize((new_width, new_height))
                width, height = new_width, new_height
            num_inference_steps = min(num_inference_steps, 20)
            print(f"⚠️  Running on CPU - using optimized settings: {width}x{height}, {num_inference_steps} steps")
        elif self.is_mps:
            # MPS can handle full SDXL resolution - keep original dimensions
            num_inference_steps = min(num_inference_steps, 30)
            print(f"🚀 Running on MPS - SDXL optimized: {width}x{height}, {num_inference_steps} steps")
        elif self.is_cuda:
            # CUDA can handle full resolution efficiently
            print(f"🚀 Running on CUDA - Full resolution: {width}x{height}, {num_inference_steps} steps")
        
        # Add LUSTIFY-specific negative prompt for better quality
        negative_prompt = kwargs.pop('negative_prompt', None)
        if not negative_prompt:
            negative_prompt = "blurry, low quality, distorted, deformed, extra limbs, bad anatomy"
        
        # Generation parameters
        generation_kwargs = {
            "image": input_image,
            "strength": strength,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt,
        }
        generation_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
        
        try:
            print("Generating... (this may take a minute)")
            with torch.no_grad():
                if self.is_mps:
                    with torch.autocast(device_type='cpu', enabled=False):
                        result = pipe(prompt, **generation_kwargs)
                elif self.is_cuda:
                    with torch.autocast(device_type='cuda', dtype=torch.float32):
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
            prompt (str): Text description of desired inpainting
            input_image (str or PIL.Image): Path to input image or PIL Image object
            mask_image (str or PIL.Image): Path to mask image or PIL Image object
                                         (white=inpaint, black=keep)
            output_path (str): Path to save generated image
            strength (float): Inpainting strength (0.0-1.0, higher=more change)
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): Guidance scale for generation
            **kwargs: Additional generation parameters
        
        Returns:
            PIL.Image: Generated image
        """
        print(f"🎭 Inpainting image with prompt: '{prompt}'")
        
        # Get pipeline
        pipe = self._get_pipeline("inpainting")
        
        # Load and prepare images
        try:
            if isinstance(input_image, str):
                input_image = Image.open(input_image).convert("RGB")
            if isinstance(mask_image, str):
                mask_image = Image.open(mask_image).convert("RGB")
        except Exception as e:
            print(f"❌ Error loading images: {e}")
            return None
        
        # Ensure images are same size
        width, height = input_image.size
        mask_image = mask_image.resize((width, height))
        
        # Adjust parameters for device capabilities
        if self.is_cpu:
            if width > 512 or height > 512:
                input_image = input_image.resize((512, 512))
                mask_image = mask_image.resize((512, 512))
                width, height = 512, 512
            num_inference_steps = min(num_inference_steps, 15)
            print(f"⚠️  Running on CPU - using reduced settings: {width}x{height}, {num_inference_steps} steps")
        elif self.is_mps:
            if width > 768 or height > 768:
                input_image = input_image.resize((768, 768))
                mask_image = mask_image.resize((768, 768))
                width, height = 768, 768
            num_inference_steps = min(num_inference_steps, 30)
            print(f"🚀 Running on MPS - optimized settings: {width}x{height}, {num_inference_steps} steps")
        elif self.is_cuda:
            # CUDA can handle larger images better
            if width > 1024 or height > 1024:
                input_image = input_image.resize((1024, 1024))
                mask_image = mask_image.resize((1024, 1024))
                width, height = 1024, 1024
            print(f"🚀 Running on CUDA - high resolution: {width}x{height}, {num_inference_steps} steps")
        
        # Generation parameters
        generation_kwargs = {
            "image": input_image,
            "mask_image": mask_image,
            "strength": strength,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        generation_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
        
        try:
            print("Generating... (this may take a minute)")
            with torch.no_grad():
                if self.is_mps:
                    with torch.autocast(device_type='cpu', enabled=False):
                        result = pipe(prompt, **generation_kwargs)
                elif self.is_cuda:
                    with torch.autocast(device_type='cuda', dtype=torch.float32):
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
            return None


# Example usage
if __name__ == "__main__":
    
    # Initialize the generator
    #generator = ImageGenerator()
    #BEST - PROMPT STRUCTURE GUIDE - using a consistent [subject] [important feature], [more details] description will create similar subject"
    "[style of photo] photo of a [subject], [important feature], [more details], [pose or action], [framing], [setting/background], [lighting], [camera angle], "
    style = "photograph, photo of "
    subject = "a sexy woman, "
    skin = "tan skin, "
    hair = "short face-framing blond hair with bangs, "
    face = "high cheekbones, "
    eyes = "brown eyes, "
    attribute = "long eyelashes, "#glasses, etc
    lips = "full lips, "
    chest = "c-cup full breasts, "#don't always need this?
    pose = "keeling down, "#laying
    #action = "twisting nipples with her mouth open and tounge out, " #x1
    #action = "sucking a penis in her mouth, " #x2
    #action = "deepthroat a penis her lips near the base, "#x4
    #action = "cums in to her open mouth, cum on her tounge and there is cum on her face, " #x3
    #action = "ejaculation in to her open mouth, cum on her tounge and there is cum on her face, " #x5
    #action = "performing oral sex on a penis inside of her mouth her, " #x6
    #action = "holding a penis with bother her hands jerking it off, " #x7
    #action = "in front of a man his cock in her mouth, " #x8 DO NOT USE !!! Don't say 'man' SO BADD - Lady and the Tramp style sharing a single cock BAD PROMPT
    #action = "her mouth around a penis, " #x9
    #action = "a penis entering her open mouth as she swallows it, " #x10
    #action = "her hand around the base of a penis her mouth around the end of it, " #x11
    #action = "fellatio, " #x12
    action = "fellatio deep, " #x13
    framing = "looking up at the camera seductivly, "
    clothes = "wearing the black lace lingerie, "# or naked, nude
    lighting = "soft dim lighting, 8k"
    print(len(style+subject+skin+hair+face+eyes+attribute+lips+chest+pose+action+framing+clothes+lighting))

    # Example 1: Text-to-image generation
    """image1 = generator.text_to_image(
         
        #prompt="photograph,photo of a sexy woman, white skin, short face framing red hair, glasses, high cheekbones, full lips and blue eyes with long lashes, kneeling down sucking a cock, looking up at the camera. wearing the black lace lingerie, 8k",
        #prompt="photograph,photo of a sexy woman, white skin, short face framing red hair, glasses, high cheekbones, full lips and blue eyes with long lashes, she laying on a bed topless her breast are naked and nipples showing, her legs are spread apart the camera is looking at her from the foot of the bed her vagina is visible, soft lighting 8k",
        #prompt="photograph,photo of a sexy woman, white skin, short face framing red hair, glasses, high cheekbones, full lips and blue eyes with long lashes, she laying on a bed topless her breast are naked and nipples showing, her legs are spread apart seductive look, the camera is looking at her from the foot of the bed her vagina is visible, soft lighting 8k",
        #prompt="photograph,photo of a sexy woman, white skin, short face framing red hair, glasses, high cheekbones, full lips and blue eyes with long lashes, she laying on a bed topless her breast are naked and nipples showing, her legs are spread apart seductive look, the camera is looking at her from the foot of the bed her vagina is being penetrated by a cock, soft lighting 8k",
        #prompt="photograph,photo of a sexy woman, white skin, short face framing red hair, glasses, high cheekbones, full lips and blue eyes with long lashes, she laying on a bed topless her breast are naked and nipples showing, her legs are spread apart climax look, the camera is looking at her from the foot of the bed her vagina is being penetrated by a cock, soft lighting 8k",
        #prompt="photograph,photo of a sexy woman, white skin, short face framing red hair, glasses, high cheekbones, full lips and blue eyes with long lashes, she is naked on all fours on a bed from the side of the view a cock is entering her vagina from behind, the camera is looking at her from the side, soft lighting 8k",
        #prompt="photograph,photo of a sexy woman, white skin, short face framing red hair, glasses, high cheekbones, full lips and blue eyes with long lashes, she is naked on all fours on a bed a penis is entering her vagina from behind, the camera is looking at her from the side, soft lighting 8k",
        #prompt="photograph,photo of a sexy woman, white skin, short face framing red hair, glasses, high cheekbones, full lips and blue eyes with long lashes, she is naked on all fours on a bed a penis is entering her vagina from behind, the camera is looking at her from the side, her back arched and looking up at the ceiling, soft lighting 8k",
        #prompt="photograph,photo of a sexy woman, white skin, short face framing red hair, glasses, high cheekbones, full lips and blue eyes with long lashes, she is straddling a man as she puts his cock inside of her and slides down on it, she is looking down at the camera in a passionate gaze, soft lighting 8k",
        #prompt="photograph,photo of a sexy woman, white skin, short face framing red hair, glasses, high cheekbones, full lips and blue eyes with long lashes, she is looking down at the camera as she positions his penis inside of her vagina, soft lighting 8k",
        #prompt="photograph,photo of a sexy woman, white skin, short face framing red hair, glasses, high cheekbones, full lips and blue eyes with long lashes, she is kneeling looking up at the camera as his penis cums in to her open mouth, cum on her tounge and there is cum on her face and glasses, soft lighting 8k",
        prompt=style+subject+skin+hair+face+eyes+attribute+lips+chest+pose+action+framing+clothes+lighting,

        output_path=f"{uuid.uuid4()}.png"
    )"""
    
    #Image to Image does NOT work well DO NOT BOTHER USING
    #Leaving in place as it may just need more work but research online suggest it's not very effective
    """if image1:
        # Example 2: Image-to-image transformation with LOWER strength for consistency
        image2 = generator.image_to_image(
            prompt="photograph, photo of the same ape swinging from a tree wide angle camera, same scene, same lighting, 8k",
            input_image="step1_text2img.png",
            output_path="step2_img2img.png",
            strength=0.35  # Much lower for better consistency!
        )
        
        if image2:
            print("🎉 Two-step demo complete!")
            print("Generated images:")
            print(f"  - Original: step1_text2img.png")
            print(f"  - Transformed: step2_img2img.png")"""
    
            
    def yseries():
        generator = ImageGenerator()
        style = "photograph, photo of "
        subject = "a sexy woman,"
        skin = "tan skin,"
        hair = "short face-framing blond hair with bangs,"
        face = "high cheekbones,"
        eyes = "brown eyes,"
        attribute = "long eyelashes,"#glasses, etc
        lips = "full lips,"
        chest = "c-cup full breasts,"#don't always need this?
        pose = {"p1":"laying on a bed,",
                "p2":"laying on a bed topless,"}
        #her legs are spread apart the camera is looking at her from the foot of the bed her vagina is visible and a penis is entering her vagina
        action = {"a1": "legs spread apart,",
                  "a2": "legs spread apart her vagina is visible,",
                  "a3": "legs spread apart her vagina is being entered by a penis,",
                  "a4": "legs spread apart having sex,",
                  "a5": "legs spread apart ejaculate is dripping out of her vagina,",
                  "a6": "legs spread apart missionary sex," } 
        framing ={"f1":"the camera is looking at her from the foot of the bed,",
                  "f2":"the camera is above her looking down from the foot of the bed,"}
        clothes = {"c1":"wearing black lace lingerie,",
                   "c2":"she is naked"}# or naked, she is naked nude #wearing the black lace lingerie,
        lighting = "soft lighting"

        combos = ["p1a1f1c1","p2a125f1c2","p2a346f2c2"]

        folder = "yseries"
        for i in range(len(combos)):
            pafc = parse_combination(combos[i],pose,action,framing, clothes)
            for i in range(len(pafc)):
                prompt = " ".join([style, subject, skin, hair, face, eyes, attribute, lips, chest, pafc[i]['prompt'], lighting])
                print(f"Prompt Char Length (Stay under 300! or 77 tokens):{len(prompt)}")
                print(f"Prompt :{prompt}")
                image1 = generator.text_to_image(
                prompt=prompt,
                output_path=f"./{folder}/y_{pafc[i]['keys']}.png"
                )
    def zseries():
        generator = ImageGenerator()
        style = "photograph, photo of "
        subject = "a sexy woman,"
        skin = "tan skin,"
        hair = "short face-framing blond hair with bangs,"
        face = "high cheekbones,"
        eyes = "brown eyes,"
        attribute = "long eyelashes,"#glasses, etc
        lips = "full lips,"
        chest = ""#don't always need this?
        pose = {"p1":"leaning against a brick wall,",
                "p2":"leaning over a table,",
                "p3":"sitting in an arm chair,",
                "p4":"sitting in a white arm chair,",
                "p5":"standing next to a white arm chair,",
                "p6":"standing on a balcony,",
                "p7":"standing in front of a bathroom mirror,"} #BAD!!! don't use mirror
        #her legs are spread apart the camera is looking at her from the foot of the bed her vagina is visible and a penis is entering her vagina
        action = {"a1": "one knee raised,",
                  "a2": "one leg raised,",
                  "a3": "legs spread apart,", #generates NSFW
                  "a4": "legs crossed,",
                  "a5": "legs infront,",
                  "a6": "laying upside down, legs in the air back of chair,",
                  "a7":"arm resting across the back of the chair,",
                  "a8":"leaning over railing looking out at city,",
                  "a9":"putting on lipstick,",
                  "a0":"brushing hair,"} 
        framing ={"f1":"seductive stare at camera looking at her from the side,",
                  "f2":"intimate stare at camera looking at her across the table,",
                  "f3":"sexual stare at camera looking directly at her,",
                  "f4":"sexual stare at camera finger in her mouth,",
                  "f5":"sexual stare at camera hand on her chest,",
                  "f6":"sexual stare at camera hand touching her thigh,",
                  "f7":"camera is behind her viewing her body from the side,",
                  "f8":"camera is looking through doorway viewing her body from the side,",}
        clothes = {"c1":"wearing black mini skirt white tank top shirt, high heals",
                   "c2":"wearing tight short red dress, high heals",
                   "c3":"wearing short sequin dress, bare feet",
                   "c4":"wearing short sequin dress, high heals",
                   "c5":"wearing strapless sequin dress, high heals"}# or naked, she is naked nude #wearing the black lace lingerie,
        lighting = "sunlight from window"

        #combos = ["p1a1f1c1","p2a2f2c2","p3a34f3c3"]
        #combos = ["p4a56f3c3"]
        #combos = ["p5a7f4c3","p5a7f5c3","p5a7f6c3"]
        #combos = ["p6a8f7c5","p6a8f7c4"]
        combos = ["p7a90f7c5","p7a90f8c5"]
        

        folder = "zseries"
        for i in range(len(combos)):
            pafc = parse_combination(combos[i],pose,action,framing, clothes)
            for i in range(len(pafc)):
                prompt = " ".join([style, subject, skin, hair, face, eyes, attribute, lips, chest, pafc[i]['prompt'], lighting])
                print(f"Prompt Char Length (Stay under 300! or 77 tokens):{len(prompt)}")
                print(f"Prompt :{prompt}")
                image1 = generator.text_to_image(
                prompt=prompt,
                output_path=f"./{folder}/z_{pafc[i]['keys']}.png"
                )
    def xseries():
        style = "photograph, photo of "
        subject = "a sexy woman, "
        skin = "tan skin, "
        hair = "short face-framing blond hair with bangs, "
        face = "high cheekbones, "
        eyes = "brown eyes, "
        attribute = "long eyelashes, "#glasses, etc
        lips = "full lips, "
        chest = "c-cup full breasts, "#don't always need this?
        pose = "keeling down, "#laying
        action = {"x1": "twisting nipples with her mouth open and tounge out, ", #x1
            "x2": "sucking a penis in her mouth, ", #x2
            "x4": "deepthroat a penis her lips near the base, ",#x4
            "x3": "cums in to her open mouth, cum on her tounge and there is cum on her face, ", #x3
            "x5": "ejaculation in to her open mouth, cum on her tounge and there is cum on her face, ", #x5
            "x6": "performing oral sex on a penis inside of her mouth her, ", #x6
            "x7": "holding a penis with bother her hands jerking it off, ", #x7
            "x8": "in front of a man his cock in her mouth, ", #x8 DO NOT USE !!! Don't say 'man' SO BADD - Lady and the Tramp style sharing a single cock BAD PROMPT
            "x9": "her mouth around a penis, ", #x9
            "x10": "a penis entering her open mouth as she swallows it, ", #x10
            "x11": "her hand around the base of a penis her mouth around the end of it, ", #x11
            "x12": "fellatio, ", #x12
            "x13": "fellatio deep, "} 
        framing = "looking up at the camera seductivly, "
        clothes = "wearing the black lace lingerie, "# or naked, nude
        lighting = "soft dim lighting, 8k"
        print(f"Prompt Char Length (Stay under 300!):{len(style+subject+skin+hair+face+eyes+attribute+lips+chest+pose+action+framing+clothes+lighting)}")
    
    

    def photoshoot(combos,style,lighting,subject,skin,hair,face,eyes,attribute,lips,chest,pose,action,framing,clothes,shoot_folder="demo_shoot",name="demo",main_output_dir = "xserver",test=False):
        """
        Image Generator Function
        Prompting Guide/Tips
         IMPORTANT - text_to_image() of generator already has ['hands'] as a negative prompt, this has been shown to help SDXL 
         (https://www.reddit.com/r/StableDiffusion/comments/18rn3aq/guide_hands_and_how_to_fix_them/)
         
         #PROMPT STRUCTURE - using a consistent [subject] [important feature], [more details] description will create similar subject"
            - With prompts, colors like for eyes, clothes etc. seem to heavily influence between images. So mentioning some colors helps to keep consitent look across images

        args:
            combos: (list) this is a coded list such as 'p1a123f1c1' that indicates what prompts to combine to create an image series
            
            style: (str) "photograph, photo of"
            lighting: (str) "soft lighting 8k"
            subject: (str) "sexy woman"
            skin: (str) "white skin"
            hair: (str) "short face-framing blond hair with bangs"
            face: (str) "high cheekbones"
            eyes: (str) "brown eyes"
            attribute: (str) "long eyelashes"
            lips: (str) "full lips"
            chest: (str) "c-cup full breasts"
            
            shoot_folder (str) save directory
            name (str) file naming
            
        - 
        """
        generator = ImageGenerator()
        # Or explicitly enable/disable CUDA
        # generator = ImageGenerator(use_cuda=True)  # Force CUDA if available
        # generator = ImageGenerator(use_cuda=False)  # Disable CUDA

        if test:
            print("test shoot >>>>>>>>>>>>")
            #create director to save results
            os.makedirs(f"{main_output_dir}", exist_ok=True)
            os.makedirs(f"{main_output_dir}/{name}", exist_ok=True)  
            os.makedirs(f"{main_output_dir}/{name}/{shoot_folder}", exist_ok=True)
            output_path = f"./{main_output_dir}/{name}/{shoot_folder}/{name}_{str(uuid.uuid4())}.png"
            image1 = generator.text_to_image(
                    prompt=combos,
                    output_path=output_path
                )
            return True
            
        print("========= STARTING PHOTO SHOOT ==========")
        print(f"=========   {shoot_folder}    ==========")
        print("================= with ===================")
        print(f"=============   {name}    ==============")
        print("=========================================")

        #create director to save results
        os.makedirs(f"{main_output_dir}", exist_ok=True)
        os.makedirs(f"{main_output_dir}/{name}", exist_ok=True)  
        os.makedirs(f"{main_output_dir}/{name}/{shoot_folder}", exist_ok=True)


        

        folder = shoot_folder

        for i in range(len(combos)):
            pafc = parse_combination(combos[i], pose, action, framing, clothes)
            for j in range(len(pafc)):  
                output_path = f"./{main_output_dir}/{name}/{folder}/{name}_in_{folder}-{pafc[j]['keys']}.png"
                
                # Check if file already exists
                if os.path.exists(output_path):
                    print(f"File already exists, skipping: {output_path}")
                    continue
                
                prompt = ", ".join([style, subject, skin, hair, face, eyes, attribute, lips, chest, pafc[j]['prompt'], lighting])
                #print(f"Prompt Char Length (Stay under 300! or 77 tokens): {len(prompt)}")
                print(f"Prompt: {prompt}")
                print(f"Photoshoot {folder} - Model: {name} - series {i} of {len(combos)} , series image {j} of {len(pafc)}")
                
                image1 = generator.text_to_image(
                    prompt=prompt,
                    output_path=output_path
                )

        return True

    def dionysus(subject,skin,hair,face,eyes,attribute,lips,chest,name="demo",fabric="white"):
        generator = ImageGenerator()

        #defaults
        style = "photograph, photo of "
        lighting = "soft lighting, 8k"

        pose = {"p1":"on hands and knees,",
                "p2":"standing bent over a table,"}
        action = {"a1": "penis entering her anus,", #x1
            "a2": "penis inside her vagina,", #x2
            "a3": "hair is being pulled back arched penis in her vagina,",#x4
            "a4": "ejaculation dripping from her vagina,"} 
        framing = {"f1":"looking at her ass from above and behind,",
                    "f2":"looking at her from the side,",
                    "f3":"she is looking back over her shoulder"}
        clothes = {"c1":f"wearing {fabric} stockings,"}# or naked, nude
        combos = ["p1a1234f1c1","p2a1234f2c1","p1a1234f3c1"]
        
        folder = "dionysus"
        
        result = photoshoot(combos,style,lighting,subject,skin,hair,face,eyes,attribute,lips,chest,pose,action,framing,clothes,shoot_folder=folder,name=name)
        print(result)
        
        """for i in range(len(combos)):
            pafc = parse_combination(combos[i], pose, action, framing, clothes)
            for j in range(len(pafc)):  
                output_path = f"./{name}/{folder}/{name}_in_{folder}-{pafc[j]['keys']}.png"
                
                # Check if file already exists
                if os.path.exists(output_path):
                    print(f"File already exists, skipping: {output_path}")
                    continue
                
                prompt = " ".join([style, subject, skin, hair, face, eyes, attribute, lips, chest, pafc[j]['prompt'], lighting])
                print(f"Prompt Char Length (Stay under 300! or 77 tokens): {len(prompt)}")
                print(f"Prompt: {prompt}")
                print(f"Photo shoot {folder} - - - generating {j} of {len(pafc)}")
                
                image1 = generator.text_to_image(
                    prompt=prompt,
                    output_path=output_path
                )"""

    def apollo(subject,skin,hair,face,eyes,attribute,lips,chest,name="demo",fabric="white"):
        generator = ImageGenerator()
        """
        style = "photograph, "
        subject = "a sexy woman, "
        skin = "tan skin, "
        hair = "short face-framing blond hair with bangs, "
        face = "high cheekbones, "
        eyes = "brown eyes, "
        attribute = "long eyelashes, "#glasses, etc
        lips = "full lips, "
        chest = "c-cup full breasts, "#don't always need this?
        pose = "keeling down, "#laying
        """
        #defaults
        style = "photograph, photo of "
        lighting = "soft dim lighting, 8k"

        pose = {"p1":"keeling down,"}
        action = {"a1": "twisting nipples with her mouth open and tounge out,", #x1
            "a2": "sucking a penis in her mouth,", #x2
            "a3": "deepthroat a penis her lips near the base, ",#x4
            "a4": "cums in to her open mouth, cum on her tounge and there is cum on her face,", #x3
            "a5": "ejaculation in to her open mouth, cum on her tounge and there is cum on her face,", #x5
            "a6": "performing oral sex on a penis inside of her mouth her,", #x6
            "a7": "holding a penis with bother her hands jerking it off,", #x7
            "a8": "a penis entering her open mouth as she swallows it,", #x10
            "a9": "her hand around the base of a penis her mouth around the end of it,", #x11
            "a0": "fellatio,"} 
        framing = {"f1":"looking up at the camera seductivly,"}
        clothes = {"c1":f"wearing the {fabric} lace lingerie,"}# or naked, nude
        combos = ["p1a1234567890f1c1"]

        folder = "apollo"
        result = photoshoot(combos,style,lighting,subject,skin,hair,face,eyes,attribute,lips,chest,pose,action,framing,clothes,shoot_folder=folder,name=name)
        print(result)
        """
        for i in range(len(combos)):
            pafc = parse_combination(combos[i], pose, action, framing, clothes)
            for j in range(len(pafc)):  
                output_path = f"./{name}/{folder}/{name}_in_{folder}-{pafc[j]['keys']}.png"
                
                # Check if file already exists
                if os.path.exists(output_path):
                    print(f"File already exists, skipping: {output_path}")
                    continue
                
                prompt = " ".join([style, subject, skin, hair, face, eyes, attribute, lips, chest, pafc[j]['prompt'], lighting])
                print(f"Prompt Char Length (Stay under 300! or 77 tokens): {len(prompt)}")
                print(f"Prompt: {prompt}")
                print(f"Photo shoot {folder} - - - generating {j} of {len(pafc)}")
                
                image1 = generator.text_to_image(
                    prompt=prompt,
                    output_path=output_path
                )"""

    def athena(subject,skin,hair,face,eyes,attribute,lips,chest,name="demo",fabric="white"):
        generator = ImageGenerator()
        """
        style = "photograph, "
        subject = "a sexy woman, "
        skin = "tan skin, "
        hair = "short face-framing blond hair with bangs, "
        face = "high cheekbones, "
        eyes = "brown eyes, "
        attribute = "long eyelashes, "#glasses, etc
        lips = "full lips, "
        chest = "c-cup full breasts, "#don't always need this?
        pose = "keeling down, "#laying
        """
        #defaults
        style = "photograph, photo of"
        lighting ="soft lighting"

        pose = {"p1":"laying on a bed,",
                "p2":"laying on a bed topless,"}
        #her legs are spread apart the camera is looking at her from the foot of the bed her vagina is visible and a penis is entering her vagina
        action = {"a1": "legs spread apart,",
                  "a2": "legs spread apart her vagina is visible,",
                  "a3": "legs spread apart her vagina is being entered by a penis,",
                  "a4": "legs spread apart having sex,",
                  "a5": "legs spread apart ejaculate is dripping out of her vagina,",
                  "a6": "legs spread apart missionary sex," } 
        framing ={"f1":"the camera is looking at her from the foot of the bed,",
                  "f2":"the camera is above her looking down from the foot of the bed,"}
        clothes = {"c1":f"wearing {fabric} lace lingerie,",
                   "c2":"naked"}# or naked, she is naked nude #wearing the black lace lingerie,
        

        combos = ["p1a1f1c1","p2a125f1c2","p2a346f2c2"]
        folder = "athena"
        result = photoshoot(combos,style,lighting,subject,skin,hair,face,eyes,attribute,lips,chest,pose,action,framing,clothes,shoot_folder=folder,name=name)
        print(result)
        """
        for i in range(len(combos)):
            pafc = parse_combination(combos[i], pose, action, framing, clothes)
            for j in range(len(pafc)):  
                output_path = f"./{name}/{folder}/{name}_in_{folder}-{pafc[j]['keys']}.png"
                
                # Check if file already exists
                if os.path.exists(output_path):
                    print(f"File already exists, skipping: {output_path}")
                    continue
                
                prompt = " ".join([style, subject, skin, hair, face, eyes, attribute, lips, chest, pafc[j]['prompt'], lighting])
                print(f"Prompt Char Length (Stay under 300! or 77 tokens): {len(prompt)}")
                print(f"Prompt: {prompt}")
                print(f"Photo shoot {folder} - - - generating {j} of {len(pafc)}")
                
                image1 = generator.text_to_image(
                    prompt=prompt,
                    output_path=output_path
                )"""
    
    def achlys(subject,skin,hair,face,eyes,attribute,lips,chest,name="demo",fabric="hot pink"):
        generator = ImageGenerator()
        
        #defaults
        style = "photograph, photo of"
        lighting = "rays of sun light, 8k"

        pose = {"p1":"keeling down in a vast desert at sunset,",
                "p2":"on knees in vast desert at sunset,"}
        action = {"a1":"mouth around dildo,",
                  "a2":"sucking dildo,",
                  "a3":"dildo penetrating mouth",
                  "a4":"dildo deepthroat",
                  "a5":"ejaculating in her open mouth,",
                  "a6":"covered with dripping seman cum spray,",
                  "a7":"spreading her ass cheeks her vagina,", 
                  "a8":"dildo inside her vagina from behind,", 
                  "a9":"sex toy entering her vagina,",
                  "a0":"cum and ejaculation oozing from her vagina,"}

        framing = {"f1":"view is looking down at her,",
                   "f2":"looking back over her shoulder,",
                   "f3":"seductive look up into the camera,",}
        clothes = {"c1":f"breast exposed, wearing {fabric} lace lingerie bottom",
                   "c2":f"naked"}# or naked, nude
        combos = ["p1a123f1c1","p1a456f3c2","p2a7890f2c2"]

        folder = "achlys"
        result = photoshoot(combos,style,lighting,subject,skin,hair,face,eyes,attribute,lips,chest,pose,action,framing,clothes,shoot_folder=folder,name=name)
        print(result)

    def pontus(subject,skin,hair,face,eyes,attribute,lips,chest,name="demo",fabric="hot pink"):
        generator = ImageGenerator()
        
        #defaults
        style = "photograph, photo of"
        lighting = "rays of sunlight, 8k"

        pose = {"p1":"laying in grassy meadow at sunset,",}
        action = {"a1":"vaginal sex,",
                  "a2":"vaginal intercourse,",
                  "a3":"penitrating vaginal orgasim,",
                  "a4":"deep vaginal penitration,"}
        framing = {"f1":"view from her feet, she is looking at camera orgasiming",
                   "f2":"view from her feet, she is looking up with pleasure",
                   "f3":"view from her feet, she is,looking sexually at viewr"}
        clothes = {"c1":f"nude"}# or naked, nude
        combos = ["p1a1234f2c1","p1a1234f1c1","p1a1234f3c1"]

        folder = "pontus"
        result = photoshoot(combos,style,lighting,subject,skin,hair,face,eyes,attribute,lips,chest,pose,action,framing,clothes,shoot_folder=folder,name=name)
        print(result)

    def tartarus(subject,skin,hair,face,eyes,attribute,lips,chest,name="demo",fabric="yellow"):
        #generator = ImageGenerator()
        
        #defaults
        style = "photograph, photo of"
        lighting = "sunny day, 8k"

        pose = {"p1":"on a yacht deck, open ocean, laying on lounge chair,",
                "p2":"on a yacht deck,kneeling,open ocean,",}
        action = {"a1":"leg slightly spread apart",
                  "a2":"touching her breasts",
                  "a3":"touching her thigh",
                  "a4":"one knee bent",
                  "a5":"touching between her legs",
                  "a6":"lips around shaft of cock,",
                  "a7":"deepthroating,",
                  "a8":"stroking and sucking cock,",
                  "a9":"vaginal sex,",
                  "a0": "ejaculation in her open mouth, cum on her tounge and on her face,", #x5,
                  }
        framing = {"f1":"full body view from the side",
                   "f2":"full body view from foot of chair",
                   "f3":"full body view from behind her chair",
                   "f4":"POV",
                   "f5":"looking down at her",
                   "f6":"ejaculation dripping from lips around penis",
                   "f7":"facial expression orgasim",
                   "f8":"gagging on penis",
                   "f9":"POV leg in air",
                   "f0":"cum dripping from vagina, screaming with pleasure"}
        clothes = {"c1":f"topless, {fabric} bikini bottom only",
                   "c2":f"sun bathing nude",
                   "c3":f"naked"}# or naked, nude
        combos = ["p1a123f1c1","p1a45f2c1","p1a3f2c2","p2a678f4c3","p1a7f4c3","p1a8f5c3","p1a9f4c3","p2a0f5c3","p2a678f6c3","p1a9f9c3","p1a9f0c3","p2a7f8c1"]

        folder = "tartarus"
        result = photoshoot(combos,style,lighting,subject,skin,hair,face,eyes,attribute,lips,chest,pose,action,framing,clothes,shoot_folder=folder,name=name)
        print(result)
    
    def asclepius(subject,skin,hair,face,eyes,attribute,lips,chest,name="demo",fabric="black"):
        generator = ImageGenerator()
        
        #defaults
        style = "photograph, photo of"
        lighting = "sunset light rays"

        """ pose = {"p1":"sitting on a bench on a cobblestone street,"}
        action = {"a1":"her legs are slightly spread apart, panties",
                  "a2":"her hands squeezing her breasts",
                  "a3":"hand pulling dress up her thigh",
                  "a4":"legs spread, vagina pubic area visible, panties around her ankle",
                  "a5":"opening her legs wide, vagina pubic area is visible",
                  "a6":"man from waist down, penis performing fellatio",
                  "a7":"POV, engaged in fellatio on his penis",
                  "a8":"engaged in stroking and sucking his penis, fellatio",
                  "a9":"intercourse, vagina pubic area",
                  "a0": "man from wasit down, penis oozing ejaculation in her open mouth and tounge,", #x5,
                  }
        framing = {"f1":"sensual gaze,",
                   "f2":"view from behind her, she is looking back over shoulder orgasim",
                   "f3":"looking down at her,"}
        clothes = {"c1":f"strapless tight {fabric} dress"}# or naked, nude

        combos = ["p1a123459f1c1","p1a6780f3c1","p1a9f2c1"]"""
        pose = {"p1":"on a bench, cobblestone street, sitting,",
                "p2":"on a cobblestone street, kneeling,",
                "p3":"sitting on a bench on a cobblestone street,",
                "p4":"walking on a cobblestone street towards viewer,"}
        action = {"a1":"her legs are slightly spread apart, panties",
                  "a2":"she is grabbing her breasts",
                  "a3":"touching her inner thigh",
                  "a4":"legs up on bench, vagina pubic hair,",
                  "a5":"touching between her legs",
                  "a6":"lips around shaft of his penis,",
                  "a7":"deepthroating,",
                  "a8":"stroking and sucking his penis,",
                  "a9":"vaginal sex, she has orgasim,",
                  "a0": "ejaculation dripping from mouth, milk on her face,", #x5,
                  }
        framing = {"f1":"full body view from the side,",
                   "f2":"full body view from front of bench,",
                   "f3":"full body view from behind her bench,",
                   "f4":"POV,",
                   "f5":"looking down at her,",
                   "f6":"looking at her,"}
        clothes = {"c1":f"strapless tight {fabric} dress",
                   "c2":f"topless, {fabric} mini skirt",
                   "c3":f"naked"}# or naked, nude
        combos = ["p4a3f6c1","p3a2f5c1","p3a13f2c1","p1a45f2c1","p1a3f6c2","p2a678f4c3","p1a7f4c3","p1a8f5c3","p3a09f6c3","p2a0f5c3","p2a8f4c1","p1a0f6c2","p3a9f5c2","ap3a9f2c2","p3a9f4c3"]

        folder = "asclepius"
        result = photoshoot(combos,style,lighting,subject,skin,hair,face,eyes,attribute,lips,chest,pose,action,framing,clothes,shoot_folder=folder,name=name)
        print(result)

    def poseidon(subject,skin,hair,face,eyes,attribute,lips,chest,name="demo",fabric="blue"):
        generator = ImageGenerator()
       
        style = "photograph, photo of"
        lighting = "sunlight rays"

        pose = {"p1":"laying on a towel at the beech,",
                "p2":"laying on stomach on towel at beach,",
                "p4":"sitting on towel legs out streched at beach,",
                "p5":"wading in water at beach,",
                "p6":"splashing in water at beach,",
                "p7":"kneeling on towel at beach,",
                "p8":"on hands and knees on towel at beach,"} 
        #her legs are spread apart the camera is looking at her from the foot of the bed her vagina is visible and a penis is entering her vagina
        action = {"a1":"legs slightly spread apart,",
                  "a2":"touching her breasts,",
                  "a3":"touching her thigh,",
                  "a4":"arms back,",
                  "a5":"touching between her legs,",
                  "a6":"lips around shaft of cock,",
                  "a7":"deepthroating,",
                  "a8":"stroking and sucking cock,",
                  "a9": "ejaculation on her stomach and chest,",
                  "a0":"standing,",}
        framing ={"f1":"smile at viewer from her side,",
                  "f2":"intimate stare looking back at viewer,",
                  "f3":"POV,",
                  "f4":"laughing,",
                  "f5":"looking up at viewer,",
                  "f6":"looking down at her,",
                  "f7":"camera is behind her viewing her body from the side,",
                  "f8":"looking at her from behind,",}
        clothes = {"c1":f"wearing {fabric} bikini,",
                   "c2":f"topless, only wearing {fabric} bikini bottom,",
                   "c3":f"nude,"}# or naked, she is naked nude #wearing the black lace lingerie,

        combos = ["p1a123f5c1","p2a13f2c2","p4a45f2c2","p5a0f7c3","p6a0f4c1","p7a78f6c2","p7a9f6c3","p2a1f8c3","p1a9f5c3"]
        
        

        folder = "poseidon"

        result = photoshoot(combos,style,lighting,subject,skin,hair,face,eyes,attribute,lips,chest,pose,action,framing,clothes,shoot_folder=folder,name=name)
        print(result)
        
    def hera(subject,skin,hair,face,eyes,attribute,lips,chest,name="demo",fabric="black"):
        generator = ImageGenerator()
        
        """subject = "a sexy woman,"
        skin = "tan skin,"
        hair = "long wavy brown hair,"
        face = "dimple cheeks,"
        eyes = "blue eyes,"
        attribute = "black frame glasses,"#glasses, etc
        lips = "full lips,"
        chest = ""#don't always need this?"""
        #defaults
        style = "photograph, photo of"
        lighting = "sunlight from window"

        pose = {"p1":"leaning against a white wall,",
                "p2":"leaning over a table,",
                "p4":"sitting in a brown leather arm chair,",
                "p5":"standing next brown leather arm chair,",
                "p6":"standing on a balcony,",
                "p7":"sitting on a bed,"} 
        #her legs are spread apart the camera is looking at her from the foot of the bed her vagina is visible and a penis is entering her vagina
        action = {"a1": "one knee raised,",
                  "a2": "one leg raised,",
                  "a3": "legs spread apart,", #generates NSFW
                  "a4": "legs crossed,",
                  "a5": "legs infront,",
                  "a6": "laying on back, legs in the air,",
                  "a7":"arm resting across the back of the chair,",
                  "a8":"leaning over railing looking out at city,",
                  "a9":"putting on lipstick,",
                  "a0":"brushing hair,"} 
        framing ={"f1":"seductive stare at camera looking at her from the side,",
                  "f2":"intimate stare at camera looking at her across the table,",
                  "f3":"sexual stare at camera looking directly at her,",
                  "f4":"sexual stare at camera finger in her mouth,",
                  "f5":"sexual stare at camera hand on her chest,",
                  "f6":"sexual stare at camera hand touching her thigh,",
                  "f7":"camera is behind her viewing her body from the side,",
                  "f8":"looking at camera from across the room,",}
        clothes = {"c1":f"wearing {fabric} skirt white tee, high heals",
                   "c3":f"wearing {fabric} skirt white tee, bare feet"}# or naked, she is naked nude #wearing the black lace lingerie,

        combos = ["p1a1f1c1","p2a2f2c1","p4a34f3c1","p4a56f3c3","p5a7f4c1","p5a7f5c3","p5a7f6c1","p6a8f7c1","p6a8f8c1","p7a90f7c3","p7a90f8c1","p7a46f3c1","p7a3f4c1"]
        
        

        folder = "hera"

        result = photoshoot(combos,style,lighting,subject,skin,hair,face,eyes,attribute,lips,chest,pose,action,framing,clothes,shoot_folder=folder,name=name)
        print(result)
        """
        for i in range(len(combos)):
            pafc = parse_combination(combos[i], pose, action, framing, clothes)
            for j in range(len(pafc)):  
                output_path = f"./{name}/{folder}/{name}_in_{folder}-{pafc[j]['keys']}.png"
                
                # Check if file already exists
                if os.path.exists(output_path):
                    print(f"File already exists, skipping: {output_path}")
                    continue
                
                prompt = " ".join([style, subject, skin, hair, face, eyes, attribute, lips, chest, pafc[j]['prompt'], lighting])
                print(f"Prompt Char Length (Stay under 300! or 77 tokens): {len(prompt)}")
                print(f"Prompt: {prompt}")
                print(f"Photo shoot {folder} - - - generating {j} of {len(pafc)}")
                
                image1 = generator.text_to_image(
                    prompt=prompt,
                    output_path=output_path
                )"""

    def zeus(subject,skin,hair,face,eyes,attribute,lips,chest,name="demo",fabric="sequin"):
        generator = ImageGenerator()
        """
        style = "photograph, photo of "
        subject = "a sexy woman, "
        skin = "tan skin, "
        hair = "short face-framing blond hair with bangs, "
        face = "high cheekbones, "
        eyes = "brown eyes, "
        attribute = "long eyelashes, "#glasses, etc
        lips = "full lips, "
        chest = "c-cup full breasts, "#don't always need this?
        pose = "keeling down, "#laying
        """
        #defaults
        style = "photograph,"
        lighting = "sunlight from window, 8k"

        pose = {"p1":"leaning against white wall,",
                "p2":"leaning over a table,",
                "p4":"sitting in a white arm chair,",
                "p5":"standing next to a white arm chair,",
                "p6":"standing on a balcony,"}
        #her legs are spread apart the camera is looking at her from the foot of the bed her vagina is visible and a penis is entering her vagina
        action = {"a1": "one knee raised,",
                  "a2": "one leg raised,",
                  "a3": "legs spread apart,", #generates NSFW
                  "a4": "legs crossed,",
                  "a5": "legs infront,",
                  "a6": "laying upside down, legs in the air back of chair,",
                  "a7":"arm resting across the back of the chair,",
                  "a8":"leaning over railing looking out at city,",
                  "a9":"putting on lipstick,",
                  "a0":"brushing hair,"} 
        framing ={"f1":"seductive stare at camera looking at her from the side,",
                  "f2":"intimate stare at camera looking at her across the table,",
                  "f3":"sexual stare at camera looking directly at her,",
                  "f4":"sexual stare at camera finger in her mouth,",
                  "f5":"sexual stare at camera hand on her chest,",
                  "f6":"sexual stare at camera hand touching her thigh,",
                  "f7":"camera is behind her viewing her body from the side,",
                  "f8":"camera is looking through doorway viewing her body from the side,",}
        clothes = {"c1":f"wearing strapless {fabric} dress, high heals",
                   "c2":f"wearing strapless {fabric} dress, bare feet"}
        
        combos = ["p1a1f1c1","p2a2f2c1","p4a34f3c2","p4a90f7c1","p4a90f8c1","p4a56f3c2","p5a7f4c2","p5a7f5c2","p5a7f6c2","p6a8f7c1","p6a8f7c1"]

        folder = "zeus"

        result = photoshoot(combos,style,lighting,subject,skin,hair,face,eyes,attribute,lips,chest,pose,action,framing,clothes,shoot_folder=folder,name=name)
        print(result)
        """
        for i in range(len(combos)):
            pafc = parse_combination(combos[i], pose, action, framing, clothes)
            for j in range(len(pafc)):  
                output_path = f"./{name}/{folder}/{name}_in_{folder}-{pafc[j]['keys']}.png"
                
                # Check if file already exists
                if os.path.exists(output_path):
                    print(f"File already exists, skipping: {output_path}")
                    continue
                
                prompt = " ".join([style, subject, skin, hair, face, eyes, attribute, lips, chest, pafc[j]['prompt'], lighting])
                print(f"Prompt Char Length (Stay under 300! or 77 tokens): {len(prompt)}")
                print(f"Prompt: {prompt}")
                print(f"Photo shoot {folder} - - - generating {j} of {len(pafc)}")
                
                image1 = generator.text_to_image(
                    prompt=prompt,
                    output_path=output_path
                )"""

    def hades(subject,skin,hair,face,eyes,attribute,lips,chest,name="demo",fabric="white"):
        #defaults
        style = "photograph, photo of"
        lighting ="soft lighting"

        pose = {"p1":"laying on a bed,",
                "p2":"laying on a bed topless,"}
        #her legs are spread apart the camera is looking at her from the foot of the bed her vagina is visible and a penis is entering her vagina
        action = {"a1": "legs spread apart, man's waist down, erect penis, sex,",
                  "a2": "missionary sex, intercours,",
                  "a3": "his penis pushing in to her vagina lips, pubic area,",
                  "a4": "ejaculation oozing from her vagina, penis entering, sex,",
                  "a5": "hardcore sexual intercourse,",
                  "a6": "sodomy, his penis in her anal sex,",
                   "a7": "his penis ejaculates cum on to her stomach,"  } 
        framing ={"f1":"POV, having orgasim,",
                  "f2":"she is gazing up at viewer, POV,",
                  "f3":"looking down at her, face orgasim,",
                  "f4":"view from over her, she is delighed",}
        clothes = {"c1":f"wearing {fabric} lace lingerie,",
                   "c2":"naked"}# or naked, she is naked nude #wearing the black lace lingerie,
        

        combos = ["p1a134567f1c1","p1a134567f2c1","p1a134567f1c2","p1a134567f2c2","p1a2f3c1","p2a2f3c2","p2a7f2c1"]
        folder = "hades"
        result = photoshoot(combos,style,lighting,subject,skin,hair,face,eyes,attribute,lips,chest,pose,action,framing,clothes,shoot_folder=folder,name=name)
        print(result)

    
    def parse_combination(combination_string, pose, action, framing, clothes):
    #def parse_combination():
        import re
        """
        Parse a combination string like 'p2a346f1c3' and return assembled prompts.
        
        Args:
            combination_string: String like 'p2a346f1c3' (pose, action, framing, clothes)
            pose: Dict with keys like 'p1', 'p2'
            action: Dict with keys like 'a1', 'a2', 'a3'
            framing: Dict with keys like 'f1', 'f2'
            clothes: Dict with keys like 'c1', 'c2', 'c3'
        
        Returns:
            List of dictionaries with 'keys' and 'prompt' for each action
        """
        
        # Extract components using regex
        pose_match = re.search(r'p(\d+)', combination_string)
        action_match = re.search(r'a(\d+)', combination_string)
        framing_match = re.search(r'f(\d+)', combination_string)
        clothes_match = re.search(r'c(\d+)', combination_string)
        
        if not all([pose_match, action_match, framing_match, clothes_match]):
            raise ValueError(f"Invalid combination string: {combination_string}")
        
        # Get the component IDs
        pose_id = f"p{pose_match.group(1)}"
        framing_id = f"f{framing_match.group(1)}"
        clothes_id = f"c{clothes_match.group(1)}"
        
        # Extract individual action numbers from the digits after 'a'
        action_digits = action_match.group(1)
        action_numbers = [int(digit) for digit in action_digits]
        
        print(f"Parsing: {combination_string}")
        print(f"  Components: {pose_id}, actions={action_numbers}, {framing_id}, {clothes_id}")
        
        # Look up values in dictionaries
        if pose_id not in pose:
            raise KeyError(f"Pose {pose_id} not found")
        if framing_id not in framing:
            raise KeyError(f"Framing {framing_id} not found")
        if clothes_id not in clothes:
            raise KeyError(f"Clothes {clothes_id} not found")
        
        pose_value = pose[pose_id]
        framing_value = framing[framing_id]
        clothes_value = clothes[clothes_id]
        
        # Generate combinations for each action
        combinations = []
        for action_num in action_numbers:
            action_id = f"a{action_num}"
            
            if action_id not in action:
                print(f"  Warning: Action {action_id} not found, skipping")
                continue
            
            action_value = action[action_id]
            
            # Build the prompt
            prompt_parts = [pose_value, action_value, framing_value, clothes_value]
            prompt = " ".join(prompt_parts)
            
            # Create combination key
            combination_key = f"{pose_id}_{action_id}_{framing_id}_{clothes_id}"
            
            combinations.append({
                'keys': combination_key,
                'prompt': prompt
            })
            
            print(f"    Created: {combination_key}")
        
        print(combinations)
        return combinations

    #yseries()
    #zseries()
    #Lseries()
    
    #Photoshoot Rhea DONE:
    def Rhea():
        """
        DONE
        """
        name="rhea"

        subject = "sexy woman"
        skin = "tan skin"
        hair = "short face-framing blond hair with bangs"
        face = "high cheekbones"
        eyes = "brown eyes"
        attribute = "long eyelashes"#glasses, etc
        lips = "full lips"
        chest = "c-cup full breasts"#don't always need this?

        return subject,skin,hair,face,eyes,attribute,lips,chest, name

    #Photoshoot Aphrodite:
    def Aphrodite():
        name = "aphrodite"

        subject = "young woman"
        skin = "white skin"
        hair = "short face-framing pink hair with bangs"
        face = "strong facial features"
        eyes = ""
        attribute = "long eyelashes"#glasses, etc
        lips = ""
        chest = "small tits"#don't always need this?
        
        return subject,skin,hair,face,eyes,attribute,lips,chest, name

    #Photoshoot Eros:
    def Eros():
        name ="eros"

        subject = "older woman"
        skin = "tan skin"
        hair = "long blond hair"
        face = "sharp features"
        eyes = "almond eyes"
        attribute = "dark eyeliner"#glasses, etc
        lips = "full lips"
        chest = "cup C breasts"#don't always need this?

        return subject,skin,hair,face,eyes,attribute,lips,chest, name

    #Photoshoot Demetra DONE:
    def Demetra():
        """
        DONE
        """

        name ="demetra"
        
        subject = "young woman"
        skin = "tan skin"
        hair = "pony tail blond hair"
        face = "thin eyebrows"
        eyes = "sexy eyes"
        attribute = "long eyelash"#glasses, etc
        lips = ""
        chest = ""#don't always need this?

        return subject,skin,hair,face,eyes,attribute,lips,chest, name

    def Thalassa():
        """
        DONE
        """

        name ="thalassa"
        
        subject = "young woman"
        skin = "tan skin"
        hair = "pony tail blond black"
        face = "thin eyebrows"
        eyes = "sexy blue eyes"
        attribute = "long eyelash"#glasses, etc
        lips = ""
        chest = ""#don't always need this?

        return subject,skin,hair,face,eyes,attribute,lips,chest, name
    
    #Photoshoot Hesperus:
    def Hesperus():
        """ DONE """
        name ="hesperus"
        
        subject = "young woman"
        skin = ""
        hair = "blond pony tail with curtain bangs"
        face = "hard-angled eye brows, light freckles"
        eyes = "sharp features"
        attribute = "long eyelash"#glasses, etc
        lips = "wide lips"
        chest = "small breasts"#can actually be anything

        return subject,skin,hair,face,eyes,attribute,lips,chest, name
        
    def Ares():
        """ DONE """
        name ="ares"
        
        subject = "sexy young woman"
        skin = "tan skin"
        hair = "platinum blond hair pixi cut"
        face = "dark eyeshadow"
        eyes = "sharp features"
        attribute = ""#glasses, etc
        lips = "wide lips"
        chest = "small breasts"#can actually be anything

        return subject,skin,hair,face,eyes,attribute,lips,chest, name

    def Hemera():
        name ="hemera"
        
        subject = "curvy milf"
        skin = "tan skin"
        hair = "long blond hair, curtain bangs"
        face = "dark eyeshadow"
        eyes = "green eyes"
        attribute = "long eyelashes"#glasses, etc
        lips = "puckered lips"
        chest = "natural breasts"#can actually be anything

        return subject,skin,hair,face,eyes,attribute,lips,chest, name
    
    def Nesoi():
        name ="nesoi"
        
        subject = "curvy woman"
        skin = "tan skin"
        hair = "brown pigtails with curtain bangs"
        face = "lots of eyeshadow"
        eyes = "light blue eyes"
        attribute = "eyelashes"#glasses, etc
        lips = "eyeliner"
        chest = "natural tits"#can actually be anything

        return subject,skin,hair,face,eyes,attribute,lips,chest, name

    def Theia():
        name ="theia"
        subject = "a sexy nymph elf"
        skin = ""
        hair = "long flowing brown hair"
        face = "high cheek bones"
        eyes = "green eyes"
        attribute = "long eyelashes"#glasses, etc
        lips = "puckered lips"
        chest = "natural breasts"#can actually be anything

        return subject,skin,hair,face,eyes,attribute,lips,chest, name

    def Phoebe():
        #photograph of  , blue eyes dark eye liner and mascara, puckered lips and high cheek bones, small breasts. wearing a towel wrapped around her chest, standing in a luxury bathroom, looking at camera sexually, rays of sunlight window,
        name ="phoebe"
        subject = "a sexy woman"
        skin = "with long flowing silver hair"
        hair = "long flowing brown hair"
        face = "high cheek bones"
        eyes = "blue eyes"
        attribute = "dark eye liner and mascara"#glasses, etc
        lips = "puckered lips"
        chest = "small breasts"#can actually be anything

        return subject,skin,hair,face,eyes,attribute,lips,chest, name

    def Eileithyia():
            #photograph of  , blue eyes dark eye liner and mascara, puckered lips and high cheek bones, small breasts. wearing a towel wrapped around her chest, standing in a luxury bathroom, looking at camera sexually, rays of sunlight window,
            name ="eileithyia"
            subject = "a woman"
            skin = "tan skin"
            hair = "short bixi cut blond hair"
            face = "soft features"
            eyes = "light blue eyes"
            attribute = "long eyelashes, eyeshadow"#glasses, etc
            lips = "puckered lips"
            chest = "small breasts"#can actually be anything

            return subject,skin,hair,face,eyes,attribute,lips,chest, name

    def Lefkothea():
            #prompt="photograph,photo of a sexy woman, white skin, , glasses, soft features, light blue eyes with , kneeling down deepthroat a cock, open jaw, pov looking up at the camera. wearing black lace lingerie, 8k",

            name ="lefkothea"
            subject = "a sexy woman"
            skin = "white skin"
            hair = "flowing black hair"
            face = "soft features"
            eyes = "light blue eyes"
            attribute = "long eyelashes"#glasses, etc
            lips = ""
            chest = "small breasts"#can actually be anything

            return subject,skin,hair,face,eyes,attribute,lips,chest, name
    
    def Evrynomi():
            #prompt="photograph,photo of a sexy woman, white skin, , glasses, soft features, light blue eyes with , kneeling down deepthroat a cock, open jaw, pov looking up at the camera. wearing black lace lingerie, 8k",

            name ="evrynomi"
            subject = "a curvy woman"
            skin = "dark tan skin"
            hair = "short bixi cut blond hair"
            face = "soft features"
            eyes = "brown eyes"
            attribute = "long eyelashes"#glasses, etc
            lips = "puckered lips"
            chest = "natural breasts"#can actually be anything

            return subject,skin,hair,face,eyes,attribute,lips,chest, name
    
    def Calliope():
        """ DONE """
        name ="calliope"
        
        subject = "an Irish woman"
        skin = "tan skin, with freckles"
        hair = "short wavy black hair with bangs"#"face-framing blond hair"
        face = "hard facial features"#"dark eyeshadow"
        eyes = "light blue eye"#"blue eys"
        attribute = "long eyelashes"#glasses, etc
        lips = "lips"
        chest = "small breasts"#"small breasts"#can actually be anything

        return subject,skin,hair,face,eyes,attribute,lips,chest, name
    
   

    def photographModel(xmodel,xshoot,fabric=None):

        #get the model attributes
        subject,skin,hair,face,eyes,attribute,lips,chest, name = xmodel()
        #run the photo shoot with default fabric selection
        if not fabric:
            xshoot(subject,skin,hair,face,eyes,attribute,lips,chest,name=name)
        else:
            xshoot(subject,skin,hair,face,eyes,attribute,lips,chest,name=name,fabric=fabric)
        return

    #Demetra() 
    #Aphrodite()
    #Hesperus()
    #Ares()
    #Eros()
    #Rhea() 
    

    #NEW photo shoot flow
    #send model function and the scene to the photographModel
    #_ = photographModel(Hesperus,dionysus,fabric="hot pink")

    # Availible Models:
    #   - Rhea
    #   - Demetra
    #   - Aphrodite
    #   - Hesperus
    #   - Ares
    #   - Eros
    
    # Avilible Shoots
    #   hera
    #   zeus
    #   apollo
    #   athena
    #   dionysus

    #Starter Series
    
    """
    1. High-definition photo of a [tall, curvy woman], [with long, wavy blonde hair], [wearing a tight, black leather corset and matching thigh-high boots], [leaning against a wall with one hand on her hip], [full body shot], [in a dimly lit room with red velvet drapes], [soft and moody lighting], [low-angle shot]
2. Ultra-realistic photo of a [brunette woman], [with piercing blue eyes], [wearing a lacy lingerie set with a matching garter belt], [sitting on a chair with legs spread wide open], [close-up shot], [in a cozy bedroom with soft pillows and blankets], [natural sunlight filtering through the window], [medium-angle shot]
 3. Photo of a [blonde woman], [with full lips and a seductive expression], [wearing a see-through silk robe that barely covers her breasts], [standing with one hand on her hip and the other holding a glass of wine], [full body shot], [in a luxurious, modern living room with plush furniture and large windows], [soft and warm lighting], [high-angle shot]
 4. High-resolution photo of a [redhead woman], [with a tattooed back], [wearing a revealing black bodysuit], [kneeling on the floor with one hand on her head], [low-angle shot], [in a dark, industrial-style loft with exposed brick walls], [cool and stark lighting], [low-angle shot]
 5. Photo of a [woman with a curvaceous figure], [with a tight, shiny bodysuit], [standing with one leg raised and one hand on her hip], [full body shot], [in a steamy bathroom with fogged mirrors and tiles], [soft and steamy lighting], [low-angle shot]
    """
    ##NEW JUMP HERE
    #_ = photographModel(Scarlett,asclepius,fabric="red")
    #_ = photographModel(Margot,asclepius,fabric="white")
    #_ = photographModel(Taylor,asclepius,fabric="silver")

    _ = photographModel(Demetra,hades,fabric="pink")
    #_ = photographModel(Calliope,hades,fabric="blue")



    # ALL MODELS ALL SHOOTS
    def runAll():
        


        _ = photographModel(Evrynomi,poseidon,fabric="gold")
        _ = photographModel(Evrynomi,tartarus,fabric="gold")
        _ = photographModel(Evrynomi,dionysus,fabric="gold")
        _ = photographModel(Evrynomi,apollo,fabric="gold")
        _ = photographModel(Evrynomi,athena,fabric="gold")
        _ = photographModel(Evrynomi,zeus,fabric="gold")
        _ = photographModel(Evrynomi,pontus,fabric="gold")
        _ = photographModel(Evrynomi,hera,fabric="gold")
        _ = photographModel(Evrynomi,achlys,fabric="gold")

        _ = photographModel(Lefkothea,poseidon,fabric="teal")
        _ = photographModel(Lefkothea,tartarus,fabric="teal")
        _ = photographModel(Lefkothea,dionysus,fabric="teal")
        _ = photographModel(Lefkothea,apollo,fabric="teal")
        _ = photographModel(Lefkothea,athena,fabric="teal")
        _ = photographModel(Lefkothea,zeus,fabric="teal")
        _ = photographModel(Lefkothea,pontus,fabric="teal")
        _ = photographModel(Lefkothea,hera,fabric="teal")
        _ = photographModel(Lefkothea,achlys,fabric="teal")

        _ = photographModel(Eileithyia,poseidon,fabric="blue")
        _ = photographModel(Eileithyia,tartarus,fabric="blue")
        _ = photographModel(Eileithyia,dionysus,fabric="blue")
        _ = photographModel(Eileithyia,apollo,fabric="blue")
        _ = photographModel(Eileithyia,athena,fabric="blue")
        _ = photographModel(Eileithyia,zeus,fabric="blue")
        _ = photographModel(Eileithyia,pontus,fabric="blue")
        _ = photographModel(Eileithyia,hera,fabric="blue")
        _ = photographModel(Eileithyia,achlys,fabric="blue")

        _ = photographModel(Phoebe,dionysus,fabric="gold")
        _ = photographModel(Phoebe,hera,fabric="gold")
        _ = photographModel(Phoebe,apollo,fabric="gold")
        _ = photographModel(Phoebe,athena,fabric="gold")
        _ = photographModel(Phoebe,zeus,fabric="gold")
        _ = photographModel(Phoebe,pontus,fabric="gold")
        _ = photographModel(Phoebe,tartarus,fabric="gold")
        _ = photographModel(Phoebe,achlys,fabric="gold")
        _ = photographModel(Phoebe,poseidon,fabric="gold")

        _ = photographModel(Theia,dionysus,fabric="green")
        _ = photographModel(Theia,hera,fabric="green")
        _ = photographModel(Theia,apollo,fabric="green")
        _ = photographModel(Theia,athena,fabric="green")
        _ = photographModel(Theia,zeus,fabric="green")
        _ = photographModel(Theia,pontus,fabric="green")
        _ = photographModel(Theia,tartarus,fabric="purple")
        _ = photographModel(Theia,achlys,fabric="green")
        _ = photographModel(Theia,poseidon,fabric="green")
        
        _ = photographModel(Rhea,hera)
        _ = photographModel(Rhea,zeus)
        _ = photographModel(Rhea,apollo)
        _ = photographModel(Rhea,athena)
        _ = photographModel(Rhea,dionysus)
        _ = photographModel(Rhea,tartarus,fabric="green")
        _ = photographModel(Rhea,pontus)
        _ = photographModel(Rhea,achlys)
        _ = photographModel(Rhea,poseidon,fabric="purple")

        _ = photographModel(Demetra,hera)
        _ = photographModel(Demetra,zeus)
        _ = photographModel(Demetra,apollo)
        _ = photographModel(Demetra,athena)
        _ = photographModel(Demetra,dionysus)
        _ = photographModel(Demetra,tartarus)
        _ = photographModel(Demetra,pontus)
        _ = photographModel(Demetra,achlys)
        _ = photographModel(Demetra,poseidon,fabric="yellow")

        _ = photographModel(Aphrodite,hera)
        _ = photographModel(Aphrodite,zeus)
        _ = photographModel(Aphrodite,apollo)
        _ = photographModel(Aphrodite,athena)
        _ = photographModel(Aphrodite,dionysus)
        _ = photographModel(Aphrodite,tartarus,fabric="pink")
        _ = photographModel(Aphrodite,pontus)
        _ = photographModel(Aphrodite,achlys)
        _ = photographModel(Aphrodite,poseidon,fabric="pink")

        _ = photographModel(Hesperus,hera)
        _ = photographModel(Hesperus,zeus)
        _ = photographModel(Hesperus,apollo)
        _ = photographModel(Hesperus,athena)
        _ = photographModel(Hesperus,dionysus)
        _ = photographModel(Hesperus,tartarus,fabric="red")
        _ = photographModel(Hesperus,pontus)
        _ = photographModel(Hesperus,achlys)
        _ = photographModel(Hesperus,poseidon,fabric="red")
        
        _ = photographModel(Ares,hera,fabric="gold")
        _ = photographModel(Ares,zeus,fabric="light blue")
        _ = photographModel(Ares,apollo)
        _ = photographModel(Ares,athena)
        _ = photographModel(Ares,dionysus)
        _ = photographModel(Ares,tartarus,fabric="black")
        _ = photographModel(Ares,pontus)
        _ = photographModel(Ares,achlys)
        _ = photographModel(Ares,poseidon,fabric="black")

        _ = photographModel(Eros,hera)
        _ = photographModel(Eros,zeus)
        _ = photographModel(Eros,apollo)
        _ = photographModel(Eros,athena)
        _ = photographModel(Eros,dionysus)
        _ = photographModel(Eros,tartarus,fabric="gold")
        _ = photographModel(Eros,pontus)
        _ = photographModel(Eros,achlys)
        _ = photographModel(Eros,poseidon,fabric="white")

        _ = photographModel(Nesoi,dionysus)
        _ = photographModel(Nesoi,hera)
        _ = photographModel(Nesoi,apollo)
        _ = photographModel(Nesoi,athena)
        _ = photographModel(Nesoi,zeus)
        _ = photographModel(Nesoi,pontus)
        _ = photographModel(Nesoi,tartarus,fabric="silver")
        _ = photographModel(Nesoi,achlys)
        _ = photographModel(Nesoi,poseidon,fabric="silver")

        _ = photographModel(Hemera,dionysus)
        _ = photographModel(Hemera,hera)
        _ = photographModel(Hemera,apollo)
        _ = photographModel(Hemera,athena)
        _ = photographModel(Hemera,zeus)
        _ = photographModel(Hemera,pontus)
        _ = photographModel(Hemera,tartarus,fabric="white")
        _ = photographModel(Hemera,achlys)
        _ = photographModel(Hemera,poseidon,fabric="orange")


        
        _ = photographModel(Thalassa,dionysus)
        _ = photographModel(Thalassa,hera)
        _ = photographModel(Thalassa,apollo)
        _ = photographModel(Thalassa,athena)
        _ = photographModel(Thalassa,zeus)
        _ = photographModel(Thalassa,pontus)
        _ = photographModel(Thalassa,tartarus,fabric="purple")
        _ = photographModel(Thalassa,achlys)
        _ = photographModel(Thalassa,poseidon,fabric="yellow")


    def Test():
            """prompt=["photograph of a sexy woman with long flowing silver hair, blue eyes dark eye liner and mascara, puckered lips and small breasts. wearing gold mini skirt and bikini top, sitting legs together barefoot drinking tea, hotel room, in a white leather chair staring sexually at viewer, sunlight window, 8k",
                    "photograph of a young woman with long flowing silver hair, blue eyes dark eye liner and mascara, puckered lips and small breasts. wearing gold mini skirt and bikini top, sitting with her legs together crosslegged toughing mouth, hotel room, in a white leather chair staring sexually at viewer, sunlight window, 8k",
                    "photograph of a young woman with long flowing silver hair, blue eyes dark eye liner and mascara, puckered lips and small breasts. wearing gold mini skirt and bikini top, sitting with her legs together crosslegged touching thigh, hotel room, in a white leather chair staring sensually at viewer, sunlight window, 8k",
                    "photograph of a young woman with long flowing silver hair, blue eyes dark eye liner and mascara, puckered lips and small breasts. wearing gold mini skirt and bikini top, sitting with her legs together crosslegged barefoot, hotel room, in a white leather chair staring seductivly at viewer, sunlight window, 8k"]"""
            
            """prompt=["photograph of a sexy woman with long flowing silver hair, blue eyes dark eye liner and mascara, puckered lips and small breasts. wearing silver sequin mini skirt and white tube-top, walking down a city side walk towards viewer, looking at camera smiling happy, rays of sunlight above, 8k",
                    "photograph of a sexy woman with long flowing silver hair, blue eyes dark eye liner and mascara, puckered lips and small breasts. wearing silver sequin mini skirt and black tube-top, walking down a city side walk towards viewer, looking at camera laughing, rays of sunlight above, 8k",
                    "photograph of a sexy woman with long flowing silver hair, blue eyes dark eye liner and mascara, puckered lips and small breasts. wearing silver sequin mini skirt and red tube-top, walking down a city side walk towards viewer, looking at seductivly, rays of sunlight above, 8k",
                    "photograph of a sexy woman with long flowing silver hair, blue eyes dark eye liner and mascara, puckered lips and small breasts. wearing silver sequin mini skirt and yellow tube-top, walking down a city side walk towards viewer, she is looking to her side not looking at viewer, rays of sunlight above, 8k"]
            """
            """prompt=["photograph of a sexy woman with long flowing silver hair, blue eyes dark eye liner and mascara, puckered lips and high cheek bones, small breasts. wearing a towel wrapped around her chest, standing in a luxury bathroom, looking at camera sexually, rays of sunlight window, 8k",
                    "photograph of a sexy woman with long flowing silver hair, blue eyes dark eye liner and mascara, puckered lips and high cheek bones, small breasts. towel wrapped around her chest, barefoot, standing in a luxury bathroom, looking out window, rays of sunlight window, 8k",
                    "photograph of a sexy woman with long flowing silver hair, blue eyes dark eye liner and mascara, puckered lips and high cheek bones, small breasts.  towel near her feet, standing in a luxury bathroom, looking at camera sexually rubbing her thigh, rays of sunlight window, 8k",
                    "photograph of a sexy woman with long flowing silver hair, blue eyes dark eye liner and mascara, puckered lips and high cheek bones, small breasts.  towel near her feet, standing in a luxury bathroom, looking at camera sexually rubbing her both hands covering her nipples, rays of sunlight window, 8k"]
            """
            prompt=[#"photograph of a sexy nymph elf with long flowing brown hair, green eyes, long eyelashes, puckered lips and high cheek bones, natural breasts. wearing a bikin made of leaves, standing at the edge of a ethereal pool, looking at camera sexually, rays of sunlight reflecting, 8k",
                    #"photograph of a sexy nymph elf with long flowing brown hair, green eyes, long eyelashes, puckered lips and high cheek bones, natural breasts. wearing a bikin bottom made of leaves, standing at the edge of a ethereal pool, looking at camera smiling, rays of sunlight reflecting, whisps floating, 8k",
                    #"photograph of a sexy nymph elf with long flowing brown hair, green eyes, long eyelashes, puckered lips and high cheek bones, natural breasts. wearing a bikin made of flowers, standing at the edge of a ethereal pool, looking at camera laughing, rays of sunlight reflecting, whisps floating, 8k",
                    #"photograph of a sexy nymph elf with long flowing brown hair, green eyes, long eyelashes, puckered lips and high cheek bones, natural breasts. her bikini top is clam shells and the bottoms are leaves, standing at the edge of a ethereal pool, looking at camera laughing, rays of sunlight reflecting, sunset, 8k",
                    #"photograph of a sexy nymph elf with long flowing brown hair, green eyes, long eyelashes, puckered lips and high cheek bones, natural breasts. her bikini top is clam shells and the bottoms are leaves, standing in a ethereal pool, deepthroat, looking at camera sexually, rays of sunlight reflecting, sunset, 8k",
                    #"photograph of a sexy nymph elf with long flowing brown hair, green eyes, long eyelashes, puckered lips and high cheek bones, natural breasts. her bikini top is clam shells and the bottoms are leaves, standing in a ethereal pool, ejaculation is oozing on to her smiling face, rays of sunlight reflecting, sunset, 8k",
                    #"photograph of a sexy nymph elf with long flowing brown hair, green eyes, long eyelashes, puckered lips and high cheek bones, natural breasts, standing at the edge of an ethereal pool, legs spread, hardcore sexual intercourse, looking over her shoulder, rays of sunlight reflecting, sunset, 8k",
                    #"photograph of a sexy nymph elf with long flowing brown hair, green eyes, long eyelashes, puckered lips and high cheek bones, natural breasts, keeling at the edge of an ethereal pool, penis deep in her throat, looking up at viewer, rays of sunlight reflecting, sunset, 8k",
                    #"photograph of a sexy nymph elf with long flowing brown hair, green eyes, long eyelashes, puckered lips and high cheek bones, natural breasts, standing at the edge of an ethereal pool, legs spread, hardcore sexual intercourse, looking over her shoulder, rays of sunlight reflecting, sunset, 8k",
                    #"photograph of a sexy nymph elf with long flowing brown hair, green eyes, long eyelashes, puckered lips and high cheek bones, natural breasts, keeling at the edge of an ethereal pool, sucking a penis in her mouth, looking up at viewer, rays of sunlight reflecting, sunset, 8k",
                    #"photograph of a sexy nymph elf with long flowing brown hair, green eyes, long eyelashes, puckered lips and high cheek bones, natural breasts, laying on a grassy hilltop, wearing a strapless blue skirt, looking up at viewer seductivly, rays of sunlight reflecting, sunset, 8k",
                    #"photograph of a sexy nymph elf with long flowing brown hair, green eyes, long eyelashes, puckered lips and high cheek bones, natural breasts, laying on a grassy hilltop, legs outstreched, arms back, wearing a strapless blue skirt, looking up at viewer seductivly, rays of sunlight reflecting, sunset, 8k",
                    #"photograph of a sexy nymph elf with long flowing brown hair, green eyes, long eyelashes, puckered lips and high cheek bones, natural breasts, sitting crosslegged on a boulder on a grassy hilltop, wearing a strapless blue skirt, looking up at viewer intimately, rays of sunlight reflecting, sunset, 8k",
                    #"photograph of a sexy nymph elf with long flowing brown hair, green eyes, long eyelashes, puckered lips and high cheek bones, natural breasts, sitting on a grassy hilltop, knees bent, arms back, wearing a strapless blue skirt, looking up at viewer sentually, rays of sunlight reflecting, sunset, 8k",
                    #"photograph of a sexy nymph elf with long flowing brown hair, green eyenugent
                    # s, long eyelashes, puckered lips and high cheek bones, natural breasts, bent over on a boulder on a grassy hilltop, wearing a strapless blue dress, penis entering her vagina from behind, she is looking back over her shoulder, rays of sunlight reflecting, sunset, 8k",
                    #"photograph of a sexy nymph elf with long flowing brown hair, green eyes, long eyelashes, puckered lips and high cheek bones, natural breasts, bent over on a boulder on a grassy hilltop, wearing a strapless blue dress, penis penitrating her vagina from behind, she is looking down, rays of sunlight reflecting, sunset, 8k",
                    #"photograph of Adriana Chechik, on her knees looking up at the camera, her hands are around his penis which is deep inside her mouth, 8k image",
                    "photograph, photo of a sexy woman, short blond hair with bangs, high cheekbones, full lips, blue eyes with long eyelashes, small breasts, athletic physique, open jaw deepthroating a penis her lips near the base, kneeling down looking up at viewer, soft lighting from window, 8k image"]



            for i, prmpt in enumerate(prompt):
                photoshoot(prmpt,"style","lighting","subject","skin","hair","face","eyes","attribute","lips","chest","pose","action","framing","clothes",shoot_folder="demo_shoot",name="demo",test=True)
    #Test()

    #runAll()

    """while True:
        runAll()
        time.sleep(1)"""

    from collections import defaultdict, Counter

    def analyze_favorites(file_path):
        """
        
        DESIGNED TO WORK WITH server.py which puts out the fav.txt file

        Analyze favorites to find popular photoshoots/arrangements across all models
        """
        
        # Read the favorites file
        with open(file_path, 'r') as f:
            favorites = [line.strip() for line in f if line.strip()]
        
        # Extract model names and photoshoot/arrangement combinations
        model_photoshoots = defaultdict(set)
        all_photoshoots = []
        all_models = set()
        
        for favorite in favorites:
            # Parse: model_name/photoshoot/arrangement
            parts = favorite.split('/')
            if len(parts) >= 3:
                model = parts[0]
                photoshoot = parts[1]
                arrangement = parts[2].split('-')[0]  # Remove the detailed parameters
                
                photoshoot_combo = f"{photoshoot}/{arrangement}"
                model_photoshoots[model].add(photoshoot_combo)
                all_photoshoots.append(photoshoot_combo)
                all_models.add(model)
        
        # Count frequency of each photoshoot/arrangement
        photoshoot_counts = Counter(all_photoshoots)
        
        # Find photoshoots that appear for ALL models
        total_models = len(all_models)
        universal_photoshoots = []
        
        for photoshoot, count in photoshoot_counts.items():
            models_with_this_photoshoot = sum(1 for model_set in model_photoshoots.values() 
                                            if photoshoot in model_set)
            if models_with_this_photoshoot == total_models:
                universal_photoshoots.append((photoshoot, count))
        
        # Results
        print("=== FAVORITES ANALYSIS ===\n")
        print(f"Total models: {total_models}")
        print(f"Models: {', '.join(sorted(all_models))}\n")
        
        print("=== PHOTOSHOOTS POPULAR WITH ALL MODELS ===")
        if universal_photoshoots:
            for photoshoot, count in sorted(universal_photoshoots, key=lambda x: x[1], reverse=True):
                print(f"  {photoshoot}: {count} favorites")
        else:
            print("  No photoshoots are favorites for ALL models")
        
        print(f"\n=== TOP 10 MOST POPULAR PHOTOSHOOTS/ARRANGEMENTS ===")
        for photoshoot, count in photoshoot_counts.most_common(10):
            models_with_this = sum(1 for model_set in model_photoshoots.values() 
                                if photoshoot in model_set)
            print(f"  {photoshoot}: {count} favorites ({models_with_this}/{total_models} models)")
        
        print(f"\n=== BREAKDOWN BY MODEL ===")
        for model in sorted(all_models):
            photoshoots = model_photoshoots[model]
            print(f"  {model}: {len(photoshoots)} unique photoshoot/arrangement combinations")
            
        # Find photoshoots by popularity level
        print(f"\n=== PHOTOSHOOTS BY POPULARITY LEVEL ===")
        popularity_levels = defaultdict(list)
        
        for photoshoot, count in photoshoot_counts.items():
            models_with_this = sum(1 for model_set in model_photoshoots.values() 
                                if photoshoot in model_set)
            popularity_levels[models_with_this].append((photoshoot, count))
        
        for level in sorted(popularity_levels.keys(), reverse=True):
            print(f"  Popular with {level}/{total_models} models:")
            for photoshoot, count in sorted(popularity_levels[level], key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {photoshoot}: {count} favorites")

    #analyze_favorites('./xserver/fav.txt')