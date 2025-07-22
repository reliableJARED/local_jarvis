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


class ImageGenerator:
    """
    A class for generating images using diffusion models.
    Supports text-to-image, image-to-image, and inpainting capabilities.
    """
    
    def __init__(self, 
                 model_name="TheImposterImposters/LUSTIFY-v2.0",
                 inpaint_model="andro-flock/LUSTIFY-SDXL-NSFW-checkpoint-v2-0-INPAINTING",
                 cache_dir=None,
                 use_mps=True):
        """
        Initialize the ImageGenerator with model loading and dependency checking.
        
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
        
        print(f"✅ ImageGenerator initialized")
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
        from diffusers import DiffusionPipeline, AutoPipelineForInpainting, AutoPipelineForImage2Image
        
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
            if pipeline_type == "inpainting":
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
            print(f"💡 Try lowering strength (current: {strength}) for more consistency")
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
    generator = ImageGenerator()
    #BEST - PROMPT STRUCTURE GUIDE - using a consistent [subject] [important feature], [more details] description will create similar subject"
    "[style of photo] photo of a [subject], [important feature], [more details], [pose or action], [framing], [setting/background], [lighting], [camera angle], "
    
    # Example 1: Text-to-image generation
    image1 = generator.text_to_image(
        prompt="photograph, photo of a sexy woman, tan skin, pixie cut blond hair, green eyes, naked and on top of a man, cum on her lips and chin, hands gripping his hips, looking into the camera with a satisfied smile, soft lighting, 8k",
        output_path="step1_text2img.png"
    )
    
    #Image to Image does NOT work well DO NOT BOTHER USING
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
    
