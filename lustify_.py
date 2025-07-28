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

class ImageGenerator:
    """
    A class for generating images using diffusion models.
    Supports text-to-image, image-to-image, and inpainting capabilities with optimized model selection.
    """
    
    def __init__(self, 
                 text2img_model="TheImposterImposters/LUSTIFY-v2.0",
                 img2img_model="stabilityai/stable-diffusion-xl-base-1.0",
                 inpaint_model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                 cache_dir=None,
                 use_mps=True,
                 use_cuda=None):
        """
        Initialize the ImageGenerator with separate models for different tasks.
        
        Args:
            text2img_model (str): Model for text-to-image generation (LUSTIFY for quality)
            img2img_model (str): Model for image-to-image generation (SDXL for compatibility)
            inpaint_model (str): Model for inpainting tasks (Official SDXL inpainting)
            cache_dir (str): Cache directory for model storage
            use_mps (bool): Use Metal Performance Shaders on Mac (if available)
            use_cuda (bool): Use CUDA if available (None=auto-detect)
        """
        self.text2img_model = text2img_model
        self.img2img_model = img2img_model
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
        
        print(f"✅ ImageGenerator initialized with multi-model setup")
        print(f"📝 Text-to-Image: {self.text2img_model}")
        print(f"🖼️  Image-to-Image: {self.img2img_model}")
        print(f"🎭 Inpainting: {self.inpaint_model}")
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
        Load the appropriate diffusion pipeline with model selection
        
        Args:
            pipeline_type (str): "text-to-image", "image-to-image", or "inpainting"
        
        Returns:
            Pipeline object
        """
        from diffusers import DiffusionPipeline, AutoPipelineForInpainting, AutoPipelineForImage2Image
        
        # Select the appropriate model based on pipeline type
        if pipeline_type == "inpainting":
            model_name = self.inpaint_model
            print(f"Loading {pipeline_type} pipeline: {model_name} (Official SDXL Inpainting)")
        elif pipeline_type == "image-to-image":
            model_name = self.img2img_model
            print(f"Loading {pipeline_type} pipeline: {model_name} (SDXL for compatibility)")
        else:  # text-to-image
            model_name = self.text2img_model
            print(f"Loading {pipeline_type} pipeline: {model_name} (LUSTIFY for quality)")
        
        try:
            # Configure model loading parameters with device-specific optimizations
            kwargs = {
                "cache_dir": self.cache_dir,
                "use_safetensors": True,
            }
            
            # Set torch_dtype based on device
            if self.is_cuda:
                kwargs["torch_dtype"] = torch.float16  # CUDA works well with float16
                kwargs["variant"] = "fp16"  # Use fp16 variant if available
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
            
            # Provide specific guidance based on the error
            if "not found" in str(e).lower():
                print(f"💡 Model '{model_name}' might not exist or be accessible.")
                print("💡 Check the model name and ensure it's available on Hugging Face.")
            elif "memory" in str(e).lower():
                print("💡 Try reducing image size or using CPU mode if you're running out of memory.")
            
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
        Generate image from text prompt using LUSTIFY v2.0 for high quality results
        
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
        print(f"🎨 Generating image from text using LUSTIFY v2.0: '{prompt}'")
        
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
            print("Generating with LUSTIFY v2.0... (this may take a minute)")
            with torch.no_grad():
                # Device-specific inference optimizations
                if self.is_mps:
                    # MPS sometimes has issues with certain operations
                    with torch.autocast(device_type='cpu', enabled=False):
                        result = pipe(prompt, **generation_kwargs)
                elif self.is_cuda:
                    # Use autocast for CUDA to improve performance
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
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
            print(f"💡 Try lowering image dimensions or reducing inference steps")
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
                      strength=0.75,           # Higher default for SDXL compatibility
                      num_inference_steps=30,  # Standard SDXL steps
                      guidance_scale=7.5,     # Standard SDXL guidance
                      use_enhanced_prompting=False,  # Don't use LUSTIFY tags for SDXL
                      **kwargs):
        """
        Transform an existing image based on text prompt using SDXL for compatibility
        
        Args:
            prompt (str): Text description of desired transformation
            input_image (str or PIL.Image): Path to input image or PIL Image object
            output_path (str): Path to save generated image
            strength (float): Transformation strength (0.0-1.0, HIGHER=more change for SDXL)
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): Guidance scale for generation
            use_enhanced_prompting (bool): Add style tags (disabled for SDXL)
            **kwargs: Additional generation parameters
        
        Returns:
            PIL.Image: Generated image
        """
        print(f"🖼️  Transforming image with SDXL: '{prompt}'")
        print(f"💡 Using strength: {strength} (SDXL typically needs higher values)")
        
        # For SDXL, we don't use LUSTIFY-specific enhancements
        if use_enhanced_prompting:
            print("🔧 Enhanced prompting disabled for SDXL compatibility")
        
        # Get pipeline (will load SDXL model)
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
        
        # Add standard negative prompt for SDXL
        negative_prompt = kwargs.pop('negative_prompt', None)
        if not negative_prompt:
            negative_prompt = "blurry, low quality, distorted, deformed, bad anatomy, extra limbs"
        
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
            print("Generating with SDXL... (this may take a minute)")
            with torch.no_grad():
                if self.is_mps:
                    with torch.autocast(device_type='cpu', enabled=False):
                        result = pipe(prompt, **generation_kwargs)
                elif self.is_cuda:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
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
            print(f"💡 Try adjusting strength (current: {strength}) - SDXL often needs 0.5-0.8")
            print(f"💡 Or try reducing guidance_scale (current: {guidance_scale})")
            return None
    
    def inpaint(self, 
               prompt,
               input_image,
               mask_image,
               output_path="output.png",
               strength=0.99,
               num_inference_steps=30,  # Standard SDXL steps
               guidance_scale=7.5,     # Standard SDXL guidance
               **kwargs):
        """
        Inpaint parts of an image based on a mask and text prompt using official SDXL inpainting
        
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
        print(f"🎭 Inpainting image with official SDXL inpainting: '{prompt}'")
        
        # Get pipeline (will load official SDXL inpainting model)
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
            if width > 1024 or height > 1024:
                # Scale down to 1024 max for MPS
                scale = min(1024/width, 1024/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                input_image = input_image.resize((new_width, new_height))
                mask_image = mask_image.resize((new_width, new_height))
                width, height = new_width, new_height
            num_inference_steps = min(num_inference_steps, 30)
            print(f"🚀 Running on MPS - optimized settings: {width}x{height}, {num_inference_steps} steps")
        elif self.is_cuda:
            # CUDA can handle larger images better
            if width > 1536 or height > 1536:
                scale = min(1536/width, 1536/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                input_image = input_image.resize((new_width, new_height))
                mask_image = mask_image.resize((new_width, new_height))
                width, height = new_width, new_height
            print(f"🚀 Running on CUDA - high resolution: {width}x{height}, {num_inference_steps} steps")
        
        # Standard negative prompt for SDXL
        negative_prompt = kwargs.pop('negative_prompt', None)
        if not negative_prompt:
            negative_prompt = "blurry, low quality, distorted, deformed, bad anatomy"
        
        # Generation parameters
        generation_kwargs = {
            "image": input_image,
            "mask_image": mask_image,
            "strength": strength,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt,
        }
        generation_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
        
        try:
            print("Generating with official SDXL inpainting... (this may take a minute)")
            with torch.no_grad():
                if self.is_mps:
                    with torch.autocast(device_type='cpu', enabled=False):
                        result = pipe(prompt, **generation_kwargs)
                elif self.is_cuda:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
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
            print(f"💡 Try adjusting the mask or prompt")
            print(f"💡 Ensure mask has clear white (inpaint) and black (keep) areas")
            return None

    def get_model_info(self):
        """
        Display information about the loaded models
        """
        print("\n" + "="*60)
        print("🤖 MODEL CONFIGURATION")
        print("="*60)
        print(f"📝 Text-to-Image: {self.text2img_model}")
        print(f"   └─ Purpose: High-quality photorealistic generation")
        print(f"   └─ Strengths: NSFW/SFW content, natural language prompts")
        print(f"")
        print(f"🖼️  Image-to-Image: {self.img2img_model}")
        print(f"   └─ Purpose: Reliable image transformation")
        print(f"   └─ Strengths: Stable results, broad compatibility")
        print(f"")
        print(f"🎭 Inpainting: {self.inpaint_model}")
        print(f"   └─ Purpose: Professional inpainting capabilities")
        print(f"   └─ Strengths: Official SDXL inpainting support")
        print(f"")
        print(f"💾 Device: {self.device}")
        print(f"📁 Cache: {self.cache_dir}")
        print("="*60)

# Example usage
if __name__ == "__main__":
    
    # Initialize the generator (will auto-detect CUDA)
    generator = ImageGenerator()
    
    # Or explicitly enable/disable CUDA
    # generator = ImageGenerator(use_cuda=True)  # Force CUDA if available
    # generator = ImageGenerator(use_cuda=False)  # Disable CUDA
    id = uuid.uuid4()
    output_path=f"{id}.png"

    # Example 1: Text-to-image generation
    image1 = generator.text_to_image(
        prompt="photograph,photo of a sexy woman, white skin, flowing black hair, glasses, soft features, light blue eyes with long eyelashes, laying on a bed, seductive stare looking at viewer. wearing black lace lingerie, sunlight rays from window 8k",
        output_path=output_path
    )
    

    #Leaving in place as it may just need more work but research online suggest it's not very effective
    if image1:
        print("IMAGE CREATED, NOW LETS MODIFY IT")

    # Example 2: Image-to-image transformation with LOWER strength for consistency
    image2 = generator.image_to_image(
            prompt="photograph,photo of the woman, sitting in a, seductive stare looking at viewer. wearing black lace lingerie, sunlight rays from window 8k",
            #prompt="photograph, photo of the woman laying on a bed looking at the camera, 8k",
            input_image=output_path,
            output_path=f"{id}_i2i.png",
            strength=0.7 # Increase for more variation
        )
        
    if image2:
            print("🎉 Two-step demo complete!")
            print("Generated images:")
            print(f"  - Original: step1_text2img.png")
            print(f"  - Transformed: step2_img2img.png")