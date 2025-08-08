"""
Wan2.2 5B Image-to-Video Generator
A modular Python wrapper for Wan2.2 TI2V-5B model optimized for 16GB GPUs
Supports both official generate.py script and Diffusers integration

Author: Jared
Date: August 2025
GPU: 16GB RTX 5060 Ti
"""

import os
import sys
import torch
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from PIL import Image
import subprocess
import json
import time
from dataclasses import dataclass
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

@dataclass
class GenerationConfig:
    """Configuration for video generation parameters"""
    size: str = "1280*704"
    duration: int = 5
    fps: int = 24
    seed: Optional[int] = None
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    use_diffusers: bool = True
    use_prompt_extend: bool = False
    prompt_extend_method: str = "local_qwen"  # "local_qwen" or "dashscope"
    prompt_extend_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"

class Wan22ImageToVideo:
    """
    Wan2.2 5B Image-to-Video Generator
    
    Optimized for 16GB GPUs with memory-efficient settings
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 optimize_for_low_memory: bool = True,
                 use_diffusers: bool = True):
        """
        Initialize the Wan2.2 Image-to-Video generator
        
        Args:
            model_path: Path to the Wan2.2-TI2V-5B model directory
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            optimize_for_low_memory: Enable memory optimizations for 16GB GPUs
            use_diffusers: Use official Diffusers integration (recommended)
        """
        self.model_path = model_path or self._get_default_model_path()
        self.device = device or self._detect_device()
        self.optimize_for_low_memory = optimize_for_low_memory
        self.use_diffusers = use_diffusers
        self.model_loaded = False
        self.pipeline = None
        
        # Initialize CUDA memory management
        self._setup_cuda_memory()
        
        # Memory optimization settings for 16GB GPU (updated for 2025)
        self.memory_settings = {
            'offload_model': True,
            'convert_model_dtype': True,
            't5_cpu': True,
            'enable_model_cpu_offload': True,
            'enable_sequential_cpu_offload': False,  # Use model_cpu_offload instead
            'enable_attention_slicing': True,
            'enable_xformers_memory_efficient_attention': True
        }
        
        logger.info(f"Initialized Wan22ImageToVideo with device: {self.device}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Memory optimization: {self.optimize_for_low_memory}")
        logger.info(f"Using Diffusers: {self.use_diffusers}")
    
    def _get_default_model_path(self) -> str:
        """Get the default model path"""
        return os.path.join(os.getcwd(), "Wan2.2-TI2V-5B")
    
    def _detect_device(self) -> str:
        """Auto-detect the best available device"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"CUDA available. GPU memory: {gpu_memory:.1f}GB")
            return "cuda"
        else:
            logger.warning("CUDA not available, using CPU")
            return "cpu"
    
    def _setup_cuda_memory(self):
        """Setup CUDA memory management for optimal performance"""
        if self.device == "cuda" and torch.cuda.is_available():
            # Set CUDA memory allocation configuration
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            # Clear any existing cache
            torch.cuda.empty_cache()
            
            # Set memory fraction to use at most 90% of GPU memory
            torch.cuda.set_per_process_memory_fraction(0.9)
            
            # Enable memory efficient attention if available
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                logger.info("Enabled memory efficient attention backends")
            except AttributeError:
                logger.warning("Memory efficient attention not available in this PyTorch version")
            
            # Log memory status
            if torch.cuda.is_available():
                memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                logger.info(f"GPU Memory - Total: {memory_total:.2f}GB, "
                           f"Reserved: {memory_reserved:.2f}GB, "
                           f"Allocated: {memory_allocated:.2f}GB")
    
    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("Cleared GPU memory cache")
    
    def check_gpu_memory_available(self, required_gb: float = 2.0) -> bool:
        """
        Check if sufficient GPU memory is available
        
        Args:
            required_gb: Minimum required memory in GB
            
        Returns:
            True if sufficient memory is available
        """
        if not torch.cuda.is_available():
            return False
            
        # Clear cache first
        torch.cuda.empty_cache()
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
        available_memory = total_memory - allocated_memory
        
        logger.info(f"GPU Memory - Total: {total_memory:.2f}GB, "
                   f"Allocated: {allocated_memory:.2f}GB, "
                   f"Available: {available_memory:.2f}GB")
        
        if available_memory < required_gb:
            logger.warning(f"Insufficient GPU memory. Available: {available_memory:.2f}GB, "
                          f"Required: {required_gb:.2f}GB")
            return False
        
        return True
    
    def optimize_generation_settings(self, size: str, duration: int) -> Dict[str, Any]:
        """
        Optimize generation settings based on available GPU memory
        
        Args:
            size: Video resolution
            duration: Video duration
            
        Returns:
            Optimized settings dictionary
        """
        settings = {}
        
        # Valid Wan2.2 resolution options (only these two are supported for ti2v-5B)
        valid_sizes = [
            '704*1280',   # Portrait - smaller memory footprint
            '1280*704'    # Landscape - larger memory usage
        ]
        
        # Check available GPU memory
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            if total_memory < 12:
                # Very aggressive optimization for <12GB GPUs
                settings.update({
                    'guidance_scale': 4.5,
                    'num_inference_steps': 15,
                    'size': '704*1280'  # Use portrait mode for lower memory usage
                })
            elif total_memory < 16:
                # Moderate optimization for 12-16GB GPUs  
                settings.update({
                    'guidance_scale': 5.5,
                    'num_inference_steps': 20,
                    'size': '704*1280' if '1280*704' not in size else '1280*704'
                })
            else:
                # Minimal optimization for >16GB GPUs
                settings.update({
                    'guidance_scale': 6.5,
                    'num_inference_steps': 25,
                    'size': '1280*704' if '1280*704' in size else '704*1280'
                })
                
            # Adjust for duration - longer videos need more conservative settings
            if duration > 3:
                settings['num_inference_steps'] = max(10, settings.get('num_inference_steps', 25) - 5)
                settings['guidance_scale'] = max(3.5, settings.get('guidance_scale', 6.5) - 0.5)
                
        return settings
    
    def download_model(self, use_modelscope: bool = False) -> bool:
        """
        Download the Wan2.2-TI2V-5B model
        
        Args:
            use_modelscope: Use ModelScope instead of HuggingFace
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Model already exists at {self.model_path}")
                return True
            
            logger.info("Downloading Wan2.2-TI2V-5B model...")
            
            if use_modelscope:
                # Install modelscope if not available
                subprocess.run([sys.executable, "-m", "pip", "install", "modelscope"], 
                             check=True, capture_output=True)
                
                # Download using modelscope
                cmd = [
                    sys.executable, "-c",
                    f"from modelscope import snapshot_download; "
                    f"snapshot_download('Wan-AI/Wan2.2-TI2V-5B', cache_dir='{os.path.dirname(self.model_path)}', "
                    f"local_dir='{self.model_path}')"
                ]
            else:
                # Install huggingface-hub if not available
                subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub[cli]"], 
                             check=True, capture_output=True)
                
                # Download using huggingface-cli
                cmd = [
                    "huggingface-cli", "download", 
                    "Wan-AI/Wan2.2-TI2V-5B",
                    "--local-dir", self.model_path
                ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Model download completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download model: {e}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during model download: {e}")
            return False
    
    def check_dependencies(self) -> Dict[str, bool]:
        """
        Check if all required dependencies are installed
        
        Returns:
            Dictionary with dependency status
        """
        dependencies = {
            'torch': False,
            'torchvision': False,
            'diffusers': False,
            'transformers': False,
            'accelerate': False,
            'opencv-python': False,
            'imageio': False,
            'flash_attn': False,
            'xformers': False,
            'sentencepiece': False
        }
        
        for dep in dependencies:
            try:
                if dep == 'opencv-python':
                    import cv2
                elif dep == 'flash_attn':
                    # flash_attn is optional for memory optimization
                    try:
                        __import__(dep)
                    except ImportError:
                        dependencies[dep] = False
                        continue
                elif dep == 'xformers':
                    # xformers is optional for memory optimization
                    try:
                        __import__(dep)
                    except ImportError:
                        dependencies[dep] = False
                        continue
                else:
                    __import__(dep)
                dependencies[dep] = True
            except ImportError:
                dependencies[dep] = False
                if dep not in ['flash_attn', 'xformers']:  # Don't warn about optional packages
                    logger.warning(f"Missing dependency: {dep}")
        
        return dependencies
    
    def install_dependencies(self) -> bool:
        """
        Install required dependencies
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Installing dependencies...")
            
            # Read requirements from the Wan2.2 directory if available
            requirements_path = os.path.join(
                os.path.dirname(self.model_path), "Wan2.2", "requirements.txt"
            )
            
            if os.path.exists(requirements_path):
                cmd = [sys.executable, "-m", "pip", "install", "-r", requirements_path]
            else:
                # Fallback to manual requirements
                requirements = [
                    "torch>=2.4.0",
                    "torchvision>=0.19.0", 
                    "opencv-python>=4.9.0.80",
                    "diffusers>=0.31.0",
                    "transformers>=4.49.0",
                    "tokenizers>=0.20.3",
                    "accelerate>=1.1.1",
                    "tqdm",
                    "imageio[ffmpeg]",
                    "easydict",
                    "ftfy",
                    "dashscope",
                    "imageio-ffmpeg",
                    "numpy>=1.23.5,<2"
                ]
                cmd = [sys.executable, "-m", "pip", "install"] + requirements
            
            # Install flash_attn separately as it often fails
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "flash_attn"], 
                             check=True, capture_output=True)
            except subprocess.CalledProcessError:
                logger.warning("Failed to install flash_attn, continuing without it")
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def _load_diffusers_pipeline(self):
        """Load the Diffusers pipeline for WANv2.2"""
        try:
            from diffusers import WanPipeline
            from diffusers.utils import load_image
            
            # Use the official Diffusers model
            diffusers_model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
            
            logger.info(f"Loading Diffusers pipeline: {diffusers_model_id}")
            
            # Load pipeline with optimizations
            self.pipeline = WanPipeline.from_pretrained(
                diffusers_model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            )
            
            if self.device == "cuda":
                self.pipeline = self.pipeline.to("cuda")
                
                # Apply memory optimizations for 16GB GPUs
                if self.optimize_for_low_memory:
                    if self.memory_settings.get('enable_model_cpu_offload'):
                        self.pipeline.enable_model_cpu_offload()
                        logger.info("Enabled model CPU offload")
                    
                    if self.memory_settings.get('enable_attention_slicing'):
                        self.pipeline.enable_attention_slicing(1)
                        logger.info("Enabled attention slicing")
                    
                    # Try to enable xformers if available
                    try:
                        if self.memory_settings.get('enable_xformers_memory_efficient_attention'):
                            self.pipeline.enable_xformers_memory_efficient_attention()
                            logger.info("Enabled xformers memory efficient attention")
                    except Exception as e:
                        logger.warning(f"Could not enable xformers: {e}")
            
            self.model_loaded = True
            logger.info("Diffusers pipeline loaded successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Diffusers not available or WANv2.2 not supported: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load Diffusers pipeline: {e}")
            return False
    
    def _generate_with_diffusers(self, 
                                image_path: str, 
                                prompt: str = "",
                                config: GenerationConfig = None) -> Optional[str]:
        """Generate video using Diffusers pipeline"""
        if not self.pipeline:
            if not self._load_diffusers_pipeline():
                return None
        
        config = config or GenerationConfig()
        
        try:
            from diffusers.utils import load_image
            
            # Load and preprocess image
            input_image = load_image(image_path)
            
            # Resize image to match expected input size
            width, height = map(int, config.size.split('*'))
            input_image = input_image.resize((width, height))
            
            # Set seed for reproducibility
            if config.seed is not None:
                torch.manual_seed(config.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(config.seed)
            
            logger.info(f"Generating video with Diffusers...")
            logger.info(f"Size: {config.size}, Steps: {config.num_inference_steps}")
            logger.info(f"Guidance Scale: {config.guidance_scale}")
            
            # Generate video
            start_time = time.time()
            result = self.pipeline(
                prompt=prompt,
                image=input_image,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                height=height,
                width=width,
                num_frames=config.fps * config.duration,
                generator=torch.Generator().manual_seed(config.seed) if config.seed else None
            )
            
            generation_time = time.time() - start_time
            logger.info(f"Generation completed in {generation_time:.2f} seconds")
            
            # Save video
            output_path = f"wan22_diffusers_output_{int(time.time())}.mp4"
            
            # Convert frames to video using imageio
            import imageio
            frames = result.frames[0]  # Get first (and only) video
            imageio.mimsave(output_path, frames, fps=config.fps)
            
            logger.info(f"Video saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Diffusers generation failed: {e}")
            return None
    
    def generate_video(self,
                      image_path: str,
                      prompt: str = "",
                      output_path: Optional[str] = None,
                      size: str = "1280*704",
                      duration: int = 5,
                      fps: int = 24,
                      seed: Optional[int] = None,
                      guidance_scale: float = 7.5,
                      num_inference_steps: int = 50,
                      use_diffusers: Optional[bool] = None,
                      use_prompt_extend: Optional[bool] = None,
                      prompt_extend_method: Optional[str] = None,
                      prompt_extend_model: Optional[str] = None,
                      enable_flash_attn: Optional[bool] = None,
                      enable_xformers: Optional[bool] = None) -> Optional[str]:
        """
        Generate video from image
        
        Args:
            image_path: Path to input image
            prompt: Text prompt describing the desired video
            output_path: Path for output video (auto-generated if None)
            size: Video resolution (default: 1280*704 for 720p)
            duration: Video duration in seconds
            fps: Frames per second
            seed: Random seed for reproducibility
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of denoising steps
            use_diffusers: Use Diffusers pipeline directly
            use_prompt_extend: Enable prompt extension
            prompt_extend_method: Method for prompt extension
            prompt_extend_model: Model for prompt extension
            enable_flash_attn: Enable flash attention
            enable_xformers: Enable xformers memory efficient attention
        
        Returns:
            Path to generated video file or None if failed
        """
        if not os.path.exists(image_path):
            logger.error(f"Input image not found: {image_path}")
            return None
        
        if not os.path.exists(self.model_path):
            logger.error(f"Model not found at {self.model_path}. Please download first.")
            return None
        
        # Check GPU memory availability
        if not self.check_gpu_memory_available(2.0):
            logger.warning("Low GPU memory detected. Consider reducing video resolution or duration.")
        
        # Optimize settings based on available memory
        if self.optimize_for_low_memory:
            optimized_settings = self.optimize_generation_settings(size, duration)
            
            # Update parameters with optimized values if not explicitly set
            if guidance_scale == 7.5:  # Default value
                guidance_scale = optimized_settings.get('guidance_scale', guidance_scale)
            if num_inference_steps == 50:  # Default value
                num_inference_steps = optimized_settings.get('num_inference_steps', num_inference_steps)
            if size == "1280*704":  # Default value
                size = optimized_settings.get('size', size)
                
            logger.info(f"Optimized settings - Size: {size}, Steps: {num_inference_steps}, "
                       f"Guidance: {guidance_scale}")
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = str(int(torch.rand(1).item() * 1000000))
            output_path = f"wan22_i2v_output_{timestamp}.mp4"
        
        # Prompt extension support
        if use_prompt_extend or self.memory_settings.get('use_prompt_extend'):
            try:
                # Example: extend prompt using local_qwen or dashscope
                if prompt_extend_method == "local_qwen":
                    # Placeholder for local Qwen extension
                    prompt = f"[Qwen-extended] {prompt}"
                elif prompt_extend_method == "dashscope":
                    # Placeholder for dashscope API call
                    prompt = f"[Dashscope-extended] {prompt}"
                logger.info(f"Prompt after extension: {prompt}")
            except Exception as e:
                logger.warning(f"Prompt extension failed: {e}")
        
        # Use Diffusers pipeline if requested
        use_diffusers_final = use_diffusers if use_diffusers is not None else self.use_diffusers
        if use_diffusers_final:
            logger.info("Using Diffusers pipeline for generation.")
            config = GenerationConfig(
                size=size,
                duration=duration,
                fps=fps,
                seed=seed,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                use_diffusers=True,
                use_prompt_extend=use_prompt_extend or False,
                prompt_extend_method=prompt_extend_method or "local_qwen",
                prompt_extend_model=prompt_extend_model or "Qwen/Qwen2.5-VL-3B-Instruct"
            )
            result = self._generate_with_diffusers(image_path, prompt, config)
            if result:
                return result
            else:
                logger.warning("Diffusers pipeline failed, falling back to generate.py script.")
        
        # Otherwise, use subprocess to call generate.py
        try:
            self.clear_gpu_memory()
            cmd, wan_dir = self._build_generation_command(
                image_path=image_path,
                prompt=prompt,
                output_path=output_path,
                size=size,
                seed=seed,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
            # Add new memory flags if requested
            if enable_flash_attn:
                cmd.extend(["--enable_flash_attn"])
            if enable_xformers:
                cmd.extend(["--enable_xformers"])
            logger.info(f"Starting video generation...")
            logger.info(f"Input image: {image_path}")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Output: {output_path}")
            logger.info(f"Memory optimization enabled: {self.optimize_for_low_memory}")
            env = os.environ.copy()
            if self.optimize_for_low_memory:
                env.update({
                    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
                    'CUDA_LAUNCH_BLOCKING': '1'
                })
            result = subprocess.run(cmd, cwd=wan_dir, env=env,
                                  capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                logger.info(f"Video generation completed: {output_path}")
                self.clear_gpu_memory()
                return output_path
            else:
                logger.error(f"Generation failed. Return code: {result.returncode}")
                logger.error(f"Stdout: {result.stdout}")
                logger.error(f"Stderr: {result.stderr}")
                self.clear_gpu_memory()
                # Fallback to Diffusers if not already tried
                if not use_diffusers_final:
                    logger.info("Trying Diffusers pipeline as fallback.")
                    config = GenerationConfig(
                        size=size,
                        duration=duration,
                        fps=fps,
                        seed=seed,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        use_diffusers=True,
                        use_prompt_extend=use_prompt_extend or False,
                        prompt_extend_method=prompt_extend_method or "local_qwen",
                        prompt_extend_model=prompt_extend_model or "Qwen/Qwen2.5-VL-3B-Instruct"
                    )
                    result = self._generate_with_diffusers(image_path, prompt, config)
                    return result
                return None
        except subprocess.TimeoutExpired:
            logger.error("Generation timed out")
            return None
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return None
    
    def _build_generation_command(self, **kwargs) -> tuple:
        """Build the command for video generation"""
        wan_dir = os.path.join(os.path.dirname(self.model_path), "Wan2.2")
        generate_script = os.path.join(wan_dir, "generate.py")
        
        # Check if generate.py exists in the expected location
        if not os.path.exists(generate_script):
            # Try alternative location
            alt_wan_dir = os.path.join(os.getcwd(), "Wan2.2")
            alt_generate_script = os.path.join(alt_wan_dir, "generate.py")
            if os.path.exists(alt_generate_script):
                wan_dir = alt_wan_dir
                generate_script = alt_generate_script
            else:
                raise FileNotFoundError(f"generate.py not found. Tried:\n- {generate_script}\n- {alt_generate_script}")
        
        cmd = [
            sys.executable, generate_script,
            "--task", "ti2v-5B",
            "--size", kwargs.get('size', '1280*704'),
            "--ckpt_dir", self.model_path,
            "--image", kwargs['image_path'],
            "--prompt", kwargs.get('prompt', ''),
            "--save_file", kwargs['output_path']
        ]
        
        # Add only the memory optimization flags that the script actually supports
        if self.optimize_for_low_memory:
            cmd.extend([
                "--offload_model", "True",
                "--convert_model_dtype",
                "--t5_cpu"
            ])
        
        # Add optional parameters
        if kwargs.get('seed') is not None:
            cmd.extend(["--base_seed", str(kwargs['seed'])])
        
        if kwargs.get('guidance_scale') is not None:
            cmd.extend(["--sample_guide_scale", str(kwargs['guidance_scale'])])
            
        if kwargs.get('num_inference_steps') is not None:
            cmd.extend(["--sample_steps", str(kwargs['num_inference_steps'])])
        
        return cmd, wan_dir
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for debugging"""
        info = {
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'device': self.device,
            'model_path': self.model_path,
            'model_exists': os.path.exists(self.model_path),
            'memory_optimization': self.optimize_for_low_memory,
            'cuda_alloc_conf': os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')
        }
        
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
            reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)
            
            info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_total': f"{total_memory:.1f}GB",
                'gpu_memory_allocated': f"{allocated_memory:.1f}GB", 
                'gpu_memory_reserved': f"{reserved_memory:.1f}GB",
                'gpu_memory_free': f"{total_memory - allocated_memory:.1f}GB",
                'cuda_version': torch.version.cuda
            })
        
        return info
    
    def monitor_memory_during_generation(self, log_interval: int = 30):
        """Monitor GPU memory usage (for debugging)"""
        if not torch.cuda.is_available():
            return
            
        def log_memory():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        
        import threading
        import time
        
        def memory_monitor():
            while hasattr(self, '_monitoring'):
                log_memory()
                time.sleep(log_interval)
        
        self._monitoring = True
        monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
        monitor_thread.start()
        return monitor_thread
    
    def stop_memory_monitoring(self):
        """Stop memory monitoring"""
        if hasattr(self, '_monitoring'):
            delattr(self, '_monitoring')
    
    def cleanup(self):
        """Cleanup resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        logger.info("Cleanup completed")

# Convenience function for quick usage
def generate_video_from_image(image_path: str, 
                            prompt: str = "",
                            output_path: Optional[str] = None,
                            model_path: Optional[str] = None) -> Optional[str]:
    """
    Quick function to generate video from image
    
    Args:
        image_path: Path to input image
        prompt: Text prompt
        output_path: Output video path
        model_path: Model directory path
    
    Returns:
        Path to generated video or None if failed
    """
    generator = Wan22ImageToVideo(model_path=model_path)
    return generator.generate_video(
        image_path=image_path,
        prompt=prompt,
        output_path=output_path
    )

if __name__ == "__main__":
    # Example usage with specific image and prompt
    generator = Wan22ImageToVideo()
    
    # Print system info
    print("System Information:")
    for key, value in generator.get_system_info().items():
        print(f"  {key}: {value}")
    
    # Check dependencies
    print("\nDependency Status:")
    deps = generator.check_dependencies()
    for dep, status in deps.items():
        print(f"  {dep}: {'✓' if status else '✗'}")
    
    # Generate video with specified image and prompt
    image_path = r"C:\Users\jared\Documents\code\local_jarvis\xserver\demetra\zeus\demetra_in_zeus-p4_a4_f3_c2.png"
    prompt = "the woman gets up from the chair and walks towards the camera"
    
    print(f"\nGenerating video from image: {image_path}")
    print(f"Prompt: {prompt}")
    
    # Generate a fast video with very aggressive memory optimization for 16GB GPU
    output_video = generator.generate_video(
        image_path=image_path,
        prompt=prompt,
        duration=2,  # Very short 2-second video to minimize memory usage
        fps=24,      # Standard FPS
        size="704*1280",  # Smallest supported resolution to minimize memory usage
        guidance_scale=4.5,  # Lower guidance for faster inference and less memory
        num_inference_steps=12  # Minimal steps for fastest generation and lowest memory
    )
    
    if output_video:
        print(f"\n✓ Video generated successfully: {output_video}")
    else:
        print("\n✗ Video generation failed")
