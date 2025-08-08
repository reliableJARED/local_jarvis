"""

Modular Python implementation for Wan 14B Text-to-Video model
Optimized for NSFW-API/NSFW_Wan_14b with CUDA acceleration and memory optimization.

Features:
- Clean class-based interface
- Optimized download logic (selective downloading)
- Automatic model and dependency downloading
- Memory optimization for consumer GPUs
- Offline operation after initial setup
"""

import os
import torch
import gc
import logging
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass
import json
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WanConfig:
    """Configuration class for Wan model parameters"""
    # Use proper Wan 2.2 models
    repo_id: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"  # Official Wan 2.2 model
    cache_dir: str = "./wan22_cache" 
    device: str = "cuda"  # Force CUDA if available
    dtype: str = "float16"
    max_memory_gb: float = 16.0
    offload_to_cpu: bool = False
    use_cpu_offload: bool = False
    optimize_memory: bool = True
    
    # Generation parameters for Wan 2.2
    num_frames: int = 81
    fps: int = 24  # Wan 2.2 uses 24 fps
    height: int = 720
    width: int = 1280
    guidance_scale: float = 7.0
    num_inference_steps: int = 50
    sample_shift: float = 8.0


class WanModelDownloader:
    """Handles downloading of Wan 2.2 diffusers model"""
    
    def __init__(self, config: WanConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def check_model_exists(self) -> bool:
        """Check if the essential model files are already downloaded"""
        # Check the actual huggingface cache structure
        model_cache_path = self.cache_dir / "models--" / self.config.repo_id.replace("/", "--")
        
        if not model_cache_path.exists():
            return False
            
        # Look for files in the snapshots directory
        snapshots_dir = model_cache_path / "snapshots"
        if not snapshots_dir.exists():
            return False
            
        # Check latest snapshot
        try:
            latest_snapshot = max(snapshots_dir.iterdir(), key=os.path.getctime)
            
            # Check for essential files
            essential_files = [
                "model_index.json",
                "scheduler/scheduler_config.json", 
                "transformer/config.json"
            ]
            
            for file in essential_files:
                file_path = latest_snapshot / file
                if not file_path.exists():
                    return False
            
            # Check if transformer model files exist (either single file or shards)
            transformer_dir = latest_snapshot / "transformer"
            if not transformer_dir.exists():
                return False
                
            # Look for either single safetensors file or sharded files
            has_single_file = (transformer_dir / "diffusion_pytorch_model.safetensors").exists()
            has_sharded_files = any(transformer_dir.glob("diffusion_pytorch_model-*-of-*.safetensors"))
            has_index_file = (transformer_dir / "diffusion_pytorch_model.safetensors.index.json").exists()
            
            # We need either a single file OR (sharded files AND index file)
            if not (has_single_file or (has_sharded_files and has_index_file)):
                return False
                
            return True
        except (ValueError, OSError):
            return False
            
    def download_model(self, force_download: bool = False) -> str:
        """Download the Wan 2.2 diffusers model"""
        logger.info(f"Downloading Wan 2.2 model from {self.config.repo_id}")
        
        # Since the model requires all sharded files, we need to do a full download
        # The selective download was causing issues because it was missing required shards
        logger.warning("This model requires all checkpoint shards - doing full download...")
        logger.warning("This will download several GB of data. Please be patient.")
        
        try:
            model_path = snapshot_download(
                repo_id=self.config.repo_id,
                cache_dir=str(self.cache_dir),
                resume_download=True,
                force_download=force_download
            )
            logger.info("Full model download completed successfully")
            logger.info(f"Downloaded to: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            raise e



class MemoryOptimizer:
    """Handles memory optimization for CUDA GPUs"""
    def __init__(self, config: WanConfig):
        self.config = config
        self.device_info = self._get_device_info()

    def _get_device_info(self) -> Dict[str, Any]:
        info = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "total_memory": 0,
            "available_memory": 0,
            "torch_version": torch.__version__
        }
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["total_memory"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            info["available_memory"] = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9
        return info

    def optimize_for_device(self) -> Dict[str, Any]:
        settings = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "dtype": torch.float16 if self.config.dtype == "float16" else torch.float32,
            "offload_to_cpu": False,
            "use_t5_cpu": False,
            "enable_attention_slicing": True,
            "enable_vae_slicing": True,
            "enable_sequential_cpu_offload": False
        }
        # If VRAM is low, fallback to CPU offload
        if torch.cuda.is_available():
            available_gb = self.device_info.get("available_memory", 16)
            if available_gb < 10:
                logger.warning("Low VRAM detected, enabling CPU offload and slicing")
                settings["offload_to_cpu"] = True
                settings["use_t5_cpu"] = True
                settings["enable_sequential_cpu_offload"] = True
        return settings

    def clear_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU cache cleared")



class WanTextToVideoGenerator:
    """Main class for Wan 14B Text-to-Video generation"""
    def __init__(self, config: Optional[WanConfig] = None):
        self.config = config or WanConfig()
        self.downloader = WanModelDownloader(self.config)
        self.optimizer = MemoryOptimizer(self.config)
        self.pipeline = None
        self.is_initialized = False
        logger.info(f"Initialized Wan 14B T2V Generator")
        logger.info(f"Target repository: {self.config.repo_id}")

    def setup(self, force_download: bool = False) -> None:
        logger.info("Setting up Wan 2.2 Text-to-Video model...")
        
        # Check if we have a complete model download
        model_complete = self.downloader.check_model_exists()
        
        if not model_complete or force_download:
            logger.info("Downloading complete model (including all required shards)...")
            self.model_path = self.downloader.download_model(force_download)
        else:
            logger.info("Complete model already exists, skipping download")
            # Get the path from cache
            self.model_path = snapshot_download(self.config.repo_id, cache_dir=str(self.config.cache_dir))
        
        self._load_model()
        self._apply_optimizations()
        self.is_initialized = True
        logger.info("Setup completed successfully")

    def _load_model(self) -> None:
        logger.info("Loading Wan 2.2 diffusers pipeline...")
        try:
            from diffusers import WanPipeline
            
            opt_settings = self.optimizer.optimize_for_device()
            
            # Load the diffusers pipeline with correct class
            load_kwargs = {
                "torch_dtype": opt_settings["dtype"]
            }
            
            # Only add device_map if using CPU offload, otherwise load normally
            if opt_settings["offload_to_cpu"]:
                load_kwargs["device_map"] = "balanced"  # Use "balanced" instead of "auto"
            
            self.pipeline = WanPipeline.from_pretrained(
                self.model_path,
                **load_kwargs
            )
            
            # Move to device if not using device_map
            if not opt_settings["offload_to_cpu"]:
                self.pipeline = self.pipeline.to(opt_settings["device"])
                
            logger.info("Wan 2.2 pipeline loaded successfully")
        except ImportError as e:
            logger.error(f"WanPipeline not found: {e}")
            # Fallback to generic diffusion pipeline without device_map="auto"
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16
            )
            # Move to device manually
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
            logger.info("Fallback diffusion pipeline loaded")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _apply_optimizations(self) -> None:
        if not self.pipeline:
            return
        opt_settings = self.optimizer.optimize_for_device()
        logger.info("Applying memory optimizations...")
        
        try:
            # Apply attention slicing if available and enabled
            if opt_settings.get("enable_attention_slicing", True) and hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()
                logger.info("Applied: enable_attention_slicing")
                
            # Apply VAE slicing if available and enabled  
            if opt_settings.get("enable_vae_slicing", True) and hasattr(self.pipeline, 'enable_vae_slicing'):
                self.pipeline.enable_vae_slicing()
                logger.info("Applied: enable_vae_slicing")
                
            # Apply sequential CPU offload if available and enabled
            if opt_settings.get("enable_sequential_cpu_offload", False) and hasattr(self.pipeline, 'enable_sequential_cpu_offload'):
                self.pipeline.enable_sequential_cpu_offload()
                logger.info("Applied: enable_sequential_cpu_offload")
                
        except Exception as e:
            logger.warning(f"Some optimizations could not be applied: {e}")

    def generate_video(self, prompt: str, negative_prompt: Optional[str] = None, **kwargs) -> List:
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call setup() first.")
        logger.info(f"Generating video for prompt: {prompt[:100]}...")
        self.optimizer.clear_cache()
        
        if negative_prompt is None:
            negative_prompt = self._get_default_negative_prompt()
            
        generation_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_frames": kwargs.get("num_frames", self.config.num_frames),
            "height": kwargs.get("height", self.config.height),
            "width": kwargs.get("width", self.config.width),
            "guidance_scale": kwargs.get("guidance_scale", self.config.guidance_scale),
            "num_inference_steps": kwargs.get("num_inference_steps", self.config.num_inference_steps),
            "generator": torch.Generator().manual_seed(kwargs.get("seed", 42)),
            "output_type": "np"  # Ensure we get numpy arrays
        }
        
        try:
            logger.info("Generating video with Wan 2.2 pipeline...")
            result = self.pipeline(**generation_params)
            
            # Handle different pipeline outputs
            if hasattr(result, 'frames'):
                frames = result.frames[0]  # Get first batch
            elif hasattr(result, 'videos'):
                frames = result.videos[0]  # Get first batch
            elif isinstance(result, list):
                frames = result
            else:
                # Fallback for unknown output format
                frames = result
                
            logger.info("Video generation completed")
            return frames
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            self.optimizer.clear_cache()
            raise

    def _get_default_negative_prompt(self) -> str:
        return ("low quality, worst quality, blurry, pixelated, static image, "
                "deformed, disfigured, ugly, bad anatomy, extra limbs, "
                "malformed, artifacts, oversaturated, distorted")

    def save_video(self, frames: List, output_path: str) -> None:
        try:
            # Handle different frame formats
            if isinstance(frames, list) and len(frames) > 0:
                if hasattr(frames[0], 'save'):  # PIL Images
                    # Convert PIL images to video
                    import imageio
                    with imageio.get_writer(output_path, fps=self.config.fps) as writer:
                        for frame in frames:
                            writer.append_data(np.array(frame))
                else:
                    # Use diffusers export_to_video for numpy arrays
                    export_to_video(frames, output_path, fps=self.config.fps)
            else:
                # Fallback to diffusers export
                export_to_video(frames, output_path, fps=self.config.fps)
                
            logger.info(f"Video saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save video: {e}")
            # Try a simpler approach
            try:
                import imageio
                frames_array = []
                for frame in frames:
                    if hasattr(frame, 'save'):  # PIL Image
                        frames_array.append(np.array(frame))
                    else:
                        frames_array.append(frame)
                imageio.mimsave(output_path, frames_array, fps=self.config.fps)
                logger.info(f"Video saved using fallback method to: {output_path}")
            except Exception as e2:
                logger.error(f"Fallback save also failed: {e2}")
                raise e

    def get_model_info(self) -> Dict[str, Any]:
        model_exists = self.downloader.check_model_exists()
        device_info = self.optimizer.device_info
        return {
            "repository": self.config.repo_id,
            "model_type": "Wan 2.2 Diffusers",
            "device_info": device_info,
            "model_downloaded": model_exists,
            "is_initialized": self.is_initialized,
            "config": self.config.__dict__
        }

    def cleanup(self) -> None:
        logger.info("Cleaning up resources...")
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        self.optimizer.clear_cache()
        self.is_initialized = False
        logger.info("Cleanup completed")


# Example usage and convenience functions

def create_wan_generator(repo_id: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                        cache_dir: str = "./wan22_cache",
                        optimize_memory: bool = True) -> WanTextToVideoGenerator:
    """Convenience function to create a Wan 2.2 generator"""
    config = WanConfig(
        repo_id=repo_id,
        cache_dir=cache_dir,
        optimize_memory=optimize_memory
    )
    return WanTextToVideoGenerator(config)



def quick_generate(prompt: str,
                  output_path: str = "output.mp4",
                  repo_id: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                  **kwargs) -> str:
    """Quick generation function for Wan 2.2 simple use cases"""
    generator = create_wan_generator(repo_id)
    try:
        generator.setup()
        frames = generator.generate_video(prompt, **kwargs)
        generator.save_video(frames, output_path)
        return output_path
    finally:
        generator.cleanup()



if __name__ == "__main__":
    # Example usage for Wan 2.2
    config = WanConfig(
        repo_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        cache_dir="./wan22_cache",
        height=720,
        width=1280,
        num_frames=81
    )
    generator = WanTextToVideoGenerator(config)
    try:
        generator.setup()
        info = generator.get_model_info()
        print(json.dumps(info, indent=2))
        prompt = "A beautiful woman dancing in a flowing dress, cinematic lighting"
        frames = generator.generate_video(prompt)
        generator.save_video(frames, "generated_video.mp4")
        print("Video generation completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        generator.cleanup()