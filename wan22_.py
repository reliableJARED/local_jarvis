#!/usr/bin/env python3
"""
Wan 2.2 I2V Wrapper - Windows Fix (Environment Variables Set First)
Critical: Environment variables MUST be set before any PyTorch imports
"""

# CRITICAL: Set environment variables BEFORE any imports
import os
os.environ["USE_LIBUV"] = "0"
"""os.environ["NCCL_BLOCKING_WAIT"] = "1" 
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"""
import torch
import sys
# Now safe to import other modules
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import json
import shutil
from huggingface_hub import snapshot_download
import time


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Wan22Config:
    
    """Configuration for dual RTX 5060 Ti 16GB setup"""
    # Model configuration
    repo_id: str = "Wan-AI/Wan2.2-I2V-A14B"
    
    # Paths
    work_dir: str = "./wan22_official"
    checkpoint_dir: str = "./wan22_official/Wan2.2-I2V-A14B"
    
    # Generation parameters (optimized for dual RTX 5060 Ti 16GB)
    task: str = "i2v-A14B"
    size: str = "1280*720"  # Full 720p with 32GB total VRAM
    
    # Memory optimization (disabled for 32GB total VRAM)
    offload_model: bool = False
    convert_model_dtype: bool = True
    t5_cpu: bool = False  # Keep T5 on GPU with 16GB per card
    
    # Performance settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5


class Wan22Generator:
    """Windows-compatible wrapper with early environment variable setup"""
    
    def __init__(self, config: Optional[Wan22Config] = None):
        self.config = config or Wan22Config()
        self.work_dir = Path(self.config.work_dir)
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.wan_repo_dir = self.work_dir / "Wan2.2"
        self.is_setup = False
        
        # Windows-compatible configuration
        self.gpu_count = 2
        self.vram_gb = 32.0  # 2x 16GB
        self.use_distributed = self._check_distributed_support()
        self.quality_preset = "quality"
        
        logger.info("Wan 2.2 I2V Generator - Windows Compatible (Environment Pre-configured)")
        logger.info(f"USE_LIBUV: {os.environ.get('USE_LIBUV')}")
        logger.info(f"GPU Count: {self.gpu_count}")
        logger.info(f"Total VRAM: {self.vram_gb} GB")
        logger.info(f"Distributed training: {self.use_distributed}")
        
    def _check_distributed_support(self) -> bool:
        """Check if distributed training is supported on this system"""
        try:
            
            if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
                logger.info(f"CUDA devices detected: {torch.cuda.device_count()}")
                return True
            else:
                logger.warning("Distributed training not available - falling back to single GPU")
                return False
        except Exception as e:
            logger.warning(f"Could not check distributed support: {e}")
            return False
        
    def setup(self, force_clone: bool = False) -> None:
        """Setup the official Wan 2.2 repository and download checkpoint"""
        logger.info("Setting up Wan 2.2 I2V for Windows...")
        
        # Step 1: Setup directories
        self.work_dir.mkdir(exist_ok=True)
        
        # Step 2: Clone/update official repository
        self._setup_wan_repository(force_clone)
        
        # Step 3: Install requirements
        self._install_requirements()
        
        # Step 4: Download checkpoint
        self._download_checkpoint()
        
        self.is_setup = True
        logger.info("Setup completed successfully!")
        
    def _setup_wan_repository(self, force_clone: bool) -> None:
        """Clone or update the official Wan 2.2 repository"""
        
        if self.wan_repo_dir.exists() and not force_clone:
            logger.info("Wan 2.2 repository already exists")
            # Try to pull latest changes
            try:
                result = subprocess.run(
                    ["git", "pull"],
                    cwd=str(self.wan_repo_dir),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    logger.info("Repository updated successfully")
                else:
                    logger.warning("Could not update repository, using existing version")
            except Exception as e:
                logger.warning(f"Could not update repository: {e}")
        else:
            if self.wan_repo_dir.exists():
                logger.info("Removing existing repository for fresh clone")
                shutil.rmtree(self.wan_repo_dir)
                
            logger.info("Cloning official Wan 2.2 repository...")
            try:
                subprocess.run(
                    ["git", "clone", "https://github.com/Wan-Video/Wan2.2.git"],
                    cwd=str(self.work_dir),
                    check=True,
                    timeout=120
                )
                logger.info("Repository cloned successfully")
            except subprocess.TimeoutExpired:
                raise RuntimeError("Repository cloning timed out")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to clone repository: {e}")
    
    def _install_requirements(self) -> None:
        """Install requirements optimized for Windows RTX 5060 Ti"""
        requirements_file = self.wan_repo_dir / "requirements.txt"
        
        if not requirements_file.exists():
            logger.warning("requirements.txt not found, installing Windows compatible requirements")
            cuda_requirements = [
                "torch>=2.1.0,<2.5.0",
                "torchvision>=0.16.0", 
                "torchaudio>=2.1.0",
                "transformers>=4.45.0",
                "accelerate>=0.34.0", 
                "diffusers>=0.30.0",
                "safetensors>=0.4.0",
                "Pillow>=10.0.0",
                "numpy>=1.24.0",
                "huggingface_hub>=0.24.0"
            ]
            
            for req in cuda_requirements:
                try:
                    logger.info(f"Installing {req}...")
                    subprocess.run(
                        ["pip", "install", req],
                        check=True, capture_output=True, text=True, timeout=60
                    )
                except Exception as e:
                    logger.warning(f"Failed to install {req}: {e}")
            return
            
        logger.info("Installing Wan 2.2 requirements...")
        try:
            subprocess.run(
                ["pip", "install", "-r", str(requirements_file)],
                check=True, capture_output=True, text=True, timeout=300
            )
            logger.info("Requirements installed successfully")
        except subprocess.TimeoutExpired:
            logger.warning("Requirements installation timed out")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Some requirements failed to install: {e}")
    
    def _download_checkpoint(self) -> str:
        """Download the Wan2.2-I2V-A14B checkpoint"""
        if self.checkpoint_dir.exists() and any(self.checkpoint_dir.iterdir()):
            logger.info(f"Checkpoint already exists: {self.checkpoint_dir}")
            return str(self.checkpoint_dir)
            
        logger.info(f"Downloading Wan 2.2 I2V checkpoint...")
        logger.info("This will download ~50GB - please be patient")
        
        try:
            downloaded_path = snapshot_download(
                repo_id=self.config.repo_id,
                local_dir=str(self.checkpoint_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            logger.info(f"Checkpoint downloaded: {downloaded_path}")
            return downloaded_path
        except Exception as e:
            logger.error(f"Failed to download checkpoint: {e}")
            try:
                logger.info("Trying alternative download method...")
                subprocess.run([
                    "huggingface-cli", "download", 
                    self.config.repo_id,
                    "--local-dir", str(self.checkpoint_dir)
                ], check=True)
                logger.info("Checkpoint downloaded via CLI")
                return str(self.checkpoint_dir)
            except Exception as e2:
                logger.error(f"Alternative download also failed: {e2}")
                raise RuntimeError(f"Failed to download model: {e}")
    
    def generate_video(self, 
                      image_path: str,
                      prompt: str = "",
                      output_path: Optional[str] = None,
                      resolution: Optional[str] = None,
                      force_single_gpu: bool = True,
                      **kwargs) -> str:
        """Generate video using Windows-compatible setup"""
        
        if not self.is_setup:
            raise RuntimeError("Generator not setup. Call setup() first.")
            
        # Prepare paths
        image_path = Path(image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        if output_path is None:
            timestamp = int(time.time())
            output_path = f"wan22_generated_{timestamp}.mp4"
        
        output_path = Path(output_path).resolve()
        
        # Resolution options
        if resolution:
            size_map = {
                "fast": "832*480",
                "balanced": "1280*720", 
                "quality": "1280*720",
                "max": "1280*720"
            }
            target_size = size_map.get(resolution, self.config.size)
        else:
            target_size = self.config.size
        
        # Determine execution mode
        use_distributed = self.use_distributed and not force_single_gpu
            
        logger.info(f"Generating video (Windows mode)...")
        logger.info(f"Image: {image_path}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Resolution: {target_size}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Mode: {'Distributed' if use_distributed else 'Single GPU'}")
        
        # Try distributed first, fallback to single GPU
        success = False
        result = None
        
        if use_distributed:
            try:
                logger.info("Attempting distributed generation...")
                cmd = self._build_distributed_command(image_path, prompt, target_size, **kwargs)
                result = self._run_command_with_env(cmd)
                success = True
                logger.info("✅ Distributed generation successful!")
            except Exception as e:
                logger.warning(f"❌ Distributed generation failed: {e}")
                logger.info("Falling back to single GPU mode...")
        
        if not success:
            logger.info("Using single GPU mode...")
            cmd = self._build_single_gpu_command(image_path, prompt, target_size, **kwargs)
            result = self._run_command_with_env(cmd)
        
        # Process results
        if result.returncode != 0:
            logger.error(f"Generation failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            raise RuntimeError(f"Generation failed: {result.stderr}")
            
        logger.info("Generation completed successfully!")
        
        # Find and move the generated video
        output_video = self._find_generated_video(output_path)
        return str(output_video)
    
    def _run_command_with_env(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run command with proper environment variables"""
        # Ensure environment variables are set in the subprocess
        env = os.environ.copy()
        env.update({
            "USE_LIBUV": "0",
            "NCCL_BLOCKING_WAIT": "1", 
            "TORCH_NCCL_BLOCKING_WAIT": "1",
            "CUDA_LAUNCH_BLOCKING": "0",
            "OMP_NUM_THREADS": "1"
        })
        
        logger.info(f"Running command: {' '.join(cmd[:8])}...")  # Show first 8 args
        logger.info(f"Environment: USE_LIBUV={env.get('USE_LIBUV')}")
        
        try:
            start_time = time.time()
            
            result = subprocess.run(
                cmd,
                cwd=str(self.wan_repo_dir),
                capture_output=True,
                text=True,
                timeout=1200,  # 20 minute timeout
                env=env
            )
            
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Command completed in {duration:.1f} seconds ({duration/60:.2f} minutes)")
            
            return result
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Generation timed out (20 minutes)")
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise
    
    def _build_distributed_command(self, 
                                 image_path: Path,
                                 prompt: str,
                                 size: str,
                                 **kwargs) -> List[str]:
        """Build distributed command"""
        cmd = [
            "torchrun", 
            "--nproc_per_node=2",
            "--master_port=29500",
            "--standalone",  # Add standalone flag for Windows
            "generate.py",
            "--task", self.config.task,
            "--size", size,
            "--ckpt_dir", str(self.checkpoint_dir.resolve()),
            "--image", str(image_path),
            "--prompt", prompt,
            "--dit_fsdp", "--t5_fsdp",
            "--ulysses_size", "2"
        ]
        
        # Add generation parameters
        self._add_common_params(cmd, **kwargs)
        return cmd
    
    def _build_single_gpu_command(self, 
                                image_path: Path,
                                prompt: str,
                                size: str,
                                **kwargs) -> List[str]:
        """Build single GPU command"""
        cmd = [
            sys.executable, "generate.py",
            "--task", self.config.task,
            "--size", size,
            "--ckpt_dir", str(self.checkpoint_dir.resolve()),
            "--image", str(image_path),
            "--prompt", prompt
        ]
        # Only add offload_model and t5_cpu if config says so
        if self.config.offload_model:
            cmd.extend(["--offload_model", "True"])
        if self.config.t5_cpu:
            cmd.append("--t5_cpu")
        if self.config.convert_model_dtype:
            cmd.append("--convert_model_dtype")
        # Add generation parameters
        self._add_common_params(cmd, **kwargs)
        return cmd
    
    def _add_common_params(self, cmd: List[str], **kwargs) -> None:
        """Add common generation parameters to command"""
        # Use correct argument names for generate.py
        if "guidance_scale" in kwargs:
            cmd.extend(["--sample_guide_scale", str(kwargs["guidance_scale"])])
        else:
            cmd.extend(["--sample_guide_scale", str(self.config.guidance_scale)])

        if "num_inference_steps" in kwargs:
            cmd.extend(["--sample_steps", str(kwargs["num_inference_steps"])])
        else:
            cmd.extend(["--sample_steps", str(self.config.num_inference_steps)])

        if "seed" in kwargs:
            cmd.extend(["--base_seed", str(kwargs["seed"])])
    
    def _find_generated_video(self, target_path: Path) -> Path:
        """Find and move the generated video file"""
        if target_path.exists():
            return target_path
            
        # Look for video in outputs directory
        output_dir = self.wan_repo_dir / "outputs"
        if output_dir.exists():
            video_files = list(output_dir.glob("*.mp4"))
            if video_files:
                latest_video = max(video_files, key=lambda p: p.stat().st_mtime)
                shutil.move(str(latest_video), str(target_path))
                logger.info(f"Video moved from outputs to: {target_path}")
                return target_path
        
        # Look for video files in the main directory
        for ext in ["*.mp4", "*.avi", "*.mov"]:
            video_files = list(self.wan_repo_dir.glob(ext))
            if video_files:
                latest_video = max(video_files, key=lambda p: p.stat().st_mtime)
                shutil.move(str(latest_video), str(target_path))
                logger.info(f"Found and moved video to: {target_path}")
                return target_path
        
        raise RuntimeError(f"Output video file not found. Expected: {target_path}")
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run benchmark"""
        if not self.is_setup:
            raise RuntimeError("Generator not setup. Call setup() first.")
            
        logger.info("🏁 Running Windows benchmark...")
        
        # Create test image
        test_image_path = self.work_dir / "test_image.jpg"
        self._create_test_image(test_image_path)
        
        results = {}
        test_configs = [
            ("Fast Single GPU", "832*480", "peaceful lake", 30, True),
            ("Fast Distributed", "832*480", "peaceful lake", 30, False),
        ]
        
        for name, resolution, prompt, steps, force_single in test_configs:
            if force_single or not self.use_distributed:
                # Skip distributed test if not supported
                if not force_single and not self.use_distributed:
                    continue
                    
            logger.info(f"🧪 Testing {name}...")
            
            try:
                start_time = time.time()
                output = self.generate_video(
                    image_path=str(test_image_path),
                    prompt=prompt,
                    num_inference_steps=steps,
                    force_single_gpu=force_single
                )
                
                end_time = time.time()
                duration = end_time - start_time
                
                results[name] = {
                    'duration': duration,
                    'resolution': resolution,
                    'steps': steps,
                    'output_file': output,
                    'speed_rating': self._get_speed_rating(duration),
                    'mode': 'single_gpu' if force_single else 'distributed'
                }
                
                logger.info(f"   ✅ Completed in {duration:.1f}s ({duration/60:.2f} min)")
                
            except Exception as e:
                results[name] = {'error': str(e)}
                logger.error(f"   ❌ Failed: {e}")
        
        return results
    
    def _create_test_image(self, path: Path):
        """Create a simple test image for benchmarking"""
        try:
            from PIL import Image, ImageDraw
            
            img = Image.new('RGB', (512, 512), color='lightblue')
            draw = ImageDraw.Draw(img)
            
            draw.ellipse([100, 100, 400, 400], fill='white', outline='blue')
            draw.rectangle([200, 200, 300, 300], fill='yellow')
            
            img.save(path)
            logger.info(f"Test image created: {path}")
            
        except Exception as e:
            logger.error(f"Failed to create test image: {e}")
            raise
    
    def _get_speed_rating(self, duration: float) -> str:
        """Get a speed rating based on generation time"""
        if duration < 300:  # < 5 minutes
            return "Excellent"
        elif duration < 600:  # < 10 minutes  
            return "Good"
        elif duration < 900:  # < 15 minutes
            return "Acceptable"
        else:
            return "Slow"
    
    def get_status(self) -> Dict[str, Any]:
        """Get setup and model status"""
        return {
            "is_setup": self.is_setup,
            "wan_repository_exists": self.wan_repo_dir.exists(),
            "checkpoint_exists": self.checkpoint_dir.exists(),
            "checkpoint_path": str(self.checkpoint_dir),
            "work_directory": str(self.work_dir),
            "hardware": {
                "gpu_count": self.gpu_count,
                "vram_gb": self.vram_gb,
                "distributed_support": self.use_distributed,
                "quality_preset": self.quality_preset
            },
            "environment": {
                "use_libuv": os.environ.get("USE_LIBUV"),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "nccl_blocking_wait": os.environ.get("NCCL_BLOCKING_WAIT"),
                "platform": "windows"
            }
        }
    
    def cleanup(self) -> None:
        """Clean up any temporary resources"""
        logger.info("Cleanup completed")


# Convenience functions
def quick_generate_wan22(image_path: str,
                        prompt: str = "",
                        output_path: str = "output.mp4",
                        resolution: str = "quality",
                        force_single_gpu: bool = True,
                        **kwargs) -> str:
    """Quick video generation (Windows compatible)"""
    
    generator = Wan22Generator()
    generator.setup()
    
    result_path = generator.generate_video(
        image_path=image_path, 
        prompt=prompt,
        output_path=output_path,
        resolution=resolution,
        force_single_gpu=force_single_gpu,
        **kwargs
    )
    
    return result_path


def setup_wan22(work_dir: str = "./wan22_official", 
               force_clone: bool = False) -> Wan22Generator:
    """Setup Wan 2.2 (Windows compatible) and return generator"""
    
    config = Wan22Config(work_dir=work_dir)
    generator = Wan22Generator(config)
    generator.setup(force_clone=force_clone)
    
    return generator

if __name__ == "__main__":
    print("🎥 Wan 2.2 I2V - Dual RTX 5060 Ti 16GB Optimized")
    print("=" * 60)
    
    # Demo configuration
    demo_image = r"C:\Users\jared\Documents\code\local_jarvis\xserver\demetra\zeus\demetra_in_zeus-p4_a4_f3_c2.png"
    demo_prompt = "the woman stands up from the chair and walks towards viewer"
    demo_quality = "fast"
    demo_output = "demetra_demo_video.mp4"
    
    generator = Wan22Generator()
    
    try:
        print("🔧 Setting up Wan 2.2 I2V...")
        generator.setup()
        print("✅ Setup completed successfully!")
        
        print(f"\n🎬 Running demo generation:")
        print(f"   Image: {demo_image}")
        print(f"   Prompt: {demo_prompt}")
        print(f"   Quality: {demo_quality}")
        print(f"   Output: {demo_output}")
        
        # Check if demo image exists
        if not Path(demo_image).exists():
            print(f"\n⚠️  Demo image not found: {demo_image}")
            print("   Creating a test image for demo...")
            test_dir = Path("./test_images")
            test_dir.mkdir(exist_ok=True)
            demo_image = str(test_dir / "demo_test.jpg")
            generator._create_test_image(Path(demo_image))
            print(f"   Using test image: {demo_image}")
        
        output = generator.generate_video(
            image_path=demo_image,
            prompt=demo_prompt,
            output_path=demo_output,
            resolution=demo_quality
        )
        
        print(f"\n✅ Demo completed! Video generated: {output}")
        print("\n💡 Dual RTX 5060 Ti (Blackwell) Performance:")
        print("   • 'fast' preset (480p): ~6-8 minutes")
        print("   • 'quality' preset (720p): ~9-12 minutes")
        print("   • Multi-GPU provides ~2x speedup over single GPU")
        print("   ⚠️ Performance may be lower due to missing xformers/flash_attn")
        
        print("\n📖 To use this module in your code:")
        print("   from wan22_ import quick_generate_wan22, Wan22Generator")
        print("   result = quick_generate_wan22('image.jpg', 'prompt', 'output.mp4')")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        generator.cleanup()
