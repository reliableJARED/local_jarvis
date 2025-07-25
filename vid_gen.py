
 ####
#
#       THIS WILL NOT WORK WITHOUT CUDA!!!!
#    Not designed for mac would take hours to run
#       As such I am abandoning this for now
#
####
"""
Official Wan2.1 Wrapper
Uses the official Wan2.1 implementation to properly load wan_1.3B_exp_e14.safetensors

This wrapper:
1. Clones/uses the official Wan2.1 repository
2. Downloads only the wan_1.3B_exp_e14.safetensors checkpoint
3. Provides a clean Python interface to the official generate.py script
4. Handles all the command-line complexity behind a simple API
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import json
import shutil
from huggingface_hub import hf_hub_download
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OfficialWanConfig:
    """Configuration for the official Wan2.1 implementation"""
    # Model configuration
    repo_id: str = "NSFW-API/NSFW_Wan_1.3b"
    checkpoint_file: str = "wan_1.3B_exp_e14.safetensors"
    
    # Paths
    work_dir: str = "./wan_official"
    checkpoint_dir: str = "./wan_official/checkpoints"
    
    # Generation parameters (optimized for 1.3B model)
    task: str = "t2v-1.3B"
    size: str = "832*480"  # Recommended stable resolution
    sample_shift: float = 8.0  # Recommended 8-12 range
    sample_guide_scale: float = 6.0  # Recommended for 1.3B
    num_inference_steps: int = 50
    fps: int = 16
    
    # Memory optimization
    offload_model: bool = True  # For consumer GPUs
    t5_cpu: bool = True  # Move T5 to CPU to save VRAM
    
    # Optional features
    use_prompt_extend: bool = False  # Disable by default (requires API key)
    prompt_extend_method: str = "local_qwen"  # Use local model if enabled


class OfficialWanGenerator:
    """Wrapper for the official Wan2.1 implementation"""
    
    def __init__(self, config: Optional[OfficialWanConfig] = None):
        self.config = config or OfficialWanConfig()
        self.work_dir = Path(self.config.work_dir)
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.wan_repo_dir = self.work_dir / "Wan2.1"
        self.is_setup = False
        
        logger.info("Official Wan2.1 Generator initialized")
        logger.info(f"Target checkpoint: {self.config.checkpoint_file}")
        
    def setup(self, force_clone: bool = False) -> None:
        """Setup the official Wan2.1 repository and download checkpoint"""
        logger.info("Setting up official Wan2.1 implementation...")
        
        # Step 1: Setup directories
        self.work_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Step 2: Clone/update official repository
        self._setup_wan_repository(force_clone)
        
        # Step 3: Install requirements
        self._install_requirements()
        
        # Step 4: Download checkpoint
        self._download_checkpoint()
        
        # Step 5: Download prompting guide
        self._download_prompting_guide()
        
        self.is_setup = True
        logger.info("Setup completed successfully!")
        
    def _setup_wan_repository(self, force_clone: bool) -> None:
        """Clone or update the official Wan2.1 repository"""
        
        if self.wan_repo_dir.exists() and not force_clone:
            logger.info("Wan2.1 repository already exists")
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
                
            logger.info("Cloning official Wan2.1 repository...")
            try:
                subprocess.run(
                    ["git", "clone", "https://github.com/Wan-Video/Wan2.1.git"],
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
        """Install the requirements for Wan2.1"""
        requirements_file = self.wan_repo_dir / "requirements.txt"
        
        if not requirements_file.exists():
            logger.warning("requirements.txt not found, skipping automatic installation")
            return
            
        logger.info("Installing Wan2.1 requirements...")
        try:
            subprocess.run(
                ["pip", "install", "-r", str(requirements_file)],
                check=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            logger.info("Requirements installed successfully")
        except subprocess.TimeoutExpired:
            logger.warning("Requirements installation timed out")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Some requirements failed to install: {e}")
            logger.warning("You may need to install them manually")
    
    def _download_checkpoint(self) -> str:
        """Download the wan_1.3B_exp_e14.safetensors checkpoint"""
        checkpoint_path = self.checkpoint_dir / self.config.checkpoint_file
        
        if checkpoint_path.exists():
            logger.info(f"Checkpoint already exists: {checkpoint_path}")
            return str(checkpoint_path)
            
        logger.info(f"Downloading checkpoint: {self.config.checkpoint_file}")
        logger.info("This will download ~2.84GB")
        
        try:
            downloaded_path = hf_hub_download(
                repo_id=self.config.repo_id,
                filename=self.config.checkpoint_file,
                local_dir=str(self.checkpoint_dir),
                resume_download=True
            )
            logger.info(f"Checkpoint downloaded: {downloaded_path}")
            return downloaded_path
        except Exception as e:
            logger.error(f"Failed to download checkpoint: {e}")
            raise
    
    def _download_prompting_guide(self) -> None:
        """Download the prompting guide"""
        guide_path = self.checkpoint_dir / "prompting-guide.json"
        
        if guide_path.exists():
            logger.info("Prompting guide already exists")
            return
            
        try:
            logger.info("Downloading prompting guide...")
            hf_hub_download(
                repo_id=self.config.repo_id,
                filename="prompting-guide.json",
                local_dir=str(self.checkpoint_dir),
                resume_download=True
            )
            logger.info("Prompting guide downloaded")
        except Exception as e:
            logger.warning(f"Could not download prompting guide: {e}")
    
    def generate_video(self, 
                      prompt: str,
                      output_path: Optional[str] = None,
                      negative_prompt: Optional[str] = None,
                      seed: Optional[int] = None,
                      **kwargs) -> str:
        """Generate video using the official implementation"""
        
        if not self.is_setup:
            raise RuntimeError("Generator not setup. Call setup() first.")
            
        # Prepare output path
        if output_path is None:
            output_path = f"generated_video_{hash(prompt) % 10000}.mp4"
        
        output_path = Path(output_path).resolve()
        
        logger.info(f"Generating video with official Wan2.1...")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Output: {output_path}")
        
        # Build command
        cmd = self._build_generation_command(prompt, output_path, negative_prompt, seed, **kwargs)
        
        # Execute generation
        try:
            logger.info("Starting generation process...")
            result = subprocess.run(
                cmd,
                cwd=str(self.wan_repo_dir),
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Generation failed with return code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                raise RuntimeError(f"Generation failed: {result.stderr}")
                
            logger.info("Generation completed successfully!")
            if result.stdout:
                logger.info(f"Generation output: {result.stdout}")
                
            # Check if output file was created
            if not output_path.exists():
                raise RuntimeError(f"Output file not created: {output_path}")
                
            return str(output_path)
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Generation timed out (10 minutes)")
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _build_generation_command(self, 
                                prompt: str,
                                output_path: Path,
                                negative_prompt: Optional[str],
                                seed: Optional[int],
                                **kwargs) -> List[str]:
        """Build the command line for generation"""
        
        cmd = [
            "python", "generate.py",
            "--task", self.config.task,
            "--size", kwargs.get("size", self.config.size),
            "--ckpt_dir", str(self.checkpoint_dir.resolve()),
            "--sample_shift", str(kwargs.get("sample_shift", self.config.sample_shift)),
            "--sample_guide_scale", str(kwargs.get("sample_guide_scale", self.config.sample_guide_scale)),
            "--prompt", prompt
        ]
        
        # Add output path if the script supports it
        # Note: Check the official generate.py to see if it has an output parameter
        if "output" in kwargs:
            cmd.extend(["--output", str(output_path)])
        
        # Memory optimizations
        if self.config.offload_model:
            cmd.append("--offload_model")
            cmd.append("True")
            
        if self.config.t5_cpu:
            cmd.append("--t5_cpu")
        
        # Negative prompt
        if negative_prompt:
            cmd.extend(["--negative_prompt", negative_prompt])
        
        # Seed
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
            
        # Inference steps
        if "num_inference_steps" in kwargs:
            cmd.extend(["--num_inference_steps", str(kwargs["num_inference_steps"])])
        
        # Prompt extension (if enabled)
        if self.config.use_prompt_extend:
            cmd.append("--use_prompt_extend")
            cmd.extend(["--prompt_extend_method", self.config.prompt_extend_method])
            
        logger.debug(f"Generation command: {' '.join(cmd)}")
        return cmd
    
    def load_prompting_guide(self) -> Optional[Dict]:
        """Load the prompting guide"""
        guide_path = self.checkpoint_dir / "prompting-guide.json"
        
        if guide_path.exists():
            try:
                with open(guide_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load prompting guide: {e}")
        
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get setup and model status"""
        return {
            "is_setup": self.is_setup,
            "wan_repository_exists": self.wan_repo_dir.exists(),
            "checkpoint_exists": (self.checkpoint_dir / self.config.checkpoint_file).exists(),
            "prompting_guide_exists": (self.checkpoint_dir / "prompting-guide.json").exists(),
            "checkpoint_file": self.config.checkpoint_file,
            "checkpoint_path": str(self.checkpoint_dir / self.config.checkpoint_file),
            "work_directory": str(self.work_dir),
            "config": {
                "task": self.config.task,
                "size": self.config.size,
                "sample_guide_scale": self.config.sample_guide_scale,
                "sample_shift": self.config.sample_shift,
                "memory_optimizations": {
                    "offload_model": self.config.offload_model,
                    "t5_cpu": self.config.t5_cpu
                }
            }
        }
    
    def cleanup(self) -> None:
        """Clean up any temporary resources"""
        logger.info("Cleanup completed")


# Convenience functions
def quick_generate_official(prompt: str, 
                          output_path: str = "output.mp4",
                          seed: Optional[int] = None,
                          **kwargs) -> str:
    """Quick video generation using official implementation"""
    
    generator = OfficialWanGenerator()
    
    # Setup if needed
    generator.setup()
    
    # Generate
    result_path = generator.generate_video(prompt, output_path, seed=seed, **kwargs)
    
    return result_path


def setup_wan_official(work_dir: str = "./wan_official", 
                      force_clone: bool = False) -> OfficialWanGenerator:
    """Setup the official Wan implementation and return generator"""
    
    config = OfficialWanConfig(work_dir=work_dir)
    generator = OfficialWanGenerator(config)
    generator.setup(force_clone=force_clone)
    
    return generator


if __name__ == "__main__":
    # Example usage
    generator = OfficialWanGenerator()
    
    try:
        # Setup - clone repo and download wan_1.3B_exp_e14.safetensors
        generator.setup()
        
        # Show status
        status = generator.get_status()
        print("Setup Status:")
        print(json.dumps(status, indent=2))
        
        # Load prompting guide
        guide = generator.load_prompting_guide()
        if guide:
            print(f"\nPrompting guide loaded with {len(guide)} entries")
            
        # Generate video
        prompt = "A beautiful woman dancing in a flowing dress, cinematic lighting"
        output_file = generator.generate_video(
            prompt=prompt,
            output_path="official_generated.mp4",
            seed=42
        )
        
        print(f"\nSuccess! Video generated: {output_file}")
        print("Used official Wan2.1 implementation with wan_1.3B_exp_e14.safetensors")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        generator.cleanup()