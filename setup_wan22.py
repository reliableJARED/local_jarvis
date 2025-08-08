"""
Setup script for Wan2.2 5B Image-to-Video model
Handles model download, dependency installation, and environment setup

Usage:
    python setup_wan22.py --download-model --install-deps
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Wan22Setup:
    """Setup handler for Wan2.2 environment"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or os.getcwd()
        self.model_dir = os.path.join(self.base_dir, "Wan2.2-TI2V-5B")
        self.wan_repo_dir = os.path.join(self.base_dir, "Wan2.2")
    
    def install_dependencies(self) -> bool:
        """Install all required dependencies"""
        logger.info("Installing Wan2.2 dependencies...")
        
        try:
            # Basic requirements
            basic_requirements = [
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
                "numpy>=1.23.5,<2",
                "huggingface_hub[cli]",
                "Pillow"
            ]
            
            logger.info("Installing basic requirements...")
            cmd = [sys.executable, "-m", "pip", "install"] + basic_requirements
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Basic requirements installed successfully")
            
            # Try to install flash_attn (optional)
            try:
                logger.info("Installing flash_attn (optional)...")
                subprocess.run([sys.executable, "-m", "pip", "install", "flash_attn"], 
                             check=True, capture_output=True, timeout=300)
                logger.info("flash_attn installed successfully")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                logger.warning("Failed to install flash_attn - continuing without it")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            logger.error(f"Stderr: {e.stderr}")
            return False
    
    def clone_wan_repo(self) -> bool:
        """Clone the Wan2.2 repository for generate.py"""
        if os.path.exists(self.wan_repo_dir):
            logger.info(f"Wan2.2 repository already exists at {self.wan_repo_dir}")
            return True
        
        try:
            logger.info("Cloning Wan2.2 repository...")
            cmd = [
                "git", "clone", 
                "https://github.com/Wan-Video/Wan2.2.git",
                self.wan_repo_dir
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Wan2.2 repository cloned successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
            return False
    
    def download_model_huggingface(self) -> bool:
        """Download model using Hugging Face"""
        try:
            logger.info("Downloading Wan2.2-TI2V-5B model from Hugging Face...")
            cmd = [
                "huggingface-cli", "download",
                "Wan-AI/Wan2.2-TI2V-5B",
                "--local-dir", self.model_dir
            ]
            
            subprocess.run(cmd, check=True)
            logger.info(f"Model downloaded successfully to {self.model_dir}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download model via Hugging Face: {e}")
            return False
    
    def download_model_modelscope(self) -> bool:
        """Download model using ModelScope"""
        try:
            # Install modelscope
            subprocess.run([sys.executable, "-m", "pip", "install", "modelscope"], 
                         check=True, capture_output=True)
            
            logger.info("Downloading Wan2.2-TI2V-5B model from ModelScope...")
            
            # Use Python to download via modelscope
            download_script = f"""
import os
from modelscope import snapshot_download
snapshot_download(
    'Wan-AI/Wan2.2-TI2V-5B', 
    cache_dir='{os.path.dirname(self.model_dir)}',
    local_dir='{self.model_dir}'
)
print("Model downloaded successfully")
"""
            
            result = subprocess.run([sys.executable, "-c", download_script], 
                                  check=True, capture_output=True, text=True)
            logger.info(f"Model downloaded successfully to {self.model_dir}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download model via ModelScope: {e}")
            return False
    
    def check_cuda_setup(self) -> dict:
        """Check CUDA setup and GPU memory"""
        info = {
            'cuda_available': False,
            'gpu_count': 0,
            'gpu_memory': [],
            'torch_version': None
        }
        
        try:
            import torch
            info['torch_version'] = torch.__version__
            info['cuda_available'] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                info['gpu_count'] = torch.cuda.device_count()
                for i in range(info['gpu_count']):
                    props = torch.cuda.get_device_properties(i)
                    info['gpu_memory'].append({
                        'device': i,
                        'name': props.name,
                        'memory_gb': props.total_memory / (1024**3)
                    })
        except ImportError:
            logger.warning("PyTorch not installed")
        
        return info
    
    def verify_setup(self) -> bool:
        """Verify the complete setup"""
        logger.info("Verifying setup...")
        
        # Check if model exists
        if not os.path.exists(self.model_dir):
            logger.error(f"Model directory not found: {self.model_dir}")
            return False
        
        # Check if Wan2.2 repo exists
        if not os.path.exists(self.wan_repo_dir):
            logger.error(f"Wan2.2 repository not found: {self.wan_repo_dir}")
            return False
        
        # Check if generate.py exists
        generate_script = os.path.join(self.wan_repo_dir, "generate.py")
        if not os.path.exists(generate_script):
            logger.error(f"generate.py not found: {generate_script}")
            return False
        
        # Check CUDA setup
        cuda_info = self.check_cuda_setup()
        logger.info(f"CUDA available: {cuda_info['cuda_available']}")
        
        if cuda_info['cuda_available'] and cuda_info['gpu_memory']:
            for gpu in cuda_info['gpu_memory']:
                logger.info(f"GPU {gpu['device']}: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
                
                if gpu['memory_gb'] < 16:
                    logger.warning(f"GPU {gpu['device']} has less than 16GB memory")
        
        logger.info("Setup verification completed successfully")
        return True
    
    def create_example_config(self):
        """Create example configuration file"""
        config = {
            "model_path": self.model_dir,
            "wan_repo_path": self.wan_repo_dir,
            "default_settings": {
                "size": "1280*704",
                "optimize_for_low_memory": True,
                "offload_model": True,
                "convert_model_dtype": True,
                "t5_cpu": True
            }
        }
        
        config_path = os.path.join(self.base_dir, "wan22_config.json")
        
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Example configuration created: {config_path}")

def main():
    parser = argparse.ArgumentParser(description="Setup Wan2.2 5B Image-to-Video model")
    parser.add_argument("--install-deps", action="store_true", 
                       help="Install dependencies")
    parser.add_argument("--download-model", action="store_true", 
                       help="Download the model")
    parser.add_argument("--clone-repo", action="store_true",
                       help="Clone Wan2.2 repository")
    parser.add_argument("--use-modelscope", action="store_true",
                       help="Use ModelScope instead of Hugging Face")
    parser.add_argument("--verify", action="store_true",
                       help="Verify setup")
    parser.add_argument("--all", action="store_true",
                       help="Run all setup steps")
    parser.add_argument("--base-dir", type=str, default=None,
                       help="Base directory for installation")
    
    args = parser.parse_args()
    
    if not any([args.install_deps, args.download_model, args.clone_repo, 
               args.verify, args.all]):
        parser.print_help()
        return
    
    setup = Wan22Setup(args.base_dir)
    
    success = True
    
    if args.all or args.install_deps:
        if not setup.install_dependencies():
            success = False
    
    if args.all or args.clone_repo:
        if not setup.clone_wan_repo():
            success = False
    
    if args.all or args.download_model:
        if args.use_modelscope:
            if not setup.download_model_modelscope():
                success = False
        else:
            if not setup.download_model_huggingface():
                success = False
    
    if args.all or args.verify:
        if not setup.verify_setup():
            success = False
        else:
            setup.create_example_config()
    
    if success:
        logger.info("Setup completed successfully!")
        logger.info("You can now use the wan22_video_i2v.py module")
    else:
        logger.error("Setup completed with errors")
        sys.exit(1)

if __name__ == "__main__":
    main()
