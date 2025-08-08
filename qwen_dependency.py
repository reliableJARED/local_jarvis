import torch
import os
import socket
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests

class QwenChatDependencyManager:
    """Handles model loading, dependency management, and offline/online detection."""
    
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", model_path=None, force_offline=False):
        """Initialize the dependency manager with model loading logic."""
        self.model_name = model_name
        self.force_offline = force_offline
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        # Load the model and tokenizer
        self._load_dependencies()
    
    def _check_internet_connection(self, timeout=5):
        """Check if internet connection is available."""
        try:
            socket.create_connection(("huggingface.co", 443), timeout)
            print("Internet connection detected")
            return True
        except (socket.timeout, socket.error, OSError):
            print("No internet connection detected")
            return False
    
    def _set_hf_offline_mode(self, offline=True):
        """Set Hugging Face environment variables for offline/online mode."""
        if offline:
            print("Setting Hugging Face to offline mode...")
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
        else:
            print("Setting Hugging Face to online mode...")
            # Remove offline flags if they exist
            os.environ.pop('HF_HUB_OFFLINE', None)
            os.environ.pop('TRANSFORMERS_OFFLINE', None) 
            os.environ.pop('HF_DATASETS_OFFLINE', None)
            # Explicitly set to allow online access
            os.environ.pop('HF_TOKEN', None)  # Remove any existing token that might cause issues
    
    def _load_dependencies(self):
        """Load model and tokenizer based on availability."""
        # Determine if we should use online or offline mode
        if self.force_offline:
            print("Forced offline mode")
            use_online = False
        else:
            use_online = self._check_internet_connection()
        
        if use_online:
            self._set_hf_offline_mode(offline=False)
            print("Online mode: Will download from Hugging Face if needed")
            self._load_model_online(self.model_name)
        else:
            self._set_hf_offline_mode(offline=True)
            print("Offline mode: Using local files only")
            if self.model_path is None:
                self.model_path = self._find_cached_model()
            self._load_model_offline(self.model_path)
        
        print("Model loaded successfully!")
    
    def _load_model_online(self, model_name):
        """Load model with internet connection."""
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"CUDA detected! Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("CUDA not available. Using CPU.")
            
        print("Loading model and tokenizer...")
        try:
            # Load without any token - this should work for public models
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,  # Add this for safety
                use_auth_token=False     # Explicitly set to False
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,  # Add this for safety
                use_auth_token=False     # Explicitly set to False
            )
            
            # Print GPU memory usage after loading
            if torch.cuda.is_available():
                print(f"Model loaded on GPU! Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            
        except Exception as e:
            print(f"Error loading model online: {e}")
            if "authentication" in str(e).lower() or "token" in str(e).lower():
                print("Authentication error detected. Trying alternative approaches...")
                
                # Try with explicit no-auth parameters
                try:
                    print("Attempting load without authentication...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype="auto",
                        device_map="auto",
                        trust_remote_code=True,
                        token=False  # Use newer parameter name
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        token=False  # Use newer parameter name
                    )
                    print("Successfully loaded without authentication!")
                    return
                except Exception as e2:
                    print(f"Second attempt failed: {e2}")
                
                # Try clearing any existing HF cache that might have auth issues
                try:
                    print("Clearing HF cache and retrying...")
                    from huggingface_hub import scan_cache_dir, delete_revisions
                    
                    # This will show what's in the cache
                    cache_info = scan_cache_dir()
                    
                    # Try to delete any problematic cached versions
                    for repo in cache_info.repos:
                        if model_name.replace("/", "--") in repo.repo_id:
                            print(f"Found cached repo: {repo.repo_id}")
                            # Delete and re-download
                            delete_strategy = delete_revisions(cache_info, repo.repo_id)
                            delete_strategy.execute()
                            break
                    
                    # Retry after cache clear
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype="auto",
                        device_map="auto",
                        trust_remote_code=True,
                        force_download=True  # Force fresh download
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        force_download=True  # Force fresh download
                    )
                    print("Successfully loaded after cache clear!")
                    return
                except ImportError:
                    print("huggingface_hub not available for cache management")
                except Exception as e3:
                    print(f"Cache clear attempt failed: {e3}")
                    
            print("Falling back to offline mode...")
            model_path = self._find_cached_model()
            self._load_model_offline(model_path)
    
    def _load_model_offline(self, model_path):
        """Load model from local files only."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}.\n"
                f"Please either:\n"
                f"1. Connect to internet to download the model automatically\n"
                f"2. Download the model manually using: python {__file__} download\n"
                f"3. Specify the correct local model path\n"
                f"4. Clear your HF cache if you have authentication issues"
            )
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"CUDA detected! Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("CUDA not available. Using CPU.")
            
        print(f"Loading model from: {model_path}")
        try:
            self._set_hf_offline_mode(offline=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                local_files_only=True,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            
            # Print GPU memory usage after loading
            if torch.cuda.is_available():
                print(f"Model loaded on GPU! Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        except Exception as e:
            print(f"Failed to load model from local files: {e}")
            raise
    
    def _find_cached_model(self):
        """Try to find cached model in common Hugging Face cache locations."""
        import platform
        
        # Common cache locations
        if platform.system() == "Windows":
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        else:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        
        print(f"Searching for cached models in: {cache_dir}")
        
        # Also check for custom downloaded models in current directory
        local_paths = [
            "./Qwen2.5-7B-Instruct",
            "./qwen2.5-7b-instruct",
            f"./{self.model_name.split('/')[-1]}"
        ]
        
        for path in local_paths:
            if os.path.exists(path) and self._validate_model_files(path):
                print(f"Found valid local model at: {path}")
                return path
        
        # Look for Qwen model folders in HF cache
        model_patterns = [
            "models--Qwen--Qwen2.5-7B-Instruct",
            f"models--{self.model_name.replace('/', '--')}"
        ]
        
        if os.path.exists(cache_dir):
            for pattern in model_patterns:
                model_dir = os.path.join(cache_dir, pattern)
                
                if os.path.exists(model_dir):
                    snapshots_dir = os.path.join(model_dir, "snapshots")
                    
                    if os.path.exists(snapshots_dir):
                        snapshots = os.listdir(snapshots_dir)
                        
                        for snapshot in snapshots:
                            snapshot_path = os.path.join(snapshots_dir, snapshot)
                            
                            if self._validate_model_files(snapshot_path):
                                print(f"Found valid cached model at: {snapshot_path}")
                                return snapshot_path
        
        raise FileNotFoundError(
            f"Could not find a valid cached model for '{self.model_name}'.\n"
            f"Options:\n"
            f"1. Download model: python {__file__} download\n"
            f"2. Connect to internet and let the script download automatically\n"
            f"3. Clear HF cache if you have authentication issues:\n"
            f"   rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
        )
    
    def _validate_model_files(self, model_path):
        """Check if a model directory has the required files."""
        if not os.path.exists(model_path):
            return False
        
        required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        model_files = [f for f in os.listdir(model_path) if f.endswith(('.bin', '.safetensors'))]
        
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                return False
        
        return len(model_files) > 0
    
    def get_model(self):
        """Get the loaded model."""
        return self.model
    
    def get_tokenizer(self):
        """Get the loaded tokenizer."""
        return self.tokenizer
    
    @staticmethod
    def clear_hf_cache(model_name="Qwen/Qwen2.5-7B-Instruct"):
        """Helper function to clear Hugging Face cache for this model."""
        import shutil
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache_pattern = f"models--{model_name.replace('/', '--')}"
        model_cache_path = os.path.join(cache_dir, model_cache_pattern)
        
        if os.path.exists(model_cache_path):
            print(f"Clearing cache at: {model_cache_path}")
            shutil.rmtree(model_cache_path)
            print("Cache cleared successfully!")
        else:
            print(f"No cache found for {model_name}")
    
    @staticmethod
    def download_model(model_name="Qwen/Qwen2.5-7B-Instruct", save_path=None):
        """Helper function to download the model for offline use."""
        if save_path is None:
            save_path = f"./{model_name.split('/')[-1]}"
        
        print(f"Downloading {model_name} for offline use...")
        print(f"Save location: {save_path}")
        
        # Clear any problematic environment variables
        os.environ.pop('HF_HUB_OFFLINE', None)
        os.environ.pop('TRANSFORMERS_OFFLINE', None)
        os.environ.pop('HF_DATASETS_OFFLINE', None)
        os.environ.pop('HF_TOKEN', None)
        
        try:
            print("Downloading model and tokenizer...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype="auto",
                trust_remote_code=True,
                use_auth_token=False
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_auth_token=False
            )
            
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            print(f"Model downloaded successfully to: {save_path}")
            
        except Exception as e:
            print(f"Error downloading model: {e}")
            if "authentication" in str(e).lower() or "token" in str(e).lower():
                print("\nIf you're getting authentication errors for this public model:")
                print("1. Try clearing the HF cache: QwenChatDependencyManager.clear_hf_cache()")
                print("2. Make sure you don't have HF_TOKEN set in environment")
                print("3. Update transformers: pip install --upgrade transformers")
                print("4. The model should be accessible without any token")

# Usage example with error handling
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        QwenChatDependencyManager.download_model()
    elif len(sys.argv) > 1 and sys.argv[1] == "clear-cache":
        QwenChatDependencyManager.clear_hf_cache()
    else:
        try:
            manager = QwenChatDependencyManager()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("\nTroubleshooting steps:")
            print("1. Run: python this_file.py clear-cache")
            print("2. Run: python this_file.py download")
            print("3. Check your internet connection")
            print("4. Update transformers: pip install --upgrade transformers")