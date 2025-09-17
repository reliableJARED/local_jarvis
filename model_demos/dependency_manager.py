#!/usr/bin/env python3
"""
Dependency Manager for Python Projects
Handles automatic installation and checking of required packages
Enhanced with model download functionality
"""

import subprocess
import sys
import importlib.util
import os
import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path

# For unzipping files for windows libvips setup
import zipfile



class Orenda_DependencyManager:
    """
    A class to manage Python package dependencies with automatic installation
    Enhanced with model download capabilities
    """
    
    def __init__(self):
        """
        Initialize the dependency manager
       
        """
        self.logger = logging.getLogger(__name__) # Use module-level logger
        _ = self._setup_default_logger() #setup default logger settings
        self.failed_installs: List[str] = []
        self.successful_installs: List[str] = []
        self.already_installed: List[str] = []
        
        # Model-related tracking
        self.downloaded_models: List[str] = []
        self.failed_downloads: List[str] = []
        self.cached_models: List[str] = []
    
    def _setup_default_logger(self) -> bool:
        """Setup settings for dependency manager logger """
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        return True
    
    def is_package_installed(self, import_name: str) -> bool:
        """
        Check if a package is already installed
        
        Args:
            import_name: The name to use when importing the package
            
        Returns:
            True if package is installed, False otherwise
        """
        spec = importlib.util.find_spec(import_name)
        is_installed = spec is not None
        self.logger.debug(f"Package {import_name} installed: {is_installed}")
        return is_installed
    
    def get_model_cache_path(self, model_name: str) -> Path:
        """
        Get the cache path for a specific model
        
        Args:
            model_name: The model name (e.g., "vikhyatk/moondream2")
            
        Returns:
            Path object pointing to the model cache directory
        """
        # Use the same cache directory as transformers
        cache_dir = os.environ.get('HF_HOME', 
                                 os.path.join(os.path.expanduser('~'), '.cache', 'huggingface'))
        
        # Convert model name to safe directory name
        safe_model_name = model_name.replace('/', '--')
        model_path = Path(cache_dir) / 'hub' / f'models--{safe_model_name}'
        
        return model_path
    
    def is_model_cached(self, model_name: str, revision: str = "main") -> bool:
        """
        Check if a model is already cached locally
        
        Args:
            model_name: The model name (e.g., "vikhyatk/moondream2")
            revision: The model revision/branch (default: "main")
            
        Returns:
            True if model is cached, False otherwise
        """
        try:
            # Try to import transformers to check cache
            from transformers import AutoModelForCausalLM
            from transformers.utils import cached_file
            
            # Check if model files exist in cache
            try:
                config_path = cached_file(
                    model_name, 
                    "config.json", 
                    revision=revision,
                    local_files_only=True  # Only check local cache
                )
                if config_path:
                    self.logger.debug(f"Model {model_name} found in cache")
                    return True
            except Exception:
                # Model not in cache or other issue
                pass
            
            return False
            
        except ImportError:
            self.logger.warning("transformers not installed, cannot check model cache")
            return False
    
    def download_model(self, model_name: str, revision: str = "main", 
                      force_download: bool = False) -> bool:
        """
        Download a model and cache it locally
        
        Args:
            model_name: The model name (e.g., "vikhyatk/moondream2")
            revision: The model revision/branch (default: "main")
            force_download: Force re-download even if cached
            
        Returns:
            True if model is available (cached or downloaded), False if download failed
        """
        self.logger.info(f"Checking model: {model_name}")
        
        # Check if already cached (unless forcing download)
        if not force_download and self.is_model_cached(model_name, revision):
            self.logger.info(f"Model {model_name} is already cached")
            self.cached_models.append(model_name)
            return True
        
        # Import required libraries
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as e:
            self.logger.error(f"Required libraries not installed: {e}")
            self.failed_downloads.append(model_name)
            return False
        
        self.logger.info(f"Downloading model: {model_name} (revision: {revision})")
        self.logger.info("This may take several minutes for large models...")
        
        try:
            # Download model with progress (config and tokenizer first for quick validation)
            self.logger.info("Downloading model configuration...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                revision=revision,
                trust_remote_code=True
            )
            
            self.logger.info("Downloading model weights...")
            # Download model without loading it fully into memory
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision=revision,
                trust_remote_code=True,
                torch_dtype=torch.float16,  # Use float16 to save memory during download
                device_map=None,  # Don't load to device yet
                low_cpu_mem_usage=True  # Use less CPU memory during loading
            )
            
            # Clean up memory
            del model
            del tokenizer
            
            self.logger.info(f"Successfully downloaded model: {model_name}")
            self.downloaded_models.append(model_name)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download model {model_name}: {e}")
            self.failed_downloads.append(model_name)
            return False
    
    def download_predefined_models(self, force_download: bool = False) -> bool:
        """
        Download all predefined models used by the project
        
        Args:
            force_download: Force re-download even if cached
            
        Returns:
            True if all models are available, False if any failed
        """
        models_to_download = [
            ("vikhyatk/moondream2", "2025-01-09"),
            ("Qwen/Qwen-Audio-Chat", "main"),
            ("Qwen/Qwen2.5-7B-Instruct", "main"),
        ]
        
        all_success = True
        
        for model_name, revision in models_to_download:
            success = self.download_model(model_name, revision, force_download)
            if not success:
                all_success = False
        
        return all_success
    
    def get_model_info(self, model_name: str) -> Dict[str, any]:
        """
        Get information about a model's cache status and size
        
        Args:
            model_name: The model name to check
            
        Returns:
            Dictionary with model information
        """
        info = {
            "model_name": model_name,
            "is_cached": False,
            "cache_path": None,
            "cache_size_mb": 0
        }
        
        cache_path = self.get_model_cache_path(model_name)
        info["cache_path"] = str(cache_path)
        
        if cache_path.exists():
            info["is_cached"] = True
            
            # Calculate cache size
            total_size = 0
            try:
                for file_path in cache_path.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                info["cache_size_mb"] = round(total_size / (1024 * 1024), 2)
            except Exception as e:
                self.logger.warning(f"Could not calculate cache size: {e}")
        
        return info
    
    def list_cached_models(self) -> List[Dict[str, any]]:
        """
        List all cached models and their information
        
        Returns:
            List of dictionaries with model information
        """
        #model names used by Orenda to check
        common_models = [
            "vikhyatk/moondream2",
        ]
        
        cached_models = []
        
        # Check transformers cache directory
        try:
            cache_dir = os.environ.get('HF_HOME', 
                                     os.path.join(os.path.expanduser('~'), '.cache', 'huggingface'))
            hub_dir = Path(cache_dir) / 'hub'
            
            if hub_dir.exists():
                # Find all model directories
                for model_dir in hub_dir.iterdir():
                    if model_dir.is_dir() and model_dir.name.startswith('models--'):
                        # Convert directory name back to model name
                        model_name = model_dir.name.replace('models--', '').replace('--', '/')
                        model_info = self.get_model_info(model_name)
                        if model_info["is_cached"]:
                            cached_models.append(model_info)
            
        except Exception as e:
            self.logger.warning(f"Error listing cached models: {e}")
        
        return cached_models
    
    def install_package(self, package_name: str, import_name: Optional[str] = None, 
                       custom_command: Optional[List[str]] = None) -> bool:
        """
        Install a package if it's not already installed
        
        Args:
            package_name: The name of the package to install (pip name)
            import_name: The name to use when importing (defaults to package_name)
            custom_command: Custom pip command to use instead of standard install
            
        Returns:
            True if package is available (already installed or successfully installed), False otherwise
        """
        if import_name is None:
            import_name = package_name
        
        # Check if package is already installed
        if self.is_package_installed(import_name):
            self.logger.info(f"{package_name} is already installed")
            self.already_installed.append(package_name)
            return True
        
        self.logger.info(f"Installing {package_name}")
        
        # Use custom command if provided, otherwise use standard pip install
        if custom_command:
            command = custom_command
            self.logger.debug(f"Running custom command: {' '.join(command)}")
        else:
            command = [sys.executable, "-m", "pip", "install","--verbose", package_name]
            self.logger.debug(f"Running pip install {package_name}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            self.logger.debug(f"pip install stdout: {result.stdout}")
            self.logger.info(f"Successfully installed {package_name}")
            self.successful_installs.append(package_name)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install {package_name}: {e}")
            self.logger.debug(f"pip install stderr: {e.stderr}")
            self.failed_installs.append(package_name)
            return False
    
    def install_pytorch(self) -> bool:
        """
        Install PyTorch with platform-specific configurations:
        - Windows: PyTorch nightly build with CUDA 12.8 support
        - macOS/Linux: Regular PyTorch installation
        
        Returns:
            True if PyTorch packages are available, False otherwise
        """
        pytorch_packages = ["torch", "torchvision", "torchaudio"]
        
        # Determine platform
        is_windows = os.name == 'nt'
        platform_name = "Windows" if is_windows else "macOS/Linux"
        
        self.logger.info(f"Installing PyTorch for {platform_name}...")
        
        if is_windows:
            return self._install_pytorch_windows_nightly()
        else:
            return self._install_pytorch_standard(pytorch_packages)

    def _install_pytorch_windows_nightly(self) -> bool:
        """
        Install PyTorch nightly build with CUDA 12.8 support for Windows
        Forces reinstallation if wrong version is detected
        
        Returns:
            True if PyTorch packages are available, False otherwise
        """
        pytorch_packages = ["torch", "torchvision", "torchaudio"]
        
        # Check if PyTorch is installed and get version info
        try:
            import torch
            current_version = torch.__version__
            cuda_available = torch.cuda.is_available()
            
            self.logger.info(f"Found PyTorch version: {current_version}")
            self.logger.info(f"CUDA available: {cuda_available}")
            
            # Check if it's the correct nightly version with CUDA
            if "cu128" in current_version and cuda_available:
                self.logger.info("Correct PyTorch nightly with CUDA 12.8 is already installed")
                for pkg in pytorch_packages:
                    self.already_installed.append(pkg)
                return True
            else:
                self.logger.warning(f"Wrong PyTorch version detected: {current_version}")
                self.logger.info("Will reinstall PyTorch nightly with CUDA 12.8...")
                
                # Uninstall existing PyTorch first
                uninstall_command = [
                    sys.executable, "-m", "pip", "uninstall", 
                    "torch", "torchvision", "torchaudio", "-y"
                ]
                
                try:
                    subprocess.run(uninstall_command, check=True)
                    self.logger.info("Existing PyTorch uninstalled")
                except subprocess.CalledProcessError as e:
                    self.logger.warning(f"Failed to uninstall existing PyTorch: {e}")
                
        except ImportError:
            self.logger.info("PyTorch not found, will install nightly build...")
        
        self.logger.info("Installing PyTorch nightly build with CUDA 12.8 support...")
        
        # Custom command for PyTorch nightly installation
        pytorch_command = [
            sys.executable, "-m", "pip", "install", "--pre",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/nightly/cu128",
            "--force-reinstall"  # Force reinstall to ensure we get the right version
        ]
        
        self.logger.debug(f"Running: {' '.join(pytorch_command)}")
        
        try:
            result = subprocess.run(
                pytorch_command,
                capture_output=True,
                text=True,
                check=True
            )
            self.logger.debug(f"PyTorch nightly install stdout: {result.stdout}")
            self.logger.info("Successfully installed PyTorch nightly build")
            
            # Verify installation
            try:
                # Clear any cached imports
                if 'torch' in sys.modules:
                    del sys.modules['torch']
                if 'torchvision' in sys.modules:
                    del sys.modules['torchvision'] 
                if 'torchaudio' in sys.modules:
                    del sys.modules['torchaudio']
                
                import torch
                new_version = torch.__version__
                cuda_available = torch.cuda.is_available()
                
                self.logger.info(f"Verification - PyTorch version: {new_version}")
                self.logger.info(f"Verification - CUDA available: {cuda_available}")
                
                if "cu128" in new_version and cuda_available:
                    for pkg in pytorch_packages:
                        self.successful_installs.append(pkg)
                    return True
                else:
                    self.logger.error("Installation verification failed - wrong version installed")
                    for pkg in pytorch_packages:
                        self.failed_installs.append(pkg)
                    return False
                    
            except ImportError as e:
                self.logger.error(f"Failed to import PyTorch after installation: {e}")
                for pkg in pytorch_packages:
                    self.failed_installs.append(pkg)
                return False
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install PyTorch nightly: {e}")
            self.logger.debug(f"PyTorch nightly install stderr: {e.stderr}")
            
            # Add all PyTorch packages to failed installs
            for pkg in pytorch_packages:
                self.failed_installs.append(pkg)
            
            return False

    def _install_pytorch_standard(self, pytorch_packages: List[str]) -> bool:
        """
        Install standard PyTorch for macOS/Linux
        
        Args:
            pytorch_packages: List of PyTorch package names
            
        Returns:
            True if PyTorch packages are available, False otherwise
        """
        # Check if PyTorch is already installed
        try:
            import torch
            current_version = torch.__version__
            
            self.logger.info(f"Found PyTorch version: {current_version}")
            
            # Check device availability
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.logger.info("MPS (Apple Silicon GPU) is available")
            elif torch.cuda.is_available():
                self.logger.info("CUDA is available")
            else:
                self.logger.info("Using CPU backend")
            
            # If PyTorch is already installed, we're good
            self.logger.info("PyTorch is already installed")
            for pkg in pytorch_packages:
                self.already_installed.append(pkg)
            return True
            
        except ImportError:
            self.logger.info("PyTorch not found, will install standard version...")
        
        self.logger.info("Installing standard PyTorch...")
        
        # Install each package individually to better track success/failure
        all_success = True
        
        for package in pytorch_packages:
            if self.install_package(package):
                self.logger.debug(f"Successfully handled {package}")
            else:
                self.logger.error(f"Failed to install {package}")
                all_success = False
        
        if all_success:
            # Verify installation
            try:
                # Clear any cached imports
                for pkg_name in ['torch', 'torchvision', 'torchaudio']:
                    if pkg_name in sys.modules:
                        del sys.modules[pkg_name]
                
                import torch
                new_version = torch.__version__
                
                self.logger.info(f"Verification - PyTorch version: {new_version}")
                
                # Check device availability
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.logger.info("Verification - MPS (Apple Silicon GPU) is available")
                elif torch.cuda.is_available():
                    self.logger.info("Verification - CUDA is available")
                else:
                    self.logger.info("Verification - Using CPU backend")
                
                return True
                
            except ImportError as e:
                self.logger.error(f"Failed to import PyTorch after installation: {e}")
                return False
        
        return all_success

    def install_dependencies(self, dependencies: List[Tuple[str, str]], 
                           skip_pytorch: bool = False) -> bool:
        """
        Install multiple dependencies
        
        Args:
            dependencies: List of tuples (package_name, import_name)
            skip_pytorch: If True, skip PyTorch packages from regular installation
            
        Returns:
            True if all dependencies are available, False if any failed
        """
        self.logger.info("Checking dependencies")
        self.logger.debug(f"Dependencies to check: {dependencies}")
        
        # Reset tracking lists
        self.failed_installs = []
        self.successful_installs = []
        self.already_installed = []
        
        # Filter out PyTorch packages if skip_pytorch is True
        pytorch_packages = {"torch", "torchvision", "torchaudio"}
        
        for package_name, import_name in dependencies:
            if skip_pytorch and package_name in pytorch_packages:
                self.logger.debug(f"Skipping {package_name} (will be handled by PyTorch nightly installer)")
                continue
            
            self.install_package(package_name, import_name)
        
        # Report results
        if self.already_installed:
            self.logger.info(f"Already installed: {', '.join(self.already_installed)}")
        
        if self.successful_installs:
            self.logger.info(f"Successfully installed: {', '.join(self.successful_installs)}")
        
        if self.failed_installs:
            self.logger.error(f"Failed to install: {', '.join(self.failed_installs)}")
            self.logger.error("Please install them manually and try again")
            return False
        
        self.logger.info("All dependencies are available")
        return True
    
    def _unzip_file(self, zip_path: str, extract_to: str) -> bool:
        """
        Unzip a file to the specified directory
        
        Args:
            zip_path (str): Path to the zip file
            extract_to (str): Directory to extract to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info(f"Extracting {zip_path} to {extract_to}...")
            
            # Create extraction directory if it doesn't exist
            os.makedirs(extract_to, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            
            self.logger.info("Extraction completed successfully!")
            return True
            
        except FileNotFoundError:
            self.logger.error(f"Error: Zip file not found at {zip_path}")
            return False
        except zipfile.BadZipFile:
            self.logger.error(f"Error: Invalid zip file at {zip_path}")
            return False
        except Exception as e:
            self.logger.error(f"Error during extraction: {str(e)}")
            return False
    
    def _add_to_system_path(self, new_path: str) -> bool:
        """
        Add a directory to the system PATH environment variable (Windows)
        
        Args:
            new_path (str): Path to add to system PATH
            
        Returns:
            bool: True if successful, False otherwise
        """
        import winreg
        try:
            # Check if path exists
            if not os.path.exists(new_path):
                self.logger.warning(f"Path {new_path} does not exist!")
                return False
            
            # Open the system environment variables registry key
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                              r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
                              0, winreg.KEY_ALL_ACCESS) as key:
                
                # Get current PATH value
                try:
                    current_path, _ = winreg.QueryValueEx(key, "PATH")
                except FileNotFoundError:
                    current_path = ""
                
                # Check if path is already in PATH
                if new_path.lower() in current_path.lower():
                    self.logger.info(f"Path {new_path} is already in system PATH")
                    return True
                
                # Add new path to PATH
                if current_path and not current_path.endswith(';'):
                    new_system_path = f"{current_path};{new_path}"
                else:
                    new_system_path = f"{current_path}{new_path}"
                
                # Set the new PATH value
                winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, new_system_path)
                
                self.logger.info(f"Successfully added {new_path} to system PATH")
                return True
                
        except PermissionError:
            self.logger.warning("Administrator privileges required to modify system PATH")
            return False
        except Exception as e:
            self.logger.error(f"Error adding to system PATH: {str(e)}")
            return False
    
    def _add_to_user_path(self, new_path: str) -> bool:
        """
        Add a directory to the user PATH environment variable (Windows)
        
        Args:
            new_path (str): Path to add to user PATH
            
        Returns:
            bool: True if successful, False otherwise
        """
        import winreg
        try:
            # Check if path exists
            if not os.path.exists(new_path):
                self.logger.warning(f"Path {new_path} does not exist!")
                return False
            
            # Open the user environment variables registry key
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                              r"Environment",
                              0, winreg.KEY_ALL_ACCESS) as key:
                
                # Get current PATH value
                try:
                    current_path, _ = winreg.QueryValueEx(key, "PATH")
                except FileNotFoundError:
                    current_path = ""
                
                # Check if path is already in PATH
                if new_path.lower() in current_path.lower():
                    self.logger.info(f"Path {new_path} is already in user PATH")
                    return True
                
                # Add new path to PATH
                if current_path and not current_path.endswith(';'):
                    new_user_path = f"{current_path};{new_path}"
                else:
                    new_user_path = f"{current_path}{new_path}"
                
                # Set the new PATH value
                winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, new_user_path)
                
                self.logger.info(f"Successfully added {new_path} to user PATH")
                return True
                
        except Exception as e:
            self.logger.error(f"Error adding to user PATH: {str(e)}")
            return False
    
    def setup_windows_libvips(self) -> bool:
        """
        Setup libvips on Windows by checking, extracting if needed, and adding to PATH.
        Looks for libvips in ../mono-workspace-repo/dependencies/libvips/ relative to current directory.
        
        Returns:
            bool: True if libvips is ready to use, False otherwise
        """
        if os.name != 'nt':  # Only run on Windows
            self.logger.debug("libvips setup is Windows-only")
            return True
        
        """WINDOWS ONLY: Fix libvips issues by adding to PATH
        FIRST: download libvisp here: https://github.com/libvips/build-win64-mxe/releases/download/v8.17.2/vips-dev-w64-web-8.17.2.zip
        Extract to a folder, then update the path below to point to the 'bin' folder
        """

        # Get current directory and construct paths
        current_dir = os.getcwd()
        workspace_dir = os.path.join(current_dir, "dependencies")
        libvips_base = os.path.join(workspace_dir, "libvips")
        bin_path = os.path.join(libvips_base, "vips-dev-8.17", "bin")
        
        self.logger.info("Setting up libvips for Windows...")
        self.logger.debug(f"Looking for libvips at: {bin_path}")
        
        # Check if libvips bin directory exists
        if os.path.exists(bin_path):
            self.logger.info("LibVIPS found, checking PATH...")
            
            # Add to PATH if not already there
            if not self._add_to_system_path(bin_path):
                self.logger.info("Falling back to user PATH...")
                if self._add_to_user_path(bin_path):
                    self.logger.info("LibVIPS added to user PATH. Restart command prompt for changes to take effect.")
                    return True
                else:
                    self.logger.error("Failed to add libvips to PATH")
                    return False
            else:
                self.logger.info("LibVIPS is ready to use")
                return True
        else:
            self.logger.info("LibVIPS not found, looking for zip file to extract...")
            
            # Look for zip file in libvips directory
            if not os.path.exists(libvips_base):
                self.logger.error(f"LibVIPS directory not found: {libvips_base}")
                self.logger.error("Please ensure the libvips zip file is in ../mono-workspace-repo/dependencies/libvips/")
                return False
            
            # Find zip file
            zip_files = [f for f in os.listdir(libvips_base) if f.endswith('.zip') and 'vips' in f.lower()]
            
            if not zip_files:
                self.logger.error("No libvips zip file found in libvips directory")
                self.logger.error(f"Please place the libvips zip file in: {libvips_base}")
                return False
            
            if len(zip_files) > 1:
                self.logger.warning(f"Multiple zip files found, using first one: {zip_files[0]}")
            
            zip_path = os.path.join(libvips_base, zip_files[0])
            self.logger.info(f"Found zip file: {zip_files[0]}")
            
            # Extract the zip file
            if self._unzip_file(zip_path, libvips_base):
                self.logger.info("Extraction successful, checking for bin directory...")
                
                # Check if bin directory now exists
                if os.path.exists(bin_path):
                    self.logger.info("LibVIPS extracted successfully, adding to PATH...")
                    
                    # Add to PATH
                    if not self._add_to_system_path(bin_path):
                        self.logger.info("Falling back to user PATH...")
                        if self._add_to_user_path(bin_path):
                            self.logger.info("LibVIPS setup complete! Restart command prompt for changes to take effect.")
                            return True
                        else:
                            self.logger.error("Failed to add libvips to PATH")
                            return False
                    else:
                        self.logger.info("LibVIPS setup complete!")
                        return True
                else:
                    self.logger.error("LibVIPS bin directory not found after extraction")
                    self.logger.error(f"Expected path: {bin_path}")
                    return False
            else:
                self.logger.error("Failed to extract libvips zip file")
                return False
    
    def get_install_summary(self) -> Dict[str, List[str]]:
        """
        Get a summary of installation results
        
        Returns:
            Dictionary with lists of already_installed, successful_installs, and failed_installs
        """
        summary = {
            'already_installed': self.already_installed.copy(),
            'successful_installs': self.successful_installs.copy(),
            'failed_installs': self.failed_installs.copy(),
            'downloaded_models': self.downloaded_models.copy(),
            'cached_models': self.cached_models.copy(),
            'failed_downloads': self.failed_downloads.copy()
        }
        self.logger.debug(f"Installation summary: {summary}")
        return summary
    
    def run(self, download_models: bool = True) -> bool:
        """
        Run the complete dependency and model setup process
        
        Args:
            download_models: Whether to download models after installing dependencies
            
        Returns:
            True if all setup completed successfully, False otherwise
        """
        # Predefined dependency sets (excluding PyTorch packages)
        GENERAL_DEPENDENCIES = [("numpy", "numpy"),
                                ("transformers", "transformers"),
                                ("hf_xet", "hf_xet")]
        VISION_DEPENDENCIES = [
            ("Pillow", "PIL"),
            ("opencv-python", "cv2"),
            ("accelerate", "accelerate"),
            ("pyvips", "pyvips"),
            ("einops", "einops")
        ]

        AUDIO_DEPENDENCIES = [
            ("sounddevice", "sounddevice"),
            ("soundfile", "soundfile"),
            ("torch-vggish-yamnet", "torch_vggish_yamnet"),
            ("torchaudio", "torchaudio"),
            ("kokoro", "kokoro")
        ]

        #TODO: Add more dependency sets as needed
        """
        WEB_DEPENDENCIES = [
            ("requests", "requests"),
            ("flask", "flask"),
            ("fastapi", "fastapi"),
            ("uvicorn", "uvicorn")
        ]"""
        
        # Setup Windows libvips automatically
        libvips_ready = self.setup_windows_libvips()
        if libvips_ready:
            self.logger.info("LibVIPS is ready!")
        else:
            self.logger.error("LibVIPS setup failed")

        # Install PyTorch nightly build first
        pytorch_success = self.install_pytorch()
        if not pytorch_success:
            #nightly builds is because of Nvidia driver issues with CUDA 12.8 blackwell architecture on 5000 series cards
            self.logger.error("PyTorch nightly installation failed")
            return False

        # Install other dependencies (skip PyTorch packages since they're handled above)
        reqs = GENERAL_DEPENDENCIES + VISION_DEPENDENCIES + AUDIO_DEPENDENCIES
        other_success = self.install_dependencies(reqs, skip_pytorch=True)

        # Download models if requested
        models_success = True
        if download_models:
            self.logger.info("Downloading required models...")
            models_success = self.download_predefined_models()

        if pytorch_success and other_success and models_success:
            self.logger.info("All dependencies and models ready! You can now import and use them.")
            
            # Verify PyTorch installation and show version info
            try:
                import torch
                self.logger.info(f"PyTorch version: {torch.__version__}")
                self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    self.logger.info(f"CUDA version: {torch.version.cuda}")
                    self.logger.info(f"GPU device count: {torch.cuda.device_count()}")
            except Exception as e:
                self.logger.warning(f"Could not verify PyTorch installation: {e}")
            
            # Show model status
            if download_models:
                if self.downloaded_models:
                    self.logger.info(f"Downloaded models: {', '.join(self.downloaded_models)}")
                if self.cached_models:
                    self.logger.info(f"Already cached models: {', '.join(self.cached_models)}")
                if self.failed_downloads:
                    self.logger.warning(f"Failed model downloads: {', '.join(self.failed_downloads)}")
            
            return True
        else:
            self.logger.error("Some dependencies or models failed to install.")
            summary = self.get_install_summary()
            if summary['failed_installs']:
                self.logger.error(f"Failed packages: {summary['failed_installs']}")
            if summary['failed_downloads']:
                self.logger.error(f"Failed model downloads: {summary['failed_downloads']}")
            return False


# Example usage and testing functions
def test_model_downloads():
    """Test the model download functionality"""
    print("Testing model download functionality...")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    dep_manager = Orenda_DependencyManager()
    
    # Test model cache checking
    print("\n2. Testing model cache checking:")
    is_cached = dep_manager.is_model_cached("vikhyatk/moondream2", "2025-01-09")
    print(f"Moondream2 is cached: {is_cached}")
    
    # Test model info
    print("\n3. Testing model info:")
    info = dep_manager.get_model_info("vikhyatk/moondream2")
    print(f"Model info: {info}")
    
    # Test listing cached models
    print("\n4. Testing cached models list:")
    cached = dep_manager.list_cached_models()
    print(f"Found {len(cached)} cached models:")
    for model in cached:
        print(f"  - {model['model_name']}: {model['cache_size_mb']} MB")


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example 1: Full setup with model downloads
    print("=== Example 1: Full setup with model downloads ===")
    dep_manager = Orenda_DependencyManager()
    if dep_manager.run(download_models=True):
        print("All dependencies and models are ready to use!")
    else:
        print("There were issues installing some dependencies or downloading models.")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Just dependencies without models
    print("=== Example 2: Dependencies only ===")
    dep_manager2 = Orenda_DependencyManager()
    if dep_manager2.run(download_models=False):
        print("All dependencies are ready! Models can be downloaded later.")
    else:
        print("There were issues installing some dependencies.")
    
    print("\n" + "="*50 + "\n")
    

    # Example 4: Test model functionality (if dependencies are ready)
    print("=== Example 4: Testing model functionality ===")
    try:
        # This would be the actual usage in your moondream_.py
        from transformers import AutoModelForCausalLM
        import torch
        
        print("Testing if Moondream2 can be loaded from cache...")
        
        # Try to load from cache (local_files_only=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2",
                revision="2025-01-09",
                trust_remote_code=True,
                dtype=torch.float16,
                local_files_only=True  # Only use cached files
            )
            print("✅ Moondream2 loaded successfully from cache!")
            del model  # Free memory
        except Exception as e:
            print(f"❌ Could not load from cache: {e}")
            print("Model may need to be downloaded first.")
    
    except ImportError as e:
        print(f"Dependencies not available for testing: {e}")
    
    # Uncomment to run model download tests
    # test_model_downloads()