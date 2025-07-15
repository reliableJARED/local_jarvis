
#used from requirements.txt
#torch
#transformers
#sounddevice
#numpy

import subprocess
import sys
from kokoro_ import KokoroDependencyManager

def install_dependencies():
    """Install required packages"""
    packages = ["torch", "transformers", "sounddevice", "numpy"]
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
            return False
        
    # Check and install dependencies for kokoro
    if not KokoroDependencyManager.check_dependencies():
        print("❌ Failed to install dependencies. Exiting.")
        return 1
    
    return True

def verify():
    """Verify installation"""
    try:
        import torch
        import transformers
        import sounddevice
        import numpy
        print("✓ All packages installed successfully!")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

if __name__ == "__main__":
    if install_dependencies() and verify():
        print("\nReady to use! Run: python local_jarvis.py")
    else:
        print("\nInstallation failed. Please check errors above.")

