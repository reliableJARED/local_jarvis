#!/usr/bin/env python3
"""
Kokoro-82M Text-to-Speech - Refactored Version
Separated into a clean Kokoro class and dependency management.

https://github.com/hexgrad/kokoro
"""

import subprocess
import sys
import os
import platform
import tempfile
from pathlib import Path
from kokoro import KPipeline
import soundfile as sf
import torch


class Kokoro:
    """
    A interface for Kokoro-82M Text-to-Speech synthesis.
    """
    
    # Voice definitions
    VOICES_FEMALE = [
        "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", 
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"
    ]
    
    VOICES_MALE = [
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", 
        "am_michael", "am_onyx", "am_puck", "am_santa"
    ]
    
    def __init__(self, lang_code='a'):
        """
        Initialize the Kokoro TTS engine.
        
        Args:
            lang_code (str): 'a' for American English, 'b' for British English
        """
        self.lang_code = lang_code
        self.pipeline = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the Kokoro pipeline with MPS fallback for Mac."""
        try:
            # Set MPS fallback for Mac M1/M2/M3/M4
            if platform.system() == "Darwin" and torch.backends.mps.is_available():
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            
            self.pipeline = KPipeline(self.lang_code)
            print("✅ Kokoro pipeline initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Kokoro pipeline: {e}")
            raise
    
    def get_available_voices(self):
        """
        Get lists of available voices.
        
        Returns:
            dict: Dictionary with 'female' and 'male' voice lists
        """
        return {
            'female': self.VOICES_FEMALE,
            'male': self.VOICES_MALE,
            'all': self.VOICES_FEMALE + self.VOICES_MALE
        }
    
    def print_voices(self):
        """Print available voices in a formatted way."""
        print("Female voices:")
        for voice in self.VOICES_FEMALE:
            print(f"   • {voice}")
        
        print("\nMale voices:")
        for voice in self.VOICES_MALE:
            print(f"   • {voice}")
    
    def generate_speech(self, text, voice="af_sky", speed=1.0, output_file=None):
        """
        Generate speech from text.
        
        Args:
            text (str): Text to convert to speech
            voice (str): Voice to use for generation
            speed (float): Speech speed multiplier
            output_file (str): If provided, save to this file. Otherwise use temporary file.
            
        Returns:
            tuple: (success: bool, file_path: str, cleanup_needed: bool)
        """
        if not self.pipeline:
            print("❌ Pipeline not initialized")
            return False, None, False
        
        try:
            print(f"🗣️  Generating speech for: '{text}'")
            print(f"🎵 Using voice: {voice}")
            print(f"⚡ Speed: {speed}x")
            
            # Determine output file path
            cleanup_needed = False
            if output_file:
                file_path = output_file
                print(f"💾 Will save audio to: {file_path}")
            else:
                # Create temporary file
                temp_fd, file_path = tempfile.mkstemp(suffix='.wav', prefix='kokoro_temp_')
                os.close(temp_fd)  # Close file descriptor
                cleanup_needed = True
                print(f"🗂️  Using temporary file: {file_path}")
            
            # Generate audio
            generator = self.pipeline(text, voice=voice, speed=speed)
            
            # Process and save audio
            for i, (graphemes, phonemes, audio) in enumerate(generator):
                print(f"📝 Graphemes: {graphemes}")
                print(f"🔤 Phonemes: {phonemes}")
                
                # Save to file using soundfile
                sf.write(file_path, audio, 24000)
                
                if output_file:
                    print(f"💾 Audio saved to: {file_path}")
                else:
                    print(f"🎵 Audio generated to temporary file")
                
                break  # Only process the first chunk
            
            return True, file_path, cleanup_needed
            
        except Exception as e:
            print(f"❌ Error generating speech: {e}")
            return False, None, False
    
    def play_audio(self, file_path):
        """
        Play audio file using sounddevice.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            bool: True if played successfully, False otherwise
        """
        try:
            import sounddevice as sd
            import soundfile as sf
            
            # Read the audio file
            data, sample_rate = sf.read(file_path)
            
            print(f"🔊 Playing audio: {file_path}")
            print(f"   Sample rate: {sample_rate} Hz")
            print(f"   Duration: {len(data)/sample_rate:.2f} seconds")
            
            # Play the audio
            sd.play(data, sample_rate)
            
            # Wait for playback to finish
            sd.wait()
            
            print("✅ Audio playback completed")
            return True
            
        except ImportError as e:
            print(f"❌ Could not import audio libraries: {e}")
            print("   Please ensure sounddevice and soundfile are installed")
            return False
        except Exception as e:
            print(f"❌ Could not play audio: {e}")
            return False
    
    def speak(self, text, voice="af_sky", speed=1.0, save_to_file=None, play_audio=True):
        """
        Convenience method to generate speech and optionally play it.
        
        Args:
            text (str): Text to convert to speech
            voice (str): Voice to use
            speed (float): Speech speed multiplier
            save_to_file (str): If provided, save audio to this file
            play_audio (bool): Whether to play the audio after generation
            
        Returns:
            bool: True if successful, False otherwise
        """
        success, file_path, cleanup_needed = self.generate_speech(
            text, voice=voice, speed=speed, output_file=save_to_file
        )
        
        if not success:
            return False
        
        played_successfully = True
        if play_audio:
            played_successfully = self.play_audio(file_path)
        
        # Clean up temporary file if needed
        if cleanup_needed:
            try:
                os.unlink(file_path)
                print(f"🗑️  Temporary file cleaned up")
            except OSError as e:
                print(f"⚠️  Could not clean up temporary file: {e}")
        
        if not played_successfully and not save_to_file:
            print(f"⚠️  Audio could not be played and temporary file was cleaned up.")
        elif not played_successfully and save_to_file:
            print(f"⚠️  Audio saved to {save_to_file} but could not be played automatically.")
            print(f"   You can manually play the file: {os.path.abspath(save_to_file)}")
        
        return success


class KokoroDependencyManager:
    """
    Handles dependency installation and system setup for Kokoro TTS.
    """
    
    @staticmethod
    def install_package(package):
        """Install a Python package using pip."""
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Successfully installed {package}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    @staticmethod
    def install_espeak():
        """Install espeak-ng based on the operating system."""
        system = platform.system().lower()
        
        if system == "linux":
            try:
                # Try apt-get first (Ubuntu/Debian)
                subprocess.check_call(["sudo", "apt-get", "update"], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.check_call(["sudo", "apt-get", "install", "-y", "espeak-ng"], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("✅ Successfully installed espeak-ng via apt-get")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                try:
                    # Try yum (CentOS/RHEL/Fedora)
                    subprocess.check_call(["sudo", "yum", "install", "-y", "espeak-ng"], 
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print("✅ Successfully installed espeak-ng via yum")
                    return True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    print("❌ Could not install espeak-ng. Please install it manually:")
                    print("   Ubuntu/Debian: sudo apt-get install espeak-ng")
                    print("   CentOS/RHEL/Fedora: sudo yum install espeak-ng")
                    return False
        
        elif system == "darwin":  # macOS
            try:
                subprocess.check_call(["brew", "install", "espeak-ng"], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("✅ Successfully installed espeak-ng via Homebrew")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("❌ Could not install espeak-ng. Please install Homebrew first, then run:")
                print("   brew install espeak-ng")
                return False
        
        elif system == "windows":
            print("⚠️  On Windows, please manually install espeak-ng:")
            print("   1. Go to: https://github.com/espeak-ng/espeak-ng/releases")
            print("   2. Download the latest .msi file (e.g., espeak-ng-x64.msi)")
            print("   3. Run the installer")
            print("   4. Restart this script after installation")
            return False
        
        else:
            print(f"❌ Unsupported operating system: {system}")
            return False
    
    @staticmethod
    def check_dependencies():
        """Check and install required dependencies."""
        print("🔍 Checking dependencies...")
        
        # Check if espeak-ng is available
        try:
            result = subprocess.run(["espeak-ng", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ espeak-ng is already installed")
            else:
                raise FileNotFoundError
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("📦 Installing espeak-ng...")
            if not KokoroDependencyManager.install_espeak():
                return False
        
        # List of required packages
        packages = [
            "kokoro>=0.9.4",
            "soundfile",
            "sounddevice",  # For audio playback
            "torch",
            "numpy"
        ]
        
        # Install main packages
        for package in packages:
            if not KokoroDependencyManager.install_package(package):
                return False
        
        return True


def main():
    """Main function to run the TTS demo and handle dependencies."""
    print("🎤 Kokoro-82M Text-to-Speech Demo")
    print("=" * 40)
    
    # Check and install dependencies
    if not KokoroDependencyManager.check_dependencies():
        print("❌ Failed to install dependencies. Exiting.")
        return 1
    
    print("\n🚀 Starting speech generation...")
    
    try:
        # Initialize Kokoro TTS
        tts = Kokoro(lang_code='a')  # American English
        
        # Show available voices
        tts.print_voices()
        
        # Test text
        text = "this is a test, it is only a test..... the real question is will you pass the test? We will find out soon"
        
        # Generate and play speech
        if tts.speak(text, voice="af_sky", speed=1.0, play_audio=True):
            print("\n✅ Demo completed successfully!")
            print("🎉 Your text has been converted to speech and played!")
        else:
            print("\n❌ Demo failed. Please check the error messages above.")
            return 1
        
        print(f"\n💡 Usage examples:")
        print(f"   • Basic usage: tts.speak('Hello world')")
        print(f"   • Save to file: tts.speak('Hello', save_to_file='output.wav')")
        print(f"   • Change voice: tts.speak('Hello', voice='am_adam')")
        print(f"   • Change speed: tts.speak('Hello', speed=1.5)")
        
    except Exception as e:
        print(f"❌ Error initializing TTS: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())