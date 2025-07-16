#!/usr/bin/env python3
"""
Kokoro-82M Text-to-Speech - Thread-Safe Audio Version
Modified to prevent conflicts with microphone access in other threads.
"""

import subprocess
import sys
import os
import platform
import tempfile
import threading
import time
from pathlib import Path
from kokoro import KPipeline
import soundfile as sf
import torch
from typing import Optional, Callable, Tuple
from dataclasses import dataclass
import sounddevice as sd


@dataclass
class AudioData:
    """Container for audio data and metadata."""
    data: any
    sample_rate: int
    file_path: str
    cleanup_needed: bool = False


class ThreadSafeAudioManager:
    """
    Manages audio devices to prevent conflicts between playback and recording.
    """
    
    def __init__(self):
        self._output_stream = None
        self._stream_lock = threading.Lock()
        self._find_audio_devices()
    
    def _find_audio_devices(self):
        """Find separate input and output devices if available."""
        devices = sd.query_devices()
        
        # Find default devices
        self.default_input = sd.default.device[0]   # Input device
        self.default_output = sd.default.device[1]  # Output device
        
        print(f"🎵 Default input device: {self.default_input}")
        print(f"🔊 Default output device: {self.default_output}")
        
        # Store device info for reference
        if self.default_input is not None:
            input_info = sd.query_devices(self.default_input)
            print(f"🎤 Input: {input_info['name']} (channels: {input_info['max_input_channels']})")
        
        if self.default_output is not None:
            output_info = sd.query_devices(self.default_output)
            print(f"🔊 Output: {output_info['name']} (channels: {output_info['max_output_channels']})")
    
    def play_audio(self, audio_data, sample_rate, stop_event, pause_event):
        """
        Play audio using only the output device with proper stream management.
        """
        with self._stream_lock:
            try:
                # Ensure audio data is in the correct format
                import numpy as np
                
                # Convert to numpy array if not already
                if not isinstance(audio_data, np.ndarray):
                    audio_data = np.array(audio_data)
                
                # Ensure audio is float32 format (required by sounddevice)
                if audio_data.dtype != np.float32:
                    # Normalize if it's integer format
                    if np.issubdtype(audio_data.dtype, np.integer):
                        if audio_data.dtype == np.int16:
                            audio_data = audio_data.astype(np.float32) / 32768.0
                        elif audio_data.dtype == np.int32:
                            audio_data = audio_data.astype(np.float32) / 2147483648.0
                        else:
                            audio_data = audio_data.astype(np.float32)
                    else:
                        audio_data = audio_data.astype(np.float32)
                
                # Ensure audio is in correct range [-1.0, 1.0]
                if np.max(np.abs(audio_data)) > 1.0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                # Ensure correct shape (mono or stereo)
                if len(audio_data.shape) == 1:
                    channels = 1
                    audio_shape = (len(audio_data), 1)
                    audio_data = audio_data.reshape(audio_shape)
                else:
                    channels = audio_data.shape[1]
                
                print(f"🎵 Audio format: shape={audio_data.shape}, dtype={audio_data.dtype}, range=[{np.min(audio_data):.3f}, {np.max(audio_data):.3f}]")
                
                # Create output-only stream with explicit format
                self._output_stream = sd.OutputStream(
                    samplerate=sample_rate,
                    channels=channels,
                    device=self.default_output,  # Explicitly use output device
                    dtype=np.float32  # Explicitly set to float32
                )
                
                self._output_stream.start()
                
                # Calculate chunk size for streaming
                chunk_size = int(sample_rate * 0.1)  # 100ms chunks
                total_samples = len(audio_data)
                current_pos = 0
                
                print(f"🔊 Starting stream playback (chunks of {chunk_size} samples)")
                
                while current_pos < total_samples and not stop_event.is_set():
                    # Handle pause
                    if pause_event.is_set():
                        print("⏸️  Playback paused")
                        while pause_event.is_set() and not stop_event.is_set():
                            time.sleep(0.1)
                        if stop_event.is_set():
                            break
                        print("▶️  Playback resumed")
                    
                    # Get next chunk
                    end_pos = min(current_pos + chunk_size, total_samples)
                    chunk = audio_data[current_pos:end_pos]
                    
                    # Write chunk to stream
                    self._output_stream.write(chunk)
                    current_pos = end_pos
                    
                    # Small delay to prevent busy waiting
                    time.sleep(0.01)
                
                # Wait for buffer to empty
                if not stop_event.is_set():
                    self._output_stream.stop()
                    print("✅ Stream playback completed")
                else:
                    print("⏹️  Stream playback stopped")
                
            except Exception as e:
                print(f"❌ Error in stream playback: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Clean up stream
                if self._output_stream:
                    try:
                        self._output_stream.close()
                    except:
                        pass
                    self._output_stream = None
    
    def stop_playback(self):
        """Stop any active playback."""
        with self._stream_lock:
            if self._output_stream:
                try:
                    self._output_stream.stop()
                    self._output_stream.close()
                except:
                    pass
                finally:
                    self._output_stream = None


class Kokoro:
    """
    A thread-safe interface for Kokoro-82M Text-to-Speech synthesis.
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
    
    def __init__(self, lang_code='a', voice ="af_sky", save_audio_as_wav=False, play_audio_immediately=True):
        """
        Initialize the Kokoro TTS engine with thread-safe audio management.
        
        Args:
            lang_code (str): 'a' for American English, 'b' for British English
            save_audio_as_wav (bool): If True, save audio as WAV files. If False, use temporary files.
            play_audio_immediately (bool): If True, play audio as soon as it's ready.
        """
        self.lang_code = lang_code
        self.voice = voice
        self.save_audio_as_wav = save_audio_as_wav
        self.play_audio_immediately = play_audio_immediately
        self.pipeline = None
        self.queued_speech_audio: Optional[AudioData] = None
        
        # Thread-safe audio manager
        self.audio_manager = ThreadSafeAudioManager()
        
        # Threading controls
        self._generation_thread: Optional[threading.Thread] = None
        self._playback_thread: Optional[threading.Thread] = None
        self._stop_playback = threading.Event()
        self._pause_playback = threading.Event()
        self._playback_lock = threading.Lock()
        
        # Garbage collection tracking
        self._temp_files_to_cleanup = []
        self._audio_data_history = []
        
        # Callbacks
        self.speech_audio_ready_callback: Optional[Callable[[AudioData], None]] = None
        self.speech_audio_playback_complete_callback: Optional[Callable[[], None]] = None
        
        self._initialize_pipeline()
    
    def print_voices(self):
        """Print available voices in a formatted way."""
        print("Female voices:")
        for voice in self.VOICES_FEMALE:
            print(f"   • {voice}")
        
        print("\nMale voices:")
        for voice in self.VOICES_MALE:
            print(f"   • {voice}")

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
    
    def _cleanup_previous_audio_data(self):
        """Clean up previous audio data and temporary files."""
        if self.queued_speech_audio and self.queued_speech_audio.cleanup_needed:
            try:
                if os.path.exists(self.queued_speech_audio.file_path):
                    os.unlink(self.queued_speech_audio.file_path)
                    print(f"🗑️  Cleaned up previous temp file: {self.queued_speech_audio.file_path}")
            except OSError as e:
                print(f"⚠️  Could not clean up previous temp file: {e}")
        
        # Clean up any orphaned temp files
        for temp_file in self._temp_files_to_cleanup[:]:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    print(f"🗑️  Cleaned up orphaned temp file: {temp_file}")
                self._temp_files_to_cleanup.remove(temp_file)
            except OSError as e:
                print(f"⚠️  Could not clean up orphaned temp file: {e}")
        
        # Clear audio data history to free memory
        self._audio_data_history.clear()
    
    def _force_stop_playback_thread(self):
        """Force stop the playback thread with proper cleanup."""
        if self._playback_thread and self._playback_thread.is_alive():
            self._stop_playback.set()
            
            # Stop audio manager playback
            self.audio_manager.stop_playback()
            
            # Give it time to cleanup gracefully
            self._playback_thread.join(timeout=3.0)
            
            if self._playback_thread.is_alive():
                print("⚠️  Playback thread didn't terminate gracefully")
            
            self._playback_thread = None
    
    def set_speech_audio_ready_callback(self, callback: Callable[[AudioData], None]):
        """Set callback for when speech audio is ready."""
        self.speech_audio_ready_callback = callback
    
    def set_speech_audio_playback_complete_callback(self, callback: Callable[[], None]):
        """Set callback for when speech audio playback completes."""
        self.speech_audio_playback_complete_callback = callback
    
    def _generate_speech_thread(self, text: str, voice: str, speed: float, output_file: Optional[str] = None):
        """
        Thread function for generating speech.
        
        Args:
            text (str): Text to convert to speech
            voice (str): Voice to use for generation
            speed (float): Speech speed multiplier
            output_file (str): If provided, save to this file
        """
        try:
            print(f"🗣️  Generating speech for: '{text}'")
            print(f"🎵 Using voice: {voice}")
            print(f"⚡ Speed: {speed}x")
            
            # Clean up previous audio data before generating new
            self._cleanup_previous_audio_data()
            
            # Determine output file path
            cleanup_needed = False
            if output_file:
                file_path = output_file
                print(f"💾 Will save audio to: {file_path}")
            elif self.save_audio_as_wav:
                # Create WAV file with timestamp
                timestamp = int(time.time())
                file_path = f"kokoro_speech_{timestamp}.wav"
                print(f"💾 Will save audio to: {file_path}")
            else:
                # Create temporary file
                temp_fd, file_path = tempfile.mkstemp(suffix='.wav', prefix='kokoro_temp_')
                os.close(temp_fd)  # Close file descriptor
                cleanup_needed = True
                self._temp_files_to_cleanup.append(file_path)  # Track for cleanup
                print(f"🗂️  Using temporary file: {file_path}")
            
            # Generate audio
            generator = self.pipeline(text, voice=voice, speed=speed)
            
            # Process and save audio
            for i, (graphemes, phonemes, audio) in enumerate(generator):
                print(f"📝 Graphemes: {graphemes}")
                print(f"🔤 Phonemes: {phonemes}")
                
                # Save to file using soundfile
                sf.write(file_path, audio, 24000)
                
                # Read the audio data for playback
                audio_data = AudioData(
                    data=audio,
                    sample_rate=24000,
                    file_path=file_path,
                    cleanup_needed=cleanup_needed
                )
                
                # Set queued audio
                self.queued_speech_audio = audio_data
                self._audio_data_history.append(audio_data)  # Track for memory management
                
                # Call ready callback if set
                if self.speech_audio_ready_callback:
                    self.speech_audio_ready_callback(audio_data)
                
                # Play immediately if configured
                if self.play_audio_immediately:
                    self._start_playback_thread(audio_data)
                
                print(f"✅ Speech generation completed")
                break  # Only process the first chunk
            
        except Exception as e:
            print(f"❌ Error generating speech: {e}")
    
    def _playback_thread_func(self, audio_data: AudioData):
        """
        Thread function for audio playback using the thread-safe audio manager.
        
        Args:
            audio_data (AudioData): Audio data to play
        """
        try:
            print(f"🔊 Starting thread-safe audio playback: {audio_data.file_path}")
            print(f"   Sample rate: {audio_data.sample_rate} Hz")
            print(f"   Duration: {len(audio_data.data)/audio_data.sample_rate:.2f} seconds")
            
            # Reset stop and pause events
            self._stop_playback.clear()
            self._pause_playback.clear()
            
            # Use the thread-safe audio manager for playback
            self.audio_manager.play_audio(
                audio_data.data, 
                audio_data.sample_rate,
                self._stop_playback,
                self._pause_playback
            )
            
            if not self._stop_playback.is_set():
                print("✅ Audio playback completed")
            
        except Exception as e:
            print(f"❌ Error during audio playback: {e}")
        finally:
            # Always cleanup temp file in finally block
            if audio_data.cleanup_needed:
                try:
                    if os.path.exists(audio_data.file_path):
                        os.unlink(audio_data.file_path)
                        print(f"🗑️  Temporary file cleaned up: {audio_data.file_path}")
                    # Remove from cleanup tracking
                    if audio_data.file_path in self._temp_files_to_cleanup:
                        self._temp_files_to_cleanup.remove(audio_data.file_path)
                except OSError as e:
                    print(f"⚠️  Could not clean up temporary file: {e}")
            
            # Call completion callback if set
            if self.speech_audio_playback_complete_callback:
                self.speech_audio_playback_complete_callback()
    
    def _start_playback_thread(self, audio_data: AudioData):
        """Start the playback thread."""
        with self._playback_lock:
            # Force stop any existing playback with proper cleanup
            self._force_stop_playback_thread()
            
            # Start new playback thread
            self._playback_thread = threading.Thread(
                target=self._playback_thread_func,
                args=(audio_data,),
                daemon=True
            )
            self._playback_thread.start()
    
    def generate_speech_async(self, text: str, voice: str = False, speed: float = 1.0, output_file: Optional[str] = None):
        """
        Generate speech asynchronously in a separate thread.
        
        Args:
            text (str): Text to convert to speech
            voice (str): Voice to use for generation
            speed (float): Speech speed multiplier
            output_file (str): If provided, save to this file
        """
        if not voice:
            voice = self.voice
            
        if not self.pipeline:
            print("❌ Pipeline not initialized")
            return
        
        # Stop any existing generation
        if self._generation_thread and self._generation_thread.is_alive():
            print("⚠️  Previous generation still running, starting new one anyway")
        
        # Start generation thread
        self._generation_thread = threading.Thread(
            target=self._generate_speech_thread,
            args=(text, voice, speed, output_file),
            daemon=True
        )
        self._generation_thread.start()
    
    def speak(self, audio_data: Optional[AudioData] = None):
        """
        Play queued speech audio or provided audio data.
        
        Args:
            audio_data (AudioData, optional): Audio data to play. If None, uses queued_speech_audio.
        """
        target_audio = audio_data or self.queued_speech_audio
        
        if not target_audio:
            print("❌ No audio data available to play")
            return
        
        self._start_playback_thread(target_audio)
    
    def stop_playback(self):
        """Stop audio playback immediately."""
        self._force_stop_playback_thread()
    
    def pause_playback(self):
        """Pause audio playback."""
        self._pause_playback.set()
        print("⏸️  Playback pause requested")
    
    def resume_playback(self):
        """Resume audio playback."""
        self._pause_playback.clear()
        print("▶️  Playback resume requested")
    
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._playback_thread and self._playback_thread.is_alive()
    
    def is_generating(self) -> bool:
        """Check if speech generation is in progress."""
        return self._generation_thread and self._generation_thread.is_alive()
    
    def wait_for_generation(self, timeout: Optional[float] = None):
        """Wait for speech generation to complete."""
        if self._generation_thread:
            self._generation_thread.join(timeout=timeout)
    
    def wait_for_playback(self, timeout: Optional[float] = None):
        """Wait for audio playback to complete."""
        if self._playback_thread:
            self._playback_thread.join(timeout=timeout)
    
    def cleanup(self):
        """Clean up threads and resources."""
        print("🧹 Cleaning up Kokoro TTS...")
        
        # Force stop playback with proper cleanup
        self._force_stop_playback_thread()
        
        # Wait for generation thread to complete
        if self._generation_thread and self._generation_thread.is_alive():
            self._generation_thread.join(timeout=5.0)
            if self._generation_thread.is_alive():
                print("⚠️  Generation thread didn't terminate in time")
        
        # Clean up all remaining temporary files
        self._cleanup_previous_audio_data()
        
        # Clear all references
        self.queued_speech_audio = None
        self._audio_data_history.clear()
        self._temp_files_to_cleanup.clear()
        
        print("✅ Cleanup completed")


# Example of how to use microphone in a separate thread
class MicrophoneRecorder:
    """
    Example class showing how to use microphone in a separate thread
    without conflicting with Kokoro TTS playback.
    """
    
    def __init__(self):
        self.audio_manager = ThreadSafeAudioManager()
        self.recording = False
        self.record_thread = None
        self._stop_recording = threading.Event()
    
    def record_audio(self, duration=5):
        """Record audio from microphone."""
        def _record():
            try:
                print(f"🎤 Starting microphone recording for {duration} seconds...")
                
                # Use input device explicitly
                recording = sd.rec(
                    int(duration * 44100),  # samples
                    samplerate=44100,
                    channels=1,
                    device=self.audio_manager.default_input,  # Explicitly use input device
                    dtype='float32'
                )
                
                # Wait for recording to complete or stop event
                for i in range(int(duration * 10)):  # Check every 0.1 seconds
                    if self._stop_recording.is_set():
                        sd.stop()
                        break
                    time.sleep(0.1)
                
                sd.wait()  # Wait for recording to complete
                
                if not self._stop_recording.is_set():
                    print("✅ Recording completed")
                    # Process recorded audio here
                    print(f"📊 Recorded {len(recording)} samples")
                else:
                    print("⏹️  Recording stopped")
                
            except Exception as e:
                print(f"❌ Error recording audio: {e}")
            finally:
                self.recording = False
        
        if not self.recording:
            self.recording = True
            self._stop_recording.clear()
            self.record_thread = threading.Thread(target=_record, daemon=True)
            self.record_thread.start()
    
    def stop_recording(self):
        """Stop recording."""
        self._stop_recording.set()
        if self.record_thread:
            self.record_thread.join(timeout=2.0)




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
    """Main function to run the threaded TTS demo with pause/resume functionality."""
    print("🎤 Kokoro-82M Text-to-Speech Threaded Demo")
    print("=" * 50)
    
    # Check and install dependencies
    if not KokoroDependencyManager.check_dependencies():
        print("❌ Failed to install dependencies. Exiting.")
        return 1
    
    print("\n🚀 Starting threaded speech generation...")
    
    try:
        # Initialize Kokoro TTS with threading
        tts = Kokoro(
            lang_code='a',
            save_audio_as_wav=False,  # Use temporary files
            play_audio_immediately=False  # Don't auto-play for demo control
        )
        
        # Set up callbacks
        def on_audio_ready(audio_data):
            print(f"🎵 Audio ready callback: {audio_data.file_path}")
        
        def on_playback_complete():
            print("🎉 Playback complete callback triggered!")
        
        tts.set_speech_audio_ready_callback(on_audio_ready)
        tts.set_speech_audio_playback_complete_callback(on_playback_complete)
        
        # Show available voices
        tts.print_voices()
        
        # Test text - longer for better pause/resume demo
        text = ("This is a comprehensive threaded test of the Kokoro TTS system. "
                "Audio generation and playback happen in separate threads, allowing for "
                "real-time control of the speech synthesis process. We can pause, resume, "
                "and stop playback at any time during the speech generation.")
        
        # Generate speech asynchronously
        print(f"\n🔄 Starting async generation...")
        tts.generate_speech_async(text, voice="af_sky", speed=1.0)
        
        # Wait for generation to complete
        print("⏳ Waiting for generation to complete...")
        tts.wait_for_generation()
        
        # Demo 1: Normal playback
        print("\n🎬 Demo 1: Normal playback")
        print("▶️  Starting playback...")
        tts.speak()
        
        # Let it play for a bit
        time.sleep(2)
        
        # Demo 2: Pause and resume
        print("\n🎬 Demo 2: Pause and resume functionality")
        print("⏸️  Pausing playback in 1 second...")
        time.sleep(1)
        tts.pause_playback()
        
        print("⏳ Paused for 3 seconds...")
        time.sleep(3)
        
        print("▶️  Resuming playback...")
        tts.resume_playback()
        
        # Let it play for a bit more
        time.sleep(2)
        
        # Demo 3: Stop and restart
        print("\n🎬 Demo 3: Stop and restart playback")
        print("⏹️  Stopping playback...")
        tts.stop_playback()
        
        print("⏳ Stopped for 2 seconds...")
        time.sleep(2)
        
        print("🔄 Restarting playback from beginning...")
        tts.speak()  # This will restart from the beginning
        
        # Let it play for a bit
        time.sleep(3)
        
        # Demo 4: Generate new audio while playing
        print("\n🎬 Demo 4: Generate new audio while current is playing")
        new_text = "This is a new speech sample generated while the previous one was playing."
        
        print("🔄 Generating new audio...")
        tts.generate_speech_async(new_text, voice="am_adam", speed=1.2)
        
        # Wait for new generation
        tts.wait_for_generation()
        
        print("⏹️  Stopping current playback...")
        tts.stop_playback()
        
        print("▶️  Playing new audio...")
        tts.speak()
        
        # Demo 5: Status checking
        print("\n🎬 Demo 5: Status monitoring")
        time.sleep(1)
        
        print(f"📊 Is playing: {tts.is_playing()}")
        print(f"📊 Is generating: {tts.is_generating()}")
        
        # Final demo: Let current playback complete
        print("\n🎬 Final: Letting playback complete naturally...")
        tts.wait_for_playback()
        
        print("\n✅ Threaded demo completed successfully!")
        
        print(f"\n💡 Demo showcased:")
        print(f"   • ✅ Async speech generation")
        print(f"   • ✅ Threaded playback")
        print(f"   • ✅ Pause/resume functionality")
        print(f"   • ✅ Stop/restart playback")
        print(f"   • ✅ Status monitoring")
        print(f"   • ✅ Audio ready callbacks")
        print(f"   • ✅ Playback complete callbacks")
        print(f"   • ✅ Multi-voice support")
        
        # Cleanup
        tts.cleanup()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        print("🧹 Cleaning up...")
        if 'tts' in locals():
            tts.cleanup()
        return 1
    except Exception as e:
        print(f"❌ Error in threaded demo: {e}")
        if 'tts' in locals():
            tts.cleanup()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())