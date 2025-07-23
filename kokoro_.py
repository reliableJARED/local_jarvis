"""
Kokoro-82M Text-to-Speech - Thread-Safe Audio Version with Multi-line Support
Modified to handle newlines by generating separate speech for each line.
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
from typing import Optional, Callable, Tuple, List
from dataclasses import dataclass
import sounddevice as sd
import re

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
        """Find separate input and output devices, prioritizing Bluetooth headphones."""
        devices = sd.query_devices()
        
        # Find default devices as fallback
        self.default_input = sd.default.device[0]   # Input device
        self.default_output = sd.default.device[1]  # Output device
        
        # Look for Bluetooth headphones or better audio devices
        bluetooth_output = None
        headphone_output = None
        
        print("🎵 Available audio devices:")
        for i, device in enumerate(devices):
            device_name = device['name'].lower()
            max_outputs = device['max_output_channels']
            
            print(f"   {i}: {device['name']} (out: {max_outputs}, in: {device['max_input_channels']})")
            
            # Skip devices with no output channels
            if max_outputs == 0:
                continue
                
            # Prioritize Bluetooth devices
            if any(keyword in device_name for keyword in ['bluetooth', 'airpods', 'headphones', 'headset', 'wireless']):
                if bluetooth_output is None:
                    bluetooth_output = i
                    print(f"   🎧 Found Bluetooth device: {device['name']}")
            
            # Look for headphone-like devices
            elif any(keyword in device_name for keyword in ['headphone', 'headset', 'earphone', 'buds']):
                if headphone_output is None:
                    headphone_output = i
                    print(f"   🎧 Found headphone device: {device['name']}")
        
        # Choose the best available output device
        if bluetooth_output is not None:
            self.default_output = bluetooth_output
            print(f"✅ Using Bluetooth device for output: {devices[bluetooth_output]['name']}")
        elif headphone_output is not None:
            self.default_output = headphone_output
            print(f"✅ Using headphone device for output: {devices[headphone_output]['name']}")
        else:
            print(f"🔊 Using default output device: {devices[self.default_output]['name']}")
        
        print(f"🎵 Final input device: {self.default_input}")
        print(f"🔊 Final output device: {self.default_output}")
        
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
        
        # Multi-line speech processing
        self.generating_multi_line_audio: List[str] = []
        self._multi_line_voice = None
        self._multi_line_speed = 1.0
        self._multi_line_output_file = None
        self._next_line_audio: Optional[AudioData] = None  # Pre-generated next line
        self._next_line_generation_thread: Optional[threading.Thread] = None
        
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
            
            # Only join if we're not the current playback thread (prevent self-join)
            current_thread = threading.current_thread()
            if self._playback_thread != current_thread:
                # Give it time to cleanup gracefully
                self._playback_thread.join(timeout=3.0)
                
                if self._playback_thread.is_alive():
                    print("⚠️  Playback thread didn't terminate gracefully")
            else:
                print("⚠️  Skipping self-join in playback thread")
            
            self._playback_thread = None
    
    def set_speech_audio_ready_callback(self, callback: Callable[[AudioData], None]):
        """Set callback for when speech audio is ready."""
        self.speech_audio_ready_callback = callback
    
    def set_speech_audio_playback_complete_callback(self, callback: Callable[[], None]):
        """Set callback for when speech audio playback completes."""
        self.speech_audio_playback_complete_callback = callback

    def _split_text_by_newlines(self, text: str) -> List[str]:
        """
        Split text by newlines and return list of non-empty lines.
        
        Args:
            text (str): Input text to split
            
        Returns:
            List[str]: List of text lines, empty lines removed
        """
        lines = text.split('\n')
        # Remove empty lines and strip whitespace
        lines = [line.strip() for line in lines if line.strip()]
        return lines

    def _process_next_multiline_chunk(self):
        """Process the next chunk of multi-line text."""
        if not self.generating_multi_line_audio:
            return
        
        # Get the next line
        next_line = self.generating_multi_line_audio.pop(0)
        
        print(f"📝 Processing multi-line chunk ({len(self.generating_multi_line_audio)} remaining): {next_line[:50]}...")
        
        # Generate speech for this line
        self._generate_single_line_speech(
            next_line, 
            self._multi_line_voice, 
            self._multi_line_speed, 
            self._multi_line_output_file
        )
        
        # Start generating the next line in background if there are more lines
        self._start_next_line_generation()
    
    def _start_next_line_generation(self):
        """Start generating the next line in background while current line plays."""
        if not self.generating_multi_line_audio:
            return
        
        # Don't start if already generating next line
        if self._next_line_generation_thread and self._next_line_generation_thread.is_alive():
            return
        
        # Get the next line without removing it from queue (peek)
        next_line = self.generating_multi_line_audio[0]
        
        print(f"🔄 Pre-generating next line in background: {next_line[:30]}...")
        
        # Start background generation thread
        self._next_line_generation_thread = threading.Thread(
            target=self._generate_next_line_background,
            args=(next_line,),
            daemon=True
        )
        self._next_line_generation_thread.start()
    
    def _generate_next_line_background(self, text: str):
        """Generate the next line in background without triggering callbacks or playback."""
        try:
            print(f"🎵 Background generation for: '{text[:50]}...'")
            
            # Determine output file path for background generation
            cleanup_needed = False
            if self._multi_line_output_file:
                # For multi-line, append timestamp to avoid overwriting
                timestamp = int(time.time() * 1000)  # millisecond precision
                base, ext = os.path.splitext(self._multi_line_output_file)
                file_path = f"{base}_bg_{timestamp}{ext}"
                print(f"💾 Background audio will save to: {file_path}")
            elif self.save_audio_as_wav:
                # Create WAV file with timestamp
                timestamp = int(time.time() * 1000)  # millisecond precision
                file_path = f"kokoro_speech_bg_{timestamp}.wav"
                print(f"💾 Background audio will save to: {file_path}")
            else:
                # Create temporary file
                temp_fd, file_path = tempfile.mkstemp(suffix='.wav', prefix='kokoro_bg_temp_')
                os.close(temp_fd)  # Close file descriptor
                cleanup_needed = True
                self._temp_files_to_cleanup.append(file_path)  # Track for cleanup
                print(f"🗂️  Background generation using temporary file: {file_path}")
            
            # Generate audio
            generator = self.pipeline(text, voice=self._multi_line_voice, speed=self._multi_line_speed)
            
            # Process and save audio
            for i, (graphemes, phonemes, audio) in enumerate(generator):
                print(f"📝 Background - Graphemes: {graphemes}")
                print(f"🔤 Background - Phonemes: {phonemes}")
                
                # Save to file using soundfile
                sf.write(file_path, audio, 24000)
                
                # Store the pre-generated audio data
                self._next_line_audio = AudioData(
                    data=audio,
                    sample_rate=24000,
                    file_path=file_path,
                    cleanup_needed=cleanup_needed
                )
                
                print(f"✅ Background speech generation completed and cached")
                break  # Only process the first chunk
            
        except Exception as e:
            print(f"❌ Error in background speech generation: {e}")
            self._next_line_audio = None
    
    def _generate_single_line_speech(self, text: str, voice: str, speed: float, output_file: Optional[str] = None):
        """
        Generate speech for a single line of text.
        
        Args:
            text (str): Text to convert to speech
            voice (str): Voice to use for generation
            speed (float): Speech speed multiplier
            output_file (str): If provided, save to this file
        """
        try:
            # Clean up previous audio data before generating new
            self._cleanup_previous_audio_data()
            
            # Determine output file path
            cleanup_needed = False
            if output_file:
                # For multi-line, append timestamp to avoid overwriting
                timestamp = int(time.time() * 1000)  # millisecond precision
                base, ext = os.path.splitext(output_file)
                file_path = f"{base}_{timestamp}{ext}"
                print(f"💾 Will save audio to: {file_path}")
            elif self.save_audio_as_wav:
                # Create WAV file with timestamp
                timestamp = int(time.time() * 1000)  # millisecond precision
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
                
                print(f"✅ Speech generation completed for line")
                break  # Only process the first chunk
            
        except Exception as e:
            print(f"❌ Error generating speech for line: {e}")
    
    def _generate_speech_thread(self, text: str, voice: str, speed: float, output_file: Optional[str] = None):
        """
        Thread function for generating speech with multi-line support.
        
        Args:
            text (str): Text to convert to speech
            voice (str): Voice to use for generation
            speed (float): Speech speed multiplier
            output_file (str): If provided, save to this file
        """
        try:
            print(f"Using voice: {voice}")
            print(f"Speed: {speed}x")
            
            # Check if text contains newlines
            lines = self._split_text_by_newlines(text)
            
            if len(lines) > 1:
                print(f"📄 Multi-line text detected: {len(lines)} lines")
                
                # Store multi-line parameters
                self.generating_multi_line_audio = lines
                self._multi_line_voice = voice
                self._multi_line_speed = speed
                self._multi_line_output_file = output_file
                
                # Process the first line
                self._process_next_multiline_chunk()
            else:
                # Single line processing (original behavior)
                single_text = lines[0] if lines else text
                self._generate_single_line_speech(single_text, voice, speed, output_file)
            
        except Exception as e:
            print(f"❌ Error generating speech: {e}")
    
    def _cleanup_multiline_state(self):
        """Clean up multi-line processing state."""
        self.generating_multi_line_audio = []
        self._next_line_audio = None
        
        # Clean up background generation thread
        if self._next_line_generation_thread and self._next_line_generation_thread.is_alive():
            # Give it a moment to finish, but don't wait too long
            self._next_line_generation_thread.join(timeout=1.0)
        self._next_line_generation_thread = None
    
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
            
            # Handle multi-line processing continuation
            if self.generating_multi_line_audio and not self._stop_playback.is_set():
                print(f"🔄 Continuing multi-line processing...")
                
                # Check if we have pre-generated audio ready
                if self._next_line_audio:
                    print("🚀 Using pre-generated next line audio (seamless transition)")
                    
                    # Remove the line from queue since we're using the pre-generated version
                    if self.generating_multi_line_audio:
                        self.generating_multi_line_audio.pop(0)
                    
                    # Use the pre-generated audio
                    next_audio = self._next_line_audio
                    self._next_line_audio = None  # Clear the cache
                    
                    # Set it as the current queued audio
                    self.queued_speech_audio = next_audio
                    self._audio_data_history.append(next_audio)
                    
                    # Start playback immediately in a new thread (don't trigger callback from playback thread)
                    if self.play_audio_immediately:
                        # Schedule playback to start after this thread completes
                        def start_next_playback():
                            time.sleep(0.1)  # Brief delay to let current thread finish
                            # Trigger callback now that we're ready to use it
                            if self.speech_audio_ready_callback:
                                self.speech_audio_ready_callback(next_audio)
                            # Start playback
                            self._start_playback_thread(next_audio)
                            # Start generating the next line in background if there are more
                            self._start_next_line_generation()
                        
                        transition_thread = threading.Thread(target=start_next_playback, daemon=True)
                        transition_thread.start()
                    else:
                        # If not auto-playing, trigger callback and start background generation
                        if self.speech_audio_ready_callback:
                            self.speech_audio_ready_callback(next_audio)
                        self._start_next_line_generation()
                else:
                    print("⏳ No pre-generated audio, falling back to sequential generation")
                    # Fall back to sequential generation
                    self._generation_thread = threading.Thread(
                        target=self._process_next_multiline_chunk,
                        daemon=True
                    )
                    self._generation_thread.start()
            else:
                # Clear multi-line state when done
                if not self.generating_multi_line_audio:
                    print("📄 Multi-line processing completed")
                    self._cleanup_multiline_state()
            
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
    
    def preprocess_text_for_tts(self, text: str) -> str:
        """
        Preprocess text to handle issues that cause Kokoro TTS to stop early.
        Removes apostrophes, emojis, and other problematic characters.
        NOTE: Newlines are now handled by multi-line processing, not removed here.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text ready for TTS
        """
        if not text or not text.strip():
            return ""
        
        print(f"🔍 Original text: {repr(text[:100])}...")
        
        # Remove emojis using comprehensive emoji pattern
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002600-\U000027BF"  # miscellaneous symbols
            "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
            "\U00002700-\U000027BF"  # dingbats
            "\U0001F170-\U0001F251"  # enclosed characters
            "]+", 
            flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)
        

        # Remove any remaining apostrophes and similar characters
        text = text.replace("'", "")
        text = text.replace("'", "")  # curly apostrophe
        text = text.replace("`", "")  # backtick
        
        #Only replace \r, keep \n for multi-line processing
        text = text.replace('\r', '')
        
        #Handle other special characters that might cause issues
        text = text.replace('—', '-')  # em-dash
        text = text.replace('–', '-')  # en-dash
        text = text.replace('’', "'")
        text = text.replace('*', ".")
        text = text.replace('…', '...')  # ellipsis
        text = text.replace('•', '-')   # bullet points
        text = text.replace('#', '.')   # hashtag

        #some conjucation specific
        text = text.replace("you’re", "your")   
        text = text.replace("I’d", "I would")   
        text = text.replace("I'd", "I would")   
        text = text.replace("I’ll", "I will")   
        text = text.replace("I'll", "I will")   
        text = text.replace("you're", "your")
        text = text.replace("I'm","I am" )   
        text = text.replace("I’m","I am" )
        text = text.replace("we'd","we would" )   
        text = text.replace("we’d","we would" )      
        
        #Clean up but preserve newlines
        text = text.strip()
        
        # Don't add ending punctuation if it's multi-line (will be handled per line)
        if '\n' not in text and text and text[-1] not in '.!?':
            text += '.'
        
        print(f"✅ Preprocessed text: {repr(text[:100])}...")
        return text

    def generate_speech_async(self, text: str, voice: str = False, speed: float = 1.0, output_file: Optional[str] = None):
        """
        Generate speech asynchronously in a separate thread with multi-line support.
        
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
        
        # Stop any existing generation and clear multi-line state
        if self._generation_thread and self._generation_thread.is_alive():
            print("⚠️  Previous generation still running, starting new one anyway")
        
        # Clear any existing multi-line processing
        self._cleanup_multiline_state()
        
        # Clean the text of special characters
        text = self.preprocess_text_for_tts(text)
        
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
        """Stop audio playback immediately and clear multi-line processing."""
        self._force_stop_playback_thread()
        # Clear multi-line processing when stopped
        self._cleanup_multiline_state()
        print("⏹️  Multi-line processing cleared")
    
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
    
    def is_multi_line_processing(self) -> bool:
        """Check if multi-line processing is active."""
        return bool(self.generating_multi_line_audio)
    
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
        
        # Clear multi-line processing
        self._cleanup_multiline_state()
        
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
    """Main function to run the multi-line TTS demo."""
    print("🎤 Kokoro-82M Text-to-Speech Multi-line Demo")
    print("=" * 50)
    
    # Check and install dependencies
    if not KokoroDependencyManager.check_dependencies():
        print("❌ Failed to install dependencies. Exiting.")
        return 1
    
    print("\n🚀 Starting multi-line speech generation demo...")
    
    try:
        # Initialize Kokoro TTS with threading
        tts = Kokoro(
            lang_code='a',
            save_audio_as_wav=False,  # Use temporary files
            play_audio_immediately=True  # Auto-play for demo
        )
        
        # Set up callbacks
        def on_audio_ready(audio_data):
            print(f"🎵 Audio ready callback: {audio_data.file_path}")
        
        def on_playback_complete():
            print("🎉 Playback complete callback triggered!")
            if tts.is_multi_line_processing():
                print(f"📄 Multi-line processing continues... ({len(tts.generating_multi_line_audio)} lines remaining)")
        
        tts.set_speech_audio_ready_callback(on_audio_ready)
        tts.set_speech_audio_playback_complete_callback(on_playback_complete)
        
        # Show available voices
        tts.print_voices()
        
        # Demo 1: Single line text (original behavior)
        print(f"\n🎬 Demo 1: Single line text")
        single_text = "This is a single line of text that will be processed normally."
        print(f"📝 Text: {single_text}")
        
        tts.generate_speech_async(single_text, voice="af_heart", speed=1.0)
        tts.wait_for_generation()
        tts.wait_for_playback()
        
        time.sleep(1)  # Brief pause between demos
        
        # Demo 2: Multi-line text with overlapped generation
        print(f"\n🎬 Demo 2: Multi-line text with overlapped generation")
        multi_text = """Sure, here's a light-hearted joke for you:
Why don't scientists trust atoms?
Because they make up everything!"""
        
        print(f"📝 Multi-line text:\n{multi_text}")
        print(f"📄 This will use overlapped generation for seamless transitions")
        
        tts.generate_speech_async(multi_text, voice="am_adam", speed=1.1)
        
        # Monitor the multi-line processing
        print("\n📊 Monitoring overlapped multi-line processing...")
        while tts.is_generating() or tts.is_playing() or tts.is_multi_line_processing():
            status = []
            if tts.is_generating():
                status.append("generating")
            if tts.is_playing():
                status.append("playing")
            if tts.is_multi_line_processing():
                status.append(f"multi-line ({len(tts.generating_multi_line_audio)} remaining)")
            if tts._next_line_audio:
                status.append("next-line-ready")
            
            if status:
                print(f"   Status: {', '.join(status)}")
            time.sleep(0.5)
        
        print("\n✅ Overlapped multi-line demo completed!")
        
        # Demo 3: Long lines to showcase overlapped generation benefit
        print(f"\n🎬 Demo 3: Long lines showcasing overlapped generation")
        long_multi_text = """This is the first line which is intentionally made quite long to demonstrate how the overlapped generation system works by starting to generate the next line while this current line is still playing, which should result in much shorter pauses between lines.
This is the second line which is also intentionally long to continue demonstrating the seamless transition capabilities of the overlapped generation system, where the next line should already be ready by the time this line finishes playing.
This final line completes our demonstration of how overlapped generation creates natural flowing speech with minimal pauses between lines, even when individual lines are quite lengthy and would normally take a long time to generate."""
        
        print(f"📝 Starting long lines demo (should have minimal pauses)...")
        start_time = time.time()
        
        tts.generate_speech_async(long_multi_text, voice="af_nova", speed=1.0)
        
        # Monitor for seamless transitions
        previous_status = ""
        transition_count = 0
        
        while tts.is_generating() or tts.is_playing() or tts.is_multi_line_processing():
            status = []
            if tts.is_generating():
                status.append("generating")
            if tts.is_playing():
                status.append("playing")
            if tts.is_multi_line_processing():
                status.append(f"multi-line ({len(tts.generating_multi_line_audio)} remaining)")
            if tts._next_line_audio:
                status.append("next-ready")
            
            current_status = ', '.join(status)
            if current_status != previous_status:
                if "next-ready" in current_status:
                    transition_count += 1
                    print(f"   🚀 Seamless transition #{transition_count}: {current_status}")
                else:
                    print(f"   Status: {current_status}")
                previous_status = current_status
            time.sleep(0.3)
        
        end_time = time.time()
        print(f"\n✅ Long lines demo completed in {end_time - start_time:.1f} seconds!")
        print(f"📊 Detected {transition_count} seamless transitions")
        
        # Demo 4: Test stopping multi-line processing
        print(f"\n🎬 Demo 3: Stopping multi-line processing")
        long_multi_text = """This is the first line that will start playing.
This is the second line that should be interrupted.
This is the third line that should never play.
This is the fourth line that should also never play."""
        
        print(f"📝 Starting long multi-line text, will stop after first line...")
        tts.generate_speech_async(long_multi_text, voice="af_nova", speed=1.0)
        
        # Let first line play for a bit
        time.sleep(2)
        
        print("⏹️  Stopping multi-line processing...")
        tts.stop_playback()
        
        time.sleep(1)
        
        print(f"📊 Multi-line processing cleared: {not tts.is_multi_line_processing()}")
        
        print("\n💡 Overlapped multi-line demo showcased:")
        print(f"   • ✅ Single line processing (original behavior)")
        print(f"   • ✅ Multi-line text split by newlines")
        print(f"   • ✅ Overlapped generation for seamless transitions")
        print(f"   • ✅ Background pre-generation of next lines")
        print(f"   • ✅ Minimal pauses even with long lines")
        print(f"   • ✅ Proper cleanup when stopped")
        print(f"   • ✅ Status monitoring for overlapped processing")
        
        # Cleanup
        tts.cleanup()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        print("🧹 Cleaning up...")
        if 'tts' in locals():
            tts.cleanup()
        return 1
    except Exception as e:
        print(f"❌ Error in multi-line demo: {e}")
        if 'tts' in locals():
            tts.cleanup()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())