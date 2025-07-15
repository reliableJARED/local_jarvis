#!/usr/bin/env python3
"""
YAAM - Yet Another Assistant Model
A voice-activated AI assistant combining Whisper VAD, Qwen Chat, and Kokoro TTS.
"""

import time
import threading
import queue
from typing import Optional, Callable
import numpy as np
import sounddevice as sd
import torch
from collections import deque

# Import our core classes
from whispervad import WhisperVAD
from qwen_ import QwenChat
from kokoro_ import Kokoro


class YetAnotherAssistantModel:
    """
    Voice-activated AI assistant that listens, thinks, and speaks.
    
    Flow:
    1. Listen for speech using custom VAD loop
    2. When speech detected and silence follows, transcribe
    3. Send transcript to Qwen for processing
    4. Convert Qwen's response to speech using Kokoro
    5. Play the response and return to listening
    """
    
    def __init__(self, 
                 whisper_model: str = "openai/whisper-small",
                 qwen_model: str = "Qwen/Qwen2.5-7B-Instruct",
                 kokoro_voice: str = "af_sky",
                 kokoro_speed: float = 1.0,
                 silence_duration: float = 2.0,
                 vad_threshold: float = 0.5,
                 debug_mode: bool = True):
        """
        Initialize YAAM with all components.
        
        Args:
            whisper_model: Whisper model for speech recognition
            qwen_model: Qwen model for text processing
            kokoro_voice: Kokoro voice for text-to-speech
            kokoro_speed: Speech speed multiplier
            silence_duration: Seconds of silence before processing
            vad_threshold: Voice activity detection threshold
            debug_mode: Enable debug output
        """
        
        self.debug_mode = debug_mode
        self.kokoro_voice = kokoro_voice
        self.kokoro_speed = kokoro_speed
        self.silence_duration = silence_duration
        self.vad_threshold = vad_threshold
        self.sample_rate = 16000
        self.chunk_size = 1024
        
        # State management
        self._is_running = False
        self._is_listening = False
        self._is_processing = False
        self._is_speaking = False
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.audio_buffer = deque(maxlen=int(self.sample_rate * 3))  # 3 second buffer
        self.speech_audio = deque()
        self.is_speaking_detected = False
        self.last_speech_time = time.time()
        
        # Threading
        self._audio_thread = None
        self._processing_thread = None
        self._stream = None
        
        # Initialize components
        self._log("🤖 Initializing YAAM components...")
        
        # Initialize WhisperVAD (we'll use its models but not its loop)
        self._log("🎤 Loading Whisper models...")
        self.whisper_vad = WhisperVAD(
            model_name=whisper_model,
            silence_duration=silence_duration,
            vad_threshold=vad_threshold
        )
        self.whisper_vad.initialize()
        
        # Initialize Qwen Chat
        self._log("🧠 Loading Qwen Chat...")
        self.qwen_chat = QwenChat(model_name=qwen_model)
        
        # Initialize Kokoro TTS
        self._log("🗣️  Loading Kokoro TTS...")
        self.kokoro_tts = Kokoro(lang_code='a')
        
        self._log("✅ YAAM initialized successfully!")
        self._log(f"   🎤 Speech: {whisper_model}")
        self._log(f"   🧠 Brain: {qwen_model}")
        self._log(f"   🗣️  Voice: {kokoro_voice}")
    
    def _log(self, message: str):
        """Log debug messages if debug mode is enabled."""
        if self.debug_mode:
            print(f"[YAAM] {message}")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Handle incoming audio data."""
        if self._is_listening and not self._is_processing and not self._is_speaking:
            audio_data = indata.flatten()
            self.audio_queue.put(audio_data.copy())
    
    def _detect_speech(self, audio_chunk):
        """Detect speech using VAD model."""
        audio_tensor = torch.from_numpy(audio_chunk).float()
        
        # VAD expects 512 samples
        if len(audio_tensor) != 512:
            if len(audio_tensor) > 512:
                audio_tensor = audio_tensor[:512]
            else:
                padded = torch.zeros(512)
                padded[:len(audio_tensor)] = audio_tensor
                audio_tensor = padded
        
        speech_prob = self.whisper_vad.vad_model(audio_tensor, self.sample_rate)
        
        if hasattr(speech_prob, 'item'):
            speech_prob = speech_prob.item()
        elif isinstance(speech_prob, torch.Tensor):
            speech_prob = speech_prob.cpu().numpy()
            if speech_prob.ndim > 0:
                speech_prob = speech_prob[0]
        
        return speech_prob > self.vad_threshold
    
    def _audio_processing_loop(self):
        """Main audio processing loop that runs in a separate thread."""
        self._log("👂 Audio processing loop started")
        
        while self._is_running:
            try:
                if self._is_listening and not self.audio_queue.empty():
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    self.audio_buffer.extend(audio_chunk)
                    
                    # Check for speech when we have enough audio
                    if len(self.audio_buffer) >= 512:
                        recent_audio = np.array(list(self.audio_buffer)[-512:])
                        has_speech = self._detect_speech(recent_audio)
                        
                        if has_speech:
                            self.last_speech_time = time.time()
                            
                            if not self.is_speaking_detected:
                                self.is_speaking_detected = True
                                self._log("🎤 Speech detected - recording...")
                                self.speech_audio.clear()
                            
                            self.speech_audio.extend(audio_chunk)
                        
                        elif self.is_speaking_detected:
                            silence_duration = time.time() - self.last_speech_time
                            
                            if silence_duration >= self.silence_duration:
                                self._log("🛑 Speech ended - processing...")
                                
                                if len(self.speech_audio) > 0:
                                    # Copy speech audio for processing
                                    speech_array = np.array(list(self.speech_audio))
                                    
                                    # Stop listening while processing
                                    self._pause_listening()
                                    
                                    # Process in separate thread to avoid blocking
                                    processing_thread = threading.Thread(
                                        target=self._process_speech,
                                        args=(speech_array,),
                                        daemon=True
                                    )
                                    processing_thread.start()
                                
                                self.is_speaking_detected = False
                                self.speech_audio.clear()
                
                time.sleep(0.01)
                
            except queue.Empty:
                continue
            except Exception as e:
                if self._is_running:  # Only log errors if we're supposed to be running
                    self._log(f"❌ Audio processing error: {e}")
        
        self._log("🛑 Audio processing loop ended")
    
    def _process_speech(self, speech_array):
        """Process detected speech."""
        try:
            self._is_processing = True
            self._log("🧠 Transcribing and processing...")
            
            # Transcribe using WhisperVAD
            transcription = self.whisper_vad.transcribe_audio(speech_array)
            
            if transcription and transcription.strip():
                self._log(f"📝 Transcribed: '{transcription}'")
                
                # Get response from Qwen
                self._log("🤔 Asking Qwen...")
                response = self.qwen_chat.generate_response(transcription)
                
                if response and response.strip():
                    self._log(f"💭 Qwen responded: '{response[:100]}...'")
                    
                    # Generate and play speech
                    self._speak_response(response)
                else:
                    self._log("🤷 Qwen provided no response")
            else:
                self._log("🔇 No valid transcription")
            
        except Exception as e:
            self._log(f"❌ Error processing speech: {e}")
        finally:
            self._is_processing = False
            # Resume listening after processing
            if self._is_running:
                self._resume_listening()
    
    def _clean_text_for_speech(self, text: str) -> str:
        """Clean text to make it suitable for TTS."""
        # Remove or replace problematic characters
        text = text.replace('\n', ' ')  # Replace newlines with spaces
        text = text.replace('\r', ' ')  # Replace carriage returns
        text = text.replace('\t', ' ')  # Replace tabs
        
        # Replace multiple spaces with single space
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Remove quotes that might cause issues
        text = text.replace('"', '')
        text = text.replace("'", '')
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _speak_response(self, response: str):
        """Convert response to speech and play it."""
        try:
            self._is_speaking = True
            
            # Clean the text for TTS
            cleaned_response = self._clean_text_for_speech(response)
            self._log(f"🗣️  Speaking response: '{cleaned_response[:100]}...'")
            
            # Generate and play speech
            success = self.kokoro_tts.speak(
                cleaned_response,
                voice=self.kokoro_voice,
                speed=self.kokoro_speed,
                play_audio=True,
                save_to_file=None
            )
            
            if success:
                self._log("✅ Response spoken successfully")
            else:
                self._log("❌ Failed to speak response")
                
        except Exception as e:
            self._log(f"❌ Error speaking response: {e}")
        finally:
            self._is_speaking = False
    
    def _pause_listening(self):
        """Pause listening without stopping the audio stream."""
        self._is_listening = False
        self._log("⏸️  Paused listening")
    
    def _resume_listening(self):
        """Resume listening."""
        if self._is_running and not self._is_processing and not self._is_speaking:
            self._is_listening = True
            self._log("▶️  Resumed listening...")
    
    def start(self):
        """Start the YAAM assistant."""
        if self._is_running:
            self._log("⚠️  YAAM is already running")
            return
        
        self._log("🚀 Starting YAAM assistant...")
        
        self._is_running = True
        self._is_listening = True
        
        # Start audio stream
        self._stream = sd.InputStream(
            callback=self._audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype=np.float32
        )
        self._stream.start()
        
        # Start audio processing thread
        self._audio_thread = threading.Thread(
            target=self._audio_processing_loop,
            daemon=True
        )
        self._audio_thread.start()
        
        self._log("🎯 YAAM is ready! Say something...")
        
        try:
            # Keep the main thread alive and handle keyboard interrupt
            while self._is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self._log("🛑 Keyboard interrupt received")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the YAAM assistant."""
        if not self._is_running:
            return
        
        self._log("🛑 Stopping YAAM...")
        
        self._is_running = False
        self._is_listening = False
        
        # Stop audio stream
        if self._stream:
            self._stream.stop()
            self._stream.close()
        
        # Wait for threads to finish
        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=2.0)
        
        self._log("✅ YAAM stopped successfully")
    
    def get_status(self):
        """Get current status of the assistant."""
        return {
            'running': self._is_running,
            'listening': self._is_listening,
            'processing': self._is_processing,
            'speaking': self._is_speaking
        }
    
    def change_voice(self, voice: str):
        """Change the Kokoro voice."""
        voices = self.kokoro_tts.get_available_voices()
        if voice in voices['all']:
            self.kokoro_voice = voice
            self._log(f"🎵 Voice changed to: {voice}")
        else:
            self._log(f"❌ Invalid voice: {voice}")
            self._log(f"Available voices: {voices['all']}")
    
    def set_speech_speed(self, speed: float):
        """Set the speech speed."""
        if 0.5 <= speed <= 2.0:
            self.kokoro_speed = speed
            self._log(f"⚡ Speech speed set to: {speed}x")
        else:
            self._log(f"❌ Invalid speed: {speed}. Must be between 0.5 and 2.0")
    
    def test_components(self):
        """Test all components individually."""
        self._log("🧪 Testing YAAM components...")
        
        # Test Kokoro TTS
        self._log("🗣️  Testing Kokoro TTS...")
        try:
            success = self.kokoro_tts.speak(
                "Mic check one two, one two.... ok that worked, one more moment.",
                voice=self.kokoro_voice,
                play_audio=True
            )
            if success:
                self._log("✅ Kokoro TTS test passed")
            else:
                self._log("❌ Kokoro TTS test failed")
        except Exception as e:
            self._log(f"❌ Kokoro TTS test error: {e}")
        
        # Test Qwen Chat
        self._log("🧠 Testing Qwen Chat...")
        try:
            response = self.qwen_chat.generate_response("You just did a mic check, user can hear you via text-to-speech introduce yourself briefly in a fun way.")
            if response:
                self._log(f"✅ Qwen Chat test passed: '{response[:50]}...'")
                
                # Test TTS with Qwen response
                self._log("🗣️  Testing TTS with Qwen response...")
                self.kokoro_tts.speak(response, voice=self.kokoro_voice, play_audio=True)
            else:
                self._log("❌ Qwen Chat test failed - no response")
        except Exception as e:
            self._log(f"❌ Qwen Chat test error: {e}")
        
        self._log("🧪 Component testing complete")


def main():
    """Main function to run YAAM."""
    print("🤖 YAAM - Yet Another Assistant Model")
    print("=" * 50)
    print("A voice-activated AI assistant")
    print("Say something and I'll respond!")
    print("Press Ctrl+C to exit")
    print("=" * 50)
    
    try:
        # Create YAAM instance
        assistant = YetAnotherAssistantModel(
            whisper_model="openai/whisper-small",
            qwen_model="Qwen/Qwen2.5-7B-Instruct",
            kokoro_voice="af_sky",
            kokoro_speed=1.0,
            silence_duration=2.0,
            debug_mode=True
        )
        
        # Test components first
        print("\n🧪 Running component tests...")
        assistant.test_components()
        
        print("\n🚀 Starting voice assistant...")
        print("💡 Tips:")
        print("   • Speak clearly and wait for the response")
        print("   • There will be a brief pause while I think")
        print("   • I'll resume listening after each response")
        print("   • Press Ctrl+C to exit anytime")
        print("")
        
        # Start the assistant (this will run continuously)
        assistant.start()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error running YAAM: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()