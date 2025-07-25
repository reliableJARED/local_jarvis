"""
Voice Activity Detection and Speech-to-Text System  
Handles VAD, transcription, audio capture and word boundary detection
"""

import torch
import sounddevice as sd
import numpy as np
import time
from datetime import datetime
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from collections import deque
import threading
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
import sys
import select
import warnings

warnings.filterwarnings("ignore")

@dataclass
class TranscriptSegment:
    """Represents a transcribed speech segment"""
    text: str
    speaker_id: str
    start_time: float
    end_time: float
    confidence: float
    is_final: bool = False
    audio_data: Optional[np.ndarray] = None 

@dataclass
class SpeechEvent:
    """Represents a speech detection event"""
    event_type: str  # 'speech_start', 'speech_end', 'silence'
    timestamp: float
    audio_data: Optional[np.ndarray] = None

class SpeechCallback(ABC):
    """Abstract base class for speech processing callbacks"""
    
    @abstractmethod
    def on_speech_start(self, event: SpeechEvent):
        """Called when speech is detected"""
        pass
    
    @abstractmethod
    def on_speech_end(self, event: SpeechEvent):
        """Called when speech ends"""
        pass
    
    @abstractmethod
    def on_transcript_update(self, segment: TranscriptSegment):
        """Called with real-time transcript updates"""
        pass
    
    @abstractmethod
    def on_transcript_final(self, segment: TranscriptSegment):
        """Called with final transcript"""
        pass
    
    @abstractmethod
    def on_speaker_change(self, old_speaker: Optional[str], new_speaker: str):
        """Called when speaker changes"""
        pass

class AudioCapture:
    """Handles real-time audio capture"""
    
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stream = None
    
    def audio_callback(self, indata, frames, time, status):
        """Audio callback for sounddevice"""
        audio_data = indata.flatten()
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        self.audio_queue.put(audio_data.copy())
    
    def start(self):
        """Start audio capture"""
        self.is_recording = True
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype=np.float32,
            device=sd.default.device[0]  # Explicitly use input device
        )
        self.stream.start()
    
    def stop(self):
        """Stop audio capture"""
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
    
    def get_audio_chunk(self) -> Optional[np.ndarray]:
        """Get next audio chunk (non-blocking)"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

class VoiceActivityDetector:
    """Handles voice activity detection using Silero VAD ONNX model"""
    
    def __init__(self, threshold=0.3, sample_rate=16000):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.vad_history = deque(maxlen=5)
        
        # Load Silero VAD ONNX model using the official package
        self.model = self._load_vad_model()
        self._reset_states()
    
    def _load_vad_model(self):
        """Load the Silero VAD ONNX model using the official package"""
        try:
            from silero_vad import load_silero_vad
            import os
            
            # First try normal loading
            try:
                model = load_silero_vad(onnx=True)
                return model
            except Exception as e:
                print(f"Normal loading failed: {e}")
                print("Attempting offline mode...")
                
                # Force offline mode for Hugging Face components
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                os.environ['HF_DATASETS_OFFLINE'] = '1'
                
                try:
                    # Try loading again with offline mode
                    model = load_silero_vad(onnx=True)
                    return model
                except Exception as offline_error:
                    raise RuntimeError(
                        f"Failed to load VAD model both online and offline. "
                        f"Online error: {e}. Offline error: {offline_error}. "
                        f"Make sure to run the model once with internet connection to cache it."
                    )
                    
        except ImportError:
            raise ImportError(
                "silero-vad package not found. Install with: pip install silero-vad"
            )
    
    def _reset_states(self):
        """Reset the internal states of the VAD model"""
        # Initialize hidden and cell states for the LSTM
        self._h = torch.zeros((2, 1, 64))
        self._c = torch.zeros((2, 1, 64))
    
    def detect_speech(self, audio_chunk: np.ndarray) -> bool:
        """Detect if audio chunk contains speech"""
        # Convert numpy array to torch tensor
        if isinstance(audio_chunk, np.ndarray):
            audio_tensor = torch.from_numpy(audio_chunk).float()
        else:
            audio_tensor = audio_chunk.float()
        
        # Ensure correct length for VAD model
        target_length = 512 if self.sample_rate == 16000 else 256
        
        if len(audio_tensor) != target_length:
            if len(audio_tensor) > target_length:
                audio_tensor = audio_tensor[:target_length]
            else:
                padded = torch.zeros(target_length)
                padded[:len(audio_tensor)] = audio_tensor
                audio_tensor = padded
        
        try:
            # Run inference with state management
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.sample_rate).item()
            
        except Exception as e:
            print(f"VAD inference error: {e}")
            return False
        
        # Smooth VAD decisions
        has_speech_raw = speech_prob > self.threshold
        self.vad_history.append(has_speech_raw)
        
        # Majority vote for smoothing
        if len(self.vad_history) >= 3:
            recent_decisions = list(self.vad_history)[-3:]
            has_speech = sum(recent_decisions) >= 2
        else:
            has_speech = has_speech_raw
        
        return has_speech
    
    def reset(self):
        """Reset the VAD state - call this when:
        - Starting a new audio session/conversation
        - Switching audio sources
        - After long periods of silence (>5 seconds)
        - When audio conditions change significantly
        """
        self._reset_states()
        self.vad_history.clear()

class WordBoundaryDetector:
    """Detects word boundaries in audio streams"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.energy_history = deque(maxlen=20)  # Store last 20 energy values
        self.vad_history = deque(maxlen=10)     # Store last 10 VAD decisions
        self.silence_threshold = 0.001          # Energy threshold for silence
        self.min_silence_duration = 0.1        # Minimum silence duration for word boundary (100ms)
        self.energy_drop_threshold = 0.3       # Energy drop threshold for boundary detection
        
    def calculate_energy(self, audio_chunk: np.ndarray) -> float:
        """Calculate RMS energy of audio chunk"""
        if len(audio_chunk) == 0:
            return 0.0
        return np.sqrt(np.mean(audio_chunk ** 2))
    
    def detect_word_boundary(self, audio_chunk: np.ndarray, has_speech: bool) -> bool:
        """
        Detect if we're at a good word boundary based on energy history and VAD
        
        Args:
            audio_chunk: Current audio chunk
            has_speech: Current VAD decision
            
        Returns:
            True if this is a good word boundary
        """
        current_energy = self.calculate_energy(audio_chunk)
        self.energy_history.append(current_energy)
        self.vad_history.append(has_speech)
        
        # Need some history to make decisions
        if len(self.energy_history) < 5:
            return False
        
        # Convert to lists for easier manipulation
        energy_list = list(self.energy_history)
        vad_list = list(self.vad_history)
        
        # Strategy 1: Detect silence gaps (classic word boundary indicator)
        if self._detect_silence_gap(energy_list, vad_list):
            return True
        
        # Strategy 2: Detect significant energy drops
        if self._detect_energy_drop(energy_list):
            return True
        
        # Strategy 3: Detect VAD transitions (speech to non-speech)
        if self._detect_vad_transition(vad_list):
            return True
        
        return False
    
    def _detect_silence_gap(self, energy_list: List[float], vad_list: List[bool]) -> bool:
        """Detect silence gaps that indicate word boundaries"""
        if len(energy_list) < 5 or len(vad_list) < 5:
            return False
        
        # Look for recent silence followed by speech
        recent_energy = energy_list[-3:]
        recent_vad = vad_list[-3:]
        
        # Check if we have low energy AND no speech detection
        has_silence = all(e < self.silence_threshold for e in recent_energy)
        has_no_speech = not any(recent_vad)
        
        if has_silence and has_no_speech:
            # Check if we had speech before this silence
            if len(vad_list) >= 6:
                previous_vad = vad_list[-6:-3]
                had_previous_speech = any(previous_vad)
                return had_previous_speech
        
        return False
    
    def _detect_energy_drop(self, energy_list: List[float]) -> bool:
        """Detect significant energy drops that might indicate word boundaries"""
        if len(energy_list) < 5:
            return False
        
        # Calculate moving averages
        recent_avg = np.mean(energy_list[-3:])
        previous_avg = np.mean(energy_list[-6:-3])
        
        # Detect significant drop
        if previous_avg > 0:
            energy_drop_ratio = (previous_avg - recent_avg) / previous_avg
            return energy_drop_ratio > self.energy_drop_threshold
        
        return False
    
    def _detect_vad_transition(self, vad_list: List[bool]) -> bool:
        """Detect VAD transitions that indicate word boundaries"""
        if len(vad_list) < 4:
            return False
        
        # Look for speech-to-silence transitions
        # Pattern: [speech, speech, no_speech, no_speech] or similar
        recent_pattern = vad_list[-4:]
        
        # Check for transition from speech to silence
        if recent_pattern[:2] == [True, True] and recent_pattern[2:] == [False, False]:
            return True
        
        # Check for transition from silence to speech (end of pause)
        if recent_pattern[:2] == [False, False] and recent_pattern[2:] == [True, True]:
            return True
        
        return False

class SpeechTranscriber:
    """Handles speech transcription using Whisper with offline support"""
    
    def __init__(self, model_name="openai/whisper-small"):
        self.sample_rate = 16000
        self.model_name = model_name
        
        # Load model and processor with offline fallback
        self.processor, self.model = self._load_models()
    
    def _load_models(self):
        """Load Whisper models with offline support"""
        import os
        
        # Set offline mode environment variables BEFORE importing/loading
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        
        try:
            print(f"Loading Whisper model '{self.model_name}' in offline mode...")
            processor = AutoProcessor.from_pretrained(
                self.model_name,
                local_files_only=True
            )
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                local_files_only=True
            )
            print("Whisper model loaded successfully in offline mode!")
            return processor, model
            
        except Exception as e:
            print(f"Offline loading failed: {e}")
            print("Attempting online loading...")
            
            # Try removing offline flags and attempt online loading
            os.environ.pop('HF_HUB_OFFLINE', None)
            os.environ.pop('TRANSFORMERS_OFFLINE', None)
            os.environ.pop('HF_DATASETS_OFFLINE', None)
            
            try:
                processor = AutoProcessor.from_pretrained(self.model_name)
                model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name)
                print("Whisper model loaded successfully online!")
                return processor, model
                
            except Exception as online_error:
                raise RuntimeError(
                    f"Failed to load Whisper model both offline and online. "
                    f"Offline error: {e}. Online error: {online_error}. "
                    f"Make sure to run the model once with internet connection to cache it."
                )
    
    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio data to text"""
        if len(audio_data) == 0:
            return ""
        
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        inputs = self.processor(
            audio_data,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones(inputs["input_features"].shape[:-1], dtype=torch.long)
        
        with torch.no_grad():
            predicted_ids = self.model.generate(
                inputs["input_features"],
                attention_mask=inputs.get("attention_mask"),
                language="en",
                task="transcribe",
                max_length=448,
                do_sample=False
            )
        
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        return transcription.strip()

class TextInputHandler:
    """Handles text input functionality"""
    
    def __init__(self):
        self.text_input_enabled = True
        self.text_input_queue = queue.Queue()
        self.text_input_thread = None
        self.should_stop = False
    
    def start(self):
        """Start text input handling"""
        if self.text_input_enabled:
            self.should_stop = False
            self.text_input_thread = threading.Thread(target=self._text_input_loop, daemon=True)
            self.text_input_thread.start()
            print("📝 Text input enabled. You can type messages in the terminal.")
            print("Type 'quit' or press Ctrl+C to exit.")
    
    def stop(self):
        """Stop text input handling"""
        self.should_stop = True
        if self.text_input_thread and self.text_input_thread.is_alive():
            self.text_input_thread.join(timeout=1.0)
    
    def has_text_input(self) -> bool:
        """Check if there's pending text input"""
        return not self.text_input_queue.empty()
    
    def get_text_input(self) -> Optional[str]:
        """Get next text input (non-blocking)"""
        try:
            return self.text_input_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _text_input_loop(self):
        """Background thread for text input"""
        print("\n" + "="*50)
        print("📝 TEXT INPUT ACTIVE")
        print("Type messages to interact with the assistant.")
        print("Type 'quit' or press Ctrl+C to exit.")
        print("="*50 + "\n")
        
        while not self.should_stop:
            try:
                # Non-blocking input check (platform specific)
                if self._has_pending_input():
                    text_input = input("💬 ").strip()
                    
                    if text_input.lower() in ['quit', 'exit', 'q']:
                        print("Exiting...")
                        break
                    
                    if text_input:
                        # Put text input in queue for main processing loop
                        self.text_input_queue.put(text_input)
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except (EOFError, KeyboardInterrupt):
                print("\nText input interrupted.")
                break
            except Exception as e:
                print(f"Error in text input: {e}")
                time.sleep(0.5)
    
    def _has_pending_input(self):
        """Check if there's pending input without blocking"""
        try:
            if sys.platform == 'win32':
                import msvcrt
                return msvcrt.kbhit()
            else:
                # Unix-like systems
                return select.select([sys.stdin], [], [], 0.0)[0]
        except:
            # Fallback: always return True (will block on input)
            return True
    
    def enable(self):
        """Enable text input mode"""
        if not self.text_input_enabled:
            self.text_input_enabled = True
            self.start()
            print("✅ Text input enabled")
    
    def disable(self):
        """Disable text input mode"""
        self.text_input_enabled = False
        self.stop()
        print("❌ Text input disabled")

class VADTranscriptionProcessor:
    """Main processing system that combines VAD, transcription, and text input"""
    
    def __init__(self, callback: SpeechCallback):
        self.callback = callback
        
        # Initialize components
        self.audio_capture = AudioCapture()
        self.vad = VoiceActivityDetector()
        self.transcriber = SpeechTranscriber()
        self.word_boundary_detector = WordBoundaryDetector()
        self.text_input_handler = TextInputHandler()
        
        # State tracking
        self.is_speaking = False
        self.current_speaker = None
        self.speech_buffer = deque()
        self.silence_start = None
        self.speech_start_time = None
        
        # Real-time transcription
        self.realtime_buffer = deque()
        self.last_realtime_transcript = ""
        self.last_realtime_time = 0
        self.realtime_transcript_thread = None
        
        # Configuration
        self.silence_duration_threshold = 1.5
        self.min_speech_duration = 0.8
        self.max_speech_duration = 10.0
        
        # Enhanced real-time configuration
        self.realtime_update_interval = 0.5
        self.realtime_max_interval = 2.5
        self.realtime_min_duration = 1.0
        
        # Processing thread
        self.processing_thread = None
        self.should_stop = False
    
    def start(self):
        """Start the VAD and transcription processing system"""
        self.should_stop = False
        
        # Start components
        self.audio_capture.start()
        self.text_input_handler.start()
        
        # Start processing threads
        self.processing_thread = threading.Thread(target=self._process_audio, daemon=True)
        self.processing_thread.start()
        
        self.realtime_transcript_thread = threading.Thread(target=self._realtime_transcription_loop, daemon=True)
        self.realtime_transcript_thread.start()
    
    def stop(self, timeout=1.0):
        """Stop the processing system"""
        print("Stopping VAD and transcription processor...")
        
        # Set stop flag
        self.should_stop = True
        
        # Stop components
        try:
            self.audio_capture.stop()
        except Exception as e:
            print(f"Error stopping audio capture: {e}")
        
        try:
            self.text_input_handler.stop()
        except Exception as e:
            print(f"Error stopping text input handler: {e}")
        
        # Wait for threads to finish
        threads_to_stop = []
        if self.processing_thread and self.processing_thread.is_alive():
            threads_to_stop.append(("processing_thread", self.processing_thread))
        if self.realtime_transcript_thread and self.realtime_transcript_thread.is_alive():
            threads_to_stop.append(("realtime_transcript_thread", self.realtime_transcript_thread))
        
        for thread_name, thread in threads_to_stop:
            print(f"Waiting for {thread_name} to stop...")
            thread.join(timeout=timeout)
        
        print("VAD and transcription processor stopped.")
    
    def _process_text_input(self, text_input: str):
        """Process text input and send through callback system"""
        current_time = time.time()
        
        # Determine speaker ID for text input
        speaker_id = self._get_text_speaker_id()
        
        print(f"💬 📝 Processing text from {speaker_id}: '{text_input}'")
        
        # Create final transcript segment
        segment = TranscriptSegment(
            text=text_input,
            speaker_id=speaker_id,
            start_time=current_time,
            end_time=current_time,
            confidence=1.0,  # Text input has perfect confidence
            is_final=True
        )
        
        # Send through callback system
        self.callback.on_transcript_final(segment)
        
        print(f"✅ Text input processed as {speaker_id}")
    
    def _get_text_speaker_id(self) -> str:
        """Determine speaker ID for text input"""
        # Check if callback has methods to determine primary speaker
        if hasattr(self.callback, 'get_primary_speaker'):
            primary = self.callback.get_primary_speaker()
            if primary:
                return primary
        
        # Check for recent active speakers
        if hasattr(self.callback, 'get_active_speakers'):
            active = self.callback.get_active_speakers()
            if active:
                # Filter out system speakers and return most recent human speaker
                human_speakers = [s for s in active if not s.startswith('SYSTEM_') and s != 'USER_00']
                if human_speakers:
                    return human_speakers[-1]
        
        # Default to text user
        return "TEXT_USER"
    

    def _process_audio(self):
        """Enhanced main audio processing loop with text input support"""
        audio_buffer = deque(maxlen=int(16000 * 0.5))  # 0.5 second buffer
        
        while not self.should_stop:
            try:
                # Check for text input first
                if self.text_input_handler.has_text_input():
                    text_input = self.text_input_handler.get_text_input()
                    if text_input:
                        self._process_text_input(text_input)
                
                # Get audio chunk
                chunk = self.audio_capture.get_audio_chunk()
                if chunk is None:
                    time.sleep(0.01)
                    continue
                
                audio_buffer.extend(chunk)
                
                # Check for speech
                if len(audio_buffer) >= 512:
                    recent_audio = np.array(list(audio_buffer)[-512:])
                    has_speech = self.vad.detect_speech(recent_audio)
                    
                    current_time = time.time()
                    
                    if has_speech:
                        if not self.is_speaking:
                            # Speech started
                            self.is_speaking = True
                            self.speech_start_time = current_time
                            self.speech_buffer.clear()
                            self.realtime_buffer.clear()
                            self.last_realtime_transcript = ""
                            
                            # Include some pre-speech audio
                            self.speech_buffer.extend(list(audio_buffer))
                            self.realtime_buffer.extend(list(audio_buffer))
                            
                            self.callback.on_speech_start(SpeechEvent(
                                event_type='speech_start',
                                timestamp=current_time,
                                audio_data=recent_audio
                            ))
                        else:
                            # Continue speech
                            self.speech_buffer.extend(chunk)
                            self.realtime_buffer.extend(chunk)
                        
                        self.silence_start = None
                    
                    else:
                        # No speech detected
                        if self.is_speaking:
                            if self.silence_start is None:
                                self.silence_start = current_time
                            
                            silence_duration = current_time - self.silence_start
                            
                            if silence_duration >= self.silence_duration_threshold:
                                # Speech ended
                                self._finalize_speech_segment()
                        
                        if self.speech_buffer:
                            self.speech_buffer.extend(chunk)
                            self.realtime_buffer.extend(chunk)
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Error in audio processing: {e}")
                time.sleep(0.1)

    def _realtime_transcription_loop(self):
        """Enhanced background thread for real-time transcription with word boundary detection"""
        while not self.should_stop:
            try:
                current_time = time.time()
                time_since_last_update = current_time - self.last_realtime_time
                
                # Only process if we're currently speaking
                if self.is_speaking and self.realtime_buffer:
                    
                    # Get current audio buffer
                    current_audio = np.array(list(self.realtime_buffer))
                    audio_duration = len(current_audio) / 16000
                    
                    # Check if we have enough audio for processing
                    if audio_duration >= self.realtime_min_duration:
                        
                        # Determine if we should process now
                        should_process = False
                        
                        # Always process if we've hit the maximum interval
                        if time_since_last_update >= self.realtime_max_interval:
                            should_process = True
                            print(f"🕐 Processing due to max interval ({self.realtime_max_interval}s)")
                        
                        # Process if we've hit minimum interval AND found a word boundary
                        elif time_since_last_update >= self.realtime_update_interval:
                            
                            # Check for word boundary using recent audio
                            if len(current_audio) >= 1024:  # Need enough audio for boundary detection
                                recent_chunk = current_audio[-1024:]
                                has_speech = self.vad.detect_speech(recent_chunk[-512:])
                                
                                is_word_boundary = self.word_boundary_detector.detect_word_boundary(
                                    recent_chunk, has_speech
                                )
                                
                                if is_word_boundary:
                                    should_process = True
                                    print(f"🎯 Processing due to word boundary detected")
                        
                        # Process the audio if conditions are met
                        if should_process:
                            try:
                                # CRITICAL FIX: Don't set speaker_id here!
                                # The speaker identification should happen in the callback
                                # when on_transcript_update/on_transcript_final is called
                                
                                # Transcribe current buffer
                                transcript = self.transcriber.transcribe(current_audio)
                                
                                # Only send update if transcript changed significantly
                                if (transcript.strip() and 
                                    transcript != self.last_realtime_transcript and
                                    len(transcript.strip()) > 2):
                                    
                                    # Don't set speaker_id here - let the callback handle it
                                    segment = TranscriptSegment(
                                        text=transcript,
                                        speaker_id="PROCESSING",  # Temporary placeholder
                                        start_time=self.speech_start_time or current_time,
                                        end_time=current_time,
                                        confidence=0.8,  # Lower confidence for real-time
                                        is_final=False
                                    )
                                    
                                    # Store the audio data in the segment for speaker identification
                                    segment.audio_data = current_audio
                                    
                                    self.callback.on_transcript_update(segment)
                                    self.last_realtime_transcript = transcript
                                    
                                    print(f"📱 Real-time update: {len(current_audio)/16000:.1f}s audio, "
                                        f"interval: {time_since_last_update:.1f}s")
                                
                                self.last_realtime_time = current_time
                                
                            except Exception as e:
                                print(f"Error in real-time transcription: {e}")
                
                time.sleep(0.05)  # Check every 50ms for more responsive boundary detection
                
            except Exception as e:
                print(f"Error in real-time transcription loop: {e}")
                time.sleep(0.5)

    def _finalize_speech_segment(self):
        """Process and finalize a speech segment"""
        if not self.speech_buffer or not self.speech_start_time:
            return
        
        speech_audio = np.array(list(self.speech_buffer))
        speech_duration = len(speech_audio) / 16000
        
        # Check minimum duration
        if speech_duration < self.min_speech_duration:
            self._reset_speech_state()
            return
        
        try:
            # CRITICAL FIX: Don't set speaker_id here either!
            # Let the callback handle speaker identification
            
            # Final transcription (usually more accurate than real-time)
            transcript = self.transcriber.transcribe(speech_audio)
            
            if transcript.strip():
                # Create final transcript segment without speaker_id
                segment = TranscriptSegment(
                    text=transcript,
                    speaker_id="PROCESSING",  # Temporary placeholder
                    start_time=self.speech_start_time,
                    end_time=time.time(),
                    confidence=1.0,  # Higher confidence for final transcription
                    is_final=True
                )
                
                # Store the audio data for speaker identification
                segment.audio_data = speech_audio
                
                self.callback.on_transcript_final(segment)
            
            # Send speech end event with audio data
            self.callback.on_speech_end(SpeechEvent(
                event_type='speech_end',
                timestamp=time.time(),
                audio_data=speech_audio
            ))
        
        except Exception as e:
            print(f"Error processing speech segment: {e}")
        
        finally:
            self._reset_speech_state()
    def _reset_speech_state(self):
        """Reset speech processing state"""
        self.is_speaking = False
        self.speech_buffer.clear()
        self.realtime_buffer.clear()
        self.silence_start = None
        self.speech_start_time = None
        self.last_realtime_transcript = ""
        self.last_realtime_time = 0
        # Reset word boundary detector state
        self.word_boundary_detector.energy_history.clear()
        self.word_boundary_detector.vad_history.clear()
    
    def set_current_speaker(self, speaker_id: str):
        """Set the current speaker ID (called by external speaker identification)"""
        if speaker_id != self.current_speaker:
            old_speaker = self.current_speaker
            self.current_speaker = speaker_id
            self.callback.on_speaker_change(old_speaker, speaker_id)
    
    def enable_text_input(self):
        """Enable text input mode"""
        self.text_input_handler.enable()
    
    def disable_text_input(self):
        """Disable text input mode"""
        self.text_input_handler.disable()


# Example callback implementation for testing
class SimpleVADCallback(SpeechCallback):
    """Simple callback implementation for testing VAD and transcription"""
    
    def __init__(self):
        self.current_conversation = []
        self.active_speakers = set()
    
    def on_speech_start(self, event: SpeechEvent):
        print(f"🎤 Speech detected at {event.timestamp:.1f}")
    
    def on_speech_end(self, event: SpeechEvent):
        print(f"🔇 Speech ended at {event.timestamp:.1f}")
    
    def on_transcript_update(self, segment: TranscriptSegment):
        print(f"📝 [{segment.speaker_id}] (live): {segment.text}")
    
    def on_transcript_final(self, segment: TranscriptSegment):
        print(f"✅ [{segment.speaker_id}] (final): {segment.text}")
        self.current_conversation.append(segment)
        self.active_speakers.add(segment.speaker_id)
    
    def on_speaker_change(self, old_speaker: Optional[str], new_speaker: str):
        if old_speaker is not None:
            print(f"👥 Speaker change: {old_speaker} → {new_speaker}")
        else:
            print(f"👤 New speaker: {new_speaker}")
    
    def get_conversation_history(self) -> List[TranscriptSegment]:
        """Get full conversation history"""
        return self.current_conversation.copy()
    
    def get_recent_transcript(self, seconds: float = 30.0) -> str:
        """Get recent transcript as formatted string"""
        cutoff_time = time.time() - seconds
        recent_segments = [s for s in self.current_conversation if s.start_time >= cutoff_time]
        
        result = []
        for segment in recent_segments:
            result.append(f"[{segment.speaker_id}]: {segment.text}")
        
        return "\n".join(result)
    
    def get_primary_speaker(self) -> Optional[str]:
        """Get primary speaker (most recent non-system speaker)"""
        if not self.current_conversation:
            return None
        
        # Find most recent human speaker
        for segment in reversed(self.current_conversation):
            if not segment.speaker_id.startswith('SYSTEM_') and segment.speaker_id != 'USER_00':
                return segment.speaker_id
        
        return None
    
    def get_active_speakers(self) -> List[str]:
        """Get list of active speakers"""
        return list(self.active_speakers)


if __name__ == "__main__":
    # Example usage for testing VAD and transcription only
    callback = SimpleVADCallback()
    processor = VADTranscriptionProcessor(callback)
    
    print("Starting VAD and transcription processor...")
    print("Features: Voice activity detection, speech transcription, text input")
    print("Speak into the microphone or type text. Press Ctrl+C to stop.")
    
    try:
        processor.start()
        
        # Main loop - your application logic would go here
        while True:
            time.sleep(1)
            
            # Example: Get recent conversation every 10 seconds
            recent = callback.get_recent_transcript(10.0)
            if recent:
                print(f"\n--- Recent conversation ---\n{recent}\n")
    
    except KeyboardInterrupt:
        print("\nStopping...")
        processor.stop()