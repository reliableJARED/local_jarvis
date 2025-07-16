
"""
Enhanced modular speech processing system with word boundary detection for dynamic live transcripts
Also uses speaker diarization to recognize voices
"""

import torch
import sounddevice as sd
import numpy as np
import time
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from collections import deque
import threading
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable, Dict, List
import warnings
warnings.filterwarnings("ignore")

# Try to import speaker embedding models
try:
    from speechbrain.pretrained import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False

@dataclass
class TranscriptSegment:
    """Represents a transcribed speech segment"""
    text: str
    speaker_id: str
    start_time: float
    end_time: float
    confidence: float
    is_final: bool = False

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
            dtype=np.float32
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
    """Handles voice activity detection"""
    
    def __init__(self, threshold=0.3, sample_rate=16000):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.vad_history = deque(maxlen=5)
        
        # Load Silero VAD
        self.vad_model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
    
    def detect_speech(self, audio_chunk: np.ndarray) -> bool:
        """Detect if audio chunk contains speech"""
        audio_tensor = torch.from_numpy(audio_chunk).float()
        
        # Ensure correct length for VAD model
        if len(audio_tensor) != 512:
            if len(audio_tensor) > 512:
                audio_tensor = audio_tensor[:512]
            else:
                padded = torch.zeros(512)
                padded[:len(audio_tensor)] = audio_tensor
                audio_tensor = padded
        
        speech_prob = self.vad_model(audio_tensor, self.sample_rate)
        
        if hasattr(speech_prob, 'item'):
            speech_prob = speech_prob.item()
        elif isinstance(speech_prob, torch.Tensor):
            speech_prob = speech_prob.cpu().numpy()
            if speech_prob.ndim > 0:
                speech_prob = speech_prob[0]
        
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
    """Handles speech transcription using Whisper"""
    
    def __init__(self, model_name="openai/whisper-small"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        self.sample_rate = 16000
    
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

class SpeakerIdentifier:
    """Handles speaker identification and embeddings"""
    
    def __init__(self):
        self.speaker_embeddings = {}
        self.current_speaker_id = None
        self.speaker_counter = 0
        
        # Load speaker embedding model if available
        if SPEECHBRAIN_AVAILABLE:
            self.speaker_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
        else:
            self.speaker_model = None
    
    def extract_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract speaker embedding from audio"""
        if not SPEECHBRAIN_AVAILABLE or self.speaker_model is None:
            # Fallback: simple audio features
            if len(audio_data) == 0:
                return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            
            energy = np.mean(audio_data ** 2)
            fft = np.fft.fft(audio_data)
            spectral_centroid = np.mean(np.abs(fft))
            zero_crossing_rate = np.mean(np.abs(np.diff(np.sign(audio_data))))
            
            embedding = np.array([energy * 1000, spectral_centroid / 1000, zero_crossing_rate * 100, 0, 0])
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        
        try:
            # Use SpeechBrain model
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                embedding = self.speaker_model.encode_batch(audio_tensor)
                embedding = embedding.squeeze().cpu().numpy()
            
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else embedding
        except Exception:
            return np.random.normal(0, 0.01, 192) / 192
    
    def identify_speaker(self, audio_data: np.ndarray, threshold=0.3) -> str:
        """Identify speaker from audio data"""
        embedding = self.extract_embedding(audio_data)
        
        # Compare with known speakers
        best_match_id = None
        best_similarity = 0.0
        
        for speaker_id, stored_embedding in self.speaker_embeddings.items():
            from scipy.spatial.distance import cosine
            similarity = 1 - cosine(embedding, stored_embedding)
            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_match_id = speaker_id
        
        if best_match_id is not None:
            return best_match_id
        else:
            # Create new speaker
            new_speaker_id = f"SPEAKER_{self.speaker_counter:02d}"
            self.speaker_counter += 1
            self.speaker_embeddings[new_speaker_id] = embedding
            return new_speaker_id

class VoiceAssistantSpeechProcessor:
    """Main speech processing system for voice assistants"""
    
    def __init__(self, callback: SpeechCallback):
        self.callback = callback
        
        # Initialize components
        self.audio_capture = AudioCapture()
        self.vad = VoiceActivityDetector()
        self.transcriber = SpeechTranscriber()
        self.speaker_id = SpeakerIdentifier()
        self.word_boundary_detector = WordBoundaryDetector()
        
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
        
        # Enhanced real-time configuration - use min max to transcribe in real time and try dynamic word breaks in between that range
        self.realtime_update_interval = 0.5      # Minimum update interval (500ms)
        self.realtime_max_interval = 2.5         # Maximum update interval (5 * minimum)
        self.realtime_min_duration = 1.0         # Minimum audio for real-time transcription
        
        # Processing thread
        self.processing_thread = None
        self.should_stop = False
    
    def start(self):
        """Start the speech processing system"""
        self.should_stop = False
        self.audio_capture.start()
        self.processing_thread = threading.Thread(target=self._process_audio, daemon=True)
        self.processing_thread.start()
        
        # Start real-time transcription thread
        self.realtime_transcript_thread = threading.Thread(target=self._realtime_transcription_loop, daemon=True)
        self.realtime_transcript_thread.start()
    
    def stop(self):
        """Stop the speech processing system"""
        self.should_stop = True
        self.audio_capture.stop()
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        if self.realtime_transcript_thread:
            self.realtime_transcript_thread.join(timeout=1.0)
    
    def _process_audio(self):
        """Main audio processing loop"""
        audio_buffer = deque(maxlen=int(16000 * 0.5))  # 0.5 second buffer
        
        while not self.should_stop:
            try:
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
                                # Get current speaker (use cached if available)
                                if self.current_speaker is None and len(current_audio) > 8000:  # ~0.5 seconds
                                    speaker_id = self.speaker_id.identify_speaker(current_audio)
                                    if speaker_id != self.current_speaker:
                                        old_speaker = self.current_speaker
                                        self.current_speaker = speaker_id
                                        self.callback.on_speaker_change(old_speaker, speaker_id)
                                else:
                                    speaker_id = self.current_speaker or "UNKNOWN"
                                
                                # Transcribe current buffer
                                transcript = self.transcriber.transcribe(current_audio)
                                
                                # Only send update if transcript changed significantly
                                if (transcript.strip() and 
                                    transcript != self.last_realtime_transcript and
                                    len(transcript.strip()) > 2):
                                    
                                    segment = TranscriptSegment(
                                        text=transcript,
                                        speaker_id=speaker_id,
                                        start_time=self.speech_start_time or current_time,
                                        end_time=current_time,
                                        confidence=0.8,  # Lower confidence for real-time
                                        is_final=False
                                    )
                                    
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
            # Identify speaker (use full audio for final identification)
            speaker_id = self.speaker_id.identify_speaker(speech_audio)
            
            # Check for speaker change
            if speaker_id != self.current_speaker:
                old_speaker = self.current_speaker
                self.current_speaker = speaker_id
                self.callback.on_speaker_change(old_speaker, speaker_id)
            
            # Final transcription (usually more accurate than real-time)
            transcript = self.transcriber.transcribe(speech_audio)
            
            if transcript.strip():
                # Create final transcript segment
                segment = TranscriptSegment(
                    text=transcript,
                    speaker_id=speaker_id,
                    start_time=self.speech_start_time,
                    end_time=time.time(),
                    confidence=1.0,  # Higher confidence for final transcription
                    is_final=True
                )
                
                self.callback.on_transcript_final(segment)
            
            # Send speech end event
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

# Example usage for voice assistant
class VoiceAssistantCallback(SpeechCallback):
    """Example callback implementation for a voice assistant"""
    
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

if __name__ == "__main__":
    # Example usage
    callback = VoiceAssistantCallback()
    processor = VoiceAssistantSpeechProcessor(callback)
    
    print("Starting enhanced voice assistant speech processor...")
    print("Features: Dynamic word boundary detection, adaptive real-time transcription")
    print("Speak into the microphone. Press Ctrl+C to stop.")
    
    try:
        processor.start()
        
        # Main loop - your voice assistant logic would go here
        while True:
            time.sleep(1)
            
            # Example: Get recent conversation every 10 seconds
            recent = callback.get_recent_transcript(10.0)
            if recent:
                print(f"\n--- Recent conversation ---\n{recent}\n")
    
    except KeyboardInterrupt:
        print("\nStopping...")
        processor.stop()