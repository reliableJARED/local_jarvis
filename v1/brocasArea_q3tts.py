"""
QUEUE DATA TEMPLATES
"""
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, Generator, Any
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Queue, Event, Manager
import queue
import time
import sounddevice as sd
import warnings
import soundfile as sf
import torch
import numpy as np
import librosa
from kokoro import KPipeline
from typing import Optional, Generator, Tuple, Any
import platform
import os
import socket
import logging
from qwen_tts import Qwen3TTSModel

# At module level, before class definition
torch.set_float32_matmul_precision('high') #https://github.com/dffdeeq/Qwen3-TTS-streaming/blob/main/examples/test_optimized_no_streaming.py

logging.basicConfig(level=logging.INFO) #ignore everything use (level=logging.CRITICAL + 1)


# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")


BROCAS_AUDIO_TEMPLATE = {
    'transcript': "",  # what is being spoken
    'audio_data': None,  # np.ndarray - raw data of the audio
    'samplerate': 24000,  # hz of sample - smaller, slower/lower - bigger faster/higher
    'num_channels': 1  # Mono audio from Kokoro
}
"""
BrocasArea_kokoro_playback() [separate process]
  ├─> Wait for audio from internal_play_queue
  ├─> Update status dict (is_playing=True, transcript, etc.)
  ├─> Play audio via sounddevice
  ├─> Loop while playing:
  │     ├─> Check stop_dict (non-blocking)
  │     │     └─> If 'stop': sd.stop(), clear queues, clear status
  │     ├─> Check if stream still active
  │     │     └─> If done: clear status, break
  │     └─> Sleep 10ms (CPU efficiency)
  └─> Continue to next audio chunk

  stop_playback()
  └─> Update 'stop' signal in stop_dict
        └─> Playback process receives signal
              ├─> Calls sd.stop()
              ├─> Clears both queues
              └─> Resets status dict

Main Process                    Playback Process
     │                                │
     ├─> synthesize_speech()          │
     │    └─> internal_play_queue ──> │ (get audio)
     │                                │
     │                          Updates status dict
     │                                │
     ├─> is_speaking() <───────── status dict (read-only)
     ├─> get_status()   <───────── status dict (read-only)
     │                                │
     ├─> stop_playback()              │
     │    └─> stop_dict ──────────> │ (receive stop)
     │                                │
     └─> shutdown() ──> shutdown_event ─> │ (exit loop)

"""

"""
SPEECH
"""
def BrocasArea_playback(internal_play_queue, stop_dict, shutdown_event, status_dict):
    """Play audio data using sounddevice.
    
    Args:
        internal_play_queue: Queue for receiving audio data (as dicts)
        stop_dict: dict for receiving a playback stop signal
        shutdown_event: Event to signal shutdown
        status_dict: Shared dict for current playback status
    """
    # PRE-INITIALIZE: Open the default audio device and query its properties
    # This warms up the audio subsystem
    try:
        default_device = sd.query_devices(kind='output')
        logging.debug(f"Pre-initialized audio device: {default_device['name']}")
        
        # Create a dummy silent buffer to "prime" the audio device
        # This forces initialization of internal buffers
        silent_buffer = np.zeros((1024, 1), dtype=np.float32)
        sd.play(silent_buffer, samplerate=24000, blocking=False)
        sd.wait()  # Wait for the silent buffer to finish
        sd.stop()  # Clean up
        
        logging.info("Audio device pre-initialization complete")
    except Exception as e:
        logging.warning(f"Could not pre-initialize audio device: {e}")

    while not shutdown_event.is_set():
        try:
            # Get audio from queue (now a dict)
            audio_template = internal_play_queue.get_nowait()
            
            audio = audio_template['audio_data']
            samplerate = audio_template['samplerate']

            logging.debug(f"Playing audio chunk of length {len(audio)}")

            # Update status_dict with active playback info
            status_dict.update({
                'is_playing': True,
                'transcript': audio_template['transcript'],
                'samplerate': samplerate,
                'num_channels': audio_template['num_channels'],
                'audio_length': len(audio),
                'audo_data' : audio
            })
            
            # Play the audio chunk
            sd.play(audio, samplerate)
            
            # Wait for playback to finish or until stop is requested
            while not shutdown_event.is_set():
                # Check if stop signal received during playback
                try:
                    signal = stop_dict.get('interrupt', False)
                    if signal:
                        print("Stop signal received. Stopping playback.")
                        sd.stop()
                        
                        # Clear status_dict
                        status_dict.update({
                            'is_playing': False,
                            'transcript': "",
                            'audio_length':  0,
                            'audo_data' : np.array([])
                        })
                        
                        # Clear any remaining items from play queues when interrupted
                        while not internal_play_queue.empty():
                            internal_play_queue.get_nowait()
                        #reset stop signal
                        stop_dict.update({'interrupt': False})
                        break
                    
                except queue.Empty:
                    pass
                
                # Check if playback is still active - with safety check
                try:
                    stream = sd.get_stream()
                    if not stream.active:
                        # Clear status_dict when playback finishes naturally
                        status_dict.update({
                            'is_playing': False,
                            'transcript': "",
                            'audio_length': 0,
                            'audo_data' : np.array([])
                        })
                        break
                except sd.PortAudioError:
                    # No active stream, playback finished
                    status_dict.update({
                        'is_playing': False,
                        'transcript': "",
                        'audio_length': 0,
                        'audo_data' : np.array([])
                    })
                    break
                    
                
                
        except queue.Empty:
            time.sleep(0.01) # Check every 10ms, helps CPU strain
        except Exception as e:
            if not shutdown_event.is_set():
                print(f"Playback error: {e}")
                # Clear status_dict on error
                status_dict.update({
                    'is_playing': False,
                    'transcript': "",
                    'audio_length': 0,
                    'audo_data' : np.array([])
                })
            time.sleep(0.01)
    
    # Clean shutdown
    sd.stop()
    logging.debug("Playback process shutting down cleanly")


##############
# MAIN CLASS - Qwen TTS SYSTEM

def preprocess_audio(audio: np.ndarray, original_sr: int, target_sr: int = 24000) -> tuple:
    """
    Preprocess audio to match model requirements:
    - Convert to mono if stereo
    - Resample to target sample rate (24kHz)
    """
    # Convert stereo to mono if needed
    if len(audio.shape) > 1 and audio.shape[1] > 1:
        logging.info(f"Converting stereo to mono...")
        audio = np.mean(audio, axis=1)
    
    # Resample if sample rate doesn't match
    if original_sr != target_sr:
        logging.info(f"Resampling from {original_sr}Hz to {target_sr}Hz...")
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    
    return audio, target_sr


class BrocasArea():
    """
    Qwen TTS Voice Clone implementation of BrocasArea.
    
    Uses Qwen3-TTS-12Hz-1.7B-Base with voice cloning from a voiceSample.wav file
    to produce consistent speech synthesis with a cloned voice.
    
    Same public interface as BrocasArea:
        - synthesize_speech(text, auto_play, ...)
        - play_audio(audio_template)
        - stop_playback()
        - is_speaking()
        - get_status()
        - get_current_transcript()
        - shutdown()
    """

    def __init__(
        self,
        brocas_area_interrupt_dict,
        voice_sample_file: str = "voiceSample.wav",
        language: str = "English",
        device: str = None,
        design_instruct: str = None,
        voice:str = 'af_sky,af_jessica', #ignored for Qwen TTS, used by Kokoro, accepted for consistency and future-proofing
        ref_text: str = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump! Sphinx of black quartz, judge my vow. Waltz, bad nymph, for quick jigs vex.",
        dtype=torch.bfloat16,
    ) -> None:
        print("----------------------------------------------------------------")
        print(" INITIALIZING BROCA'S AREA (Qwen TTS)... ")
        print("----------------------------------------------------------------")

        # Check for internet connection
        _ = self.check_internet()

        self.voice = voice #ignored for Qwen TTS, used by Kokoro, accepted for consistency and future-proofing

        self.voice_sample_file: str = voice_sample_file
        self.language: str = language
        self.sample_rate: int = 24000
        
        # Default reference text for voice cloning
        self.ref_text: str = ref_text

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda:0"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.dtype = dtype

        # Generation parameters for consistency
        self.gen_params = {
            'max_new_tokens': 200, #12hz model can handle longer outputs, but we limit to 200 tokens (~15-20 seconds) for responsiveness
            'temperature': 0.6,
            'top_p': 0.85,
            'top_k': 40,
            'repetition_penalty': 1.0,
        }

        # Initialize Qwen TTS model and voice clone prompt
        self.clone_model = None
        self.voice_clone_prompt = None
        self._initialize_model()

        # Create multiprocessing Manager for shared objects
        self.manager = Manager()

        # Create queues for process communication
        self.stop_dict = brocas_area_interrupt_dict
        self.internal_play_queue = self.manager.Queue(maxsize=15)
        self.shutdown_event = self.manager.Event()

        # Create shared dict for current status
        self.status = self.manager.dict({
            'is_playing': False,
            'transcript': "",
            'samplerate': self.sample_rate,
            'num_channels': 1,
            'audio_length': 0,
            'audo_data': np.array([])
        })

        # Start playback process
        self.playback_process = Process(
            target=BrocasArea_playback,
            args=(self.internal_play_queue, self.stop_dict, self.shutdown_event, self.status),
            daemon=True
        )
        self.playback_process.start()

        # ---------------------------------------------------------------------------
        # Emotion configuration
        # ---------------------------------------------------------------------------

        self.emotion_instructions = {
            "neutral": "Speak naturally and clearly",
            "happy":   "Speak in a happy, upbeat, enthusiastic tone with a smile in your voice.",
            "sad":     "Speak in a sad, soft, slow, melancholic tone with subdued energy.",
            "angry":   "Speak in an angry, tense, clipped, forceful tone with controlled intensity.",
            "excited": "Speaks excited with energy, enthusiasm.",
            "humor":   "Speak in an ammused, playful tone, laughing between sentences and words.",
        }

        self.emotions = list(self.emotion_instructions.keys())  # ['neutral', 'happy', 'sad', 'angry', 'excited', 'humor']

        #Design voice if instruction provided else load default voice sample
        if design_instruct is not None:
            print("=" * 60)
            print("Designing voice with instruction:", design_instruct)
            success = self.design_voice(
                instruct=design_instruct,
                save_as=voice_sample_file,
                regenerate_emotions=False
            )
            if not success:
                logging.error("Voice design failed. Falling back to default voice sample.")
            else:
                logging.info("Voice designed and saved successfully.")


    def check_internet(self):
        """Check if internet connection is available."""
        try:
            socket.create_connection(("huggingface.co", 443), timeout=5)
            return True
        except (socket.timeout, socket.error, OSError):
            os.environ["HF_HUB_OFFLINE"] = "1"
            return False

    def _initialize_model(self) -> None:
        """
        Initialize the Qwen TTS Base model and create the voice clone prompt
        from voiceSample.wav.
        
        This loads the model once and creates a reusable voice_clone_prompt
        so that every call to synthesize_speech uses the same cloned voice.
        """
       
        # ── Load voice sample ──
        if not os.path.exists(self.voice_sample_file):
            raise FileNotFoundError(
                f"Voice sample file not found: {self.voice_sample_file}\n"
                f"Please provide a voiceSample.wav file for voice cloning."
            )

        print(f"Loading voice sample from {self.voice_sample_file}...")
        ref_wavs_array, original_sr = sf.read(self.voice_sample_file)

        # Preprocess audio (mono, 24kHz)
        ref_wavs_array, sr = preprocess_audio(ref_wavs_array, original_sr, target_sr=24000)
        print("✓ Voice sample loaded and preprocessed")

        # ── Load Qwen TTS Base model ──
        print(f"Loading Qwen TTS Base model on {self.device}...")
        self.clone_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map=self.device,
            dtype=self.dtype,
            attn_implementation="sdpa",  # built into PyTorch 2.0+, no install needed
        )
        print("✓ Qwen TTS Base model loaded")
        self.clone_model.model = torch.compile(
            self.clone_model.model, 
            mode="max-autotune"  # Upgrade from reduce-overhead
        )
                

        # ── Create reusable voice clone prompt ──
        print("Creating reusable voice clone prompt...")
        self.voice_clone_prompt = self.clone_model.create_voice_clone_prompt(
            ref_audio=(ref_wavs_array, sr),
            ref_text=self.ref_text,
            x_vector_only_mode=True,  # Speaker embedding only - faster and more consistent
        )
        print("✓ Voice clone prompt created")

        # Then run warmup after optimizations are enabled:
        print("Running warmup passes...")
        warmup_texts = ["Hello.", "Testing one two three.", "Warming up the synthesis pipeline."]
        for t in warmup_texts:
            self.clone_model.generate_voice_clone(
                text=t, language=self.language,
                voice_clone_prompt=self.voice_clone_prompt,
            )
        print("✓ Warmup complete")

    def synthesize_speech(
        self,
        text: str,
        emotion: str = "neutral",
        voice: str = None, #ignored for Qwen TTS, used by Kokoro, accepted for consistency and future-proofing
        auto_play: bool = False,
        **kwargs,
    ) -> Optional[dict]:
        """
        Synthesize speech from text using the appropriate emotional voice clone.

        For ``emotion="neutral"`` (or any unrecognised value) the default voice
        clone prompt (self.voice_clone_prompt) is used.  For any other emotion
        the corresponding pre-baked prompt is used if it exists, otherwise it
        falls back to the neutral prompt with a warning.

        Args:
            text:       Text to synthesize.
            emotion:    One of: neutral | happy | sad | angry | excited | humor.
            auto_play:  If True, audio is queued for playback and None is returned.
            **kwargs:   Override default generation params
                        (temperature, top_p, top_k, max_new_tokens,
                        repetition_penalty).

        Returns:
            Dict with audio data when auto_play=False, else None.
            Dict format::

                {
                    'transcript':   str,
                    'audio_data':   np.ndarray,
                    'samplerate':   int,
                    'num_channels': int,
                }
        """
        if voice is not None:
            logging.warning("The 'voice' parameter is currently ignored in Qwen TTS mode. It is used for Kokoro. Accepted for consistency and future-proofing.")
        if self.clone_model is None or self.voice_clone_prompt is None:
            logging.error("Qwen TTS model not initialised.")
            return None

        # ------ Select voice prompt -----------------------------------------------
        emotion = emotion.lower().strip()

        if emotion == "neutral" or emotion not in self.emotions:
            voice_prompt = self.voice_clone_prompt  # default / neutral
            if emotion not in ("neutral", "") and emotion not in self.emotions:
                logging.warning(
                    f"Unknown emotion '{emotion}' — falling back to neutral."
                )
        else:
            # Use the pre-baked emotional prompt if available
            emotional_prompt = getattr(self, f"voice_clone_prompt_{emotion}", None)
            if emotional_prompt is not None:
                voice_prompt = emotional_prompt
            else:
                logging.warning(
                    f"Emotion prompt for '{emotion}' not generated yet "
                    f"(call generate_vocal_emotions() first). Using neutral."
                )
                voice_prompt = self.voice_clone_prompt

        # ------ Merge generation parameters ---------------------------------------
        gen_params = {**self.gen_params, **kwargs}

        # ------ Generate ----------------------------------------------------------
        try:
            logging.info(
                f"[{emotion}] Generating: '{text[:60]}{'...' if len(text) > 60 else ''}'"
            )

            wavs, sr = self.clone_model.generate_voice_clone(
                text=text,
                language=self.language,
                voice_clone_prompt=voice_prompt,
                max_new_tokens=gen_params["max_new_tokens"],
                temperature=gen_params["temperature"],
                top_p=gen_params["top_p"],
                top_k=gen_params["top_k"],
                repetition_penalty=gen_params["repetition_penalty"],
            )

            audio_data = (
                wavs[0] if isinstance(wavs[0], np.ndarray) else wavs[0].cpu().numpy()
            )

            audio_template = {
                "transcript":   text,
                "audio_data":   audio_data,
                "samplerate":   sr,
                "num_channels": 1,
            }

            if auto_play:
                self.status.update({"is_playing": True})
                try:
                    self.internal_play_queue.put(audio_template)
                except Exception as e:
                    logging.error(f"Failed to queue audio: {e}")
                return None

            return audio_template

        except Exception as e:
            logging.error(f"Speech synthesis failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    # ------------------------------------------------------------------
    # Voice design (change what the AI sounds like)
    # ------------------------------------------------------------------

    def _rebuild_voice_clone_prompt(
        self,
        ref_wavs_array: np.ndarray,
        sr: int,
        ref_text: str,
        base_instruct: str = "Speak with a clear, natural voice.",
        regenerate_emotions: bool = True,
    ) -> None:
        """
        Rebuild self.voice_clone_prompt (and neutral) from a raw audio array.
        Called after design_voice() or whenever the reference audio changes.
        
        Args:
            ref_wavs_array: Raw audio array
            sr: Sample rate
            ref_text: Reference text for voice cloning
            base_instruct: Base voice instruction for emotion generation
            regenerate_emotions: If True, regenerate all emotional voice prompts
        """
        print("Rebuilding voice clone prompt...")
        ref_wavs_array, sr = preprocess_audio(ref_wavs_array, sr, target_sr=24000)
        self.voice_clone_prompt = self.clone_model.create_voice_clone_prompt(
            ref_audio=(ref_wavs_array, sr),
            ref_text=ref_text,
            x_vector_only_mode=True,
        )
        self.voice_clone_prompt_neutral = self.voice_clone_prompt
        print("✓ Voice clone prompt rebuilt — new voice is now active")
        
        # Regenerate all emotional voice prompts to match the new base voice
        if regenerate_emotions:
            print("\nRegenerating emotional voice variants...")
            self.generate_vocal_emotions(
                base_instruct=base_instruct,
                ref_text=ref_text,
            )
            print("✓ Emotional voice prompts regenerated")

    def design_voice(
        self,
        instruct: str = "Speak naturally and clearly.",
        ref_text: str = None,
        save_as: str = "voiceSample.wav",
        regenerate_emotions: bool = True,
    ) -> bool:
        """
        Design a new AI voice using the VoiceDesign model and replace the
        current cloned voice with it.

        Automatically regenerates all emotional voice variants to match the new
        voice identity (unless regenerate_emotions=False).

        Args:
            instruct:  Natural language description of the desired voice.
            ref_text:  Text spoken during generation; defaults to self.ref_text.
            save_as:   File path for the saved voice sample WAV.
            regenerate_emotions: If True, regenerate all emotional voice prompts.

        Returns:
            True on success, False on failure.
        """
        from qwen_tts import Qwen3TTSModel

        ref_text = ref_text or self.ref_text

        try:
            print(f"Loading VoiceDesign model on {self.device}...")
            design_model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                device_map=self.device,
                dtype=self.dtype,
            )

            print(f"Designing voice: '{instruct}'")
            ref_wavs, sr = design_model.generate_voice_design(
                text=ref_text,
                language=self.language,
                instruct=instruct,
            )
            print("✓ Voice designed")

            sf.write(save_as, ref_wavs[0], sr)
            self.voice_sample_file = save_as
            self.ref_text = ref_text
            print(f"✓ Voice sample saved → {save_as}")

            del design_model
            torch.cuda.empty_cache()

            self._rebuild_voice_clone_prompt(
                ref_wavs[0], sr, ref_text,
                base_instruct=instruct,
                regenerate_emotions=regenerate_emotions
            )
            return True

        except Exception as e:
            logging.error(f"Voice design failed: {e}")
            import traceback
            traceback.print_exc()
            return False
 
    # ------------------------------------------------------------------
    # Emotion library
    # ------------------------------------------------------------------

    def generate_vocal_emotions(
        self,
        base_instruct: str = "Speak with a clear, natural voice.",
        ref_text: str = (
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump!"
        ),
        save_dir: str = "emotion_voices/",
    ) -> None:
        """
        Pre-generate one voice sample per emotion using the VoiceDesign model,
        then build a reusable voice_clone_prompt for each emotion.

        The resulting prompts are stored as:
            self.voice_clone_prompt_neutral
            self.voice_clone_prompt_happy
            self.voice_clone_prompt_sad
            self.voice_clone_prompt_angry
            self.voice_clone_prompt_excited
            self.voice_clone_prompt_humor

        Call this once at startup (or after design_voice()) — it runs the
        VoiceDesign model for every emotion and then unloads it, leaving only
        the Base model in memory for fast real-time synthesis.

        Args:
            base_instruct:  Description of the AI's base voice identity
                            (timbre, gender, accent, etc.).  The per-emotion
                            style modifier is appended automatically.
            ref_text:       Text spoken during voice design generation.
                            Longer, phonetically rich sentences produce better
                            voice samples.
            save_dir:       Directory where generated WAV files are saved so
                            they can be reloaded without regenerating.
        """
        

        os.makedirs(save_dir, exist_ok=True)

        print(f"Loading VoiceDesign model on {self.device}...")
        design_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            device_map=self.device,
            dtype=self.dtype,
        )
        print("✓ VoiceDesign model loaded")

        for emotion in self.emotions:
            save_path = os.path.join(save_dir, f"{emotion}.wav")
            instruct  = f"{base_instruct} {self.emotion_instructions[emotion]}"

            print(f"  Designing '{emotion}' voice...")
            ref_wavs, sr = design_model.generate_voice_design(
                text=ref_text,
                language=self.language,
                instruct=instruct,
            )

            # Persist to disk for future reuse
            sf.write(save_path, ref_wavs[0], sr)
            print(f"  ✓ '{emotion}' voice saved → {save_path}")

            # Pre-process and build clone prompt
            audio_arr, sr2 = preprocess_audio(ref_wavs[0], sr, target_sr=24000)
            prompt = self.clone_model.create_voice_clone_prompt(
                ref_audio=(audio_arr, sr2),
                ref_text=ref_text,
                x_vector_only_mode=True,
            )

            # Assign to the matching property
            setattr(self, f"voice_clone_prompt_{emotion}", prompt)
            print(f"  ✓ '{emotion}' clone prompt ready")

        # Unload VoiceDesign model — Base model handles all synthesis
        del design_model
        torch.cuda.empty_cache()
        print("✓ Emotion library complete — VoiceDesign model unloaded")

    def load_vocal_emotions(
        self,
        ref_text: str = (
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump!"
        ),
        save_dir: str = "emotion_voices/",
    ) -> bool:
        """
        Reload voice_clone_prompts from previously saved WAV files without
        running the VoiceDesign model.  Returns True if all six emotions were
        loaded, False if any file is missing (call generate_vocal_emotions()
        in that case).

        Args:
            ref_text:  Same transcript used when the WAVs were generated.
            save_dir:  Directory containing the emotion WAV files.
        """
        all_found = True
        for emotion in self.emotions:
            wav_path = os.path.join(save_dir, f"{emotion}.wav")
            if not os.path.exists(wav_path):
                logging.warning(f"Emotion WAV not found: {wav_path}")
                all_found = False
                continue

            audio_arr, original_sr = sf.read(wav_path)
            audio_arr, sr = preprocess_audio(audio_arr, original_sr, target_sr=24000)

            prompt = self.clone_model.create_voice_clone_prompt(
                ref_audio=(audio_arr, sr),
                ref_text=ref_text,
                x_vector_only_mode=True,
            )
            setattr(self, f"voice_clone_prompt_{emotion}", prompt)
            print(f"  ✓ '{emotion}' prompt loaded from {wav_path}")

        return all_found
    
    #------------------------------------------------------------------

    def stop_playback(self) -> None:
        """Stop the current audio playback."""
        logging.debug("Stopping playback...")
        try:
            self.stop_dict.update({'interrupt': True})
        except Exception as e:
            logging.error(f"Failed to send brocasArea stop signal: {e}")

    def play_audio(self, audio_template: dict) -> None:
        """Queue audio for playback.
        
        Args:
            audio_template: Dict containing audio data to play
            Expected format: {
                'transcript': str,
                'audio_data': np.ndarray,
                'samplerate': int,
                'num_channels': int
            }
        """
        self.internal_play_queue.put(audio_template)

    def is_speaking(self) -> bool:
        """Check if audio is currently playing."""
        logging.debug("status:", self.status.get('is_playing', False))
        return self.status.get('is_playing', False)

    def get_status(self) -> dict:
        """Get the current playback status."""
        return dict(self.status)

    def get_current_transcript(self) -> str:
        """Get the transcript of currently playing audio."""
        return self.status.get('transcript', "")

    def set_generation_params(self, **kwargs) -> None:
        """Update default generation parameters.
        
        Args:
            **kwargs: Any of: temperature, top_p, top_k, 
                      max_new_tokens, repetition_penalty
        """
        valid_keys = {'temperature', 'top_p', 'top_k', 'max_new_tokens', 'repetition_penalty'}
        for key, value in kwargs.items():
            if key in valid_keys:
                self.gen_params[key] = value
                logging.info(f"Updated generation param: {key} = {value}")
            else:
                logging.warning(f"Unknown generation parameter: {key}")

    def shutdown(self) -> None:
        """Cleanly shutdown the TTS system and free GPU memory."""
        logging.debug("Shutting down TTS system...")
        self.stop_playback()
        self.shutdown_event.set()

        # Wait for playback process to finish (with timeout)
        self.playback_process.join(timeout=2.0)

        if self.playback_process.is_alive():
            logging.debug("Warning: Playback process did not shutdown cleanly")
            self.playback_process.terminate()
        else:
            logging.debug("TTS system shutdown complete")

        # Free model from GPU
        if self.clone_model is not None:
            del self.clone_model
            self.clone_model = None
            self.voice_clone_prompt = None
            torch.cuda.empty_cache()

        # Cleanup manager
        self.manager.shutdown()



if __name__ == "__main__":
    
    print("=" * 60)
    print("BrocasArea TTS Demo")
    print("=" * 60)

    # Initialize the TTS system
    print("\n[1] Initializing BrocasArea...")
    stop_dict = Manager().dict({'interrupt': False})
    tts = BrocasArea(stop_dict, voice_sample_file="voiceSample.wav")
    time.sleep(2)
    print("✓ Initialization complete\n")

    try:
        # Demo 1: Design a custom voice and generate emotions
        print("=" * 60)
        print("[2] Demo 1: Design custom voice with emotional variants")
        print("=" * 60)
        print("Designing a warm, friendly voice...")
        success = tts.design_voice(
            instruct="a mature woman's voice, speak with a sultry tone.she speaks softly and seductively, she is interested in an intimate conversation.",
            #ref_text="Hello! I'm excited to demonstrate this amazing text to speech system.",
            save_as="custom_voice.wav",
            regenerate_emotions=True
        )
        if success:
            print("✓ Custom voice designed and emotional variants generated\n")
        else:
            print("✗ Voice design failed, using default voice\n")
        time.sleep(1)

        # Demo 2: Auto-play a short sentence
        print("=" * 60)
        print("[3] Demo 2: Auto-play with status monitoring")
        print("=" * 60)
        text1 = "This is a demonstration of the Brocas Area text to speech system using Qwen TTS voice cloning."
        tts.synthesize_speech(text1, auto_play=True)

        # Monitor status during playback
        print("\nMonitoring playback status:")
        while tts.is_speaking():
            status = tts.get_status()
            print(f"  Playing: {status['is_playing']} | "
                  f"Transcript: '{status['transcript'][:50]}...' | "
                  f"Audio length: {status['audio_length']}")
            time.sleep(0.5)

        print("✓ Playback completed naturally\n")
        time.sleep(1)

        # Demo 3: Premature stop
        print("=" * 60)
        print("[4] Demo 3: Premature stop during playback")
        print("=" * 60)
        text2 = ("This sentence will be interrupted midway through playback. "
                 "We will stop the audio before it finishes speaking this entire message. "
                 "You should not hear the end of this sentence.")
        print(f"Speaking: '{text2}'")
        print("Will stop after 2 seconds...")

        tts.synthesize_speech(text2, auto_play=True)
        time.sleep(2)
        print("\n[Sending stop signal]")
        tts.stop_playback()
        time.sleep(0.5)

        status = tts.get_status()
        print(f"✓ Stopped. Current status: {status}\n")
        time.sleep(1)

        # Demo 4: Generate without playing, then play manually
        print("=" * 60)
        print("[5] Demo 4: Generate audio then play manually")
        print("=" * 60)
        text3 = "This audio was pre-generated and is now being played manually."
        print(f"Generating (not playing): '{text3}'")

        audio_template = tts.synthesize_speech(text3, auto_play=False)

        if audio_template:
            print(f"✓ Generated audio: {len(audio_template['audio_data'])} samples")
            print(f"  Samplerate: {audio_template['samplerate']} Hz")
            print(f"  Duration: {len(audio_template['audio_data']) / audio_template['samplerate']:.2f} seconds")

            print("\nNow playing the pre-generated audio...")
            tts.play_audio(audio_template)

            while tts.is_speaking():
                time.sleep(0.1)

            print("✓ Manual playback completed\n")

        time.sleep(1)

        # Demo 5: Test emotional voice variants
        print("=" * 60)
        print("[6] Demo 5: Emotional voice variants")
        print("=" * 60)
        
        emotions_to_test = tts.emotions  # ['neutral', 'happy', 'sad', 'angry', 'excited', 'humor']
        for emotion in emotions_to_test:
            text = f"This is the {emotion} emotion speaking it should sound distinct yet familiar."
            print(f"Speaking with '{emotion}' emotion: '{text}'")
            tts.synthesize_speech(text, emotion=emotion, auto_play=True)
            
            while tts.is_speaking():
                time.sleep(0.1)
            
            time.sleep(0.5)
        
        print("✓ All emotional variants tested\n")
        time.sleep(1)

        # Demo 6: Save audio to WAV file
        print("=" * 60)
        print("[7] Demo 6: Generate and save audio to WAV file")
        print("=" * 60)
        text5 = "This audio will be saved to a WAV file on your local disk."
        print(f"Generating: '{text5}'")

        audio_template = tts.synthesize_speech(text5, auto_play=False)

        if audio_template:
            output_filename = "brocas_qwen_output.wav"
            print(f"✓ Generated {len(audio_template['audio_data'])} samples")
            print(f"  Duration: {len(audio_template['audio_data']) / audio_template['samplerate']:.2f} seconds")

            sf.write(
                output_filename,
                audio_template['audio_data'],
                audio_template['samplerate'],
                subtype='PCM_16'
            )

            print(f"✓ Saved to: {output_filename}")

            print("\nPlaying back the saved audio to verify...")
            tts.play_audio(audio_template)

            while tts.is_speaking():
                time.sleep(0.1)

            print("✓ Playback verification complete\n")

        # Demo 7: Queue multiple phrases
        print("=" * 60)
        print("[8] Demo 7: Queue multiple short phrases")
        print("=" * 60)
        phrases = [
            "First phrase.",
            "Second phrase.",
            "Third and final phrase."
        ]

        for i, phrase in enumerate(phrases, 1):
            print(f"Queuing phrase {i}: '{phrase}'")
            tts.synthesize_speech(phrase, auto_play=True)

        print("\nWaiting for all phrases to complete...")
        while tts.is_speaking():
            current = tts.get_current_transcript()
            if current:
                print(f"  Currently speaking: '{current}'")
            time.sleep(0.5)

        time.sleep(1)
        if not tts.is_speaking():
            print("✓ All phrases completed\n")

    except KeyboardInterrupt:
        print("\n\n[Interrupted by user]")
    except Exception as e:
        print(f"\n\n[Error during demo: {e}]")
        import traceback
        traceback.print_exc()
    finally:
        print("=" * 60)
        print("[9] Shutting down")
        print("=" * 60)
        tts.shutdown()
        print("\n✓ Demo complete!")
    
    