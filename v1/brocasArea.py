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
from kokoro import KPipeline
from typing import Optional, Generator, Tuple, Any
import platform
import os
import socket
import logging
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
BrocasArea_playback() [separate process]
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


class BrocasArea():
    def __init__(self,brocas_area_interrupt_dict,lang_code = 'a',voice='af_sky,af_jessica') -> None:
        # Check for internet connection
        _ = self.check_internet()
        self.KPipeline = KPipeline

        # Voice definitions reference
        # https://github.com/hexgrad/kokoro/tree/main/kokoro.js/voices
        a_VOICES_FEMALE: list[str] = [
            "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", 
            "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"
        ]
        a_VOICES_MALE: list[str] = [
            "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", 
            "am_michael", "am_onyx", "am_puck", "am_santa"
        ]

        b_VOICES_FEMALE: list[str] = [
            'bf_alice','bf_emma','bf_isabella','bf_lily'
        ]
        b_VOICES_MALE: list[str] = [
            'bm_daniel','bm_fable','bm_george','bm_lewis'
        ]
        

        self.lang_code: str = lang_code  # 'a' for American English 'b' is brittish english
        self.voice: str = voice  # Single voice can be requested (e.g. 'af_sky') or multiple voices (e.g. 'af_bella,af_jessica'). If multiple voices are requested, they are averaged.
        self.speech_speed: float = 1.0  # Normal speed

        self.pipeline = None
        self._initialize_pipeline()  # will set pipeline
        
        # Create multiprocessing Manager for shared objects
        self.manager = Manager()
        
        # Create queues for process communication
        self.stop_dict = brocas_area_interrupt_dict#Multiprocessing dict, {'interrupt':True}   #self.manager.Queue(maxsize=1)
        self.internal_play_queue = self.manager.Queue(maxsize=15)#max pending audio chunks from synthesize_speech TTS for playback
        self.shutdown_event = self.manager.Event()
        
        # Create shared dict for current status (updated by BrocasArea_playback process)
        self.status = self.manager.dict({
            'is_playing': False,
            'transcript': "",
            'samplerate': 24000,
            'num_channels': 1,
            'audio_length': 0,
            'audo_data' : np.array([])
        })
        
        # Start playback process
        self.playback_process = Process(
            target=BrocasArea_playback, 
            args=(self.internal_play_queue, self.stop_dict, self.shutdown_event, self.status),
            daemon=True
        )
        self.playback_process.start()

    def check_internet(self):
        """Check if internet connection is available."""
        try:
            socket.create_connection(("huggingface.co", 443), timeout=5)
            return True
        except (socket.timeout, socket.error, OSError):

            os.environ["HF_HUB_OFFLINE"] = "1"
            return False

    def _initialize_pipeline(self) -> None:
        """Initialize the Kokoro pipeline with MPS fallback for Mac.
        https://github.com/hexgrad/kokoro/blob/main/kokoro/pipeline.py
        """

        
        try:
            # Set MPS fallback for Mac M1/M2/M3/M4
            if platform.system() == "Darwin" and torch.backends.mps.is_available():
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            
            self.pipeline = self.KPipeline(self.lang_code, repo_id='hexgrad/Kokoro-82M')
            logging.debug("Kokoro pipeline initialized successfully")
        except Exception as e:
            logging.debug(f"Failed to initialize Kokoro pipeline: {e}")
            raise


    def synthesize_speech(self, text: str, auto_play: bool = False, voice: str = None) -> Optional[dict]:
        """Synthesize speech from text and optionally play it.
        
        Args:
            text: Text to synthesize
            auto_play: If True, automatically queue audio for playback
            
        Returns:
            Dict with complete audio if auto_play=False, None otherwise
            Dict format: {
                'transcript': str,
                'audio_data': np.ndarray,
                'samplerate': int,
                'num_channels': int
            }
        """
        if voice is None:
            voice = self.voice

        # Generate audio
        #https://github.com/hexgrad/kokoro/blob/main/kokoro/pipeline.py#L361
        generator: Generator[Tuple[Any, Any, Optional[np.ndarray]], None, None] = self.pipeline(
            text, voice=voice, speed=self.speech_speed
        )
            
        # Process audio
        audio_data: np.ndarray = np.array([])

        for i, (graphemes, phonemes, audio) in enumerate(generator):
            if audio is not None:
                audio_data = np.concatenate((audio_data, audio))

                if auto_play:
                    try:
                        logging.debug(f"Queuing audio chunk {i}")
                        
                        # Create fresh audio dict for each chunk
                        audio_template = {
                            'transcript': text,  # Full text being synthesized
                            'audio_data': audio,  # This chunk's audio
                            'samplerate': 24000,
                            'num_channels': 1
                        }
                        
                        # IMPORTANT - updating status here, before the internal_play_queue because of the delay when immediately checking status
                        self.status.update({'is_playing': True})
                        
                        # Put the data in queue for playback
                        self.internal_play_queue.put(audio_template)
                        
                    except Exception as e:
                        logging.debug(f"Failed to queue audio: {e}")
                        return None

        if not auto_play:
           
            # Return complete audio data as a single dict
            return {
                'transcript': text,
                'audio_data': audio_data,
                'samplerate': 24000,
                'num_channels': 1
            }

        return None
    
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
        """Check if audio is currently playing.
        
        Returns:
            bool: True if audio is playing, False otherwise.
        """
        logging.debug("status:",self.status.get('is_playing', False))
        return self.status.get('is_playing', False)
    
    def get_status(self) -> dict:
        """Get the current playback status including transcript and other details.
        
        Returns:
            dict: Dictionary containing:
                - is_playing (bool): Whether audio is currently playing
                - transcript (str): Text being spoken
                - samplerate (int): Sample rate of audio
                - num_channels (int): Number of audio channels
                - audio_length (int): Length of current audio chunk
        """
        return dict(self.status)
    
    def get_current_transcript(self) -> str:
        """Get the transcript of currently playing audio.
        
        Returns:
            str: The transcript being spoken, or empty string if not playing.
        """
        return self.status.get('transcript', "")

    def shutdown(self) -> None:
        """Cleanly shutdown the TTS system."""
        logging.debug("Shutting down TTS system...")
        self.stop_playback()
        self.shutdown_event.set()
        
        # Wait for playback process to finish (with timeout)
        self.playback_process.join(timeout=2.0)
        
        if self.playback_process.is_alive():
            logging.debug("Warning: Playback process did not shutdown cleanly")
            self.playback_process.terminate()  # Force terminate if needed
        else:
            logging.debug("TTS system shutdown complete")
        
        # Cleanup manager
        self.manager.shutdown()


if __name__ == "__main__":
    print("=" * 60)
    print("BrocasArea TTS Demo")
    print("=" * 60)
    
    # Initialize the TTS system
    print("\n[1] Initializing BrocasArea...")
    stop_dict = Manager().dict({'interrupt': False})
    tts = BrocasArea(stop_dict)
    time.sleep(7)
    print("✓ Initialization complete\n")

    
    try:
        # Demo 1: Auto-play a short sentence
        print("=" * 60)
        print("[2] Demo 1: Auto-play with status monitoring")
        print("=" * 60)
        text1 = "This is a demonstration of the Brocas Area text to speech system. This will play to completion"
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
        
        # Demo 2: Premature stop
        print("=" * 60)
        print("[3] Demo 2: Premature stop during playback")
        print("=" * 60)
        text2 = ("This sentence will be interrupted midway through playback. "
                 "We will stop the audio before it finishes speaking this entire message. "
                 "You should not hear the end of this sentence.")
        print(f"Speaking: '{text2}'")
        print("Will stop after 2 seconds...")
        
        tts.synthesize_speech(text2, auto_play=True)
        
        # Wait a bit then stop
        time.sleep(2)
        print("\n[Sending stop signal]")
        tts.stop_playback()
        
        # Wait for stop to take effect
        time.sleep(0.5)
        
        status = tts.get_status()
        print(f"✓ Stopped. Current status: {status}\n")
        time.sleep(1)
        
        # Demo 3: Generate without playing, then play manually
        print("=" * 60)
        print("[4] Demo 3: Generate audio then play manually")
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
            
            # Wait for playback to finish
            while tts.is_speaking():
                time.sleep(0.1)
            
            print("✓ Manual playback completed\n")
        
        time.sleep(1)
        
        # Demo 4: Check status when not playing
        print("=" * 60)
        print("[5] Demo 4: Status when idle")
        print("=" * 60)
        status = tts.get_status()
        print(f"Status: {status}")
        print(f"Is speaking: {tts.is_speaking()}")
        print(f"Current transcript: '{tts.get_current_transcript()}'")
        print("✓ All status checks passed\n")
        
        # Demo 5: Multiple rapid plays
        print("=" * 60)
        print("[6] Demo 5: Queue multiple short phrases")
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
        
        # Wait a bit to ensure queue is fully processed
        time.sleep(1)
        if not tts.is_speaking():
            print("✓ All phrases completed\n")

        # Demo 6: Save audio to WAV file
        print("=" * 60)
        print("[6] Demo 6: Generate and save audio to 'WAV' file")
        print("=" * 60)
        text6 = "This audio will be saved to a WAV file on your local disk."
        print(f"Generating: '{text6}'")
        
        audio_template = tts.synthesize_speech(text6, auto_play=False)
        
        if audio_template:
            output_filename = "brocas_output.wav"
            print(f"✓ Generated {len(audio_template['audio_data'])} samples")
            print(f"  Duration: {len(audio_template['audio_data']) / audio_template['samplerate']:.2f} seconds")
            
            # Save to WAV file
            sf.write(
                output_filename,
                audio_template['audio_data'],
                audio_template['samplerate'],
                subtype='PCM_16'
            )
            
            print(f"✓ Saved to: {output_filename}")
            
            # Optionally play it back to verify
            print("\nPlaying back the saved audio to verify...")
            tts.play_audio(audio_template)
            
            while tts.is_speaking():
                time.sleep(0.1)
            
            print("✓ Playback verification complete\n")
        
        a_VOICES_FEMALE: list[str] = [
            "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", 
            "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"
        ]
        a_VOICES_MALE: list[str] = [
                "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", 
                "am_michael", "am_onyx", "am_puck", "am_santa"
            ]
        #Demo 7: Voices
        print("=" * 60)
        for voice in a_VOICES_FEMALE:
            print(f"Voice: {voice}")
            #up to 15 speech synth calls can be queued, there are only 11 voices so no risk of overflow here
            tts.synthesize_speech(f"This is a test of the {voice} voice.", auto_play=True, voice=voice)
            time.sleep(5)  # brief pause between voices to let it finish
            
            
    except KeyboardInterrupt:
        print("\n\n[Interrupted by user]")
    except Exception as e:
        print(f"\n\n[Error during demo: {e}]")
        import traceback
        traceback.print_exc()
    finally:
        # Always cleanup
        print("=" * 60)
        print("[7] Shutting down")
        print("=" * 60)
        tts.shutdown()
        print("\n✓ Demo complete!")

    
    