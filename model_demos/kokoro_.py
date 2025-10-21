import os
import platform
import warnings
import tempfile


from typing import  Optional, Generator, Tuple, Any,Tuple, TYPE_CHECKING
#ONLY for type hints, not actual imports - IDE friendly will set this to true, not at runtime
if TYPE_CHECKING:
    import numpy as np

# Add this at the top of your script, after imports
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
"""Warning 1: RNN Dropout Warning
Issue: The warning about dropout expects num_layers > 1 but got num_layers=1 from the Kokoro model's internal architecture.

Warning 2: Weight Norm Deprecation Warning
Issue: torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.
"""

class kokoroTTS:
    def __init__(self) -> None:
        from kokoro import KPipeline
        import sounddevice as sd
        import soundfile as sf
        import torch

        self.KPipeline = KPipeline
        self.sf = sf
        self.torch = torch
        self.sd = sd
        
        self.pipeline: Optional[Any] = None  # will hold the TTS pipeline once initialized

        # Voice definitions - US English
        VOICES_FEMALE: list[str] = [
            "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", 
            "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"
        ]
        
        VOICES_MALE: list[str] = [
            "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", 
            "am_michael", "am_onyx", "am_puck", "am_santa"
        ]

        self.accent: str = 'a'  # 'a' for American English, 'b' for British English
        self.voice: str = 'af_sky'  # Default
        self.speech_speed: float = 1.0  # Normal speed

    def _initialize_pipeline(self) -> None:
        """Initialize the Kokoro pipeline with MPS fallback for Mac."""
        try:
            # Set MPS fallback for Mac M1/M2/M3/M4
            if platform.system() == "Darwin" and self.torch.backends.mps.is_available():
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            
            # Explicitly specify repo_id to suppress warning
            self.pipeline = self.KPipeline(self.accent, repo_id='hexgrad/Kokoro-82M')

            print("✅ Kokoro pipeline initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Kokoro pipeline: {e}")
            raise

    def synthesize_speech(self, text: str) -> None:
        """Synthesize speech from text and play it through the default audio device."""
        # Create temporary file
        temp_fd: int
        file_path: str
        temp_fd, file_path = tempfile.mkstemp(suffix='.wav', prefix='kokoro_temp_')
        os.close(temp_fd)  # Close file descriptor
        cleanup_needed: bool = True
        print(f"Using temporary file: {file_path}")
            
        # Generate audio
        generator: Generator[Tuple[Any, Any, Optional['np.ndarray']], None, None] = self.pipeline(
            text, voice=self.voice, speed=self.speech_speed
        )
            
        # Process and save audio
        for i, (graphemes, phonemes, audio) in enumerate(generator):
            if audio is not None:
                # Save to file using soundfile
                self.sf.write(file_path, audio, 24000)
                print(f"Audio synthesized and saved to {file_path}")

                # Play audio using sounddevice
                try:
                    data: np.ndarray
                    samplerate: int
                    data, samplerate = self.sf.read(file_path, dtype='float32')
                    self.sd.play(data, samplerate)
                    self.sd.wait()  # Wait until playback is finished
                    print("Playback finished")
                    # Cleanup temporary file
                    if cleanup_needed:
                        os.remove(file_path)
                        print(f"Temporary file {file_path} deleted")
                except Exception as e:
                    os.remove(file_path)
                    print(f"Temporary file {file_path} deleted")
                    print(f"Playback failed: {e}")

if __name__ == "__main__":


    kokoro: kokoroTTS = kokoroTTS()
    kokoro._initialize_pipeline()
    sample_text: str = "Hello, this is a test of the Kokoro text to speech synthesis model."
    kokoro.synthesize_speech(sample_text)