import sys
import subprocess
import logging
import numpy as np
from typing import Union, Optional
import os


class VGGishEmbedder:
    """
    A class for generating audio embeddings using the VGGish model.
    Supports both WAV files and sounddevice audio data.
    This version is configured to use CPU only to avoid device conflicts with tensor and numpy returning from gpu/cpu.
    """
    
    def __init__(self, use_local_files: bool = False):
        """
        VGGish embedder for audio embeddings. 128-dimensional embeddings are generated from audio data.
        
        Args:
            use_local_files: Whether to use local model files only
        """
        self.use_local_files_flag = use_local_files
        self.model = None
        self.device = None
        self.torch = None
        self.vggish = None
        self.vggish_input = None
        self.soundfile = None
        self.sounddevice = None
        
        # Initialize dependencies and model
        self._check_and_install_dependencies()
        self._load_model()
    
    def use_local_files(self) -> bool:
        """Check if we should use local files only."""
        return self.use_local_files_flag
    
    def _get_cpu_device(self, torch_module):
        """Force CPU device usage only."""
        return torch_module.device("cpu")
    
    def _check_and_install_dependencies(self) -> None:
        """
        Check if required dependencies are installed and install if missing.
        Stores imported modules as instance attributes.
        """
        offline_mode = self.use_local_files()
        
        # Check and install PyTorch
        try:
            import torch
            self.torch = torch
            # Force CPU usage
            self.device = self._get_cpu_device(torch)
            logging.debug(f"Using device: {self.device}")
        except ImportError:
            if not offline_mode:
                logging.debug("PyTorch not found. Installing torch...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
                try:
                    import torch
                    self.torch = torch
                    # Force CPU usage
                    self.device = self._get_cpu_device(torch)
                    logging.debug(f"Using device: {self.device}")
                except ImportError:
                    raise ImportError("WARNING! Failed to install or import PyTorch")
            else:
                raise ImportError("PyTorch not found and offline mode is enabled")
        
        # Check and install soundfile (for audio I/O)
        try:
            import soundfile
            self.soundfile = soundfile
        except ImportError:
            if not offline_mode:
                logging.debug("soundfile not found. Installing soundfile...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "soundfile"])
                try:
                    import soundfile
                    self.soundfile = soundfile
                except ImportError:
                    raise ImportError("WARNING! Failed to install or import soundfile")
            else:
                raise ImportError("soundfile not found and offline mode is enabled")
        
        # Check and install sounddevice (for real-time audio capture)
        try:
            import sounddevice as sd
            self.sounddevice = sd
        except ImportError:
            if not offline_mode:
                logging.debug("sounddevice not found. Installing sounddevice...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "sounddevice"])
                try:
                    import sounddevice as sd
                    self.sounddevice = sd
                except ImportError:
                    logging.warning("Failed to install sounddevice. Real-time audio capture may not work.")
                    self.sounddevice = None
            else:
                logging.warning("sounddevice not found and offline mode is enabled. Real-time audio capture will not work.")
                self.sounddevice = None

        # Check and install torchvggish
        try:
            from torchvggish import vggish_input, vggish
            self.vggish = vggish
            self.vggish_input = vggish_input
        except ImportError:
            if not offline_mode:
                logging.debug("torchvggish not found. Installing torchvggish...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "torchvggish"])
                try:
                    from torchvggish import vggish_input, vggish
                    self.vggish = vggish
                    self.vggish_input = vggish_input
                except ImportError:
                    raise ImportError("WARNING! Failed to install or import torchvggish")
            else:
                raise ImportError("torchvggish not found and offline mode is enabled")
        
    def _load_model(self):
        """Load and initialize the VGGish model on CPU."""
        if self.vggish is None:
            raise RuntimeError("VGGish dependencies not properly loaded")
        
        # Force CPU usage by setting map_location
        self.model = self.vggish()
        self.model.eval()  # Set to evaluation mode
        
        # Ensure model is on CPU
        self.model = self.model.to(self.device)
        
        # Updated PyTorch 2.1+ way to set defaults (replaces deprecated set_default_tensor_type)
        self.torch.set_default_dtype(self.torch.float32)
        self.torch.set_default_device('cpu')
    
    def _preprocess_audio_data(self, audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Preprocess raw audio data for VGGish input. VGGish expects audio resampled to 16 kHz mono 
        It uses a spectrogram with window size of 25 ms, hop of 10 ms, and maps to 64 mel bins covering 125-7500 Hz
        A stabilized log mel spectrogram is computed by applying log(mel-spectrum + 0.01)
        
        Args:
            audio_data: Raw audio data as numpy array
            sample_rate: Sample rate of the audio data
            
        Returns:
            Preprocessed audio tensor ready for VGGish
        """
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample to 16kHz if needed (basic resampling)
        if sample_rate != 16000:
            # For proper resampling, you might want to use librosa
            logging.warning(f"Audio sample rate is {sample_rate}Hz, but VGGish expects 16kHz. Consider resampling.")
        
        # Use VGGish input preprocessing
        return self.vggish_input.waveform_to_examples(audio_data, sample_rate)
    
    def embed_wav_file(self, wav_path: str) -> np.ndarray:
        """
        Generate embeddings from a WAV file.
        
        Args:
            wav_path: Path to the WAV file
            
        Returns:
            Embeddings array of shape (num_frames, 128)
        """
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Audio file not found: {wav_path}")
        
        # Use VGGish's built-in preprocessing for WAV files
        input_batch = self.vggish_input.preprocess_wav(wav_path)
        
        # Generate embeddings - ensure tensor is on CPU
        with self.torch.no_grad():
            # Fixed: Use proper tensor conversion method
            if isinstance(input_batch, self.torch.Tensor):
                input_tensor = input_batch.detach().clone().to('cpu')
            else:
                input_tensor = self.torch.from_numpy(input_batch).to('cpu')
            embeddings = self.model(input_tensor)

        # Return as numpy array (already on CPU)
        return embeddings.detach().numpy()
    
    def embed_audio_data(self, audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Generate embeddings from raw audio data (e.g., from sounddevice).
        
        Args:
            audio_data: Raw audio data as numpy array
            sample_rate: Sample rate of the audio data
            
        Returns:
            Embeddings array of shape (num_frames, 128)
        """
        # Preprocess the audio data
        input_batch = self._preprocess_audio_data(audio_data, sample_rate)
        
        # Generate embeddings - ensure tensor is on CPU
        with self.torch.no_grad():
            # Fixed: Use proper tensor conversion method
            if isinstance(input_batch, self.torch.Tensor):
                input_tensor = input_batch.detach().clone().to('cpu')
            else:
                input_tensor = self.torch.from_numpy(input_batch).to('cpu')
            embeddings = self.model(input_tensor)
        
        # Return as numpy array (already on CPU)     
        return embeddings.detach().numpy()
    
    def embed_sounddevice_recording(self, duration: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
        """
        Record audio using sounddevice and generate embeddings.
        
        Args:
            duration: Duration of recording in seconds
            sample_rate: Sample rate for recording
            
        Returns:
            Embeddings array of shape (num_frames, 128)
        """
        if self.sounddevice is None:
            raise RuntimeError("sounddevice is not available. Cannot record audio.")
        
        print(f"Recording for {duration} seconds...")
        audio_data = self.sounddevice.rec(
            int(duration * sample_rate), 
            samplerate=sample_rate, 
            channels=1, 
            dtype='float32'
        )
        self.sounddevice.wait()  # Wait until recording is finished
        print("Recording finished.")
        
        # Remove the channel dimension
        audio_data = audio_data.flatten()
        
        return self.embed_audio_data(audio_data, sample_rate)
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model and device.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_loaded": self.model is not None,
            "device": str(self.device) if self.device else "Unknown",
            "vggish_available": self.vggish is not None,
            "sounddevice_available": self.sounddevice is not None,
            "forced_cpu_mode": True,
            "pytorch_version": self.torch.__version__ if self.torch else "Unknown"
        }


# Example usage
if __name__ == "__main__":
    # Initialize the embedder
    embedder = VGGishEmbedder()
    
    print("VGGish Instance args:", embedder.get_model_info())
    
    # Example 1: Embed a WAV file
    # embeddings_from_file = embedder.embed_wav_file("path/to/your/audio.wav")
    # print(f"Embeddings from file shape: {embeddings_from_file.shape}")
    
    # Example 2: Embed sounddevice recording
    # embeddings_from_recording = embedder.embed_sounddevice_recording(duration=3.0)
    # print(f"Embeddings from recording shape: {embeddings_from_recording.shape}")
    
    # Example 3: Embed raw audio data
    # Simulate some audio data
    sample_rate = 16000
    duration = 2.0
    fake_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
    embeddings_from_data = embedder.embed_audio_data(fake_audio, sample_rate)
    print(f"Embeddings from raw data shape: {embeddings_from_data.shape}")