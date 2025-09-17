
import threading
import queue
import time

from dependency_manager import Orenda_DependencyManager

from yamnet_classes import yamnet_class_names

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    from PIL import Image
    import cv2
    import torch
    import torchaudio
    import sounddevice as sd
    from transformers import AutoModelForCausalLM

class AudioClassifier:
    def __init__(self, sample_rate: int = 16000, chunk_duration: float = 1.0) -> None:

        """
        Initialize the audio classifier.
        
        Args:
            sample_rate (int): Sample rate for audio capture (YAMNet expects 16kHz)
            chunk_duration (float): Duration of each audio chunk in seconds
        """
        import sounddevice as sd
        import numpy as np
        import torch
        import torchaudio
        from torch_vggish_yamnet import yamnet
        from torch_vggish_yamnet.input_proc import WaveformToInput

        self.sd = sd
        self.np = np
        self.torch = torch
        self.torchaudio = torchaudio
        self.yamnet = yamnet
        self.WaveformToInput = WaveformToInput

        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        
        # Initialize YAMNet model
        print("Loading YAMNet model...")
        self.model = yamnet.yamnet(pretrained=True)
        self.model.eval()  # Set to evaluation mode
        
        # Initialize input converter with error handling
        try:
            self.converter = WaveformToInput()
            self.use_converter = True
        except Exception as e:
            print(f"Warning: Could not initialize WaveformToInput converter: {e}")
            self.converter = None
            self.use_converter = False
        
        # Audio buffer queue
        self.audio_queue = queue.Queue()
        
        # Control flags
        self.running = False
        
        # Debug counters
        self.callback_count = 0
        self.processing_count = 0
        
        print(f"Audio classifier initialized:")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Chunk duration: {self.chunk_duration} seconds")
        print(f"  Chunk size: {self.chunk_size} samples")
        
        # Test audio devices
        self.check_audio_devices()
    
    def check_audio_devices(self) -> None:
        """Check available audio input devices."""
        print("\n" + "="*50)
        print("Available Audio Devices:")
        print("-" * 30)
        
        devices = self.sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"Input Device {i}: {device['name']}")
                print(f"  Channels: {device['max_input_channels']}")
                print(f"  Sample Rate: {device['default_samplerate']}")
                print()
        
        # Check default device
        try:
            default_device = self.sd.query_devices(kind='input')
            print(f"Default Input Device: {default_device['name']}")
            print(f"  Max Input Channels: {default_device['max_input_channels']}")
            print(f"  Default Sample Rate: {default_device['default_samplerate']}")
        except Exception as e:
            print(f"Error querying default input device: {e}")
        
        print("="*50 + "\n")

    def manual_preprocess(self, audio_tensor: 'torch.Tensor') -> 'torch.Tensor':
        """
        Manual preprocessing when WaveformToInput converter fails.
        Creates a simple mel spectrogram using torchaudio.
        """
        try:
            
            # Create mel spectrogram transform
            mel_transform = self.torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=512,
                hop_length=160,  # 10ms hop length
                n_mels=64,       # YAMNet typically uses 64 mel bins
                f_min=125.0,
                f_max=7500.0,
            )
            
            # Add batch dimension for the transform
            batched_audio = audio_tensor.unsqueeze(0)  # [1, 16000]
            
            # Compute mel spectrogram
            mel_spec = mel_transform(batched_audio)  # [1, n_mels, time_frames]
            
            # Convert to log scale (dB)
            log_mel_spec = self.torch.log(mel_spec + 1e-6)
            
            # Transpose to match YAMNet expected format [batch, time, features]
            log_mel_spec = log_mel_spec.transpose(1, 2)  # [1, time_frames, n_mels]
            
            print(f"  Manual mel spectrogram shape: {log_mel_spec.shape}")
            return log_mel_spec
            
        except ImportError:
            print("  torchaudio not available, using raw audio...")
            # Fallback: just reshape raw audio
            # This probably won't work well but will let us test the pipeline
            reshaped = audio_tensor.view(1, -1, 1)  # [1, 16000, 1]
            return reshaped
        except Exception as e:
            print(f"  Manual preprocessing failed: {e}")
            # Last resort: minimal reshape
            return audio_tensor.unsqueeze(0).unsqueeze(-1)  # [1, 16000, 1]

    def audio_callback(self, indata: 'np.ndarray', frames: int, time: float, status: 'sd.CallbackFlags') -> None:
        """Callback function for audio input stream."""
        self.callback_count += 1
        
        # Debug: Print callback info every 10 calls
        if self.callback_count % 10 == 1:
            print(f"\n[DEBUG] Callback #{self.callback_count}")
            print(f"  Input shape: {indata.shape}")
            print(f"  Input dtype: {indata.dtype}")
            print(f"  Frames: {frames}")
            print(f"  Expected frames: {self.chunk_size}")
            print(f"  Audio range: [{indata.min():.6f}, {indata.max():.6f}]")
            print(f"  Audio RMS: {self.np.sqrt(self.np.mean(indata**2)):.6f}")
            
        if status:
            print(f"[WARNING] Audio callback status: {status}")
        
        # Debug: Check if we're getting any audio
        audio_level = self.np.sqrt(self.np.mean(indata**2))
        if self.callback_count % 10 == 1:
            if audio_level < 1e-6:
                print(f"[WARNING] Very low audio level detected: {audio_level}")
            else:
                print(f"[INFO] Audio level looks good: {audio_level}")
        
        # Convert to mono if stereo
        if len(indata.shape) > 1 and indata.shape[1] > 1:
            print(f"[DEBUG] Converting stereo to mono (shape: {indata.shape})")
            audio_data = self.np.mean(indata, axis=1)
        else:
            if len(indata.shape) > 1:
                audio_data = indata[:, 0]
            else:
                audio_data = indata
        
        # Debug: Check processed audio
        if self.callback_count % 10 == 1:
            print(f"  Processed audio shape: {audio_data.shape}")
            print(f"  Processed audio dtype: {audio_data.dtype}")
            print(f"  Queue size: {self.audio_queue.qsize()}")
        
        # Add to queue for processing
        try:
            self.audio_queue.put(audio_data.copy(), block=False)
        except queue.Full:
            print("[WARNING] Audio queue is full, dropping frame")

    def classify_audio(self, audio_data: 'np.ndarray') -> tuple:
        """
        Classify audio data using YAMNet.
        
        Args:
            audio_data (np.ndarray): Audio samples
            
        Returns:
            tuple: (embeddings, class_predictions, class_names)
        """
        try:
            print(f"\n[DEBUG] Classifying audio:")
            print(f"  Input shape: {audio_data.shape}")
            print(f"  Input dtype: {audio_data.dtype}")
            print(f"  Input range: [{audio_data.min():.6f}, {audio_data.max():.6f}]")
            print(f"  Input RMS: {self.np.sqrt(self.np.mean(audio_data**2)):.6f}")
            
            # Convert numpy array to torch tensor
            audio_tensor = self.torch.from_numpy(audio_data).float()
            print(f"  Tensor shape: {audio_tensor.shape}")
            print(f"  Tensor dtype: {audio_tensor.dtype}")
            
            # Check if audio tensor is valid
            if audio_tensor.numel() == 0:
                print("[ERROR] Empty audio tensor!")
                return None, None, None
            
            if self.torch.isnan(audio_tensor).any():
                print("[ERROR] NaN values in audio tensor!")
                return None, None, None
            
            if self.torch.isinf(audio_tensor).any():
                print("[ERROR] Infinite values in audio tensor!")
                return None, None, None
            
            # The converter might expect a different format, let's try different approaches
            print(f"  Converting waveform with sample rate: {self.sample_rate}")
            
            if not self.use_converter or self.converter is None:
                print("  Using manual preprocessing (no converter)")
                input_tensor = self.manual_preprocess(audio_tensor)
            else:
                # Try approach 1: Add batch dimension first
                try:
                    batched_audio = audio_tensor.unsqueeze(0)  # Shape: [1, 16000]
                    print(f"  Trying batched input shape: {batched_audio.shape}")
                    input_tensor = self.converter(batched_audio, self.sample_rate)
                    print(f"  Converted tensor shape: {input_tensor.shape}")
                    
                except Exception as e1:
                    print(f"  Batched approach failed: {e1}")
                    
                    # Try approach 2: Original tensor directly
                    try:
                        print(f"  Trying original input shape: {audio_tensor.shape}")
                        input_tensor = self.converter(audio_tensor, self.sample_rate)
                        print(f"  Converted tensor shape: {input_tensor.shape}")
                        
                    except Exception as e2:
                        print(f"  Original approach failed: {e2}")
                        
                        # Try approach 3: Manual preprocessing fallback
                        print("  All converter approaches failed, using manual preprocessing...")
                        input_tensor = self.manual_preprocess(audio_tensor)
                        self.use_converter = False  # Disable converter for future calls
            
            print(f"  Final input tensor shape: {input_tensor.shape}")
            
            # Ensure we have the right batch dimension for the model
            if len(input_tensor.shape) == 2:
                input_tensor = input_tensor.unsqueeze(0)  # Add batch dim: [batch, time, features]
            elif len(input_tensor.shape) == 1:
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and feature dims
            
            # Get predictions
            with self.torch.no_grad():
                print("  Running inference...")
                embeddings, logits = self.model(input_tensor)
                print(f"  Embeddings shape: {embeddings.shape}")
                print(f"  Logits shape: {logits.shape}")
            
            # Get class probabilities
            probabilities = self.torch.softmax(logits, dim=-1)  # Shape: [1, 521]
            print(f"  Probabilities shape: {probabilities.shape}")
            
            # Fix: Don't take mean across classes, take the first (and only) batch
            # Get top 5 predictions from the probability distribution
            batch_probs = probabilities[0]  # Shape: [521] - remove batch dimension
            print(f"  Batch probabilities shape: {batch_probs.shape}")
            
            # Now get top 5 from the 521 classes
            top5_probs, top5_indices = self.torch.topk(batch_probs, 5)
            
            # Get class names (YAMNet uses AudioSet class names)
            class_names = self.get_class_names(top5_indices)
            
            return embeddings, top5_probs, class_names
            
        except Exception as e:
            print(f"[ERROR] Error in classification: {e}")
            print(f"  Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def get_class_names(self, indices: 'torch.Tensor') -> list[str]:
        """
        Get class names for given indices.
        Uses the actual AudioSet class names that correspond to YAMNet's 521 classes.
        """

        # The index of each string should match the class index from the model
        class_names = yamnet_class_names

        # Validate that we have the right number of classes
        if len(class_names) != 521:
            print(f"[WARNING] Expected 521 class names, but got {len(class_names)}")
        
        result_names = []
        for idx in indices:
            idx_int = int(idx.item()) if hasattr(idx, 'item') else int(idx)
            
            if 0 <= idx_int < len(class_names):
                result_names.append(class_names[idx_int])
            else:
                print(f"[WARNING] Index {idx_int} is out of range for class_names (0-{len(class_names)-1})")
                result_names.append(f"Unknown_Class_{idx_int}")
        
        return result_names

    def process_audio(self) -> None:
        """Process audio chunks from the queue."""
        while self.running:
            try:
                # Get audio chunk with timeout
                audio_chunk = self.audio_queue.get(timeout=0.1)
                self.processing_count += 1
                
                print(f"\n[DEBUG] Processing chunk #{self.processing_count}")
                print(f"  Original chunk shape: {audio_chunk.shape}")
                print(f"  Expected chunk size: {self.chunk_size}")
                
                # Ensure chunk is the right size
                if len(audio_chunk) != self.chunk_size:
                    print(f"  [WARNING] Chunk size mismatch! Got {len(audio_chunk)}, expected {self.chunk_size}")
                    
                    # Pad or truncate to match expected size
                    if len(audio_chunk) < self.chunk_size:
                        pad_amount = self.chunk_size - len(audio_chunk)
                        print(f"  Padding with {pad_amount} zeros")
                        audio_chunk = np.pad(audio_chunk, (0, pad_amount))
                    else:
                        print(f"  Truncating to {self.chunk_size} samples")
                        audio_chunk = audio_chunk[:self.chunk_size]
                
                print(f"  Final chunk shape: {audio_chunk.shape}")
                
                # Check for valid audio data
                if self.np.all(audio_chunk == 0):
                    print("  [WARNING] All zeros in audio chunk - skipping")
                    continue
                
                # Classify the audio
                embeddings, probabilities, class_names = self.classify_audio(audio_chunk)
                
                if probabilities is not None:
                    # Print results
                    print("\n" + "="*50)
                    print(f"Audio Classification Results (Chunk #{self.processing_count}):")
                    print("-" * 30)
                    
                    #DEMO: of streaming results in to data pipeline
                    ear_results = {'audio_scape': []}
                    for i, (prob, class_name) in enumerate(zip(probabilities, class_names)):
                        #print(f"{i+1}. {class_name:15s} {prob:.3f}")
                        if prob > 0.3:  # Only log significant probabilities
                            ear_results['audio_scape'].append((class_name))
                    print(ear_results)

                    print("="*50)
                    # Print embedding shape
                    if embeddings is not None:
                        print(f"\nEmbedding shape: {embeddings.shape}")
                else:
                    print(f"[ERROR] Classification failed for chunk #{self.processing_count}")
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Error processing audio: {e}")
                import traceback
                traceback.print_exc()

    def start(self) -> None:
        """Start real-time audio classification."""
        print("\nStarting real-time audio classification...")
        print("Press Ctrl+C to stop\n")
        
        self.running = True
        
        # Start processing thread
        process_thread = threading.Thread(target=self.process_audio)
        process_thread.daemon = True
        process_thread.start()
        
        try:
            # Test audio stream parameters first
            print(f"Testing audio stream with:")
            print(f"  Sample rate: {self.sample_rate}")
            print(f"  Channels: 1")
            print(f"  Block size: {self.chunk_size}")
            print(f"  Data type: float32")
            
            # Start audio stream
            with self.sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=self.np.float32
            ) as stream:
                print(f"\nAudio stream started successfully!")
                print(f"Stream info: {stream}")
                print("Listening for audio... Make some noise!")
                
                # Keep the main thread alive
                while self.running:
                    time.sleep(1)
                    
                    # Print periodic status
                    if hasattr(self, 'callback_count') and self.callback_count > 0:
                        if self.callback_count % 50 == 0:  # Every 5 seconds at 10Hz
                            print(f"\n[STATUS] Callbacks: {self.callback_count}, Processed: {self.processing_count}, Queue: {self.audio_queue.qsize()}")
                    
        except KeyboardInterrupt:
            print("\nStopping audio classification...")
        except Exception as e:
            print(f"[ERROR] Error starting audio stream: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            print(f"\nFinal stats: Callbacks: {self.callback_count}, Processed: {self.processing_count}")


if __name__ == "__main__":
    dep_manager: Orenda_DependencyManager = Orenda_DependencyManager()
    if dep_manager.run(download_models=False):
        print("All dependencies are ready to use!")
        
    """Main function to run the audio classifier."""
    try:
        # Create and start the audio classifier
        classifier = AudioClassifier(
            sample_rate=16000,  # YAMNet expects 16kHz
            chunk_duration=1.0  # 1 second chunks
        )
        
        classifier.start()
        
    except Exception as e:
        print(f"[ERROR] {e}")
        print("Make sure you have installed the required packages:")
        print("pip install sounddevice numpy torch torch-vggish-yamnet")
        import traceback
        traceback.print_exc()