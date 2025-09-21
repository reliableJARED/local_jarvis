import threading
import queue
import time
import numpy as np
from collections import deque

#list of class names for YAMNet (AudioSet)
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
    def __init__(self, sample_rate: int = 16000, chunk_duration: float = 1.0, enable_alm: bool = True, alm_threshold: float = 0.3, audio_queue_input=None) -> None:
        """
        Initialize the audio classifier.
        
        Args:
            sample_rate (int): Sample rate for audio capture (YAMNet expects 16kHz)
            chunk_duration (float): Duration of each audio chunk in seconds
            enable_alm (bool): Enable Audio Landscape Monitoring
            alm_threshold (float): Threshold for ALM soundscape change detection
            audio_queue_input: External queue to receive audio data from (if None, creates own audio stream)
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
        
        # Audio Landscape Monitoring (ALM) settings
        self.enable_alm = enable_alm
        self.alm_threshold = alm_threshold
        self.alm_buffer_size = 15
        
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
        
        # Audio buffer queue - use external queue if provided, otherwise create internal one
        self.audio_queue_input = audio_queue_input  # External queue from auditory nerve
        self.audio_queue = queue.Queue() if audio_queue_input is None else None  # Internal queue for self-contained mode
        self.use_external_queue = audio_queue_input is not None
        
        # Control flags
        self.running = False
        
        # Debug counters
        self.callback_count = 0
        self.processing_count = 0
        
        # Initialize Audio Landscape Monitoring if enabled
        if self.enable_alm:
            self._init_audio_landscape_monitor()
        
        print(f"Audio classifier initialized:")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Chunk duration: {self.chunk_duration} seconds")
        print(f"  Chunk size: {self.chunk_size} samples")
        print(f"  External queue mode: {self.use_external_queue}")
        print(f"  ALM enabled: {self.enable_alm}")
        if self.enable_alm:
            print(f"  ALM threshold: {self.alm_threshold}")
            print(f"  ALM buffer size: {self.alm_buffer_size}")
        
        # Test audio devices only if we're creating our own stream
        if not self.use_external_queue:
            self.check_audio_devices()

    def _init_audio_landscape_monitor(self):
        """Initialize Audio Landscape Monitoring components"""
        # Buffer for storing embeddings for ALM
        self.embedding_buffer = deque(maxlen=self.alm_buffer_size)
        
        # Audio accumulator for building 1-second clips for ALM
        self.alm_audio_accumulator = []
        self.alm_target_samples = self.chunk_size  # 1 second worth of samples
        
        # ALM statistics
        self.soundscape_changes = 0
        self.last_soundscape_change = None
        
        print("Audio Landscape Monitor initialized")

    def _extract_embedding_from_model_output(self, audio_chunk):
        """Extract embedding from YAMNet model for ALM analysis"""
        try:
            # Convert numpy array to torch tensor
            audio_tensor = self.torch.from_numpy(audio_chunk).float()
            
            # Ensure audio is the right length and format
            if len(audio_tensor) != self.chunk_size:
                if len(audio_tensor) < self.chunk_size:
                    # Pad with zeros
                    pad_amount = self.chunk_size - len(audio_tensor)
                    audio_tensor = self.torch.nn.functional.pad(audio_tensor, (0, pad_amount))
                else:
                    # Truncate
                    audio_tensor = audio_tensor[:self.chunk_size]
            
            # Normalize to [-1, 1] range if needed
            if self.torch.max(self.torch.abs(audio_tensor)) > 1.0:
                audio_tensor = audio_tensor / self.torch.max(self.torch.abs(audio_tensor))
            
            # Preprocess for YAMNet
            if not self.use_converter or self.converter is None:
                input_tensor = self.manual_preprocess(audio_tensor)
            else:
                try:
                    batched_audio = audio_tensor.unsqueeze(0)
                    input_tensor = self.converter(batched_audio, self.sample_rate)
                except Exception:
                    input_tensor = self.manual_preprocess(audio_tensor)
                    self.use_converter = False
            
            # Ensure correct batch dimension
            if len(input_tensor.shape) == 2:
                input_tensor = input_tensor.unsqueeze(0)
            elif len(input_tensor.shape) == 1:
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
            
            # Get embeddings from YAMNet
            with self.torch.no_grad():
                embeddings, _ = self.model(input_tensor)
                
                # FIX: Properly process the embeddings
                # embeddings shape is [1, 1024, 1, 1] - we need to flatten and squeeze
                embedding_np = embeddings.cpu().numpy()
                
                # Remove all singleton dimensions and flatten to 1D
                embedding_np = embedding_np.squeeze()  # Remove dimensions of size 1
                
                # Ensure it's 1D
                if len(embedding_np.shape) > 1:
                    embedding_np = embedding_np.flatten()
                
                return embedding_np
                
        except Exception as e:
            print(f"Error extracting embedding for ALM: {e}")
            return None
        
    def _calculate_embedding_distance(self, embedding1, embedding2):
        """Calculate cosine distance between two embeddings using numpy only"""
        try:
            # Ensure embeddings are numpy arrays
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            # Calculate cosine similarity: (A · B) / (||A|| * ||B||)
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            # Avoid division by zero
            if norm1 == 0 or norm2 == 0:
                return 1.0  # Maximum distance if either vector is zero
            
            cosine_similarity = dot_product / (norm1 * norm2)
            
            # Clamp similarity to [-1, 1] range to handle numerical errors
            cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
            
            # Convert similarity to distance: distance = 1 - similarity
            # This gives us a range of [0, 2] where 0 = identical, 2 = opposite
            cosine_distance = 1.0 - cosine_similarity
            
            return cosine_distance
            
        except Exception as e:
            print(f"Error calculating embedding distance: {e}")
            return 0.0

    def _process_alm_audio_chunk(self, audio_chunk):
        """Process audio chunk for Audio Landscape Monitoring"""
        if not self.enable_alm:
            return False
            
        # Add chunk to ALM accumulator
        self.alm_audio_accumulator.extend(audio_chunk)
        
        soundscape_change_detected = False
        
        # Check if we have enough samples for ALM analysis
        while len(self.alm_audio_accumulator) >= self.alm_target_samples:
            # Extract 1-second clip for ALM
            alm_clip = np.array(self.alm_audio_accumulator[:self.alm_target_samples])
            
            # Remove processed samples from accumulator
            self.alm_audio_accumulator = self.alm_audio_accumulator[self.alm_target_samples:]
            
            # Use the same classification method to get consistent embeddings
            embeddings, class_predictions, class_names = self.classify_audio(alm_clip)
            
            if embeddings is not None:
                # Extract and process the embedding properly
                embedding_np = embeddings.cpu().numpy().squeeze().flatten()
                
                # Check for soundscape change if we have previous embeddings
                if len(self.embedding_buffer) > 0:
                    # Calculate distances to all previous embeddings
                    distances = []
                    for prev_embedding in self.embedding_buffer:
                        distance = self._calculate_embedding_distance(embedding_np, prev_embedding)
                        distances.append(distance)
                    
                    # Check if minimum distance exceeds threshold
                    min_distance = min(distances)
                    if min_distance > self.alm_threshold:
                        soundscape_change_detected = True
                        self.soundscape_changes += 1
                        self.last_soundscape_change = time.time()
                        print(f"SOUND SCAPE CHANGE DETECTED! Min distance: {min_distance:.3f} > {self.alm_threshold}")
                        print(f"Trigger clip contains:{class_names} with probability: {class_predictions}")
                        print(f"Total soundscape changes: {self.soundscape_changes}")
                
                # Add current embedding to buffer
                self.embedding_buffer.append(embedding_np)
        
        return soundscape_change_detected

    def get_alm_stats(self):
        """Get Audio Landscape Monitoring statistics"""
        if not self.enable_alm:
            return None
            
        return {
            'soundscape_changes': self.soundscape_changes,
            'last_soundscape_change': self.last_soundscape_change,
            'embedding_buffer_size': len(self.embedding_buffer),
            'alm_threshold': self.alm_threshold,
            'alm_enabled': self.enable_alm
        }
    
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
                # Get audio chunk with timeout from appropriate queue
                if self.use_external_queue:
                    # Get from external auditory nerve queue - expecting dict format
                    audio_data_dict = self.audio_queue_input.get(timeout=1.0)
                    # Extract audio frame from the dict structure
                    audio_chunk = audio_data_dict.get('audio_frame', audio_data_dict)
                    if isinstance(audio_chunk, dict):
                        # Handle nested dict structure if needed
                        audio_chunk = audio_chunk.get('audio_frame', audio_chunk)
                else:
                    # Get from internal queue - expecting raw numpy array
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
                
                # Process for Audio Landscape Monitoring (ALM)
                if self.enable_alm:
                    alm_change = self._process_alm_audio_chunk(audio_chunk)
                    if alm_change:
                        print(f" \n#\n#\n#\n[ALM] Soundscape change detected in chunk #{self.processing_count}\n#\n#\n#\n#\n")
                
                # Classify the audio
                embeddings, probabilities, class_names = self.classify_audio(audio_chunk)
                
                if probabilities is not None:
                    # Print results
                    print("\n" + "="*50)
                    print(f"Audio Classification Results (Chunk #{self.processing_count}):")
                    if self.enable_alm:
                        alm_stats = self.get_alm_stats()
                        print(f"ALM Stats: Changes={alm_stats['soundscape_changes']}, Buffer={alm_stats['embedding_buffer_size']}")
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
                
                # Mark task done only for internal queue
                if not self.use_external_queue:
                    self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Error processing audio: {e}")
                import traceback
                traceback.print_exc()

    def start(self, use_own_stream: bool = None) -> None:
        """
        Start real-time audio classification.
        
        Args:
            use_own_stream: If True, creates own audio stream. If False, uses external queue.
                          If None, auto-detects based on audio_queue_input parameter.
        """
        print("\nStarting real-time audio classification...")
        if self.enable_alm:
            print("Audio Landscape Monitoring (ALM) is ENABLED")
        
        # Determine stream mode
        if use_own_stream is None:
            use_own_stream = not self.use_external_queue
        
        if use_own_stream:
            print("Mode: Creating own audio stream")
        else:
            print("Mode: Using external audio queue")
            
        print("Press Ctrl+C to stop\n")
        
        self.running = True
        
        # Start processing thread
        process_thread = threading.Thread(target=self.process_audio)
        process_thread.daemon = True
        process_thread.start()
        
        try:
            if use_own_stream and not self.use_external_queue:
                # Original mode: create own audio stream
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
                                status_msg = f"\n[STATUS] Callbacks: {self.callback_count}, Processed: {self.processing_count}, Queue: {self.audio_queue.qsize()}"
                                if self.enable_alm:
                                    alm_stats = self.get_alm_stats()
                                    status_msg += f", ALM Changes: {alm_stats['soundscape_changes']}"
                                print(status_msg)
            else:
                # External queue mode: just wait for audio data from external source
                print("Waiting for audio data from external queue...")
                print("Make sure your auditory nerve worker is running and feeding the queue!")
                
                # Keep the main thread alive and show status
                while self.running:
                    time.sleep(1)
                    
                    # Print periodic status for external queue mode
                    if self.processing_count > 0:
                        if self.processing_count % 10 == 0:  # Every 10 processed chunks
                            status_msg = f"\n[STATUS] Processed: {self.processing_count}"
                            if self.use_external_queue:
                                try:
                                    queue_size = self.audio_queue_input.qsize()
                                    status_msg += f", External Queue: {queue_size}"
                                except:
                                    status_msg += ", External Queue: N/A"
                            if self.enable_alm:
                                alm_stats = self.get_alm_stats()
                                status_msg += f", ALM Changes: {alm_stats['soundscape_changes']}"
                            print(status_msg)
                    
        except KeyboardInterrupt:
            print("\nStopping audio classification...")
        except Exception as e:
            print(f"[ERROR] Error in audio processing: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            if self.enable_alm:
                final_alm_stats = self.get_alm_stats()
                print(f"\nFinal ALM stats: {final_alm_stats}")
            print(f"\nFinal stats: Processed: {self.processing_count}")
            if hasattr(self, 'callback_count'):
                print(f"Callbacks: {self.callback_count}")


if __name__ == "__main__":
    """Main function to run the audio classifier."""
    try:
        # Example 1: Standalone mode (creates own audio stream)
        print("=== STANDALONE MODE ===")
        classifier_standalone = AudioClassifier(
            sample_rate=16000,      # YAMNet expects 16kHz
            chunk_duration=1.0,     # 1 second chunks
            enable_alm=True,        # Enable Audio Landscape Monitoring
            alm_threshold=0.3,      # ALM threshold for soundscape changes
            audio_queue_input=None  # No external queue - create own stream
        )
        
        # Uncomment to run standalone mode:
        classifier_standalone.start()
        
        # Example 2: External queue mode (for integration with auditory nerve system)
        print("\n=== EXTERNAL QUEUE MODE EXAMPLE ===")
        print("To use with your auditory nerve system:")
        print("audio_nerve_queue = multiprocessing.Queue()")
        print("classifier = AudioClassifier(")
        print("    sample_rate=16000,")
        print("    chunk_duration=1.0,") 
        print("    enable_alm=True,")
        print("    alm_threshold=0.3,")
        print("    audio_queue_input=audio_nerve_queue  # Pass your existing queue")
        print(")")
        print("classifier.start()  # Will use external queue, no own audio stream")
        
        # Example setup for external queue mode:
        # audio_nerve_queue = queue.Queue()  # This would be your existing queue
        # classifier_external = AudioClassifier(
        #     sample_rate=16000,
        #     chunk_duration=1.0,
        #     enable_alm=True,
        #     alm_threshold=0.3,
        #     audio_queue_input=audio_nerve_queue
        # )
        # classifier_external.start()
        
    except Exception as e:
        print(f"[ERROR] {e}")
        print("Make sure you have installed the required packages:")
        print("pip install sounddevice numpy torch torch-vggish-yamnet")
        import traceback
        traceback.print_exc()