### Move SpeechTranscriber to it's own file after testing here
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import numpy as np
import os

class SpeechTranscriber:
    """Handles speech transcription using Whisper with offline support"""
    
    def __init__(self, model_name="openai/whisper-small"):
        self.sample_rate = 16000
        self.model_name = model_name
        
        # Load model and processor with offline fallback
        self.processor, self.model = self._load_models()
    
    def _load_models(self):
        """Load Whisper models with offline support"""
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