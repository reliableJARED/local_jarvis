#!/usr/bin/env python3
"""
Simple Modular Real-Time Speech-to-Text System
"""

import torch
import sounddevice as sd
import numpy as np
import time
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from collections import deque
import queue
from typing import Optional, Callable


class WhisperVAD:
    """Simple Real-Time Speech-to-Text System"""
    
    def __init__(self, 
                 model_name: str = "openai/whisper-small",
                 sample_rate: int = 16000,
                 silence_duration: float = 2.0,
                 vad_threshold: float = 0.5,
                 on_transcription: Optional[Callable[[str], None]] = None):
        
        # Configuration
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.silence_duration = silence_duration
        self.vad_threshold = vad_threshold
        self.on_transcription = on_transcription or self._print_transcription
        
        # Audio settings
        self.chunk_size = 1024
        self.audio_queue = queue.Queue()
        
        # Speech detection
        self.audio_buffer = deque(maxlen=int(sample_rate * 3))  # 3 second buffer
        self.speech_audio = deque()
        self.is_speaking = False
        self.last_speech_time = time.time()
        
        # Models (loaded in initialize())
        self.processor = None
        self.whisper_model = None
        self.vad_model = None
        self.device = torch.device('cpu')
        
        self._running = False
        self._stream = None
    
    def _print_transcription(self, text: str):
        """Default transcription handler"""
        if text:
            print(f"Transcription: {text}")
        else:
            print("No transcription result")
    
    def initialize(self):
        """Load all models"""
        print(f"Loading Whisper model: {self.model_name}")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name)
        
        print("Loading VAD model...")
        self.vad_model, _ = torch.hub.load('snakers4/silero-vad', model='silero_vad')
        
        print("Models loaded successfully!")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Handle incoming audio"""
        audio_data = indata.flatten()
        self.audio_queue.put(audio_data.copy())
    
    def _detect_speech(self, audio_chunk):
        """Detect speech using VAD"""
        audio_tensor = torch.from_numpy(audio_chunk).float()
        
        # VAD expects 512 samples
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
        
        return speech_prob > self.vad_threshold
    
    def _transcribe(self, audio_data):
        """Transcribe audio using Whisper"""
        if len(audio_data) == 0:
            return ""
        
        audio_data = audio_data.astype(np.float32)
        
        inputs = self.processor(
            audio_data, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt"
        )
        
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones(inputs["input_features"].shape[:-1], dtype=torch.long)
        
        with torch.no_grad():
            predicted_ids = self.whisper_model.generate(
                inputs["input_features"],
                attention_mask=inputs.get("attention_mask"),
                language="en",
                task="transcribe",
                max_length=448,
                do_sample=False
            )
        
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()
    
    def _process_audio(self):
        """Main audio processing loop"""
        print("Listening for speech...")
        
        while self._running:
            try:
                if not self.audio_queue.empty():
                    audio_chunk = self.audio_queue.get()
                    self.audio_buffer.extend(audio_chunk)
                    
                    # Check for speech when we have enough audio
                    if len(self.audio_buffer) >= 512:
                        recent_audio = np.array(list(self.audio_buffer)[-512:])
                        has_speech = self._detect_speech(recent_audio)
                        
                        if has_speech:
                            self.last_speech_time = time.time()
                            
                            if not self.is_speaking:
                                self.is_speaking = True
                                print("Speech detected - recording...")
                                self.speech_audio.clear()
                            
                            self.speech_audio.extend(audio_chunk)
                        
                        elif self.is_speaking:
                            silence_duration = time.time() - self.last_speech_time
                            
                            if silence_duration >= self.silence_duration:
                                print("Speech ended - transcribing...")
                                
                                if len(self.speech_audio) > 0:
                                    speech_array = np.array(list(self.speech_audio))
                                    transcription = self._transcribe(speech_array)
                                    self.on_transcription(transcription)
                                
                                self.is_speaking = False
                                self.speech_audio.clear()
                                print("Listening...")
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Error: {e}")
    
    def start(self):
        """Start real-time transcription"""
        if not self.processor or not self.whisper_model or not self.vad_model:
            raise RuntimeError("Models not loaded. Call initialize() first.")
        
        print("Starting real-time speech recognition...")
        self._running = True
        
        self._stream = sd.InputStream(
            callback=self._audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype=np.float32
        )
        
        self._stream.start()
        
        try:
            self._process_audio()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop transcription"""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
        print("STT system stopped")
    
    def transcribe_audio(self, audio_data: np.ndarray) -> str:
        """Transcribe audio data directly"""
        if not self.processor or not self.whisper_model:
            raise RuntimeError("Models not loaded. Call initialize() first.")
        
        return self._transcribe(audio_data)


# Utility functions
def list_audio_devices():
    """List available audio devices"""
    print("Available audio devices:")
    print(sd.query_devices())



# Example usage
def main():
    """Simple example"""

    # List audio devices
    list_audio_devices()
    print()
    
    # Custom transcription handler
    def handle_transcription(text):
        if text:
            print(f"🎤 You said: {text}")
            if "poop" in text.lower():
                print("poop command detected!")
        else:
            print("🔇 No speech detected")
    
    # Create and start STT system
    stt = WhisperVAD(
        model_name="openai/whisper-small",
        silence_duration=1.5,
        on_transcription=handle_transcription
    )
    
    stt.initialize()
    stt.start()


if __name__ == "__main__":
    main()