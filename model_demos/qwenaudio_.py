"""
Enhanced Qwen2-Audio implementation with improved audio processing and debugging
Refactored into AudioVLM class structure
"""

import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import sounddevice as sd
import numpy as np
import warnings
import gc
import os
from datetime import datetime
import soundfile as sf
from pathlib import Path

warnings.filterwarnings("ignore")


class AudioVLM:
    """Audio Vision Language Model class for Qwen2-Audio analysis"""
    
    def __init__(self, model_name="Qwen/Qwen2-Audio-7B-Instruct", save_debug_audio=True):
        """
        Initialize the AudioVLM class
        
        Args:
            model_name (str): HuggingFace model name
            save_debug_audio (bool): Whether to save debug audio files
        """
        self.model_name = model_name
        self.save_debug_audio = save_debug_audio
        self.processor = None
        self.model = None
        self.device = None
        self.dtype = None
        
        # Setup device and data type
        self._setup_device()
        
        # Create debug directory if needed
        if self.save_debug_audio:
            self.debug_dir = Path("debug_audio")
            self.debug_dir.mkdir(exist_ok=True)
    
    def _setup_device(self):
        """Setup device and data type based on availability"""
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float16
        else:
            self.device = "cpu" 
            self.dtype = torch.float32
        
        print(f"Using device: {self.device}")
    
    def clear_memory(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def normalize_audio(self, audio_array, target_rms=0.1):
        """
        Normalize audio to target RMS level
        
        Args:
            audio_array (np.ndarray): Input audio array
            target_rms (float): Target RMS level
            
        Returns:
            np.ndarray: Normalized audio array
        """
        current_rms = np.sqrt(np.mean(audio_array**2))
        if current_rms > 0:
            scaling_factor = target_rms / current_rms
            return audio_array * scaling_factor
        return audio_array
    
    def save_audio_for_debug(self, audio_array, sample_rate=16000, prefix="debug_audio"):
        """
        Save audio file for debugging purposes
        
        Args:
            audio_array (np.ndarray): Audio data to save
            sample_rate (int): Sample rate for the audio
            prefix (str): Prefix for the filename
            
        Returns:
            str: Path to saved file or None if failed
        """
        if not self.save_debug_audio:
            return None
            
        try:
            # Generate timestamp filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.debug_dir / f"{prefix}_{timestamp}.wav"
            
            # Save audio file
            sf.write(filename, audio_array, sample_rate)
            print(f"Audio saved for debugging: {filename}")
            return str(filename)
        except Exception as e:
            print(f"Warning: Could not save debug audio: {e}")
            return None
    
    def record_audio(self, duration=3, sample_rate=16000):
        """
        Record audio with improved settings
        
        Args:
            duration (int): Recording duration in seconds
            sample_rate (int): Sample rate for recording
            
        Returns:
            np.ndarray: Recorded and processed audio array or None if failed
        """
        print(f"Recording for {duration} seconds at {sample_rate}Hz... Speak clearly now!")
        
        try:
            # Record with higher quality settings
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32,
                blocking=True
            )
            
            print("Recording finished!")
            audio_array = audio_data.squeeze()
            
            # Check audio quality
            rms = np.sqrt(np.mean(audio_array**2))
            max_amplitude = np.max(np.abs(audio_array))
            
            print(f"Audio quality - RMS: {rms:.4f}, Max: {max_amplitude:.4f}")
            
            if rms < 0.001:
                print("Warning: Very quiet audio detected! Try speaking louder or closer to microphone.")
            
            # Save original audio for debugging if requested
            self.save_audio_for_debug(audio_array, sample_rate, "original")
            
            # Normalize if needed
            if max_amplitude > 0:
                audio_array_normalized = self.normalize_audio(audio_array)
                print(f"Audio normalized - New RMS: {np.sqrt(np.mean(audio_array_normalized**2)):.4f}")
                
                # Save normalized audio for debugging if requested
                self.save_audio_for_debug(audio_array_normalized, sample_rate, "normalized")
                
                return audio_array_normalized
            else:
                print("Warning: No audio signal detected!")
                return None
                
        except Exception as e:
            print(f"Error during recording: {e}")
            return None
    
    def load_model(self):
        """Load the processor and model"""
        self.clear_memory()
        
        if torch.cuda.is_available():
            print(f"GPU Memory before loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        try:
            # Load processor
            print("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            print("Loading model...")
            if self.device == "cuda":
                self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    dtype=self.dtype,
                    low_cpu_mem_usage=True,
                ).eval()
            elif self.device == "mps":
                self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    self.model_name,
                    device_map="mps",
                    dtype=self.dtype,
                    low_cpu_mem_usage=True
                ).eval()
            else:
                self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    self.model_name,
                    device_map="cpu",
                    low_cpu_mem_usage=True
                ).eval()
            
            print(f"Model loaded successfully!")
            if torch.cuda.is_available():
                print(f"GPU Memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            return True
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA Out of Memory Error: {e}")
            print("Falling back to CPU...")
            self.clear_memory()
            self.device = "cpu"
            self.dtype = torch.float32
            
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    self.model_name,
                    device_map="cpu",
                    low_cpu_mem_usage=True
                ).eval()
                print("Successfully loaded on CPU!")
                return True
            except Exception as cpu_error:
                print(f"CPU loading also failed: {cpu_error}")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def analyze_audio(self, audio_array, question="What sounds do you hear in this audio?"):
        """
        Analyze audio with improved error handling
        
        Args:
            audio_array (np.ndarray): Audio data to analyze
            question (str): Question to ask about the audio
            
        Returns:
            str: Analysis response or None if failed
        """
        if self.model is None or self.processor is None:
            print("Model not loaded. Please call load_model() first.")
            return None
            
        try:
            # Use the working conversation format
            conversation = [
                {
                    "role": "system",
                    "content": "You are an expert audio analyst. Provide clear, concise descriptions of what you hear."
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": f"<|AUDIO|>{question}"}
                    ]
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            # Process inputs with explicit sampling rate
            inputs = self.processor(
                text=text, 
                audio=[audio_array], 
                return_tensors="pt", 
                padding=True,
                sampling_rate=16000
            )
            
            # Generate with better parameters
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id if hasattr(self.processor.tokenizer, 'eos_token_id') else None
                )
                
                # Extract only new tokens
                generated_ids = generated_ids[:, inputs.input_ids.size(1):]
                response = self.processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True
                )[0]
                
            return response.strip()
            
        except Exception as e:
            print(f"Error in audio analysis: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_consistency(self, audio_array, num_tests=3, question="What sounds do you hear in this audio?"):
        """
        Test model consistency across multiple runs
        
        Args:
            audio_array (np.ndarray): Audio data to analyze
            num_tests (int): Number of consistency tests to run
            question (str): Question to ask about the audio
            
        Returns:
            list: List of results from each test
        """
        results = []
        
        print(f"\n{'='*60}")
        print(f"CONSISTENCY TEST - Running {num_tests} times")
        print(f"{'='*60}")
        
        for i in range(num_tests):
            print(f"\nTest {i+1}/{num_tests}")
            print("-" * 20)
            result = self.analyze_audio(audio_array, question)
            results.append(result)
            if result:
                print(f"Result: {result}")
            else:
                print("Result: Failed to generate response")
            
        print(f"\n{'='*40}")
        print("CONSISTENCY SUMMARY:")
        print(f"{'='*40}")
        for i, result in enumerate(results):
            print(f"Run {i+1}: {result if result else 'FAILED'}")
            
        return results
    
    def run_multiple_questions(self, audio_array, questions=None):
        """
        Run multiple specific questions on the audio
        
        Args:
            audio_array (np.ndarray): Audio data to analyze
            questions (list): List of questions to ask. If None, uses default questions.
            
        Returns:
            dict: Dictionary mapping questions to responses
        """
        if questions is None:
            questions = [
                "Is there any speech in this audio?",
                "What type of sounds are present?", 
                "Describe the audio environment and background.",
                "Generate a detailed caption for this audio clip."
            ]
        
        results = {}
        
        print(f"\n{'='*60}")
        print("MULTIPLE QUESTIONS ANALYSIS")
        print(f"{'='*60}")
        
        for i, question in enumerate(questions):
            print(f"\n--- Question {i+1}: {question} ---")
            try:
                response = self.analyze_audio(audio_array, question)
                results[question] = response
                if response:
                    print(f"Answer: {response}")
                else:
                    print("Answer: Failed to generate response")
            except Exception as e:
                print(f"Error with question {i+1}: {e}")
                results[question] = None
        
        return results
    
    def run_full_analysis(self, duration=5):
        """
        Run complete analysis pipeline: record -> analyze -> test
        
        Args:
            duration (int): Recording duration in seconds
            
        Returns:
            dict: Complete analysis results
        """
        print("Enhanced Qwen2-Audio Analysis Tool")
        print("="*50)
        
        # Load model if not already loaded
        if self.model is None:
            if not self.load_model():
                print("Failed to load model. Exiting.")
                return None
        
        print(f"\nModel ready! Debug audio saving: {'ON' if self.save_debug_audio else 'OFF'}")
        
        # Record audio
        audio_array = self.record_audio(duration=duration, sample_rate=16000)
        
        if audio_array is None:
            print("Recording failed. Exiting.")
            return None
        
        print(f"Audio shape: {audio_array.shape}")
        
        results = {}
        
        try:
            # Method 1: Enhanced conversation format
            print(f"\n{'='*60}")
            print("METHOD 1: Enhanced conversation format")
            print(f"{'='*60}")
            
            self.clear_memory()
            
            response = self.analyze_audio(audio_array)
            results['main_analysis'] = response
            
            if response:
                print("AUDIO ANALYSIS (Enhanced Method):")
                print("-" * 40)
                print(response)
            else:
                print("Failed to generate response")
            
            # Method 2: Multiple specific questions
            results['multiple_questions'] = self.run_multiple_questions(audio_array)
            
            # Method 3: Consistency test
            results['consistency_test'] = self.test_consistency(audio_array, num_tests=3)
            
        except Exception as e:
            print(f"Error during audio analysis: {e}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
            
        finally:
            self.clear_memory()
            print(f"\n{'='*50}")
            print("Analysis complete! Memory cleared.")
            if self.save_debug_audio:
                print("Check the 'debug_audio' folder for saved audio files.")
        
        return results


def main():
    """Main function to demonstrate AudioVLM usage"""
    # Check if soundfile is available for audio saving
    try:
        import soundfile as sf
    except ImportError:
        print("Warning: soundfile not installed. Audio saving will be disabled.")
        print("Install with: pip install soundfile")
    
    # Initialize AudioVLM
    audio_vlm = AudioVLM(save_debug_audio=True)
    
    # Run full analysis
    results = audio_vlm.run_full_analysis(duration=5)
    
    if results:
        print("\nFull analysis completed successfully!")
    else:
        print("\nAnalysis failed.")


if __name__ == "__main__":
    main()