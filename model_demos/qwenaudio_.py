"""
Enhanced Qwen2-Audio implementation with improved audio processing and debugging
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

def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def normalize_audio(audio_array, target_rms=0.1):
    """Normalize audio to target RMS level"""
    current_rms = np.sqrt(np.mean(audio_array**2))
    if current_rms > 0:
        scaling_factor = target_rms / current_rms
        return audio_array * scaling_factor
    return audio_array

def save_audio_for_debug(audio_array, sample_rate=16000, prefix="debug_audio"):
    """Save audio file for debugging purposes"""
    try:
        # Create debug directory if it doesn't exist
        debug_dir = Path("debug_audio")
        debug_dir.mkdir(exist_ok=True)
        
        # Generate timestamp filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = debug_dir / f"{prefix}_{timestamp}.wav"
        
        # Save audio file
        sf.write(filename, audio_array, sample_rate)
        print(f"Audio saved for debugging: {filename}")
        return str(filename)
    except Exception as e:
        print(f"Warning: Could not save debug audio: {e}")
        return None

def record_audio_improved(duration=3, sample_rate=16000, save_debug=False):
    """Improved audio recording with better settings"""
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
        if save_debug:
            save_audio_for_debug(audio_array, sample_rate, "original")
        
        # Normalize if needed
        if max_amplitude > 0:
            audio_array_normalized = normalize_audio(audio_array)
            print(f"Audio normalized - New RMS: {np.sqrt(np.mean(audio_array_normalized**2)):.4f}")
            
            # Save normalized audio for debugging if requested
            if save_debug:
                save_audio_for_debug(audio_array_normalized, sample_rate, "normalized")
            
            return audio_array_normalized
        else:
            print("Warning: No audio signal detected!")
            return None
            
    except Exception as e:
        print(f"Error during recording: {e}")
        return None

def analyze_audio_improved(processor, model, audio_array, question="What sounds do you hear in this audio?"):
    """Improved audio analysis with better error handling"""
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
        text = processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        # Process inputs with explicit sampling rate
        inputs = processor(
            text=text, 
            audio=[audio_array], 
            return_tensors="pt", 
            padding=True,
            sampling_rate=16000
        )
        
        # Generate with better parameters
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=processor.tokenizer.eos_token_id if hasattr(processor.tokenizer, 'eos_token_id') else None
            )
            
            # Extract only new tokens
            generated_ids = generated_ids[:, inputs.input_ids.size(1):]
            response = processor.batch_decode(
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

def test_consistency(processor, model, audio_array, num_tests=3):
    """Test model consistency across multiple runs"""
    results = []
    question = "What sounds do you hear in this audio?"
    
    print(f"\n{'='*60}")
    print(f"CONSISTENCY TEST - Running {num_tests} times")
    print(f"{'='*60}")
    
    for i in range(num_tests):
        print(f"\nTest {i+1}/{num_tests}")
        print("-" * 20)
        result = analyze_audio_improved(processor, model, audio_array, question)
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

def main():
    print("Enhanced Qwen2-Audio Analysis Tool")
    print("="*50)
    
    # Configuration
    SAVE_DEBUG_AUDIO = True  # Set to True to save audio files for debugging
    DURATION = 5  # Recording duration in seconds
    
    # Clear any existing memory
    clear_memory()
    
    # Setup device
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu" 
        dtype = torch.float32
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Memory before loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    try:
        # Load processor with explicit sampling rate configuration
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            # Note: feature_extractor_kwargs might not work for all processor types
            # The sampling_rate will be handled in the processor call instead
        )
        
        print("Loading model...")
        if device == "cuda":
            model = Qwen2AudioForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-Audio-7B-Instruct",
                device_map="auto",
                dtype=dtype,
                low_cpu_mem_usage=True,
            ).eval()
        elif device == "mps":
            model = Qwen2AudioForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-Audio-7B-Instruct",
                device_map="mps",
                dtype=dtype,
                low_cpu_mem_usage=True
            ).eval()
        else:
            model = Qwen2AudioForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-Audio-7B-Instruct",
                device_map="cpu",
                low_cpu_mem_usage=True
            ).eval()
        
        print(f"Model loaded successfully!")
        if torch.cuda.is_available():
            print(f"GPU Memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA Out of Memory Error: {e}")
        print("Falling back to CPU...")
        clear_memory()
        device = "cpu"
        dtype = torch.float32
        
        try:
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
            model = Qwen2AudioForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-Audio-7B-Instruct",
                device_map="cpu",
                low_cpu_mem_usage=True
            ).eval()
            print("Successfully loaded on CPU!")
        except Exception as cpu_error:
            print(f"CPU loading also failed: {cpu_error}")
            return
    
    print(f"\nModel ready! Debug audio saving: {'ON' if SAVE_DEBUG_AUDIO else 'OFF'}")
    
    # Record audio with improved method
    audio_array = record_audio_improved(
        duration=DURATION, 
        sample_rate=16000, 
        save_debug=SAVE_DEBUG_AUDIO
    )
    
    if audio_array is None:
        print("Recording failed. Exiting.")
        return
    
    print(f"Audio shape: {audio_array.shape}")
    
    try:
        print(f"\n{'='*60}")
        print("METHOD 1: Enhanced conversation format")
        print(f"{'='*60}")
        
        clear_memory()
        
        response = analyze_audio_improved(processor, model, audio_array)
        
        if response:
            print("AUDIO ANALYSIS (Enhanced Method):")
            print("-" * 40)
            print(response)
        else:
            print("Failed to generate response")
        
        # Multiple specific questions
        print(f"\n{'='*60}")
        print("METHOD 2: Multiple specific questions")
        print(f"{'='*60}")
        
        questions = [
            "Is there any speech in this audio?",
            "What type of sounds are present?", 
            "Describe the audio environment and background.",
            "Generate a detailed caption for this audio clip."
        ]
        
        for i, question in enumerate(questions):
            print(f"\n--- Question {i+1}: {question} ---")
            try:
                response = analyze_audio_improved(processor, model, audio_array, question)
                if response:
                    print(f"Answer: {response}")
                else:
                    print("Answer: Failed to generate response")
            except Exception as e:
                print(f"Error with question {i+1}: {e}")
        
        # Consistency test
        test_consistency(processor, model, audio_array, num_tests=3)
        
    except Exception as e:
        print(f"Error during audio analysis: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        clear_memory()
        print(f"\n{'='*50}")
        print("Analysis complete! Memory cleared.")
        if SAVE_DEBUG_AUDIO:
            print("Check the 'debug_audio' folder for saved audio files.")

if __name__ == "__main__":
    # Check if soundfile is available for audio saving
    try:
        import soundfile as sf
    except ImportError:
        print("Warning: soundfile not installed. Audio saving will be disabled.")
        print("Install with: pip install soundfile")
    
    main()