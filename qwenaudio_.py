#https://huggingface.co/Qwen/Qwen-Audio-Chat
#https://github.com/QwenLM/Qwen-Audio
"""
 Qwen-Audio implementation - automatically records 2 seconds and analyzes
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sounddevice as sd
import soundfile as sf
import tempfile
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def main():
    print("Loading Qwen-Audio model...")
    
    # Setup device
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu" 
        dtype = torch.float32
    
    print(f"Using device: {device}")
    
    # Load model - using the correct device_map format
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)
    
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-Audio-Chat",
            device_map="cuda",
            trust_remote_code=True
        ).eval()
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-Audio-Chat",
            device_map="mps",
            trust_remote_code=True,
            torch_dtype=dtype
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-Audio-Chat",
            device_map="cpu",
            trust_remote_code=True
        ).eval()
    
    print("Model loaded! Starting recording...")
    
    # Record 2 seconds of audio
    duration = 2
    sample_rate = 16000  # Changed to 16kHz which is more standard for speech
    print(f"Recording for {duration} seconds... Make sound now!")
    
    audio_data = sd.rec(int(duration * sample_rate), 
                       samplerate=sample_rate, 
                       channels=1, 
                       dtype=np.float32)
    sd.wait()
    print("Recording finished!")
    
    # Save to temporary file
    temp_dir = tempfile.gettempdir()
    #recorded
    audio_path = os.path.join(temp_dir, "recorded_audio.wav")
    sf.write(audio_path, audio_data, sample_rate)
    #existing
    #audio_path = "/Users/home/Documents/code/memoryPage/audio/ex_drums.wav"
    audio_path = "/Users/home/Documents/code/memoryPage/audio/ex_speaking.wav"
    
    # Different types of analysis queries optimized for recorded sounds:
    example_queries = [
            "What sounds do you hear in this audio?",
            "Describe the audio content in detail.",
            "What type of environment or scene does this audio represent?",
            "Identify the source of the sounds in this audio.",
            "What objects or activities are making these sounds?",
            "Is there any speech in this audio?",
            "What is the mood or atmosphere of this audio?",
            "Are there any musical instruments in this audio?",
            "What natural sounds can you identify?",
            "Generate a caption for this audio clip.",
        ]
    try:
        # Use the correct method from the documentation
        query = tokenizer.from_list_format([
            {'audio': audio_path},  # Path to audio file
            {'text': example_queries[0]}
        ])
        
        print("Analyzing audio...")
        
        # Use the model.chat method instead of generate
        response, history = model.chat(tokenizer, query=query, history=None)
        
        print("\n" + "="*50)
        print("AUDIO ANALYSIS:")
        print("="*50)
        print(response.strip())
        print("="*50)
        
        # Optional: Ask a follow-up question
        if response and "english" in response.lower():
            follow_up_response, _ = model.chat(tokenizer, 'What is the spoken text?', history=history)
            print("\nFOLLOW-UP ANALYSIS:")
            print("-" * 30)
            print(follow_up_response.strip())
        
    except Exception as e:
        print(f"Error during audio analysis: {e}")
        print("This might be due to:")
        print("1. Audio file format issues")
        print("2. Model not properly loaded")
        print("3. Insufficient memory")
        
    finally:
        # Cleanup
        print("uncomment if you want to delete the file")
        #if os.path.exists(audio_path):
        #    os.unlink(audio_path)

if __name__ == "__main__":
    main()