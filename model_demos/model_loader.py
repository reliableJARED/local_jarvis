import torch
from transformers import (
    AutoProcessor, 
    AutoModelForSpeechSeq2Seq, 
    AutoModelForCausalLM,
    YolosImageProcessor, 
    YolosForObjectDetection,
    Qwen2AudioForConditionalGeneration
)
import gc
import time

def cache_models(models) -> None:
    """
    Pre-download multiple HuggingFace models to the cache folder without keeping them in memory.
    """
    
    device = ""
    dtype = ""
    """Detect and configure optimal device and dtype"""
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = "cpu"
        dtype = torch.float32
        print("Using CPU")

    
    for model_name in models:
        try:
            print(f"📥 Downloading model '{model_name}' to cache...")
            
            if model_name == "Qwen/Qwen2-Audio-7B-Instruct":
                model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map="auto",
                    dtype=dtype,
                    low_cpu_mem_usage=True,
                )

            elif model_name == "openai/whisper-base":
                model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
                
            elif model_name == "hustvl/yolos-tiny":
                model = YolosForObjectDetection.from_pretrained(model_name)
                
            elif model_name == "vikhyatk/moondream2":
                model = AutoModelForCausalLM.from_pretrained(
                    "vikhyatk/moondream2",
                    revision="2025-01-09",
                    trust_remote_code=True,
                    device_map={"": device},
                    dtype=dtype
                )
                
            elif model_name == "Qwen/Qwen2.5-7B-Instruct":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map={"": device},
                    dtype=dtype,
                )
                
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            print(f"✅ Model '{model_name}' cached successfully!")
            # Clear from memory immediately after caching

            del model
            
            # force memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all CUDA operations to complete
                # Force CUDA memory defragmentation
                torch.cuda.ipc_collect()
                    
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
               
            # Give the system a moment to fully release memory
            time.sleep(2)
            
            print(f"🧹 Cleared '{model_name}', ready to load next model")
            
        except Exception as e:
            print(f"❌ Failed to cache model '{model_name}': {str(e)}")
    
    print("🎉 All models have been cached to HuggingFace cache folder!")


# Example usage
if __name__ == "__main__":
    # Define your model configurations
    model_configs = [
        "Qwen/Qwen2-Audio-7B-Instruct",
        "openai/whisper-base",
        "hustvl/yolos-tiny", 
        "vikhyatk/moondream2",
        "Qwen/Qwen2.5-7B-Instruct"
    ]
    
    # Cache models
    cache_models(model_configs)