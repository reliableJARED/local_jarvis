# Load model directly - Optimized for single GPU CUDA with 4-bit quantization
import os
import torch
from threading import Thread
from transformers import AutoProcessor, AutoModelForImageTextToText, TextIteratorStreamer, BitsAndBytesConfig
from PIL import Image


class Qwen3VLChat:
    """Qwen3 Vision-Language Chat with 4-bit quantization for efficient CUDA inference."""
    
    def __init__(self, model_id="huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated"):
        """Initialize the model with 4-bit quantization."""
        self.model_id = model_id
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        print(f"Using device: {self.device}, dtype: {self.dtype}")
        
        # Enable TF32 for faster computation on Ampere+ GPUs
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Initialize model and processor
        self._setup_model()
        
    def _setup_model(self):
        """Load model with 4-bit quantization configuration."""
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_use_double_quant=True,  # Nested quantization for extra memory savings
            bnb_4bit_quant_type="nf4",  # NormalFloat4 - best for LLMs
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        # Load model with 4-bit quantization (~4GB VRAM instead of ~16GB)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto",  # Required for bitsandbytes
            attn_implementation="sdpa",  # Use SDPA (built into PyTorch) instead of flash_attention_2
            low_cpu_mem_usage=True,
        )
        
        # Optimize for inference
        self.model.eval()
        torch.cuda.empty_cache()
    
    def generate(self, prompt, image=None, max_new_tokens=2048, streaming=False):
        """
        Generate a response from the model.
        
        Args:
            prompt: Text prompt
            image: PIL Image or path to image file (optional)
            max_new_tokens: Maximum tokens to generate
            streaming: If True, returns a generator for streaming output
            
        Returns:
            Generated text (or generator if streaming=True)
        """
        # Handle image input
        if isinstance(image, str):
            if os.path.exists(image):
                image = Image.open(image).convert("RGB")
            else:
                raise FileNotFoundError(f"Image not found: {image}")
        
        # Build message content
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        
        # Prepare inputs
        with torch.inference_mode():
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            
            if streaming:
                return self._generate_streaming(inputs, max_new_tokens)
            else:
                return self._generate_standard(inputs, max_new_tokens)
    
    def _generate_standard(self, inputs, max_new_tokens):
        """Non-streaming generation."""
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
        result = self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result
    
    def _generate_streaming(self, inputs, max_new_tokens):
        """Streaming generation with thread-based approach."""
        streamer = TextIteratorStreamer(
            self.processor.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
        )
        
        # Run generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield tokens as they're generated
        for text in streamer:
            yield text
        
        thread.join()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def chat_loop(self):
        """Interactive terminal chat loop with optional image input."""
        print("=" * 50)
        print("Qwen3 VL Chat (type 'quit' or 'exit' to stop)")
        print(f"Running on {self.device} with {self.dtype}")
        print("=" * 50)
        
        while True:
            # Get image path
            image_path = input("\nImage path (or press Enter for no image): ").strip()
            
            if image_path.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            # Validate image path if provided
            if image_path and not os.path.exists(image_path):
                print(f"Error: File not found: {image_path}")
                continue
            
            # Get prompt
            prompt = input("Prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if not prompt:
                print("Error: Please enter a prompt.")
                continue
            
            print("\nThinking...")
            try:
                # Generate with streaming
                print("\nAssistant: ", end="", flush=True)
                for text in self.generate(
                    prompt=prompt,
                    image=image_path if image_path else None,
                    max_new_tokens=1024,
                    streaming=True
                ):
                    print(text, end="", flush=True)
                print()  # Newline at the end
                    
            except Exception as e:
                print(f"\nError: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def demo_inference(self, image_path, prompt):
        """Run a demo inference with the given image and prompt."""
        print(f"\nDemo Inference:")
        print(f"Image: {image_path}")
        print(f"Prompt: {prompt}")
        print(f"Response: ", end="", flush=True)
        
        try:
            response = self.generate(prompt=prompt, image=image_path, max_new_tokens=1024)
            print(response)
            return response
        except Exception as e:
            print(f"Error: {e}")
            return None

if __name__ == "__main__":
    # Initialize the chat model
    chat_model = Qwen3VLChat()
    
    # Optional: Run demo inference
    demo_image_path = r"C:\Users\jared\Documents\code\local_jarvis\xserver\demetra\tartarus\demetra_in_tartarus-p2_a7_f8_c1.png"
    if os.path.exists(demo_image_path):
        chat_model.demo_inference(demo_image_path, "What animal is on the candy?")
    
    # Start interactive chat loop
    chat_model.chat_loop()