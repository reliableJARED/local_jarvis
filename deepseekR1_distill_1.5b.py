import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

class SimpleDeepSeekChat:
    def __init__(self, show_thoughts=False):
        # Configuration
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.show_thoughts = show_thoughts  # Set this flag in code only
        
        # Setup device
        if torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.bfloat16
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32
        
        print(f"Loading model on {self.device}...")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize with hardcoded system prompt as first user message
        self.messages = [{"role": "user", "content": "You are a helpful assistant"}]
        print("Model loaded! Type 'quit' to exit.")
    
    def generate_response(self, user_input):
        # Add user message
        self.messages.append({"role": "user", "content": user_input})
        
        # Format conversation
        try:
            conversation_text = self.tokenizer.apply_chat_template(
                self.messages, tokenize=False, add_generation_prompt=True
            )
        except:
            # Fallback format
            conversation_text = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in self.messages]) + "\nAssistant:"
        
        # Tokenize and generate
        inputs = self.tokenizer(conversation_text, return_tensors="pt", truncation=True, max_length=4000)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.6,
                    top_p=0.85,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.05
                )
            except:
                # Fallback to deterministic
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):].strip()
        
        # Handle thinking display
        if "</think>" in response:
            think_end = response.find("</think>") + 8
            thinking = response[:think_end]
            final = response[think_end:].strip()
            display = f"{thinking}\n\n{final}" if self.show_thoughts else final
            # Only add the final answer to conversation history
            self.messages.append({"role": "assistant", "content": final})
        else:
            display = response
            self.messages.append({"role": "assistant", "content": response})
        
        return display
    
    def chat(self):
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['quit', 'exit']:
                    break
                if not user_input:
                    continue
                
                response = self.generate_response(user_input)
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    # Set show_thoughts=True to see <think> tags, False to hide them
    chat = SimpleDeepSeekChat(show_thoughts=False)
    chat.chat()