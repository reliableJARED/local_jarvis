import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class SimpleDeepSeekChat:
    def __init__(self, show_thoughts=False):
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.show_thoughts = show_thoughts
        
        print(f"Loading model {self.model_name}...")
        
        # Load model and tokenizer (following documentation pattern)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Initialize conversation with system message
        self.messages = [
            {"role": "user", "content": "You are a helpful assistant"},
        ]
        
        print("Model loaded! Type 'quit' to exit.")
    
    def generate_response(self, user_input):
        # Add user message to conversation
        self.messages.append({"role": "user", "content": user_input+"<think>\n"})
        
        # Use the documented approach for chat template
        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=50,
                temperature=0.6,
                #top_p=0.85,
                #do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens (following documentation pattern)
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        # Handle thinking tags if present
        if "</think>" in response:
            think_end = response.find("</think>") + 8
            thinking = response[:think_end]
            final_answer = response[think_end:].strip()
            
            # Show thinking if enabled, otherwise just final answer
            display_response = f"{thinking}\n\n{final_answer}" if self.show_thoughts else final_answer
            
            # Add only final answer to conversation history
            self.messages.append({"role": "assistant", "content": final_answer})
        else:
            display_response = response
            self.messages.append({"role": "assistant", "content": response})
        
        return display_response
    
    def chat(self):
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                if not user_input:
                    continue
                
                response = self.generate_response(user_input)
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

# Quick test function following documentation example
def quick_test():
    """Quick test following the documentation pattern"""
    print("Running quick test...")
    
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=40)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(f"Response: {response}")

if __name__ == "__main__":
    # Uncomment the line below to run a quick test first
    # quick_test()
    
    # Set show_thoughts=True to see <think> tags, False to hide them
    chat = SimpleDeepSeekChat(show_thoughts=False)
    chat.chat()