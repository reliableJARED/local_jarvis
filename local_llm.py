import torch
import os
import urllib.request
import socket
from transformers import AutoTokenizer, AutoModelForCausalLM

class QwenChat:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", model_path=None, force_offline=False):
        """Initialize the Qwen chat model with automatic online/offline detection."""
        
        self.model_name = model_name
        self.force_offline = force_offline
        # token tracking
        self.token_stats = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'conversation_count': 0
        }
        
        # Determine if we should use online or offline mode
        if force_offline:
            print("Forced offline mode")
            use_online = False
        else:
            use_online = self._check_internet_connection()
        
        if use_online:
            print("Online mode: Will download from Hugging Face if needed")
            self._load_model_online(model_name)
        else:
            print("Offline mode: Using local files only")
            if model_path is None:
                model_path = self._find_cached_model()
            self._load_model_offline(model_path)
        
        # Initialize conversation with system prompt
        self.messages = []
        self._add_system_prompt()
        print("Model loaded successfully!")
    
    def _add_system_prompt(self):
        """Add the initial system prompt."""
        system_content = """IMPORTANT INFORMATION:
<short_term_memory>
  <visual>
    <scene> A man is seated in a black office chair in a cozy living room, wearing a red and black plaid shirt. He is holding a can of soda in his lap. The room features a vaulted ceiling with exposed wooden beams, a fireplace, and a sliding glass door that leads to another room. A dark brown or black leather couch is visible in the background, along with a coffee table with a lamp and a stack of blankets. The walls are a cream or beige color, and the floor is covered in beige carpeting.</scene>
    <people_count>1</people_count>
    <faces>['unrecognized_person']</faces>
  </visual>
  <audio>
    <sounds>['unknown_sound']</sounds>
    <speech_detected>False</speech_detected>
    <analysis>This is the sound of a car alarm going off.</analysis>
  </audio>
  <multimodal_data_id>
    <id>20250617_221036_322</id>
  </multimodal_data_id>
  <context>
    <location>"HOME"</location>
    <time>"2025-06-17_22:10:45"</time>
    <activity>"person_present"</activity>
  </context>
</short_term_memory> 
YOU ARE A ROBOT.
This Information provides context for your conversation.
ALWAYS speak in the first person, treat this information as if you experience it directly"""
        
        self.messages = [{"role": "system", "content": system_content}]
    
    def update_token_stats(self, input_tokens, output_tokens):
        """Update token usage statistics."""
        self.token_stats['total_input_tokens'] += input_tokens
        self.token_stats['total_output_tokens'] += output_tokens
        self.token_stats['total_tokens'] += (input_tokens + output_tokens)
        self.token_stats['conversation_count'] += 1

    def print_token_stats(self):
        """Print current token usage statistics."""
        stats = self.token_stats
        print(f"\n--- Token Usage Statistics ---")
        print(f"Context Window: 32,768 tokens (Qwen2.5) extensive inputs exceeding 32,768 tokens, need to utilize YaRN - NOT IMPLEMENTED ")
        print(f"Conversations: {stats['conversation_count']}")
        print(f"Input tokens: {stats['total_input_tokens']}")
        print(f"Output tokens: {stats['total_output_tokens']}")
        print(f"Total tokens: {stats['total_tokens']}")
        if stats['conversation_count'] > 0:
            print(f"Avg tokens per conversation: {stats['total_tokens'] / stats['conversation_count']:.1f}")
        print(f"----------------------------\n")

    def _check_internet_connection(self, timeout=5):
        """Check if internet connection is available."""
        try:
            socket.create_connection(("huggingface.co", 443), timeout)
            print("Internet connection detected")
            return True
        except (socket.timeout, socket.error, OSError):
            print("No internet connection detected")
            return False
    
    def _load_model_online(self, model_name):
        """Load model with internet connection."""
        print("Loading model and tokenizer...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
        except Exception as e:
            print(f"Error loading model online: {e}")
            print("Falling back to offline mode...")
            model_path = self._find_cached_model()
            self._load_model_offline(model_path)
    
    def _load_model_offline(self, model_path):
        """Load model from local files only."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}.\n"
                f"Please either:\n"
                f"1. Connect to internet to download the model automatically\n"
                f"2. Download the model manually using: python {__file__} download\n"
                f"3. Specify the correct local model path"
            )
        
        print(f"Loading model from: {model_path}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                local_files_only=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True
            )
        except Exception as e:
            print(f"Failed to load model from local files: {e}")
            raise
    
    def _find_cached_model(self):
        """Try to find cached model in common Hugging Face cache locations."""
        import platform
        
        # Common cache locations
        if platform.system() == "Windows":
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        else:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        
        print(f"Searching for cached models in: {cache_dir}")
        
        # Also check for custom downloaded models in current directory
        local_paths = [
            "./Qwen2.5-7B-Instruct",
            "./qwen2.5-7b-instruct",
            f"./{self.model_name.split('/')[-1]}"
        ]
        
        for path in local_paths:
            if os.path.exists(path) and self._validate_model_files(path):
                print(f"Found valid local model at: {path}")
                return path
        
        # Look for Qwen model folders in HF cache
        model_patterns = [
            "models--Qwen--Qwen2.5-7B-Instruct",
            f"models--{self.model_name.replace('/', '--')}"
        ]
        
        for pattern in model_patterns:
            model_dir = os.path.join(cache_dir, pattern)
            
            if os.path.exists(model_dir):
                snapshots_dir = os.path.join(model_dir, "snapshots")
                
                if os.path.exists(snapshots_dir):
                    snapshots = os.listdir(snapshots_dir)
                    
                    for snapshot in snapshots:
                        snapshot_path = os.path.join(snapshots_dir, snapshot)
                        
                        if self._validate_model_files(snapshot_path):
                            print(f"Found valid cached model at: {snapshot_path}")
                            return snapshot_path
        
        raise FileNotFoundError(
            f"Could not find a valid cached model for '{self.model_name}'.\n"
            f"Options:\n"
            f"1. Download model: python {__file__} download\n"
            f"2. Connect to internet and let the script download automatically"
        )
    
    def _validate_model_files(self, model_path):
        """Check if a model directory has the required files."""
        if not os.path.exists(model_path):
            return False
        
        required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        model_files = [f for f in os.listdir(model_path) if f.endswith(('.bin', '.safetensors'))]
        
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                return False
        
        return len(model_files) > 0
    
    def generate_response(self, user_input, max_new_tokens=512):
        """Generate a response using the simplified Qwen pattern."""
        
        # Add user message to conversation
        self.messages.append({"role": "user", "content": user_input})
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize and generate
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        input_tokens = model_inputs.input_ids.shape[1]
        print(f"Input tokens: {input_tokens}")
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        
        # Extract only the new tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Count tokens and update stats
        output_tokens = len(generated_ids[0])
        print(f"Output tokens: {output_tokens}")
        self.update_token_stats(input_tokens, output_tokens)
        
        # Add assistant response to conversation
        self.messages.append({"role": "assistant", "content": response})
        
        return response
    
    def clear_history(self):
        """Clear the conversation history but keep system prompt."""
        print("Conversation history cleared.")
        self._add_system_prompt()
    
    def save_conversation(self, filename="conversation.txt"):
        """Save the conversation to a file."""
        with open(filename, "w", encoding="utf-8") as f:
            for message in self.messages:
                role = message["role"].upper()
                content = message["content"]
                f.write(f"{role}: {content}\n\n")
        print(f"Conversation saved to {filename}")
    
    def chat_loop(self):
        """Start an interactive chat session."""
        print("\n" + "="*50)
        print("Qwen2.5-7B-Instruct Chat Interface")
        print("Commands: 'quit' to exit, 'clear' to clear history, 'save' to save conversation")
        print("="*50 + "\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    continue
                elif user_input.lower() == 'save':
                    filename = input("Enter filename (default: conversation.txt): ").strip()
                    if not filename:
                        filename = "conversation.txt"
                    self.save_conversation(filename)
                    continue
                elif not user_input:
                    print("Please enter a message.")
                    continue
                
                print("\nThinking...")
                response = self.generate_response(user_input)
                print(f"\nQwen: {response}")

                self.print_token_stats()
                
            except KeyboardInterrupt:
                print("\n\nChat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.")


def download_model(model_name="Qwen/Qwen2.5-7B-Instruct", save_path=None):
    """Helper function to download the model for offline use."""
    if save_path is None:
        save_path = f"./{model_name.split('/')[-1]}"
    
    print(f"Downloading {model_name} for offline use...")
    print(f"Save location: {save_path}")
    
    try:
        print("Downloading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        print(f"Model downloaded successfully to: {save_path}")
        
    except Exception as e:
        print(f"Error downloading model: {e}")


def main():
    """Main function to run the chat interface."""
    import sys
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "download":
            model_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-7B-Instruct"
            download_model(model_name)
            return
        elif sys.argv[1] == "offline":
            force_offline = True
        else:
            force_offline = False
    else:
        force_offline = False
    
    try:
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        print(f"Initializing {model_name}...")
        
        chat = QwenChat(
            model_name=model_name,
            force_offline=force_offline
        )
        chat.chat_loop()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. To force offline mode: python qwen_chat.py offline")
        print("2. To download model: python qwen_chat.py download")


if __name__ == "__main__":
    main()