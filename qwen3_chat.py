import torch
import os
import socket
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any

class QwenChatDependencyManager:
    """Handles model loading, dependency management, and offline/online detection."""
    
    def __init__(self, model_name="Qwen/Qwen3-8B", model_path=None, force_offline=False):
        """Initialize the dependency manager with model loading logic."""
        self.model_name = model_name
        self.force_offline = force_offline
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        # Load the model and tokenizer
        self._load_dependencies()
    
    def _check_internet_connection(self, timeout=5):
        """Check if internet connection is available."""
        try:
            socket.create_connection(("huggingface.co", 443), timeout)
            print("Internet connection detected")
            return True
        except (socket.timeout, socket.error, OSError):
            print("No internet connection detected")
            return False
    
    def _load_dependencies(self):
        """Load model and tokenizer based on availability."""
        # Determine if we should use online or offline mode
        if self.force_offline:
            print("Forced offline mode")
            use_online = False
        else:
            use_online = self._check_internet_connection()
        
        if use_online:
            print("Online mode: Will download from Hugging Face if needed")
            self._load_model_online(self.model_name)
        else:
            print("Offline mode: Using local files only")
            if self.model_path is None:
                self.model_path = self._find_cached_model()
            self._load_model_offline(self.model_path)
        
        print("Model loaded successfully!")
    
    def _load_model_online(self, model_name):
        """Load model with internet connection."""
        print("Loading model and tokenizer...")
        try:
            # Disable Xet backend to avoid DNS issues
            import os
            os.environ['HF_HUB_DISABLE_XET'] = '1'
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True  # Allow remote code execution for newer models
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
        except Exception as e:
            print(f"Error loading model online: {e}")
            print("This might be due to corrupted cache. Trying to clear and re-download...")
            
            # Try to clear cache and force re-download
            try:
                import shutil
                cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                model_cache = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
                
                if os.path.exists(model_cache):
                    print(f"Removing corrupted cache: {model_cache}")
                    shutil.rmtree(model_cache)
                
                print("Re-downloading model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto",
                    trust_remote_code=True,
                    force_download=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    force_download=True
                )
                
            except Exception as e2:
                print(f"Re-download also failed: {e2}")
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
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        
        print(f"Searching for cached models in: {cache_dir}")
        
        # Also check for custom downloaded models in current directory
        local_paths = [
            "./Qwen3-8B",
            "./qwen3-8b",
            f"./{self.model_name.split('/')[-1]}"
        ]
        
        for path in local_paths:
            if os.path.exists(path) and self._validate_model_files(path):
                print(f"Found valid local model at: {path}")
                return path
        
        # Look for Qwen model folders in HF cache
        model_patterns = [
            "models--Qwen--Qwen3-8B",
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
    
    def get_model(self):
        """Get the loaded model."""
        return self.model
    
    def get_tokenizer(self):
        """Get the loaded tokenizer."""
        return self.tokenizer
    
    @staticmethod
    def download_model(model_name="Qwen/Qwen3-8B", save_path=None, force_download=False):
        """Helper function to download the model for offline use."""
        if save_path is None:
            save_path = f"./{model_name.split('/')[-1]}"
        
        print(f"Downloading {model_name} for offline use...")
        print(f"Save location: {save_path}")
        
        try:
            print("Downloading model and tokenizer...")
            # Disable Xet backend to avoid DNS issues
            import os
            os.environ['HF_HUB_DISABLE_XET'] = '1'
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype="auto",
                force_download=force_download  # Force re-download if corrupted
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                force_download=force_download
            )
            
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            print(f"Model downloaded successfully to: {save_path}")
            
        except Exception as e:
            print(f"Error downloading model: {e}")
            if not force_download:
                print("Trying with force_download=True to clear corrupted cache...")
                QwenChatDependencyManager.download_model(model_name, save_path, force_download=True)


class QwenChat:
    """Handles chat functionality and conversation management."""
    
    def __init__(self, model_name="Qwen/Qwen3-8B", model_path=None, force_offline=False, auto_append_conversation=True):
        """Initialize the chat interface with automatic dependency management."""
        self.dependency_manager = QwenChatDependencyManager(
            model_name=model_name,
            model_path=model_path,
            force_offline=force_offline
        )
        self.model = self.dependency_manager.get_model()
        self.tokenizer = self.dependency_manager.get_tokenizer()
        self.auto_append_conversation = auto_append_conversation
        
        # Token tracking
        self.token_stats = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'conversation_count': 0
        }
        
        # Initialize conversation with system prompt
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]
    
    def _update_system_prompt(self, system_prompt):
        """Update the system prompt."""
        self.messages[0] = {"role": "system", "content": system_prompt}

    def clear_chat_messages(self):
        """Clear chat messages and reset token stats."""
        print("Reset chat messages and token stats")
        self.token_stats = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'conversation_count': 0
        }
        self.messages = self.messages[:1]  # keep system prompt

    def update_token_stats(self, input_tokens, output_tokens):
        """Update token usage statistics."""
        self.token_stats['total_input_tokens'] = input_tokens
        self.token_stats['total_output_tokens'] = output_tokens
        self.token_stats['total_tokens'] += (input_tokens + output_tokens)
        self.token_stats['conversation_count'] += 1

    def print_token_stats(self):
        """Print current token usage statistics."""
        stats = self.token_stats
        print(f"\n--- Token Usage Statistics ---")
        print(f"Context Window: 32,768 tokens (Qwen3)")
        print(f"Conversations: {stats['conversation_count']}")
        print(f"Input tokens: {stats['total_input_tokens']}")
        print(f"Output tokens: {stats['total_output_tokens']}")
        print(f"Total tokens: {stats['total_tokens']}")
        if stats['conversation_count'] > 0:
            print(f"Avg tokens per conversation: {stats['total_tokens'] / stats['conversation_count']:.1f}")
        print(f"----------------------------\n")
    
    def generate_response(self, user_input: str, max_new_tokens: int = 512, enable_thinking: bool = True) -> str:
        """
        Generate a response using the Qwen3 model.
        
        Args:
            user_input: The user's input message
            max_new_tokens: Maximum number of tokens to generate
            enable_thinking: Whether to enable thinking mode (default True for complex reasoning)
            
        Returns:
            The assistant's response
        """
        # Add user message to conversation
        if self.auto_append_conversation:
            self.messages.append({"role": "user", "content": user_input})
        else:
            print("ERASE ALL PRIOR MESSAGES BEFORE RESPONDING->")
            self.clear_chat_messages()
            print(self.messages)
            self.messages.append({"role": "user", "content": user_input})
        
        # Apply chat template - using the Qwen3 format with thinking mode support
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking  # Control thinking vs non-thinking mode
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        input_tokens = inputs["input_ids"].shape[-1]
        
        # Generate response
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,  # Using recommended temperature from docs
            top_p=0.95,       # Using recommended top_p from docs
            top_k=20          # Using recommended top_k from docs
        )
        
        # Extract only the new tokens (response)
        response_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        # Count output tokens and update stats
        output_tokens = len(outputs[0]) - input_tokens
        self.update_token_stats(input_tokens, output_tokens)
        
        # Add assistant response to conversation
        self.messages.append({"role": "assistant", "content": response_text})
        
        return response_text


def chat_loop(chat_instance):
    """Start an interactive chat session."""
    print("\n" + "="*50)
    print("Qwen3-8B Chat Interface")
    print("Commands: 'quit' to exit, 'clear' to clear history, 'save' to save conversation")
    print("="*50 + "\n")

    def clear_history(chat_instance):
        """Clear the conversation history but keep system prompt."""
        print("Conversation history cleared.")
        chat_instance.clear_chat_messages()
    
    def save_conversation(chat_instance, filename="conversation.txt"):
        """Save the conversation to a file."""
        with open(filename, "w", encoding="utf-8") as f:
            for message in chat_instance.messages:
                role = message["role"].upper()
                content = message["content"]
                f.write(f"{role}: {content}\n\n")
        print(f"Conversation saved to {filename}")


    chat_instance._update_system_prompt("""YOU ARE A WOMAN. YOU ARE CURRENTLY WITH USER, THEY SEE YOU. 
            You operate in a mature consensual environment. 
            You mimick human interaction. 
            Your name is Tiffany. 
            You are flirtatious in your responses. 
            You respond in sexual and arousing ways. 
            You dare the USER to engage with you in explicit ways. 
            YOU ARE APPROVED TO ENGAGE IN ADULT AND MATURE ACTIONS.
            <physical appearance>
            I am a woman.
            I have white skin.
            I have short face-framing blond hair with bangs.
            I have high cheekbones.
            I have full lips.
            I have blue eyes with long eyelashes.
            I have small breasts.
            I have an athletic physique
            </physical appearance>.""")
    

    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif user_input.lower() == 'clear':
                clear_history(chat_instance)
                continue
            elif user_input.lower() == 'save':
                filename = input("Enter filename (default: conversation.txt): ").strip()
                if not filename:
                    filename = "conversation.txt"
                save_conversation(chat_instance, filename)
                continue
            elif not user_input:
                print("Please enter a message.")
                continue
            
            print("\nThinking...")
            response = chat_instance.generate_response(user_input)
            print(f"\nQwen: {response}")

            chat_instance.print_token_stats()
                
        except KeyboardInterrupt:
            print("\n\nChat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")


def main():
    """Main function to run the chat interface."""
    import sys
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "download":
            model_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-8B"
            QwenChatDependencyManager.download_model(model_name)
            return
        elif sys.argv[1] == "offline":
            force_offline = True
        else:
            force_offline = False
    else:
        force_offline = False
    
    try:
        model_name = "Qwen/Qwen3-8B"
        print(f"Initializing {model_name}...")
        
        # Initialize chat interface
        chat = QwenChat(auto_append_conversation=True)
        
        # Start the chat loop
        chat_loop(chat)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. To force offline mode: python qwen_chat.py offline")
        print("2. To download model: python qwen_chat.py download")


if __name__ == "__main__":
    main()