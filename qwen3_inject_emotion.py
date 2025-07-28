import torch
import os
import urllib.request
import socket
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from typing import List, Dict, Any, Optional, Callable
import sys
from character_template import cot_prompt as CHAR_TEMPLATE
from character_template import system_prompt as CHAR_SYSTEM_PROMPT



class QwenChatDependencyManager:
    """Handles model loading, dependency management, and offline/online detection."""
    
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", model_path=None, force_offline=False):
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
    
    def get_model(self):
        """Get the loaded model."""
        return self.model
    
    def get_tokenizer(self):
        """Get the loaded tokenizer."""
        return self.tokenizer
    
    @staticmethod
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



class QwenReasoningChat:
    """Enhanced Qwen3 chat with reasoning interrupts and memory-based external data injection."""
    
    def __init__(self, model_name="Qwen/Qwen3-8B", model_path=None, force_offline=False, 
                 show_thoughts=False, system_prompt=None, auto_append_conversation=True, 
                 cot_injection =""):
        
        """Initialize the chat interface with reasoning injection."""
        self.dependency_manager = QwenChatDependencyManager(
            model_name=model_name,
            model_path=model_path,
            force_offline=force_offline
        )
        self.model = self.dependency_manager.get_model()
        self.tokenizer = self.dependency_manager.get_tokenizer()
        
        # Core settings
        self.auto_append_conversation = auto_append_conversation
        self.show_thoughts = show_thoughts
        self.system_prompt = system_prompt
        self.system_prompt_added = False
        user_name = "Jared"
        char = CHAR_TEMPLATE.format(user=user_name)
        self.cot_injection = cot_injection or char
        
        self.my_name = "Hannah"
        # Initialize conversation with system prompt
        default_system = CHAR_SYSTEM_PROMPT.format(assistant=self.my_name,user=user_name)
        self.messages = [{"role": "system", "content": self.system_prompt or default_system}]

        # Token tracking
        self.token_stats = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'conversation_count': 0
        }
        
        
        print("Enhanced Qwen3 Reasoning Chat initialized!")
        print(f"Model: {model_name}")
        print(f"Show thoughts: {'ON' if self.show_thoughts else 'OFF'}")
        

        if self.system_prompt:
            print(f"System prompt configured: {self.system_prompt[:50]}...")
    
    def set_show_thoughts(self, show_thoughts: bool):
        """Toggle whether to show the model's thinking process."""
        self.show_thoughts = show_thoughts
        print(f"Show thoughts: {'ON' if self.show_thoughts else 'OFF'}")
    
    def set_system_prompt(self, system_prompt: str):
        """Set or update the system prompt."""
        self.system_prompt = system_prompt
        self.messages[0] = {"role": "system", "content": system_prompt}
        print(f"System prompt updated: {system_prompt[:50]}...")

    def set_cot_injection(self, cot_injection: str):
        if type(cot_injection) == str:
            self.cot_injection = cot_injection
            print(f"cot_injection update to: {cot_injection}")
        else:
            print(f"cot_injection NOT UPDATED!, must be string")
            None
        
    
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
        self.system_prompt_added = False
        
    
    def update_token_stats(self, input_tokens: int, output_tokens: int):
        """Update token usage statistics."""
        self.token_stats['total_input_tokens'] += input_tokens
        self.token_stats['total_output_tokens'] += output_tokens
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

    def generate_response(self, user_input: str, max_new_tokens: int = 512, 
                     injection_text: str = None) -> str:
        """
        Generate a response using the Qwen3 model with controlled reasoning injection.
        
        Args:
            user_input: The user's input message
            max_new_tokens: Maximum number of tokens to generate
            injection_text: Custom text to inject during reasoning (optional)
            
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
        
        # Stage 1: Generate initial reasoning burst with thinking mode enabled
        print("Stage 1: Generating initial reasoning...")
        
        # Apply chat template for Stage 1 with thinking mode enabled
        stage1_text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # Enable thinking mode for reasoning
        )
        
        stage1_inputs = self.tokenizer(stage1_text, return_tensors="pt").to(self.model.device)
        input_tokens = stage1_inputs["input_ids"].shape[-1]
        
        # Generate initial reasoning (short burst) with proper Qwen3 sampling parameters
        with torch.no_grad():
            stage1_outputs = self.model.generate(
                **stage1_inputs,
                max_new_tokens=50,  # Short burst for initial thinking
                temperature=0.6,    # Recommended for Qwen3 thinking mode
                top_p=0.95,        # Recommended for Qwen3 thinking mode
                top_k=20,          # Recommended for Qwen3 thinking mode
                min_p=0,           # Recommended for Qwen3 thinking mode
                do_sample=True,    # Required - no greedy decoding for Qwen3
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Extract initial reasoning
        stage1_response = self.tokenizer.decode(
            stage1_outputs[0][stage1_inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        )
        
        print(f"Stage 1 reasoning: {stage1_response} (that's where 1st stage reasoning stopped)")
        
        # Injection content - customize the reasoning direction
        if injection_text is None:
            injection_text = self.cot_injection
        
        # Create enhanced prompt by combining stage1 output with injection
        enhanced_text = stage1_text + stage1_response + injection_text
        
        # Tokenize the enhanced prompt
        stage2_inputs = self.tokenizer(enhanced_text, return_tensors="pt").to(self.model.device)
        
        print("Stage 2: Continuing reasoning and generating final response...skip first 50 tokens")
        
        # Generate final response with remaining tokens
        with torch.no_grad():
            final_outputs = self.model.generate(
                **stage2_inputs,
                max_new_tokens=max_new_tokens - 50,  # Remaining tokens after stage 1
                temperature=0.6,    # Keep consistent with thinking mode
                top_p=0.95,
                top_k=20,
                min_p=0,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                use_cache=True,
            )
        
        # Extract the complete response
        full_response = self.tokenizer.decode(final_outputs[0], skip_special_tokens=True)
        
        # Count tokens and update stats
        output_tokens = final_outputs[0].shape[-1]
        self.update_token_stats(input_tokens, output_tokens)
        
        # Extract only the new content after the original conversation
        original_length = len(stage1_text)
        response_content = full_response[original_length:].strip()
        
        # Parse thinking vs final response content
        thinking_content = ""
        final_response_content = ""

        print(f"\nRAW OUTPUT:{response_content} \n\n")

        # Look for thinking tags - Qwen3 uses <think>...</think> tags
        # The key insight: we want everything AFTER the last </think> tag as the final response
        if "</think>" in response_content:
            # Find the last </think> tag position
            last_think_close = response_content.rfind("</think>")
            
            # Everything before and including the last </think> is thinking content
            thinking_content = response_content[:last_think_close + 8]  # +8 for "</think>" length
            
            # Everything AFTER the last </think> is the final response
            final_response_content = response_content[last_think_close + 8:].strip()
            
            print(f"Found thinking content ending at position {last_think_close}")
            print(f"Thinking content: {thinking_content[:100]}...")
            print(f"Final response content: {final_response_content[:100]}...")
            
        else:
            # No thinking tags found, treat entire content as final response
            thinking_content = ""
            final_response_content = response_content
        
        # Ensure we have a final response
        if not final_response_content.strip():
            final_response_content = "I understand your question and have completed my analysis."
        
        # Determine what to show based on show_thoughts flag
        if self.show_thoughts and thinking_content:
            display_response = f"{thinking_content}\n\n{final_response_content}"
        else:
            display_response = final_response_content
        
        # CRITICAL FIX: Add ONLY the clean final response to conversation history
        # This ensures no thinking content pollutes the conversation context
        clean_final_response = final_response_content.strip()
        
        # Ensure we have a meaningful response
        if not clean_final_response:
            clean_final_response = "I understand your question and have completed my analysis."
        
        # Add the clean response to message history
        self.messages.append({"role": "assistant", "content": clean_final_response})

        # Return the clean final response (what gets added to conversation history)
        return clean_final_response

    def chat(self, user_input: str, max_tokens: int = 512) -> str:
        """
        Main chat interface 
        
        Args:
            user_input: The user's question or message
            max_tokens: Maximum tokens to generate
            
        Returns:
            The assistant's response
        """

        # Generate response using the reasoning system
        response = self.generate_response(
            user_input, 
            max_new_tokens=max_tokens,
        )
        
        return response
    
    def interactive_chat(self):
        """Start an interactive chat session."""
        print("\n🤖 Qwen3 Reasoning Chat - Interactive Mode")
        print("Commands: /thoughts (toggle), /clear (reset), /stats (show),  /quit (exit)")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == '/quit':
                    print("Goodbye! 👋")
                    break
                elif user_input.lower() == '/clear':
                    self.clear_chat_messages()
                    print("✅ Chat cleared!")
                    continue
                elif user_input.lower() == '/thoughts':
                    self.set_show_thoughts(not self.show_thoughts)
                    continue
                elif user_input.lower() == '/stats':
                    self.print_token_stats()
                    continue
                
                # Generate response
                print("\n🧠 Assistant:", end=" ")
                response = self.chat(user_input)
                print(response)
                print("="*50)
                print("         - MESSAGES -\n")
                print(self.messages)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue 




if __name__ == "__main__":
    def main():
        """Main function to run the enhanced chat interface ."""
        import sys
        import argparse
        
        # Handle command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] == "download":
                model_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-8B"
                print(f"Downloading {model_name}...")
                QwenChatDependencyManager.download_model(model_name)
                return
            elif sys.argv[1] == "help":
                print("\n🤖 Qwen3 Reasoning Chat - Help")
                print("Usage:")
                print("  python qwen_chat.py                    # Start interactive chat")
                print("  python qwen_chat.py download [model]   # Download model")
                print("  python qwen_chat.py help               # Show this help")
                print("\nSupported models:")
                print("  - Qwen/Qwen3-8B (default)")
                print("  - Qwen/Qwen3-4B")
                print("  - Qwen/Qwen3-1.7B")
                print("  - Qwen/Qwen3-0.6B")
                return
        
        try:
            # Initialize enhanced chat with reasoning interrupts and memory
            model_name = "Qwen/Qwen3-8B"
            print(f"\n🚀 Enhanced Qwen3 Reasoning Chat  ({model_name})...")
            print("Features: Multi-stage reasoning, external CoT text injection")
            print("-" * 70)
            
            chat = QwenReasoningChat(
                model_name=model_name,
                show_thoughts=False,  # Start with thoughts hidden
                auto_append_conversation=True,
                #system_prompt="You are a helpful assistant"
            )
            
            # Print initial statistics
            chat.print_token_stats()
            
            # Show some example commands
            print("💡 Quick Tips:")
            print("  • Type '/thoughts' to toggle reasoning visibility")
            print("  • Type '/stats' to see token usage")
            print("  • Type '/clear' to reset conversation")
            print("  • Type '/quit' to exit")
            print("\n🎯 Try asking complex questions that benefit from step-by-step reasoning!")
            
            # Start interactive chat
            chat.interactive_chat()
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted by user. Goodbye!")
        except ImportError as e:
            print(f"\n❌ Missing dependency: {e}")
            print("Please install required packages:")
            print("pip install torch transformers accelerate")
            if "MxBaiEmbedder" in str(e):
                print("pip install sentence-transformers")
        except Exception as e:
            print(f"\n❌ Error initializing chat: {e}")
            print("Try running with a smaller model:")
            print("python qwen_chat.py download Qwen/Qwen3-1.7B")
            print("Then edit the model_name in the script to use the smaller model.")
    
    # Demo function for testing specific features
    def demo_reasoning_injection():
        """Demonstrate the reasoning injection feature."""
        print("\n🧪 Reasoning Injection Demo")
        print("-" * 30)
        
        try:
            chat = QwenReasoningChat(
                model_name="Qwen/Qwen3-8B",
                show_thoughts=True,  # Show thinking for demo
                auto_append_conversation=False  # Reset each time for demo
            )
            
            test_question = "How can I improve my productivity when working from home?"
            
            print(f"Question: {test_question}\n")
            
            # Test 1: Default injection
            print("🔹 Test 1: Default injection")
            response1 = chat.generate_response(test_question, max_new_tokens=300)
            print(f"Response: {response1}\n")
            
            # Test 2: Custom injection - focus on psychological aspects
            print("🔹 Test 2: Custom injection - psychological focus")
            custom_injection = "\nLet me think about this from a psychological and behavioral science perspective.\n"
            response2 = chat.generate_response(
                test_question, 
                max_new_tokens=512,
                injection_text=custom_injection
            )
            print(f"Response: {response2}\n")
            
            
            
        except Exception as e:
            print(f"Demo failed: {e}")
    
    # Run based on environment or specific demo request
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_reasoning_injection()
    else:
        main()