import os
import sys
import json
import socket
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


class MythoMaxDependencyManager:
    """Manages model loading with online/offline fallback capabilities."""
    
    def __init__(self, model_name="Gryphe/MythoMax-L2-13b", model_path=None, force_offline=False):
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
            socket.create_connection(("8.8.8.8", 53), timeout)
            return True
        except OSError:
            return False
    
    def _load_dependencies(self):
        """Load model and tokenizer based on availability."""
        print("Loading MythoMax model and tokenizer...")
        
        # If force offline or no internet, try offline loading first
        if self.force_offline or not self._check_internet_connection():
            print("Attempting offline loading...")
            if self.model_path:
                success = self._load_model_offline(self.model_path)
            else:
                # Try to find cached model
                cached_path = self._find_cached_model()
                success = self._load_model_offline(cached_path) if cached_path else False
            
            if success:
                print("Successfully loaded model offline!")
                return
            elif self.force_offline:
                raise RuntimeError("Failed to load model offline and force_offline is True")
        
        # Try online loading
        if self._check_internet_connection():
            print("Attempting online loading...")
            try:
                self._load_model_online(self.model_name)
                print("Successfully loaded model online!")
                return
            except Exception as e:
                print(f"Online loading failed: {e}")
        
        # Final fallback to cached model
        print("Trying cached model as final fallback...")
        cached_path = self._find_cached_model()
        if cached_path and self._load_model_offline(cached_path):
            print("Successfully loaded cached model!")
        else:
            raise RuntimeError("Failed to load model from all sources")
    
    def _load_model_online(self, model_name):
        """Load model from Hugging Face Hub."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model online: {e}")
    
    def _load_model_offline(self, model_path):
        """Load model from local files only."""
        if not model_path or not self._validate_model_files(model_path):
            return False
        
        try:
            print(f"Loading from: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
                use_fast=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            return True
        except Exception as e:
            print(f"Offline loading failed: {e}")
            return False
    
    def _find_cached_model(self):
        """Try to find cached model in common Hugging Face cache locations."""
        cache_dirs = [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "huggingface" / "transformers",
            Path("./models"),
            Path("./cache")
        ]
        
        model_variants = [
            self.model_name.replace("/", "--"),
            self.model_name.split("/")[-1],
            "MythoMax-L2-13b"
        ]
        
        for cache_dir in cache_dirs:
            if not cache_dir.exists():
                continue
                
            for variant in model_variants:
                # Look for exact matches
                potential_paths = list(cache_dir.glob(f"*{variant}*"))
                for path in potential_paths:
                    if path.is_dir() and self._validate_model_files(path):
                        return str(path)
        
        return None
    
    def _validate_model_files(self, model_path):
        """Check if a model directory has the required files."""
        if not model_path:
            return False
            
        path = Path(model_path)
        if not path.exists() or not path.is_dir():
            return False
        
        required_files = ["config.json"]
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json"]
        model_files = ["pytorch_model.bin", "model.safetensors"]
        
        # Check for required config
        if not any((path / f).exists() for f in required_files):
            return False
        
        # Check for tokenizer files
        if not any((path / f).exists() for f in tokenizer_files):
            return False
        
        # Check for model weights
        if not any((path / f).exists() for f in model_files):
            # Also check for sharded models
            if not list(path.glob("pytorch_model-*.bin")) and not list(path.glob("model-*.safetensors")):
                return False
        
        return True
    
    def get_model(self):
        """Get the loaded model."""
        return self.model
    
    def get_tokenizer(self):
        """Get the loaded tokenizer."""
        return self.tokenizer


class MythoMax:
    """MythoMax chat interface with conversation management."""
    
    def __init__(self, model_name="Gryphe/MythoMax-L2-13b", model_path=None, 
                 force_offline=False, auto_append_conversation=False, name="Artemis"):
        """Initialize the chat interface with automatic dependency management."""
        self.dependency_manager = MythoMaxDependencyManager(
            model_name=model_name,
            model_path=model_path,
            force_offline=force_offline
        )
        self.model = self.dependency_manager.get_model()
        self.tokenizer = self.dependency_manager.get_tokenizer()
        self.auto_append_conversation = auto_append_conversation
        self.name = name
        
        # Initialize conversation state
        self.chat_messages = []
        self.system_prompt = f"You are {self.name}, a helpful and knowledgeable AI assistant."
        
        # Token statistics
        self.token_stats = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'conversation_count': 0
        }
        
        # Generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        print(f"MythoMax initialized successfully!")
        print(f"Model: {model_name}")
        print(f"Assistant name: {self.name}")
        print(f"Device: {next(self.model.parameters()).device}")
    
    def _update_system_prompt(self, system_prompt):
        """Update the system prompt."""
        self.system_prompt = system_prompt
        print(f"System prompt updated: {system_prompt[:100]}...")
    
    def clear_chat_messages(self):
        """Clear conversation history and reset token stats."""
        self.chat_messages = []
        self.token_stats = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'conversation_count': 0
        }
        print("Reset chat messages and token stats")
    
    def update_token_stats(self, input_tokens, output_tokens):
        """Update token usage statistics."""
        self.token_stats['total_input_tokens'] = input_tokens
        self.token_stats['total_output_tokens'] = output_tokens
        self.token_stats['total_tokens'] += (input_tokens + output_tokens)
        self.token_stats['conversation_count'] += 1
    
    def print_token_stats(self):
        """Print current token usage statistics."""
        stats = self.token_stats
        context_window = getattr(self.model.config, 'max_position_embeddings', 'Unknown')
        
        print(f"\n--- Token Usage Statistics ---")
        print(f"Context Window: {context_window}")
        print(f"Conversations: {stats['conversation_count']}")
        print(f"Input tokens: {stats['total_input_tokens']}")
        print(f"Output tokens: {stats['total_output_tokens']}")
        print(f"Total tokens: {stats['total_tokens']}")
        if stats['conversation_count'] > 0:
            print(f"Avg tokens per conversation: {stats['total_tokens'] / stats['conversation_count']:.1f}")
        print(f"----------------------------\n")
    
    def _format_conversation(self):
        """Format the conversation for the model."""
        # Build conversation with system prompt
        formatted_messages = [f"System: {self.system_prompt}"]
        
        for message in self.chat_messages:
            role = message['role']
            content = message['content']
            if role == 'user':
                formatted_messages.append(f"Human: {content}")
            elif role == 'assistant':
                formatted_messages.append(f"Assistant: {content}")
        
        return "\n".join(formatted_messages) + "\nAssistant:"
    
    def generate_response(self, user_input: str, max_new_tokens: int = 512) -> str:
        """
        Generate a response using the MythoMax model.
        
        Args:
            user_input: The user's input message
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            The assistant's response
        """
        try:
            # Add user message to conversation
            if self.auto_append_conversation:
                self.chat_messages.append({'role': 'user', 'content': user_input})
            
            # Format the conversation
            prompt = self._format_conversation()
            if not self.auto_append_conversation:
                prompt += f"\nHuman: {user_input}\nAssistant:"
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncate=True,
                max_length=self.model.config.max_position_embeddings - max_new_tokens
            )
            
            # Move to model device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            input_length = inputs['input_ids'].shape[1]
            
            # Update generation config
            self.generation_config.max_new_tokens = max_new_tokens
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            # Clean up response
            response = response.strip()
            
            # Stop at common stop sequences
            stop_sequences = ['\nHuman:', '\nUser:', '\nSystem:', '\n\n\n']
            for stop_seq in stop_sequences:
                if stop_seq in response:
                    response = response.split(stop_seq)[0]
            
            # Update token statistics
            output_tokens = len(response_tokens)
            self.update_token_stats(input_length, output_tokens)
            
            # Add assistant response to conversation
            if self.auto_append_conversation:
                self.chat_messages.append({'role': 'assistant', 'content': response})
            
            return response
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return "I'm sorry, I encountered an error while generating a response."


def main():
    """Example usage of the MythoMax chat interface."""
    print("Initializing MythoMax Chat Demo...")
    
    try:
        # Initialize the chat interface
        chat = MythoMax(
            model_name="Gryphe/MythoMax-L2-13b",
            auto_append_conversation=True,
            name="Artemis"
        )
        
        print("\nMythoMax Chat Demo started!")
        print("Type 'quit' to exit, 'clear' to reset conversation, 'stats' to show token usage")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    chat.clear_chat_messages()
                    continue
                elif user_input.lower() == 'stats':
                    chat.print_token_stats()
                    continue
                elif not user_input:
                    continue
                
                print(f"\n{chat.name}: ", end="", flush=True)
                response = chat.generate_response(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                
    except Exception as e:
        print(f"Failed to initialize MythoMax: {e}")
        print("Please ensure the model is available or check your internet connection.")


if __name__ == "__main__":
    main()