import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FalconChatBot:
    """
    A modular chatbot using the Falcon-7B-Instruct model.
    """
    
    def __init__(self, model_name: str = "tiiuae/falcon-7b-instruct", 
                 cache_dir: Optional[str] = None,
                 #max_length: int = 512,
                 temperature: float = 0.7,
                 top_k: int = 10,
                 top_p: float = 0.9):
        """
        Initialize the Falcon chatbot.
        
        Args:
            model_name: HuggingFace model identifier
            cache_dir: Directory where model is cached (None for default)
            max_length: Maximum length of generated response
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/hub")
        #self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        self.tokenizer = None
        self.pipeline = None
        self.conversation_history = []
        
        self._load_model()
    
    def _load_model(self):
        """Load the tokenizer and model pipeline."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                #cache_dir=self.cache_dir
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create pipeline
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                #cache_dir=self.cache_dir
            )
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _format_prompt(self, user_input: str, include_history: bool = True) -> str:
        """
        Format the input prompt with conversation history.
        
        Args:
            user_input: The user's message
            include_history: Whether to include conversation history
            
        Returns:
            Formatted prompt string
        """
        if include_history and self.conversation_history:
            # Build conversation context
            context = ""
            for entry in self.conversation_history[-3:]:  # Last 3 exchanges
                context += f"Human: {entry['user']}\nAssistant: {entry['assistant']}\n"
            
            prompt = f"{context}Human: {user_input}\nAssistant:"
        else:
            prompt = f"Human: {user_input}\nAssistant:"
        
        return prompt
    
    def generate_response(self, user_input: str, include_history: bool = True) -> str:
        """
        Generate a response to user input.
        
        Args:
            user_input: The user's message
            include_history: Whether to include conversation history
            
        Returns:
            Generated response text
        """
        if not self.pipeline:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        prompt = self._format_prompt(user_input, include_history)
        
        try:
            # Generate response
            sequences = self.pipeline(
                prompt,
                #max_length=len(self.tokenizer.encode(prompt)) + self.max_length,
                do_sample=True,
                truncation=True,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            # Extract generated text
            generated_text = sequences[0]['generated_text'].strip()
            
            # Clean up the response (remove any remaining prompt artifacts)
            if "Human:" in generated_text:
                generated_text = generated_text.split("Human:")[0].strip()
            if "Assistant:" in generated_text:
                generated_text = generated_text.replace("Assistant:", "").strip()
            
            # Store in conversation history
            self.conversation_history.append({
                'user': user_input,
                'assistant': generated_text
            })
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared.")
    
    def save_conversation(self, filename: str):
        """Save conversation history to a file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for i, entry in enumerate(self.conversation_history, 1):
                    f.write(f"=== Exchange {i} ===\n")
                    f.write(f"Human: {entry['user']}\n")
                    f.write(f"Assistant: {entry['assistant']}\n\n")
            logger.info(f"Conversation saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")

def interactive_chat():
    """Run an interactive chat session."""
    print("🦅 Falcon-7B Chat Interface")
    print("=" * 50)
    print("Commands:")
    print("  /clear  - Clear conversation history")
    print("  /save   - Save conversation to file")
    print("  /quit   - Exit the chat")
    print("=" * 50)
    
    # Initialize chatbot
    try:
        chatbot = FalconChatBot()
    except Exception as e:
        print(f"Failed to initialize chatbot: {e}")
        return
    
    print("\nChatbot ready! Start typing your messages.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.lower() == '/quit':
                print("Goodbye!")
                break
            elif user_input.lower() == '/clear':
                chatbot.clear_history()
                print("Conversation history cleared.")
                continue
            elif user_input.lower() == '/save':
                filename = input("Enter filename (default: conversation.txt): ").strip()
                if not filename:
                    filename = "conversation.txt"
                chatbot.save_conversation(filename)
                continue
            
            # Generate and display response
            print("Assistant: ", end="", flush=True)
            response = chatbot.generate_response(user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":

    interactive_chat()