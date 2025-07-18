import torch
import os
import urllib.request
import socket
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from typing import List, Dict, Any, Optional, Callable

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




class QwenChat:
    """Handles chat functionality, conversation management, token tracking, and tool use."""
    
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", model_path=None, force_offline=False):
        """Initialize the chat interface with automatic dependency management."""
        self.dependency_manager = QwenChatDependencyManager(
            model_name=model_name,
            model_path=model_path,
            force_offline=force_offline
        )
        self.model = self.dependency_manager.get_model()
        self.tokenizer = self.dependency_manager.get_tokenizer()
        
        # Token tracking
        self.token_stats = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'conversation_count': 0
        }
        
        # Tool management
        self.tools = {}
        self.available_tools = []
        
        # Initialize conversation with system prompt
        self.messages = []
        self._add_system_prompt()
    
    def _add_system_prompt(self):
        """Add the initial system prompt."""
        system_content = """You are a robot"""
        self.messages = [{"role": "system", "content": system_content}]
    
    def _update_system_prompt(self, system_content):
        """Update the system prompt."""
        self.messages[0] = {"role": "system", "content": system_content}

    def clear_chat_messages(self):
        self.messages = self.messages[:1]

    def register_tool(self, tool_function: Callable, name: str = None, description: str = None, parameters: Dict = None):
        """
        Register a tool function for use in conversations.
        
        Args:
            tool_function: The callable function to register
            name: Name of the tool (defaults to function name)
            description: Description of what the tool does
            parameters: JSON Schema describing the function parameters
        """
        if name is None:
            name = tool_function.__name__
            
        # Store the function
        self.tools[name] = tool_function
        
        # Create tool definition for the model
        tool_def = {
            "type": "function",
            "function": {
                "name": name,
                "description": description or tool_function.__doc__ or f"Function {name}",
                "parameters": parameters or {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
        
        # Update available tools list
        self.available_tools = [t for t in self.available_tools if t["function"]["name"] != name]
        self.available_tools.append(tool_def)
    
    def _parse_tool_calls(self, content: str) -> Dict[str, Any]:
        """
        Parse tool calls from model output using the Hermes format.
        
        The Qwen2.5 model with Hermes template generates tool calls in the format:
        <tool_call>
        {"name": "function_name", "arguments": {"arg1": "value1"}}
        </tool_call>
        """
        tool_calls = []
        offset = 0
        
        # Find all tool call blocks
        for i, match in enumerate(re.finditer(r"<tool_call>\n(.+?)\n</tool_call>", content, re.DOTALL)):
            if i == 0:
                offset = match.start()
            
            try:
                # Parse the JSON inside the tool_call tags
                tool_call_json = json.loads(match.group(1).strip())
                
                # Ensure arguments is a dict, not a string
                if isinstance(tool_call_json.get("arguments"), str):
                    tool_call_json["arguments"] = json.loads(tool_call_json["arguments"])
                
                tool_calls.append({
                    "type": "function", 
                    "function": tool_call_json
                })
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse tool call: {match.group(1)} - Error: {e}")
                continue
        
        # Extract content before tool calls (if any)
        if tool_calls:
            if offset > 0 and content[:offset].strip():
                content_text = content[:offset].strip()
            else:
                content_text = ""
            
            return {
                "role": "assistant",
                "content": content_text,
                "tool_calls": tool_calls
            }
        else:
            # No tool calls found, return regular assistant message
            # Remove any trailing special tokens
            clean_content = re.sub(r"<\|im_end\|>$", "", content)
            return {
                "role": "assistant",
                "content": clean_content
            }
    
    def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute the tool calls and return tool results."""
        tool_results = []
        
        for tool_call in tool_calls:
            if function_call := tool_call.get("function"):
                function_name = function_call["name"]
                function_args = function_call["arguments"]
                
                if function_name in self.tools:
                    try:
                        # Execute the function
                        result = self.tools[function_name](**function_args)
                        
                        # Add tool result message
                        tool_results.append({
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps(result) if not isinstance(result, str) else result
                        })
                        
                    except Exception as e:
                        # Handle function execution errors
                        tool_results.append({
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error executing {function_name}: {str(e)}"
                        })
                else:
                    tool_results.append({
                        "role": "tool",
                        "name": function_name,
                        "content": f"Function {function_name} not found"
                    })
        
        return tool_results

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
        print(f"Context Window: 32,768 tokens (Qwen2.5)")
        print(f"Conversations: {stats['conversation_count']}")
        print(f"Input tokens: {stats['total_input_tokens']}")
        print(f"Output tokens: {stats['total_output_tokens']}")
        print(f"Total tokens: {stats['total_tokens']}")
        if stats['conversation_count'] > 0:
            print(f"Avg tokens per conversation: {stats['total_tokens'] / stats['conversation_count']:.1f}")
        print(f"----------------------------\n")
    
    def generate_response(self, user_input: str, max_new_tokens: int = 512, auto_execute_tools: bool = True) -> str:
        """
        Generate a response using the Qwen model with optional tool use.
        
        Args:
            user_input: The user's input message
            max_new_tokens: Maximum number of tokens to generate
            auto_execute_tools: Whether to automatically execute tool calls and generate final response
            
        Returns:
            The assistant's response (either direct response or final response after tool execution)
        """
        # Add user message to conversation
        self.messages.append({"role": "user", "content": user_input})
        
        # Apply chat template with tools if available
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tools=self.available_tools if self.available_tools else None,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize and generate
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_tokens = model_inputs.input_ids.shape[1]
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8
        )
        
        # Extract only the new tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Count tokens and update stats
        output_tokens = len(generated_ids[0])
        self.update_token_stats(input_tokens, output_tokens)
        
        # Parse the response for tool calls
        parsed_response = self._parse_tool_calls(response_text)
        self.messages.append(parsed_response)
        
        # Check if there are tool calls to execute
        if tool_calls := parsed_response.get("tool_calls"):
            if auto_execute_tools:
                # Execute the tools
                tool_results = self._execute_tool_calls(tool_calls)
                self.messages.extend(tool_results)
                
                # Generate final response based on tool results
                return self._generate_final_response(max_new_tokens)
            else:
                # Return indication that tools need to be executed
                return f"[TOOL_CALLS_PENDING] {len(tool_calls)} tool(s) need execution"
        else:
            # No tool calls, return the content directly
            return parsed_response["content"]
    
    def _generate_final_response(self, max_new_tokens: int) -> str:
        """Generate the final response after tool execution."""
        # Apply chat template again with the tool results
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tools=self.available_tools if self.available_tools else None,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize and generate
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_tokens = model_inputs.input_ids.shape[1]
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8
        )
        
        # Extract only the new tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        final_response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Count tokens and update stats
        output_tokens = len(generated_ids[0])
        self.update_token_stats(input_tokens, output_tokens)
        
        # Parse and add the final response
        parsed_final = self._parse_tool_calls(final_response)
        self.messages.append(parsed_final)
        
        return parsed_final["content"]
    
    def execute_pending_tools(self, max_new_tokens: int = 512) -> str:
        """
        Execute any pending tool calls from the last assistant message.
        Useful when auto_execute_tools=False in generate_response.
        """
        if self.messages and self.messages[-1]["role"] == "assistant":
            if tool_calls := self.messages[-1].get("tool_calls"):
                # Execute the tools
                tool_results = self._execute_tool_calls(tool_calls)
                self.messages.extend(tool_results)
                
                # Generate final response
                return self._generate_final_response(max_new_tokens)
        
        return "No pending tool calls found"
    
    def list_available_tools(self) -> List[str]:
        """Return a list of registered tool names."""
        return list(self.tools.keys())
    
    def remove_tool(self, tool_name: str) -> bool:
        """Remove a registered tool."""
        if tool_name in self.tools:
            del self.tools[tool_name]
            self.available_tools = [t for t in self.available_tools if t["function"]["name"] != tool_name]
            return True
        return False





if __name__ == "__main__":
    # Example tools
    def get_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
        """Get current weather for a location."""
        # This would normally call a real weather API
        return {
            "location": location,
            "temperature": 22,
            "unit": unit,
            "conditions": "sunny"
        }
    
    def calculate(expression: str) -> Dict[str, Any]:
        """Safely evaluate a mathematical expression."""
        try:
            # In production, use a safer evaluation method
            result = eval(expression)
            return {"expression": expression, "result": result}
        except Exception as e:
            return {"expression": expression, "error": str(e)}
        
    def chat_loop(chat_instance):
        """Start an interactive chat session."""
        print("\n" + "="*50)
        print("Qwen2.5-7B-Instruct Chat Interface")
        print("Commands: 'quit' to exit, 'clear' to clear history, 'save' to save conversation")
        print("="*50 + "\n")

        def clear_history(chat_instance):
            """Clear the conversation history but keep system prompt."""
            print("Conversation history cleared.")
            chat_instance.reset_to_system_prompt()
        
        def save_conversation(chat_instance, filename="conversation.txt"):
            """Save the conversation to a file."""
            with open(filename, "w", encoding="utf-8") as f:
                for message in chat_instance.messages:
                    role = message["role"].upper()
                    content = message["content"]
                    f.write(f"{role}: {content}\n\n")
            print(f"Conversation saved to {filename}")
        
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

                chat_instance.clear_chat_messages()

                # Register tools
                chat_instance.register_tool(
                    get_weather,
                    description="Get current weather for a specific location",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name, e.g., 'New York' or 'Tokyo'"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature unit"
                            }
                        },
                        "required": ["location"]
                    }
                )
                
                chat_instance.register_tool(
                    calculate,
                    description="Perform mathematical calculations",
                    parameters={
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Mathematical expression to evaluate, e.g., '2 + 2' or '10 * 5'"
                            }
                        },
                        "required": ["expression"]
                    }
                )

                # Example Tool conversation
                print("Available tools:", chat_instance.list_available_tools())
                
                response1 = chat_instance.generate_response("What's the weather like in Tokyo?")
                print("Assistant:", response1)
                
                response2 = chat_instance.generate_response("Can you calculate 15 * 8 + 3?")
                print("Assistant:", response2)
                
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
                model_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-7B-Instruct"
                QwenChatDependencyManager.download_model(model_name)
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
            
            # Initialize chat interface (Qwen/Qwen2.5-7B-Instruct is default if no model passed)
            chat = QwenChat()
            
            # Start the chat loop
            chat_loop(chat)
            
        except Exception as e:
            print(f"Error: {e}")
            print("\nTroubleshooting:")
            print("1. To force offline mode: python qwen_chat.py offline")
            print("2. To download model: python qwen_chat.py download")



    main()