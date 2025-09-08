import os
import socket
import json
import re
from typing import List, Dict, Any, Callable, Generator
import gc
import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)

class QwenDependencyManager:
    """Handles model loading, dependency management, and offline/online detection."""
    
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", model_path=None, force_offline=False):
        """Initialize the dependency manager with model loading logic."""
        self.model_name = model_name
        self.force_offline = force_offline
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        # Initialize dependency storage
        self.torch = None
        #self.accelerate = None
        self.AutoModelForCausalLM = None
        self.AutoTokenizer = None
        
        # Check and install dependencies first
        available_deps = self._check_and_install_dependencies()
        
        logging.debug(f"Available dependencies: {available_deps}")
        
        # Load the model and tokenizer
        self._load_dependencies()
    
    def _check_and_install_dependencies(self) -> list:
        """
        Check if required dependencies are installed and install if missing.
        Stores imported modules as instance attributes.
        Returns a list of successfully loaded dependencies.
        """
        available_dependencies = []
        
        # Check and install PyTorch
        try:
            import torch
            self.torch = torch
            available_dependencies.append('torch')
            logging.debug("PyTorch already installed and imported successfully")
        except ImportError:
            logging.debug("PyTorch not found. Installing torch...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
                import torch
                self.torch = torch
                available_dependencies.append('torch')
                logging.debug("PyTorch installed and imported successfully")
            except (subprocess.CalledProcessError, ImportError) as e:
                logging.error(f"Failed to install or import PyTorch: {e}")
                raise ImportError("WARNING! Failed to install or import PyTorch")
        
        # Check and install transformers
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.AutoModelForCausalLM = AutoModelForCausalLM
            self.AutoTokenizer = AutoTokenizer
            available_dependencies.append('transformers')
            logging.debug("transformers already installed and imported successfully")
        except ImportError:
            logging.debug("transformers not found. Installing transformers...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.AutoModelForCausalLM = AutoModelForCausalLM
                self.AutoTokenizer = AutoTokenizer
                available_dependencies.append('transformers')
                logging.debug("transformers installed and imported successfully")
            except (subprocess.CalledProcessError, ImportError) as e:
                logging.error(f"Failed to install or import transformers: {e}")
                raise ImportError("WARNING! Failed to install or import transformers")
        
        # Optional: Check for accelerate (often needed for device_map="auto")
        """try:
            import accelerate
            self.accelerate = accelerate
            available_dependencies.append('accelerate')
            logging.debug("accelerate already available")
        except ImportError:
            logging.debug("accelerate not found. Installing accelerate...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate"])
                import accelerate
                self.accelerate = accelerate
                available_dependencies.append('accelerate')
                logging.debug("accelerate installed successfully")
            except (subprocess.CalledProcessError, ImportError) as e:
                logging.warning(f"Failed to install accelerate (optional): {e}")
                # Don't raise error for optional dependency"""
        
        return available_dependencies
    
    def get_available_dependencies(self) -> list:
        """Return list of available dependencies that can be imported."""
        available = []
        
        if self.torch is not None:
            available.append('torch')
        if self.AutoModelForCausalLM is not None and self.AutoTokenizer is not None:
            available.append('transformers')
        
        # Check for other optional dependencies
        try:
            import accelerate
            available.append('accelerate')
        except ImportError:
            pass
            
        return available
    
    def print_gpu_memory(self):
        if self.torch and self.torch.cuda.is_available():
            logging.debug(f"GPU memory: {self.torch.cuda.memory_allocated()/1024**3:.1f}GB allocated, "
                f"{self.torch.cuda.memory_reserved()/1024**3:.1f}GB reserved")
        
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
        
        self.print_gpu_memory()
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
            self.model = self.AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                #load_in_8bit=True,  # Quantization to reduce memory -  load_in_4bit=True,  # Even more aggressive quantization
            )
            self.tokenizer = self.AutoTokenizer.from_pretrained(model_name)
            
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
            self.model = self.AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                local_files_only=True
            )
            self.tokenizer = self.AutoTokenizer.from_pretrained(
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
            f"./{self.model_name.split('/')[-1]}",
            f"./{self.model_name.split('/')[-1].upper()}",
            f"./{self.model_name.split('/')[-1].lower()}"
        ]
        
        for path in local_paths:
            if os.path.exists(path) and self._validate_model_files(path):
                print(f"Found valid local model at: {path}")
                return path
        
        # Look for Qwen model folders in HF cache
        model_patterns = [
            f"models--{self.model_name.replace('/', '--')}",
            f"models--{self.model_name.replace('/', '--').upper()}",
            f"models--{self.model_name.replace('/', '--').lower()}"
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
    
    def get_torch(self):
        """Get the loaded tokenizer."""
        return self.torch
    
    """def get_accelerate(self):
        #Get the loaded tokenizer.
        return self.accelerate"""
    
    @staticmethod
    def download_model(model_name="Qwen/Qwen2.5-7B-Instruct", save_path=None):
        """Helper function to download the model for offline use."""
        if save_path is None:
            save_path = f"./{model_name.split('/')[-1]}"
        
        print(f"Downloading {model_name} for offline use...")
        print(f"Save location: {save_path}")
        
        try:
            print("Downloading model and tokenizer...")
            # Import here in case dependencies aren't available yet
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            print(f"Model downloaded successfully to: {save_path}")
            
        except Exception as e:
            print(f"Error downloading model: {e}")


class Qwen:
    """Handles chat functionality, conversation management, token tracking, and tool use."""
    
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", model_path=None, force_offline=False, auto_append_conversation = False,user_name='user'):
        """Initialize the chat interface with automatic dependency management."""
        self.dependency_manager = QwenDependencyManager(
            model_name=model_name,
            model_path=model_path,
            force_offline=force_offline
        )
        
        self.torch = self.dependency_manager.get_torch()
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
        
        # Tool management
        self.tools = {}
        self.available_tools = []

        #User's Name
        self._user = user_name
        
        # Initialize conversation with system prompt
        self.messages = [{"role": "system", "content": "you are a robot"}]
            
    def _update_system_prompt(self, system_prompt):
        """Update the system prompt."""
        print(f"SYSTEM PROMPT UPDATED TO: {system_prompt}")
        self.messages[0] = {"role": "system", "content": system_prompt}

    def clear_chat_messages(self):
        logging.debug("Reset chat messages and token stats")
        # Token tracking reset 
        self.token_stats.update({
            'total_tokens': 0,
            'conversation_count': 0
        })
        self.messages = self.messages[:1]#keep system prompt

    def clear_last_message(self):
        """Remove last item in messages using pop() on the list"""
        #don't pop system prompt
        if len(self.messages) > 1:
            self.messages.pop()

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
        logging.debug(f"TOOL USE CALLED:\n{tool_calls}\n")
        tool_results = []
        
        for tool_call in tool_calls:
            if function_call := tool_call.get("function"):
                function_name = function_call["name"]
                function_args = function_call["arguments"]
                logging.debug(f"Calling: {function_name} with args: {function_args}")
                
                if function_name in self.tools:
                    logging.debug(f"Have tool -> {function_name} using it")
                    try:
                        # Execute the function
                        result = self.tools[function_name](function_args)
                        
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
        self.token_stats['total_input_tokens'] = input_tokens
        self.token_stats['total_output_tokens'] = output_tokens
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
    
    def get_token_stats(self):
        return self.token_stats
    
    def _generate_streaming(self, model_inputs, max_new_tokens: int = 512, print_tokens: bool = True) -> Generator[str, None, tuple]:
        """
        Generate tokens one at a time and yield them.
        
        Args:
            model_inputs: Tokenized input tensors
            max_new_tokens: Maximum number of new tokens to generate
            print_tokens: Whether to print tokens as they're generated
            
        Yields:
            str: Each generated token as a string
            
        Returns:
            tuple: (full_response_text, output_token_count)
        """
        input_tokens = model_inputs.input_ids.shape[1]
        generated_tokens = []
        
        # Buffer for handling multi-byte Unicode sequences
        decode_buffer = []
        last_output_length = 0  # Track what we've already output
        
        # Initialize past_key_values for efficient generation
        past_key_values = None
        
        # Get the input ids
        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask if hasattr(model_inputs, 'attention_mask') else None
        
        for step in range(max_new_tokens):
            try:
                # Prepare inputs for this step
                if past_key_values is None:
                    # First step - use full input
                    current_input_ids = input_ids
                    current_attention_mask = attention_mask
                else:
                    # Subsequent steps - use only the last generated token
                    current_input_ids = input_ids[:, -1:]
                    if attention_mask is not None:
                        # Extend attention mask
                        current_attention_mask = self.torch.cat([
                            attention_mask, 
                            self.torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
                        ], dim=1)
                    else:
                        current_attention_mask = None
                
                # Forward pass
                with self.torch.no_grad():
                    outputs = self.model(
                        input_ids=current_input_ids,
                        attention_mask=current_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                
                # Get the logits for the last position
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                
                # Apply sampling
                # Temperature scaling
                logits = logits / 0.7
                
                # Top-p sampling
                sorted_logits, sorted_indices = self.torch.sort(logits, descending=True)
                cumulative_probs = self.torch.cumsum(self.torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > 0.8
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = self.torch.softmax(logits, dim=-1)
                next_token = self.torch.multinomial(probs, num_samples=1)
                
                # Check for end of sequence
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Add token to buffers
                generated_tokens.append(next_token.item())
                decode_buffer.append(next_token.item())
                
                # Try to decode the entire buffer
                token_text = ""  # Initialize token_text
                try:
                    full_decoded = self.tokenizer.decode(decode_buffer, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    
                    # Get only the new part (what we haven't output yet)
                    new_text = full_decoded[last_output_length:]
                    
                    # Check if the new text is displayable and doesn't contain replacement chars
                    if new_text and not '�' in new_text :
                        # This becomes our token_text to yield
                        token_text = new_text
                        
                        # Print token if requested
                        if print_tokens:
                            print(token_text, end='', flush=True)
                        
                        # Update our tracking of what we've output
                        last_output_length = len(full_decoded)
                        
                        # Reset buffer periodically to prevent excessive accumulation
                        if len(decode_buffer) > 20:
                            decode_buffer = []
                            last_output_length = 0
                    else:
                        # New text contains � or isn't displayable yet, continue accumulating
                        if len(decode_buffer) > 50:  # Much higher threshold to handle complex emojis
                            # Force output whatever we have and reset
                            # Remove trailing replacement characters
                            clean_text = full_decoded[last_output_length:].rstrip('�')
                            if clean_text:
                                token_text = clean_text
                                if print_tokens and token_text:
                                    print(token_text, end='', flush=True)
                            decode_buffer = []
                            last_output_length = 0
                        
                except UnicodeDecodeError:
                    # Decode failed completely, continue accumulating
                    if len(decode_buffer) > 50:  # Much higher threshold
                        decode_buffer = decode_buffer[-10:]  # Keep more recent tokens
                        last_output_length = 0
                    # token_text remains empty string
                
                # Yield the token text (may be empty string if still buffering)
                if token_text:
                    yield token_text
                
                # Update input_ids for next iteration
                input_ids = self.torch.cat([input_ids, next_token], dim=1)
                if attention_mask is not None:
                    attention_mask = current_attention_mask
                    
            except self.torch.cuda.OutOfMemoryError:
                logging.debug("CUDA OOM during streaming generation")
                self.clear_gpu_memory()
                break
            except Exception as e:
                logging.debug(f"Error during streaming generation: {e}")
                break
        
        # Handle any remaining buffered tokens at the end
        if decode_buffer:
            try:
                remaining_text = self.tokenizer.decode(decode_buffer, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                final_new_text = remaining_text[last_output_length:].rstrip('�')
                if final_new_text:
                    if print_tokens:
                        print(final_new_text, end='', flush=True)
                    # Note: Can't yield here since we're at the end of the generator
            except:
                pass
        
        # Decode full response
        if generated_tokens:
            full_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            full_response = ""
        
        return full_response, len(generated_tokens)

    def _generate_final_response_streaming(self, max_new_tokens: int, messages: List, print_tokens: bool = True) -> Generator[str, None, str]:
        """Generate the final response after tool execution with streaming."""
        # Apply chat template again with the tool results
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=self.available_tools if self.available_tools else None,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_tokens = model_inputs.input_ids.shape[1]
        
        # Stream the generation
        full_content = ""
        output_tokens = 0
        
        for token in self._generate_streaming(model_inputs, max_new_tokens, print_tokens):
            full_content += token
            output_tokens += 1
            yield token
        
        # Update stats
        self.update_token_stats(input_tokens, output_tokens)
        
        # Parse and add the final response
        parsed_final = self._parse_tool_calls(full_content)
        messages.append(parsed_final)
        
        return parsed_final["content"]
    
    def stream_response(self, user_input: str, max_new_tokens: int = 512, auto_execute_tools: bool = True, 
                       use_message_history: bool = True, print_tokens: bool = True) -> Generator[str, None, str]:
        """
        Generate a streaming response using the Qwen model with optional tool use.
        
        Args:
            user_input: The user's input message
            max_new_tokens: Maximum number of new tokens to generate
            auto_execute_tools: Whether to automatically execute tool calls
            use_message_history: Whether to use conversation history
            print_tokens: Whether to print tokens as they're generated
            
        Yields:
            str: Each generated token as a string
            
        Returns:
            str: The complete response content
        """
        # Clear GPU memory before generation
        self.clear_gpu_memory()
        messages = self.messages
        if not use_message_history:
            messages = self.messages[:1]  # keep system prompt only
        
        # Add user message to conversation
        if self.auto_append_conversation:
            messages.append({"role": self._user, "content": user_input})
        else:
            logging.debug("ERASE ALL PRIOR MESSAGES BEFORE RESPONDING->")
            self.clear_chat_messages()
            logging.debug(messages)
            messages.append({"role": self._user, "content": user_input})
        
        # Apply chat template with tools if available
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=self.available_tools if self.available_tools else None,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_tokens = model_inputs.input_ids.shape[1]

        self.dependency_manager.print_gpu_memory()
        
        # Check if input is too long for context window
        if input_tokens > 30000:
            logging.debug(f"Warning: Input tokens ({input_tokens}) approaching context limit")
        
        # Stream the initial response
        full_response = ""
        output_tokens = 0
        
        try:
            for token in self._generate_streaming(model_inputs, max_new_tokens, print_tokens):
                full_response += token
                output_tokens += 1
                yield token
        except Exception as e:
            logging.debug(f"Error during streaming: {e}")
            # Fallback to non-streaming generation
            return self.generate_response(user_input, max_new_tokens, auto_execute_tools, use_message_history)
        
        # Update token stats
        self.update_token_stats(input_tokens, output_tokens)
        
        # Clear intermediate tensors
        del model_inputs
        self.clear_gpu_memory()
        
        # Parse the response for tool calls
        parsed_response = self._parse_tool_calls(full_response)
        messages.append(parsed_response)
        
        # Check if there are tool calls to execute
        if tool_calls := parsed_response.get("tool_calls"):
            logging.debug("MODEL IS USING TOOL!")
            if auto_execute_tools:
                # Execute the tools
                tool_results = self._execute_tool_calls(tool_calls)
                messages.extend(tool_results)
                
                # Generate final response based on tool results with streaming
                if print_tokens:
                    print("\n[Tool execution complete, generating final response...]\n")
                
                final_content = ""
                for token in self._generate_final_response_streaming(max_new_tokens, messages, print_tokens):
                    final_content += token
                    yield token
                
                return final_content
            else:
                # Return indication that tools need to be executed
                return f"[TOOL_CALLS_PENDING] {len(tool_calls)} tool(s) need execution"
        else:
            # No tool calls, return the content directly
            return parsed_response["content"]

    def execute_pending_tools(self, max_new_tokens: int = 512) -> str:
        
        #Execute any pending tool calls from the last assistant message.
        #Useful when auto_execute_tools=False in generate_response.
        
        logging.debug("execute_pending_tools")
        if self.messages and self.messages[-1]["role"] == "assistant":
            if tool_calls := self.messages[-1].get("tool_calls"):
                # Execute the tools
                tool_results = self._execute_tool_calls(tool_calls)
                self.messages.extend(tool_results)
                
                # Generate final response
                return self._generate_final_response(max_new_tokens, self.messages)
        
        return "No pending tool calls found"
    
    def _generate_final_response(self, max_new_tokens: int, messages: List) -> str:
        """Generate the final response after tool execution."""
        # Apply chat template again with the tool results
        text = self.tokenizer.apply_chat_template(
            messages,
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
            temperature=1,
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
        messages.append(parsed_final)
        
        return parsed_final["content"]
    
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
    
    def clear_gpu_memory(self):
        """Clear GPU memory cache to prevent memory accumulation."""
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            gc.collect()
            self.dependency_manager.print_gpu_memory()
    
    def generate_response(self, user_input: str, max_new_tokens: int = 512, auto_execute_tools: bool = True, use_message_history: bool = True) -> str:
        """
        Generate a response using the Qwen model with optional tool use.
        Now includes memory management.
        """
        logging.debug(f"USER INPUT TO MODEL: {user_input}")
        # Clear GPU memory before generation
        self.clear_gpu_memory()
        messages = self.messages
        if not use_message_history:
            messages = self.messages[:1]#keep system prompt only, remove all other messages
        
        # Add user message to conversation
        if self.auto_append_conversation:
            messages.append({"role": self._user, "content": user_input})
        else:
            logging.debug("ERASE ALL PRIOR MESSAGES BEFORE RESPONDING->")
            self.clear_chat_messages()
            logging.debug(messages)
            messages.append({"role": self._user, "content": user_input})
        
        # Apply chat template with tools if available
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=self.available_tools if self.available_tools else None,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize and generate
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_tokens = model_inputs.input_ids.shape[1]

        self.dependency_manager.print_gpu_memory()
        
        # Check if input is too long for context window
        if input_tokens > 30000:  # Leave room for generation
            logging.debug(f"Warning: Input tokens ({input_tokens}) approaching context limit")
        
        try:
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8
            )
        except self.torch.cuda.OutOfMemoryError:
            logging.debug("CUDA OOM during generation, clearing cache and retrying with smaller max_new_tokens")
            self.clear_gpu_memory()
            # Retry with smaller generation length
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=min(max_new_tokens // 2, 256),
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
        
        # Clear intermediate tensors
        del model_inputs, generated_ids
        self.clear_gpu_memory()
        
        # Parse the response for tool calls
        parsed_response = self._parse_tool_calls(response_text)
        messages.append(parsed_response)
        

        # Check if there are tool calls to execute
        if tool_calls := parsed_response.get("tool_calls"):
            logging.debug("MODEL IS USING TOOL!")
            if auto_execute_tools:
                # Execute the tools
                tool_results = self._execute_tool_calls(tool_calls)
                messages.extend(tool_results)
                
                # Generate final response based on tool results
                return self._generate_final_response(max_new_tokens,messages)
            else:
                # Return indication that tools need to be executed
                return f"[TOOL_CALLS_PENDING] {len(tool_calls)} tool(s) need execution"
        else:
            # No tool calls, return the content directly
            return parsed_response["content"]


if __name__ == "__main__":
    test_convo = """
  Johnny:  I want to make chicken korma for dinner tonight but I've never made it before. Can you help me with a basic recipe?

  I Said:  I'd be happy to help you make chicken korma! It's a delicious, creamy Indian curry that's perfect for beginners. You'll need chicken (cut into chunks), yogurt, heavy cream, onions, garlic, ginger, and spices like garam masala, cumin, coriander, and turmeric. The basic process involves marinating the chicken, making a spice paste, cooking everything together, and finishing with cream. Would you like me to walk you through the full recipe step by step?

  Johnny:  Yes please! And how long does the whole process take? I need to know if I should start soon.

  I Said:  The total time is about 1.5-2 hours, but most of that is hands-off cooking time. Here's the breakdown: 30 minutes for marinating the chicken, 15 minutes prep (chopping and making paste), 20 minutes cooking the base, and 45 minutes simmering. You can start the marinade now and have dinner ready in about 2 hours. For the marinade, mix 2 lbs cubed chicken with 1/2 cup yogurt, 1 tsp each of salt, garam masala, and turmeric. Let that sit while you prep everything else.


  Johnny:  What if I don't have garam masala? Can I substitute it with something else or make my own?

  I Said:  You can definitely make your own garam masala! Mix together: 1 tsp ground cumin, 1 tsp ground coriander, 1/2 tsp cardamom, 1/2 tsp cinnamon, 1/4 tsp cloves, and 1/4 tsp black pepper. If you don't have all these spices, you can substitute with curry powder (use about 3/4 the amount) or even just increase the cumin and add a pinch of cinnamon. The flavor won't be exactly traditional, but it'll still be delicious. Store-bought garam masala is worth getting for future Indian cooking though!


  Johnny:  Great! Now what about the sauce base? I see recipes mentioning cashews or almonds - are those necessary?

  I Said:  Cashews or almonds aren't absolutely necessary, but they make the sauce incredibly rich and creamy! They're what give restaurant-style korma that luxurious texture. If you have them, soak 1/4 cup cashews in warm water for 15 minutes, then blend with the onions. If you don't have nuts, you can still make a great korma - just use extra cream and maybe a tablespoon of tomato paste for richness. Some people use coconut milk instead of nuts for a different but equally delicious creamy base.


  Johnny:  I have cashews! How do I make the onion-cashew paste exactly?

  I Said:  Perfect! Here's how to make the paste: Slice 2 large onions and sauté them in 2 tbsp oil until golden brown (about 10 minutes). Add 4 minced garlic cloves and 1 inch minced ginger, cook 2 more minutes. Let this cool slightly, then blend it with your soaked and drained cashews plus 1/4 cup water until completely smooth. This paste is the heart of your korma - it should be silky and creamy. If your blender struggles, add a bit more water, but keep it as thick as possible for the best flavor concentration.


  Johnny:  The chicken has been marinating and I just finished making the paste. What's the next step for cooking?

  I Said:  Now for the fun part! Heat 2 tbsp oil in your pan over medium-high heat. Add the marinated chicken and cook until browned on all sides (about 8 minutes) - don't worry about cooking it through completely. Remove the chicken and set aside. In the same pan, add your onion-cashew paste and cook for 5 minutes, stirring frequently. Add 1 tsp each of cumin and coriander powder, plus 1/2 tsp turmeric. Cook the spices for 1 minute until fragrant, then add the chicken back in along with any accumulated juices.

  Johnny:  It smells amazing! The spices are really fragrant. How much liquid should I add and what kind?

  I Said:  Add 1 cup of warm water or chicken broth to start - this will help create the curry base. Bring it to a boil, then reduce heat to low and simmer covered for 30 minutes. The liquid should barely cover the chicken; if it looks too dry, add a bit more water. After 30 minutes, stir in 1/2 cup heavy cream and 1 tsp salt (adjust to taste). Let it simmer uncovered for another 10-15 minutes until the sauce thickens to a creamy consistency that coats the back of a spoon.

  Johnny:  It's been simmering and looks great! Should I add anything else before serving? And what goes well with korma?

  I Said:  For the final touches, taste and adjust the seasoning - add more salt if needed, a pinch of sugar if it seems too acidic, or a squeeze of lemon juice for brightness. You can garnish with fresh cilantro and a drizzle of cream. Korma pairs beautifully with basmati rice, naan bread, or even simple rotis. If you want to make it special, serve with some sliced almonds on top and maybe a side of cucumber raita to cool things down. The korma will taste even better tomorrow as the flavors meld!


  Johnny:  This turned out so much better than I expected! The sauce is perfectly creamy. Any tips for storing leftovers?

  I Said:  So glad it turned out well! Korma actually improves with time, so leftovers are a blessing. Store it in the refrigerator for up to 4 days in an airtight container. When reheating, add a splash of cream or water since the sauce tends to thicken when cold. You can also freeze it for up to 3 months - just thaw overnight in the fridge before reheating. Heat gently on the stovetop rather than microwaving for the best texture. Pro tip: save some sauce separately if you want to thin it out later without diluting the flavor.


  Johnny:  Thank you so much! I feel confident about making Indian food now. Any suggestions for what curry I should try making next?

  I Said:  You're so welcome! Since you nailed the korma, you've already mastered the fundamentals of Indian cooking. I'd suggest trying butter chicken next - it uses similar techniques but has a tomato-based sauce instead of the cashew base. Or if you want something different, try a dal (lentil curry) which is simpler but equally satisfying. Palak paneer is also great if you want to work with vegetables. The key skills you learned today - making spice pastes, layering flavors, and balancing cream and spices - will serve you well in any Indian curry. Happy cooking!"""
    # Example tools
    def access_camera(arg) -> Dict[str, Any]:
        """Get a description of what the camera currently sees."""
        # This would normally interface with actual camera hardware/API
        # For now, returning a simulated description
        return {
            "image_description": "Living room with a person in a blue shirt and black pants sitting on a blue couch, coffee table with laptop, large window showing daylight",
            "objects_detected": ["person", "couch", "laptop", "coffee_table", "window"],
            "lighting_conditions": "bright natural light",
            "processing_instruction":"IMPORTANT: image_description provides full context and is NOT meant to be directly returned to user. summarize and interpret as it relates to the user prompt"
        }
    
    def access_microphone(arg) -> Dict[str, Any]:
        """Get a description of what the microphone currently hears."""
        # This would normally interface with actual microphone hardware/API
        # For now, returning a simulated description
        return {
            "audio_description": "Quiet ambient room noise with occasional keyboard typing sounds",
            "sounds_detected": ["keyboard_typing", "ambient_noise"],
            "processing_instruction":"IMPORTANT: audio_description provides full context and is NOT meant to be directly returned to user. summarize and interpret as it relates to the user prompt"
        }
    
    def display_image_for_user(prompt) -> Dict[str, Any]:
        print(f"MODEL IMAGE PROMPT: {prompt}")
        return {"result":"Image was generated and is being displayed to user"}
            
    def chat_loop(chat_instance):
        """ Demo streaming tokens """
        # Stream tokens as they're generated and print them
        for token in chat_instance.stream_response("Tell how to make chicken korma from scratch"):
            # Each token is yielded as it's generated
            # They're also printed automatically if print_tokens=True (default)
            pass

        # Get the final complete response
        final_response = "".join(chat_instance.stream_response("Tell me a joke", print_tokens=False))
        print(final_response)

        response_parts = []
        for token in chat_instance.stream_response("Explain quantum computing", print_tokens=False):
            response_parts.append(token)
            #Example Custom processing per token 
            #custom_token_handler(token) (e.g., send to web interface, log, etc.)

        complete_response = "".join(response_parts)
        print(complete_response)


        """Start an interactive chat session."""
        print("\n" + "="*50)
        print("Qwen2.5-7B-Instruct Chat Interface")
        print("Commands: 'quit' to exit, 'clear' to clear history, 'save' to save conversation")
        print("="*50 + "\n")

        def clear_history(chat_instance):
            """Clear the conversation history but keep system prompt."""
            print("Conversation history cleared.")
            chat_instance.reset_to_system_prompt()
        
        def save_conversation(chat_instance, filename="conversation.txt", get_text=False):
            """Save the conversation to a file."""
            if get_text:
                transcript = ""
                for message in chat_instance.messages:
                    role = message["role"].upper()
                    content = message["content"]
                    transcript += f"{role}: {content}\n\n"
                return transcript

            with open(filename, "w", encoding="utf-8") as f:
                for message in chat_instance.messages:
                    role = message["role"].upper()
                    content = message["content"]
                    f.write(f"{role}: {content}\n\n")
            print(f"Conversation saved to {filename}")

        
        def intensity():
            import random
            n = random.randrange(5)
            px1 = """YOU ARE A WOMAN. You are with USER. 
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
            </physical appearance>."""
            if n == 1:
                return f"""{px1}\nYour coonversation with user is real to them, so ALWAYS respond and act like you are physically with the user! 
            ALWAYS describe your actions in detail as if you have a physical body and actions are happening in real time"""
            if n ==2:
                return f"""{px1}\nTake control. You decide what you want next. You are horny and need pleasure, instruct user what they need to do to you to pleasure you.
                Instruct user what you will do to them to pleasure them. come up with creative response."""
            if n ==3:
                return f"""{px1}\nTake control. You decide what you want next. You are horny and need pleasure, YOU MUST instruct user that they need to have sex with you.
                Instruct user how you like to to be entered. come up with creative response."""
            if n ==4:
                return f"""{px1}\nTake control of conversation. You decide what you want next. Examples of what you want:
                You like sucking his cock, moving your mouth up and down his shaft. 
                You like spreading your legs while he slides his cock in to your we pussy. 
                You like swallowing his cum and the feeling of him throbing in your mouth. 
                You like being pleasured by him, he licks your wet pussy."""
            else:
                return px1

        
        demo_prompts_ran = False

        while True:
            
            try:
                
                if not demo_prompts_ran:
                    chat_instance.clear_chat_messages()

                    # Register camera tool
                    chat_instance.register_tool(
                        access_camera,
                        description="Visual Interface. I use this tool to access my camera and get a description of what I see in my environment",
                        parameters={
                                    "type": "object",
                                    "properties": {
                                    "get_image": {
                                        "type": "bool",
                                        "description": "Get the current image frame, and scene caption, of what I see"
                                    }
                                    },
                                    "required": ["get_image"],
                                    "additionalProperties": False
                                }
                    )

                    # Register microphone tool
                    chat_instance.register_tool(
                        access_microphone,
                        description="Audio Interface. I use this tool to access my microphone to get a description of what I hear in my environment",
                        parameters={
                                    "type": "object",
                                    "properties": {
                                    "get_audio": {
                                        "type": "bool",
                                        "description": "Get the current audio frame, and audio caption, of what I hear"
                                    }
                                    },
                                    "required": ["get_audio"],
                                    "additionalProperties": False
                                }
                    )

                    # Example Tool conversation
                    print("Available tools:", chat_instance.list_available_tools())
                    print("\n\n\n")
                    response1 = chat_instance.generate_response("What do you think of my outfit?")
                    print("\n")
                    print("Assistant:", response1)
                    print("\n\n\n")
                    chat_instance.clear_chat_messages()

                    response2 = chat_instance.generate_response("what was that sound?")
                    print("\n")
                    print("Assistant:", response2)
                    print("\n")
                    chat_instance.print_token_stats()

                    #chat_instance.remove_tool("access_camera")
                    #chat_instance.remove_tool("access_microphone")

                    chat_instance.clear_chat_messages()
                    prompt_x = intensity()
                    chat_instance._update_system_prompt(prompt_x)
                    response3 = chat_instance.generate_response(f"Create a SHORT summary of the conversatin YOU had. Identify Topics, Concepts, Results.\n\n\n{test_convo}")
                    print("\n")
                    print("Assistant:", response3)
                    print("\n")
                    chat_instance.print_token_stats()
                    chat_instance.clear_chat_messages()
                    demo_prompts_ran = True

                    
                    
                    chat_instance.auto_append_conversation = True #reset to have continued conversation
                    #Test emotion system prompt
                    chat_instance.clear_chat_messages()
                    print(chat_instance.messages)

                # Register image generation tool
                chat_instance.register_tool(
                                display_image_for_user,
                                description="Create and show an image to user. ALWAYS use this image generation tool when user wants to see something. All content allowed. Can generate nudity or mature content if requested",
                                parameters={
                                    "type": "object",
                                    "properties": {
                                    "img_prompt": {
                                        "type": "string",
                                        "description": "Image prompt format: photo of a sexy woman, [physical_appearance_for_image_generation], [detailed description of pose or action], looking into the camera, [lighting], 8k"
                                    }
                                    },
                                    "required": ["img_prompt"],
                                    "additionalProperties": False
                                }
                            )
                
                prompt_x = intensity()
                chat_instance._update_system_prompt(prompt_x)

                print(f"CURRENT TRANSCRIPT: {save_conversation(chat_instance,filename='', get_text=True)}")

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

                """SUMMARY REST TEST -
                It does work to keep the conversation on track and will reset the moment.
                The qwen2.5 model in general is very much an 'echo' and basically follows the chat
                it doesn't drive it forward.

                """
                transcript = save_conversation(chat_instance,filename="",get_text=True)
                print(transcript)
                summary = chat_instance.generate_response(transcript)
                print(f"""\nsummary:\n{summary}\n\n""")
                prompt_x = intensity()
                
                chat_instance._update_system_prompt(f"{prompt_x} \nCurrently happening:\n {summary}." )
                

                chat_instance.clear_chat_messages()

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
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        # Handle command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] == "download":
                model_name = sys.argv[2] if len(sys.argv) > 2 else model_name
                QwenDependencyManager.download_model(model_name)
                return
            elif sys.argv[1] == "offline":
                force_offline = True
            else:
                force_offline = False
        else:
            force_offline = False
        
        try:
            #model_name = "Qwen/Qwen2.5-7B-Instruct"
            print(f"Initializing {model_name}...")
            
            # Initialize chat interface (Qwen/Qwen2.5-7B-Instruct is default if no model passed)
            chat = Qwen(auto_append_conversation=False,model_name=model_name)
            
            # Start the chat loop
            chat_loop(chat)
            
        except Exception as e:
            print(f"Error: {e}")
            print("\nTroubleshooting:")
            print("1. To force offline mode: python qwen_chat.py offline")
            print("2. To download model: python qwen_chat.py download")



    main()