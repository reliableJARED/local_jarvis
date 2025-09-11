import subprocess
import sys
import json
import re
from typing import List, Dict, Any, Callable

def install_dependencies():
    """Install required packages if not available."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "transformers"])
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    
    return torch, AutoModelForCausalLM, AutoTokenizer

def load_model(model_name="Qwen/Qwen2.5-7B-Instruct"):
    """Load model and tokenizer."""
    torch, AutoModelForCausalLM, AutoTokenizer = install_dependencies()
    
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer, torch

class SimpleQwen:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.model, self.tokenizer, self.torch = load_model(model_name)
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        self.tools = {}
        self.available_tools = []
    
    def register_tool(self, func: Callable, name: str = None, description: str = None):
        """Register a tool function."""
        if name is None:
            name = func.__name__
        
        self.tools[name] = func
        
        # Simple tool definition
        tool_def = {
            "type": "function",
            "function": {
                "name": name,
                "description": description or func.__doc__ or f"Function {name}",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
        
        # Update tools list
        self.available_tools = [t for t in self.available_tools if t["function"]["name"] != name]
        self.available_tools.append(tool_def)
    
    def _parse_tool_calls(self, content: str) -> Dict[str, Any]:
        """Parse tool calls from model output."""
        tool_calls = []
        
        # Look for tool call blocks: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
        for match in re.finditer(r"<tool_call>\n(.+?)\n</tool_call>", content, re.DOTALL):
            try:
                tool_call_json = json.loads(match.group(1).strip())
                tool_calls.append({
                    "type": "function", 
                    "function": tool_call_json
                })
            except json.JSONDecodeError:
                continue
        
        if tool_calls:
            # Extract content before tool calls
            offset = content.find("<tool_call>")
            content_text = content[:offset].strip() if offset > 0 else ""
            return {
                "role": "assistant",
                "content": content_text,
                "tool_calls": tool_calls
            }
        else:
            return {
                "role": "assistant",
                "content": content.strip()
            }
    
    def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute tool calls and return results."""
        tool_results = []
        
        for tool_call in tool_calls:
            function_call = tool_call.get("function")
            if function_call:
                function_name = function_call["name"]
                function_args = function_call.get("arguments", {})
                
                if function_name in self.tools:
                    try:
                        result = self.tools[function_name](function_args)
                        tool_results.append({
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps(result) if not isinstance(result, str) else result
                        })
                    except Exception as e:
                        tool_results.append({
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error: {str(e)}"
                        })
                else:
                    tool_results.append({
                        "role": "tool",
                        "name": function_name,
                        "content": f"Function {function_name} not found"
                    })
        
        return tool_results
    
    def chat(self, user_input: str) -> str:
        """Generate response with tool support."""
        # Add user message
        self.messages.append({"role": "user", "content": user_input})
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tools=self.available_tools if self.available_tools else None,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate response
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.8
        )
        
        # Extract new tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Parse response for tool calls
        parsed_response = self._parse_tool_calls(response_text)
        self.messages.append(parsed_response)
        
        # Execute tools if present
        if tool_calls := parsed_response.get("tool_calls"):
            tool_results = self._execute_tool_calls(tool_calls)
            self.messages.extend(tool_results)
            
            # Generate final response after tool execution
            text = self.tokenizer.apply_chat_template(
                self.messages,
                tools=self.available_tools if self.available_tools else None,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.8
            )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            final_response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            final_parsed = self._parse_tool_calls(final_response)
            self.messages.append(final_parsed)
            
            return final_parsed["content"]
        
        return parsed_response["content"]

# Example usage
if __name__ == "__main__":
    # Initialize chat
    chat = SimpleQwen()
    
    # Example tool
    def get_weather(args):
        """Get weather information for a location."""
        location = args.get("location", "unknown")
        return f"The weather in {location} is sunny and 75°F"
    
    # Register tool
    chat.register_tool(get_weather, description="Get current weather for a location")
    
    # Chat loop
    print("Chat started! Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            break
        
        if user_input:
            response = chat.chat(user_input)
            print(f"Assistant: {response}")