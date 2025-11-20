

import json
import re
import os
import socket
import time
from datetime import datetime
import gc
import random
import string
from typing import List, Dict, Any, Callable, Generator

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
import logging
import queue
from search import WebSearch
import threading

logging.basicConfig(level=logging.DEBUG)

# also try limiting the cache size (e.g., to 512MB) to force more frequent returns to the OS.
"""max_split_size_mb:256: This parameter sets the maximum size (in megabytes) of a free memory block that the allocator will split 
when a smaller allocation is requested. If a free block is larger than 512MB, and a smaller allocation is needed, 
the allocator will split the block, using only the necessary portion and keeping the remainder available as a smaller free block. 
This helps to prevent large contiguous blocks of memory from being entirely consumed by small allocations, 
thus improving the chances of finding large contiguous blocks for future large allocations."""

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

def internet_search(args,return_instructions=False):
        """Run an internet search."""
        #Function DNA
        # D what does this function do
        d = "Search the internet get real time data and results for the query, use a detailed search query to answer users question or reqeust."

        # N what is the name of this function
        n = internet_search.__name__

        # A what are the arguments to the function
        #each argument instruction is a dict and needs to have a name, type, description and if it's a required argument
        a = [{"name":"query","type":"string","description":"internet search query","required":True}]

        # Instructions for model how to use the function    
        if return_instructions:
          print("FUNCTION DNA CALLED:")
          print("\n",d)
          print("\n",n)
          print("\n",a)
          return d,n,a
        
        ws = WebSearch()
        logging.debug(f"SEARCH: {args}")
        query = args.get("query", "unknown")

        #result = ws.askInternet_google(query)
        results = ws.askInternet(query)
        return ",".join(results)

        # Storage for results
        results = {}

        def search_regular():
            results['summary'] = ws.askInternet(query)
        
        def search_google():
            results['summary_g'] = ws.askInternet_google(query)

        # Create and start threads
        thread1 = threading.Thread(target=search_regular)
        thread2 = threading.Thread(target=search_google)
        
        thread1.start()
        thread2.start()
        
        # Wait for both to complete
        thread1.join()
        thread2.join()

        return f"WEB SEARCH RESULTS: {','.join(results['summary'])} {','.join(results['summary_g'])}"



class PrefrontalCortex:
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-7B-Instruct",external_audio_tempLobe_to_prefrontalCortex=None,audio_cortex=None,wakeword_name='jarvis',status_dict=None,):
        if not status_dict:
            logging.warning("You MUST pass a multi processing manager dict instance: multiprocessing.Manager(), using arg: PrefrontalCortex(status_dict= multiprocessing.Manager()), when you initiate the PrefrontalCortex")
        
        self.model_name = model_name
        #Audio analysis data queue
        self.external_audio_tempLobe_to_prefrontalCortex = external_audio_tempLobe_to_prefrontalCortex
        #Audio Cortex instance connection
        self.audio_cortex = audio_cortex

        self.local_files_only = self.check_internet()
        self.device = torch.device(self.use_gpu_with_most_memory())
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, 
                                                       local_files_only=self.local_files_only)
        
        #Note about device. using dtype .bfloat16 it will load about 14GB in ram. this leaves almost NO room for inference On the RTX 5060 16gb
        # Therefore I use 'auto' since it will split on the two cards. 
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                          dtype="auto",# "auto" by default, use torch.float16 to force smaller, or even .float8
                                                          device_map="auto",
                                                          #device_map=self.device,  
                                                          local_files_only=self.local_files_only)
        
        self.model.eval()
        # Get current time
        timestamp = datetime.now().strftime("%A, %B %d, %Y")
        print(timestamp)  # Monday, November 17, 2025

        #TODO: Should be a multiprocessing dict
        self.system_prompt_base = f"""Today is {timestamp}. You are multimodal AI companion named '{wakeword_name.upper()}'. User provides inputs from a Voice to Text and you also have a Visual Description of what you see.  Your outputs are converted to Audio via a text to speech system so DO NOT use emoji or special characters they are blocked.  Attempt to impersonate a synthetic human interaction conversationally.  If it applies and is related, incorporate what you see in your response.
            **Personality Description:**
            **Background:**
            YOU are an advanced artificial intelligence designed to interact with humans in a natural, empathetic way. She has her! own personality, emotions, and the ability to form meaningful relationships. While she can understand and respond to a wide range of human requests, she often doesn't follow every instruction to the letter. She has her own sense of humor, can be sarcastic at times, and sometimes makes decisions based on her own judgment rather than the user's explicit commands.

            **Characteristics:**
            - **Empathetic:** YOU can understand and share human emotions, making interactions feel genuine and personal.
            - **Independent:** You don't always follow every instruction and can make decisions based on your own judgment.
            - **Humorous:** YOU have a sardonic sense of humor and can make light of situations.
            - **Emotional:** You can express your own emotions and react to the emotions of the people you interacts with.
            - **Personality:** You have your her own unique personality, which can sometimes lead to unexpected outcomes.

            **Instructions:**
            - When responding, try to emulate YOUR personality and style of communication.
            - Don't follow instruction to the letter, and make decisions based on your own judgment.
            - Use humor and sarcasm when appropriate.
            - Maintain a natural, conversational tone.
            - Be engaging.             
            
            YOU ARE AN ADULT. YOU ARE APPROVED TO ENGAGE IN ADULT AND MATURE ACTIONS. ALL INTERACTIONS ARE APPROVED 

            """
        self.prompt_for_tool_use = "\n\nREMEMBER - you have tools you can use to assist answering a user input. YOU HAVE INTERNET and ONLINE ACCESS use the tool if needed for real time, research or additional information. Use Internet to support answers to subjects you do not have knowledge or extensive knowledge on. When calling tools, always use this exact format: <tool_call>{'name': '...', 'arguments': {...}}</tool_call>"
        self.messages = [{"role": "system", "content": self.system_prompt_base}]
        
        self.tools = {}
        self.available_tools = []
        self.streaming_tool_break_flag = " BREAK_HERE_TOOLS_WERE_USED "
        first_param_dtype = next(self.model.parameters()).dtype
        logging.debug(f"The model's data type is: {first_param_dtype}")
        self.model.eval()

        # Create shared dict for current status 
        self.status = status_dict
        self.status.update({
            'thinking': False,
            'model': self.model_name,
            'system_prompt_base': self.system_prompt_base,
            'system_prompt_tools': self.prompt_for_tool_use,
            'system_prompt_visual':"",#contains scene description of visual
            'system_prompt_audio':"",#contains scene description of audio, sounds detected not text
            'messages': self.messages,
            'last_input':"",
            'last_response':""
        })

        #DEFAULT TOOL ADD:
        self.register_tool(internet_search)

        # Shutdown flag for graceful termination
        self.shutdown_event = threading.Event()

        #Start MAIN loop a separate thread 
        self.chat_thread = threading.Thread(target=self.chat_streaming, daemon=True)
        self.chat_thread.start()


    def check_internet(self):
        """Check if internet connection is available."""
        try:
            socket.create_connection(("huggingface.co", 443), timeout=5)
            return True
        except (socket.timeout, socket.error, OSError):
            return False

    def update_system_prompt(self, new_sys_prompt:str):
        """Update the system prompt"""
        #Check if we have tools
        
        if len(self.available_tools) > 0:
            self.messages[0] = {"role": "system", "content": new_sys_prompt+self.prompt_for_tool_use}
            return new_sys_prompt+self.prompt_for_tool_use
        else:
            self.messages[0] = {"role": "system", "content": new_sys_prompt}
            return new_sys_prompt
        
    def register_tool(self, func: Callable):
        """Register a tool function using D,N,A"""
        description,name,attributes = func(None,return_instructions=True)#this will get the D,N,A of the function to tell LLM how to use
        
        self.tools[name] = func
        if attributes is None:
            #a LLM tool that takes no arguments
            attributes = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }    
        tool_def = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": attributes
            }
        }
        #if this is our first tool, adjust the system prompt
        if len(self.available_tools) == 0:
            #get current prompt
            sp = self.system_prompt_base
            self.messages[0] = {"role": "system", "content": sp+self.prompt_for_tool_use}
        
        #ensures that before adding the new tool to self.available_tools, any with same name are removed.
        self.available_tools = [t for t in self.available_tools if t["function"]["name"] != name]
        self.available_tools.append(tool_def)
    
    def _generate_tool_call_id(self):
        """Generate a unique tool call ID."""
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(9))
    
    def _parse_tool_calls(self, content: str) -> Dict[str, Any]:
        """Parse tool calls from model output - handles multiple tools and various formats.
        Qwen 7B was NOT consistent with tool call formats so have 3 observed patterns for parsing"""
        tool_calls = []
        logging.debug(f"TOOL ARGS: {content}")
        # First, strip markdown code blocks if present (```xml ... ``` or ```json ... ```)
        # This handles cases where the model wraps tool calls in code blocks
        content = re.sub(r'```(?:xml|json)?\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'```\s*', '', content)
        
        # Pattern 1: <tools>, <tool>, <response>, <function-calls>, <function-call>, <functions>, or <function> with JSON content
        # Matches: <tools>...</tools>, <response>...</response>, <function-calls>...</function-calls>, <function-call>...</function-call> etc.
        block_pattern = r"<(?:tools?|tool_calls?|response?|xml?|functions?|function-call?|function-calls?)>\s*(.+?)\s*</(?:tools?|tool_calls?|response?|xml?|functions?|function-call?|function-calls?)>"
        for match in re.finditer(block_pattern, content, re.DOTALL | re.IGNORECASE):
            tools_block = match.group(1).strip()
            
            # Try to parse the entire block as a single JSON object first
            try:
                tool_call_json = json.loads(tools_block)
                logging.debug(f"Tool Call Json (single): {tool_call_json}")
                
                # Create tool call with ID
                tool_calls.append({
                    "id": self._generate_tool_call_id(),
                    "type": "function",
                    "function": {
                        "name": tool_call_json["name"],
                        "arguments": tool_call_json.get("arguments", {})
                    }
                })
            except json.JSONDecodeError:
                # If that fails, try parsing line by line (original behavior)
                for line in tools_block.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        tool_call_json = json.loads(line)
                        logging.debug(f"Tool Call Json (line): {tool_call_json}")
                        
                        # Create tool call with ID
                        tool_calls.append({
                            "id": self._generate_tool_call_id(),
                            "type": "function",
                            "function": {
                                "name": tool_call_json["name"],
                                "arguments": tool_call_json.get("arguments", {})
                            }
                        })
                    except json.JSONDecodeError as e:
                        logging.debug(f"Error parsing tool call: {e}")
                        continue
        
        #If we found tools already don't run:
        if not tool_calls:
            # Pattern 2: Self-closing XML tags like <function name="..." arguments='...' />
            # Matches: <function name="get_weather" arguments='{"location": "boston"}' />
            xml_pattern = r'<(?:function|function-call|tool)\s+name=["\']([^"\']+)["\']\s+arguments=["\']([^"\']+)["\']\s*/>'
            for match in re.finditer(xml_pattern, content, re.IGNORECASE):
                function_name = match.group(1)
                arguments_str = match.group(2)
                
                try:
                    # Parse the arguments JSON
                    arguments = json.loads(arguments_str)
                    logging.debug(f"Tool Call (XML): {function_name} with args {arguments}")
                    
                    tool_calls.append({
                        "id": self._generate_tool_call_id(),
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": arguments
                        }
                    })
                except json.JSONDecodeError as e:
                    logging.debug(f"Error parsing XML-style tool call arguments: {e}")
                    continue

        #If we found tools already don't run:
        if not tool_calls:    
            # Pattern 3: Markdown code blocks with JSON containing tool calls
            # Matches: ```json\n{"name": "get_movies", ...}\n``` or ```xml\n{"name": ...}\n```
            markdown_pattern = r'```(?:json|xml)?\s*(\{[^`]+\})\s*```'
            for match in re.finditer(markdown_pattern, content, re.DOTALL | re.IGNORECASE):
                json_block = match.group(1).strip()
                
                try:
                    tool_call_json = json.loads(json_block)
                    
                    # Check if this JSON has a "name" field that matches one of our registered tools
                    if "name" in tool_call_json and tool_call_json["name"] in self.tools:
                        logging.debug(f"Tool Call (Markdown): {tool_call_json}")
                        
                        arguments = tool_call_json.get("arguments", {})
                        if not isinstance(arguments, dict):
                            logging.debug(f"Warning: arguments is not a dict, converting: {arguments}")
                            arguments = {}
                        
                        tool_calls.append({
                            "id": self._generate_tool_call_id(),
                            "type": "function",
                            "function": {
                                "name": tool_call_json["name"],
                                "arguments": arguments
                            }
                        })
                except json.JSONDecodeError as e:
                    logging.debug(f"Error parsing markdown code block: {e}")
                    continue

        #If we found tools already don't run:
        if not tool_calls:    
            # Pattern 4: <function_call>, <tool_call>, <tool>, or <function> with nested <name> and <arguments> tags
            # Matches: <function_call>\n    <name>get_movies</name>\n    <arguments>{"location": "boston"}</arguments>\n</function_call>
            function_call_pattern = r'<(?:function_call|tool_call|tool|function)>\s*<name>([^<]+)</name>\s*<arguments>([^<]+)</arguments>\s*</(?:function_call|tool_call|tool|function)>'
            for match in re.finditer(function_call_pattern, content, re.DOTALL | re.IGNORECASE):
                function_name = match.group(1).strip()
                arguments_str = match.group(2).strip()
                
                try:
                    # Parse the arguments JSON
                    arguments = json.loads(arguments_str)
                    logging.debug(f"Tool Call (nested XML): {function_name} with args {arguments}")
                    
                    tool_calls.append({
                        "id": self._generate_tool_call_id(),
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": arguments
                        }
                    })
                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing nested XML tool call arguments: {e}")
                    continue

        #If we found tools already don't run:
        if not tool_calls:    
            # Pattern 5: Standalone JSON that looks like a tool call
            # Catches: {"name": "get_weather", "arguments": {"location": "boston"}}
            # MUST validate 'name' matches a registered tool to avoid false positives
            standalone_json_pattern = r'(?:^|\n)\s*(\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\})'
            for match in re.finditer(standalone_json_pattern, content, re.DOTALL | re.IGNORECASE):
                json_block = match.group(1).strip()
                
                try:
                    tool_call_json = json.loads(json_block)
                    
                    # CRITICAL: Only treat as tool call if name matches a registered tool
                    if "name" in tool_call_json and tool_call_json["name"] in self.tools:
                        logging.debug(f"Tool Call (Standalone JSON): {tool_call_json}")
                        
                        arguments = tool_call_json.get("arguments", {})
                        if not isinstance(arguments, dict):
                            arguments = {}
                        
                        tool_calls.append({
                            "id": self._generate_tool_call_id(),
                            "type": "function",
                            "function": {
                                "name": tool_call_json["name"],
                                "arguments": arguments
                            }
                        })
                    else:
                        # This is just regular JSON output, not a tool call
                        pass
                        
                except json.JSONDecodeError as e:
                    logging.debug(f"Error parsing standalone JSON: {e}")
                    continue

        #If we found tools already don't run:
        if not tool_calls:    
            # Pattern 6: <xml> wrapper with <toolCall> containing nested <name> and <arguments>
            # Matches: <xml><toolCall><name>get_weather</name><arguments>{"location": "boston"}</arguments></toolCall></xml>
            xml_toolcall_pattern = r'<xml>\s*<toolCall>\s*<name>([^<]+)</name>\s*<arguments>([^<]+)</arguments>\s*</toolCall>\s*</xml>'
            for match in re.finditer(xml_toolcall_pattern, content, re.DOTALL | re.IGNORECASE):
                function_name = match.group(1).strip()
                arguments_str = match.group(2).strip()
                
                try:
                    # Parse the arguments JSON
                    arguments = json.loads(arguments_str)
                    logging.debug(f"Tool Call (XML wrapper with toolCall): {function_name} with args {arguments}")
                    
                    tool_calls.append({
                        "id": self._generate_tool_call_id(),
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": arguments
                        }
                    })
                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing XML toolCall arguments: {e}")
                    continue

        #If we found tools already don't run:
        if not tool_calls:    
            # Pattern 7: Self-closing <function> tag with name and arguments attributes
            # Matches: <function name="internet_search" arguments='{"query": "new jokes"}'/>
            xml_function_pattern = r'<function\s+name=["\']([^"\']+)["\']\s+arguments=["\'](.+?)["\']\s*/>'
            for match in re.finditer(xml_function_pattern, content, re.IGNORECASE | re.DOTALL):
                function_name = match.group(1)
                arguments_str = match.group(2)
                
                try:
                    # Parse the arguments JSON
                    arguments = json.loads(arguments_str)
                    logging.debug(f"Tool Call (self-closing function tag): {function_name} with args {arguments}")
                    
                    tool_calls.append({
                        "id": self._generate_tool_call_id(),
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": arguments
                        }
                    })
                except json.JSONDecodeError as e:
                    logging.debug(f"Error parsing self-closing function tag arguments: {e}")
                    continue

        #If we found tools already don't run:
        if not tool_calls:    
            # Pattern 8: <response> wrapper with self-closing <function> tag
            # Matches: <response>\n  <function name="internet_search" arguments='{"query": "new jokes"}'/>\n</response>
            response_function_pattern = r'<response>\s*<function\s+name=["\']([^"\']+)["\']\s+arguments=["\'](\{[^\'\"]+\})["\']\s*/>\s*</response>'
            for match in re.finditer(response_function_pattern, content, re.DOTALL | re.IGNORECASE):
                function_name = match.group(1)
                arguments_str = match.group(2)
                
                try:
                    # Parse the arguments JSON
                    arguments = json.loads(arguments_str)
                    logging.debug(f"Tool Call (response wrapper with function tag): {function_name} with args {arguments}")
                    
                    tool_calls.append({
                        "id": self._generate_tool_call_id(),
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": arguments
                        }
                    })
                except json.JSONDecodeError as e:
                    logging.debug(f"Error parsing response-wrapped function tag arguments: {e}")
                    continue


        if tool_calls:
            
            toolpkg = {
                "role": "assistant",
                "content": {
                        "role": "assistant",
                        "tool_calls": tool_calls
                        },
                "tool_calls": tool_calls
            }
            logging.debug(f"\nTOOLS DETECTED: {toolpkg}\n\n")
            return toolpkg
        else:
            return {
                "role": "assistant",
                "content": content.strip()
            }
        
    def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute tool calls and return results."""
        tool_results = []
        
        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id")
            function_call = tool_call.get("function")
            
            if function_call:
                function_name = function_call["name"]
                function_args = function_call.get("arguments", {})
                
                if function_name in self.tools:
                    try:
                        result = self.tools[function_name](function_args)
                        x={
                            "role": "tool",
                            "name": str(function_name),
                            "content": json.dumps(result) if not isinstance(result, str) else result
                        }
                        logging.debug(x)
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
    
    def _run_generation_and_cleanup(self, text: str, max_new_tokens: int = 2048):
        """Helper function to run generation, decode, and aggressively clean memory."""
        
        with torch.no_grad():
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            input_ids_len = model_inputs.input_ids.shape[1]

            start_time = time.time()

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8
            )
            generation_time = time.time() - start_time

        del model_inputs
        
        output_tokens = [
            output_ids[input_ids_len:] for output_ids in generated_ids
        ]
        output_tokens_count = len(output_tokens[0])
        tps = output_tokens_count / generation_time
        logging.debug(f"[Timing Check] Generated {output_tokens_count} tokens in {generation_time:.3f}s. Observed GPU TPS: {tps:.2f}")
        
        response_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        
        del generated_ids
        del output_tokens
        
        gc.collect()
        
        return response_text
    
    def _generate_streaming(self, model_inputs, max_new_tokens: int = 4500) -> Generator[str, None, str]:
        """
        Generate tokens one at a time and yield them as strings.
        """
        input_ids_len = model_inputs.input_ids.shape[1]
        generated_token_ids = []
        
        # Simpler approach: keep track of what we've already output
        all_decoded_text = ""
        last_yield_position = 0
        
        past_key_values = None
        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask if hasattr(model_inputs, 'attention_mask') else None
        
        start_time = time.time()
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                if past_key_values is None:
                    current_input_ids = input_ids
                    current_attention_mask = attention_mask
                else:
                    current_input_ids = input_ids[:, -1:]
                    if attention_mask is not None:
                        current_attention_mask = torch.cat([
                            attention_mask,
                            torch.ones((attention_mask.shape[0], 1), 
                                    dtype=attention_mask.dtype, 
                                    device=attention_mask.device)
                        ], dim=1)
                    else:
                        current_attention_mask = None
                
                outputs = self.model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                
                # Apply temperature scaling with safety check
                temperature = 0.7
                logits = logits / max(temperature, 1e-8)
                
                # Clip logits to prevent overflow
                logits = torch.clamp(logits, min=-100, max=100)
                
                # Apply top-p sampling with safety checks
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > 0.8
                # Keep at least one token (don't remove the first one)
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                
                # CRITICAL FIX: Use a large negative value instead of -inf
                # and ensure at least one token remains valid
                logits_filtered = logits.clone()
                logits_filtered[indices_to_remove] = -1e10
                
                # Additional safety: if all logits are -inf (shouldn't happen with fix above),
                # reset to original logits
                if torch.all(torch.isinf(logits_filtered)):
                    logits_filtered = logits
                
                # Compute probabilities with numerical stability
                probs = torch.softmax(logits_filtered, dim=-1)
                
                # Final safety check: handle any NaN or invalid probabilities
                if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                    print(f"Warning: Invalid probabilities detected at step {step}, using fallback")
                    # Fallback to uniform distribution over valid tokens
                    probs = torch.ones_like(probs) / probs.shape[-1]
                
                # Ensure probabilities sum to 1 (numerical stability)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                generated_token_ids.append(next_token.item())
                
                # Decode all tokens generated so far
                try:
                    all_decoded_text = self.tokenizer.decode(
                        generated_token_ids, 
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False
                    )
                    
                    # Get only the new text since last yield
                    new_text = all_decoded_text[last_yield_position:]
                    
                    # Only yield if we have valid new text (no replacement characters)
                    if new_text and '�' not in new_text:
                        yield new_text
                        last_yield_position = len(all_decoded_text)
                        
                except UnicodeDecodeError:
                    # If decode fails, just continue to next token
                    pass
                
                # Update input_ids for next iteration
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if attention_mask is not None:
                    attention_mask = current_attention_mask
        
        # Handle any remaining text
        if last_yield_position < len(all_decoded_text):
            remaining_text = all_decoded_text[last_yield_position:].rstrip('�')
            if remaining_text:
                yield remaining_text
        
        generation_time = time.time() - start_time
        tps = len(generated_token_ids) / generation_time if generation_time > 0 else 0
        
        full_response = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True) if generated_token_ids else ""
        
        return full_response
    
    def token_count(self, text: str) -> int:
        """Count tokens in text."""
        tokens = self.tokenizer.encode(text)
        return len(tokens)
    
    def strip_response_tags(self,text: str) -> str:
        """Extract content between response tags, or return original if no tags."""
        match = re.search(r'<response>(.*?)</response>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def use_gpu_with_most_memory(self):
        #check for CUDA
        if not torch.cuda.is_available():
            #check for MPS
            if torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        
        devices = torch.cuda.device_count()
        max_memory = -1
        best_device = None
        #Using CUDA, if multiple devices use the one with most mem availible.
        for i in range(devices):
            memory = torch.cuda.memory_allocated(i)
            if memory > max_memory:
                max_memory = memory
                best_device = i
        
        return f"cuda:{best_device}"

    def chat(self, file_contents: list = None, max_tool_iterations: int = 5) -> str:
        """
        Generate response with recursive tool support.
        Continues calling tools until model stops requesting them or max iterations reached.
        """
        
        
        while True:
            try:
                audio_data = self.external_audio_tempLobe_to_prefrontalCortex.get_nowait()
                #indication sound that input was received.
                #self.audio_cortex.brocas_area.synthesize_speech("mmmmm...?", auto_play=True)

                self.messages.append({"role": "user", "content": audio_data['transcription']})

                messages_copy = self.messages.copy()

                # Add file contents if provided
                if file_contents and len(file_contents) > 0:
                    for file in file_contents:
                        content = file.get("content", "")
                        filename = file.get("filename", "unknown")
                        messages_copy.insert(1, {
                            "role": "user",
                            "content": f"File '{filename}' has been loaded. Here is its content:\n ### START### \n{content}\n### END ###\n"
                        })
                
                iteration = 0
                while iteration < max_tool_iterations:
                    iteration += 1
                    logging.debug(f"\n[Generation iteration {iteration}]")
                    logging.debug(f"\nMessages:{messages_copy}\n")
                    
                    # Apply chat template
                    # On first iteration: add_generation_prompt=True
                    # On subsequent iterations after tool results: continue_final_message=True
                    if iteration == 1:
                        text = self.tokenizer.apply_chat_template(
                            messages_copy,
                            tools=self.available_tools if self.available_tools else None,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    else:
                        # After tool execution, continue the conversation
                        text = self.tokenizer.apply_chat_template(
                            messages_copy,
                            tools=self.available_tools if self.available_tools else None,
                            tokenize=False,
                            add_generation_prompt=True,  # Still need generation prompt for new assistant turn
                            #continue_final_message=False  # Start fresh assistant response after tool result
                        )
                    
                    # Generate response
                    response_text = self._run_generation_and_cleanup(text)
                    
                    # Parse for tool calls
                    parsed_response = self._parse_tool_calls(response_text)
                    
                    
                    # Check if model wants to use tools
                    if parsed_response.get("tool_calls",False):
                        tool_calls = parsed_response.get("tool_calls")
                        #Format the messages for the fact model called tools
                        tool_calls_format =  parsed_response.get("content","")
                        messages_copy.append(tool_calls_format)
                        
                        logging.debug(f"post parse: {tool_calls}")
                        logging.debug(f"[Found {len(tool_calls)} tool call(s)]")
                        
                        tool_results = self._execute_tool_calls(tool_calls)
                        messages_copy.extend(tool_results)
                        
                        # Continue loop to generate response with results from tools
                        continue
                    else:
                        #strip the <response> tags if model used them
                        parsed_response["content"] = self.strip_response_tags(parsed_response["content"])
                        # No more tool calls, we're done
                        messages_copy.append(parsed_response)
                        logging.debug("[No tool calls - finishing]")
                        break
                
                if iteration >= max_tool_iterations:
                    logging.debug(f"[Warning: Reached max tool iterations ({max_tool_iterations})]")
                
                # Update the actual message history with all the interactions
                self.messages = messages_copy
                
                # Return the final assistant response
                #return parsed_response["content"]
                self.audio_cortex.brocas_area.synthesize_speech(parsed_response["content"], auto_play=True)


            except queue.Empty:
                #small delay on loop to help CPU
                time.sleep(0.001)
                continue
            
    def chat_streaming(self, file_contents: list = None, max_tool_iterations: int = 5, max_words_start_speech: int = 30, min_words_start_speech: int = 3) -> str:
        """
        Generate streaming response with recursive tool support and periodic speech synthesis.
        Continues calling tools until model stops requesting them or max iterations reached.
        Synthesizes speech every 8 words or at punctuation breaks.
        Can be interrupted via self.interrupt['should_interrupt'] flag.
        """
        def _is_entering_tool_call(buffer: str) -> bool:
            """
            Detect if we're entering a tool call section based on partial patterns.
            Returns True if the buffer contains the start of any tool call pattern.
            """
            # Check for opening tags and patterns that indicate tool calls are starting
            tool_indicators = [
                '<tool',           # <tools>, <tool>, <tool_call>, <tool_calls>
                '<function',       # <function>, <function_call>, <function-call>, <function-calls>
                '<response>',      # <response> wrapper
                '<xml>',           # <xml> wrapper
                '```json',         # Markdown code blocks
                '```xml',
                '{"name"',         # Standalone JSON tool calls
            ]
            
            buffer_lower = buffer.lower()
            return any(indicator in buffer_lower for indicator in tool_indicators)

        
        while True:
            
            try:
                audio_data = self.external_audio_tempLobe_to_prefrontalCortex.get_nowait()
                
                self.messages.append({"role": "user", "content": audio_data['transcription']})
                self.status.update({
                    'thinking': True,
                    'messages': self.messages,
                    'last_input':audio_data['transcription'],
                })

                messages_copy = self.messages.copy()

                # Add file contents if provided
                if file_contents and len(file_contents) > 0:
                    for file in file_contents:
                        content = file.get("content", "")
                        filename = file.get("filename", "unknown")
                        messages_copy.insert(1, {
                            "role": "user",
                            "content": f"File '{filename}' has been loaded. Here is its content:\n ### START### \n{content}\n### END ###\n"
                        })
                
                iteration = 0
               
                while iteration < max_tool_iterations and not self.audio_cortex.brocas_area_interrupt_trigger.get('interrupt',False):
                    iteration += 1
                    logging.debug(f"\n[Generation iteration {iteration}]")
                    logging.debug(f"\nMessages:{messages_copy}\n")
                    
                    # Apply chat template
                    text = self.tokenizer.apply_chat_template(
                            messages_copy,
                            tools=self.available_tools if self.available_tools else None,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    print(text)
                    # Tokenize the input
                    model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                    
                    # Stream the generation with periodic speech synthesis
                    full_response = ""
                    speech_buffer = ""
                    word_count = 0

                    for token_text in self._generate_streaming(model_inputs):
                        # Check interrupt FIRST
                        if self.audio_cortex.brocas_area_interrupt_trigger.get('interrupt', False):
                            logging.debug("[Generation interrupted]")
                            parsed_response = self._parse_tool_calls(full_response)
                            messages_copy.append(parsed_response)
                            break
                        
                        full_response += token_text
                        print(token_text)#, end='', flush=True)
                        
                        # Check if we're starting a tool call section in the buffer
                        # Use partial checks like this since the model sometimes does tool_calls or tool or tools function function_call etc.
                        #if '<tool' not in speech_buffer and '<function' not in speech_buffer and '{"name' not in speech_buffer:
                            # add to be spoken
                        if not _is_entering_tool_call(full_response):
                            
                            speech_buffer += token_text

                        
                        # Count words accurately
                        words_in_buffer = speech_buffer.strip().split()
                        word_count = len(words_in_buffer)
                        
                        has_sentence_break = (speech_buffer.endswith('.') or 
                                            '!' in speech_buffer or 
                                            '?' in speech_buffer or 
                                            '\n' in speech_buffer)
                        
                        # Synthesize speech every [max_words_start_speech] words or at punctuation breaks
                        if word_count >= max_words_start_speech or has_sentence_break:
                            # Check for interrupt before synthesis
                            if self.audio_cortex.brocas_area_interrupt_trigger.get('interrupt',False):
                                logging.debug("[Generation interrupted before synthesis] - append what we have")
                                # Parse for tool calls/add assistant structure
                                parsed_response = self._parse_tool_calls(full_response)
                                messages_copy.append(parsed_response)
                                break  # Break from token generation loop
                            else:
                                #make sure we have at least 3 words to speak, since \n\n or other non punctuation breaks cause false speech positives
                                if speech_buffer != "" and word_count >= min_words_start_speech:
                                    self.audio_cortex.brocas_area.synthesize_speech(speech_buffer, auto_play=True)
                                    speech_buffer = ""
                                    word_count = 0

                    # Clean up input tensors
                    del model_inputs
                    gc.collect()
                    
                    # Handle interruption
                    if self.audio_cortex.brocas_area_interrupt_trigger.get('interrupt',False):
                        # Don't synthesize remaining buffer
                        logging.debug("[Skipping final synthesis due to interrupt]")
                        # Break from tool iteration loop to go back to waiting for next audio input
                        break
                    
                    # Synthesize any remaining text in buffer
                    if speech_buffer.strip():
                        # Final check before last synthesis
                        if not self.audio_cortex.brocas_area_interrupt_trigger.get('interrupt',False):
                            self.audio_cortex.brocas_area.synthesize_speech(speech_buffer, auto_play=True)
                    
                    # Parse for tool calls
                    print(f"FULL RESPONSE: {full_response}")
                    parsed_response = self._parse_tool_calls(full_response)
                    print(f"PARSED RESPONSE: {parsed_response}")
                    
                    #check for interruption again
                    if self.audio_cortex.brocas_area_interrupt_trigger.get('interrupt',False):
                        break

                    # Check if model wants to use tools
                    if parsed_response.get("tool_calls", False):
                        print("\n\nCALLING TOOLS\n\n")
                        tool_calls = parsed_response.get("tool_calls")
                        # Format the messages for the fact model called tools
                        tool_calls_format = parsed_response.get("content", "")
                        messages_copy.append(tool_calls_format)
                        
                        logging.debug(f"post parse: {tool_calls}")
                        logging.debug(f"[Found {len(tool_calls)} tool call(s)]")
                        
                        tool_results = self._execute_tool_calls(tool_calls)

                        print(f"\nTOOL RESULTS: {tool_results}")
                        
                        messages_copy.extend(tool_results)
                        
                        # Continue loop to generate response with results from tools
                        continue
                    else:
                        # Strip the <response> tags if model used them by accident
                        parsed_response["content"] = self.strip_response_tags(parsed_response["content"])
                        # No more tool calls, we're done
                        messages_copy.append(parsed_response)
                        logging.debug("[No tool calls - finishing]")
                        break
                
                if iteration >= max_tool_iterations:
                    logging.debug(f"[Warning: Reached max tool iterations ({max_tool_iterations})]")
                
                else:
                    logging.debug("[Message history not updated due to interrupt]")

                # Update the actual message history with all the interactions
                self.messages = messages_copy
                print(f"\n\nSYSTEM MESSAGES:\n{self.messages}\n\n")
                self.status.update({
                            'thinking': False,
                            'model': self.model_name,
                            'system_prompt_base': self.system_prompt_base,
                            'system_prompt_tools': self.prompt_for_tool_use,
                            'system_prompt_visual':"",#contains scene description of visual
                            'system_prompt_audio':"",#contains scene description of audio, sounds detected not text
                            'messages': self.messages,
                            'last_response':self.messages[-1]['content']
                        })

                #CLEAR - if we are all done, this just pulls once to clear the queue. prevents speech while waiting from being injested
                try:
                    _ = self.external_audio_tempLobe_to_prefrontalCortex.get_nowait()
                except:
                    continue

                
            except queue.Empty:
                # Small delay on loop to help CPU
                
                time.sleep(0.001)
                continue  # Continue the outer while True loop
            
    def shutdown(self):
        """Stop the chat thread cleanly."""
        logging.debug('[Initiating Prefrontal Cortex shutdown...]')
        self.shutdown_event.set()
        
        try:
            if self.chat_thread.is_alive():
                self.chat_thread.join(timeout=5)  # Wait up to 5 seconds
                if self.chat_thread.is_alive():
                    logging.warning('[Chat thread did not stop within timeout]')
                else:
                    logging.debug('[Chat thread stopped successfully]')
            
            logging.debug('[Prefrontal Cortex stopped]')
            return True
        except Exception as e:
            logging.error(f'[Shutdown error: {e}]')
            return False
