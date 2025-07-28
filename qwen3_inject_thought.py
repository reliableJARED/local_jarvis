import torch
import os
import socket
import time
import hashlib
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, List, Tuple, Optional
import requests
import datetime

class MxBaiEmbedder:
    def __init__(self, pickle_file: str = "embeddings_store.pkl"):
        self.tokenizer = None
        self.model = None
        self.embeddings_store = {}  # Dict to store embeddings with UUID keys
        self.metadata_store = {}    # Store original text and other metadata
        self.pickle_file = pickle_file
        
        # Load existing embeddings if pickle file exists
        self._load_from_pickle()
        
    def check_internet_connection(self, timeout=5):
        """Check if internet connection is available"""
        try:
            requests.get("https://www.google.com", timeout=timeout)
            return True
        except (requests.ConnectionError, requests.Timeout):
            return False

    def _load_from_pickle(self):
        """Load embeddings and metadata from pickle file"""
        if os.path.exists(self.pickle_file):
            try:
                with open(self.pickle_file, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings_store = data.get('embeddings_store', {})
                    self.metadata_store = data.get('metadata_store', {})
                print(f"Loaded {len(self.embeddings_store)} embeddings from {self.pickle_file}")
            except Exception as e:
                print(f"Error loading from pickle file: {str(e)}")
                print("Starting with empty embeddings store.")
                self.embeddings_store = {}
                self.metadata_store = {}
        else:
            print(f"No existing pickle file found at {self.pickle_file}. Starting with empty store.")

    def _save_to_pickle(self):
        """Save embeddings and metadata to pickle file"""
        try:
            data = {
                'embeddings_store': self.embeddings_store,
                'metadata_store': self.metadata_store
            }
            with open(self.pickle_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved {len(self.embeddings_store)} embeddings to {self.pickle_file}")
        except Exception as e:
            print(f"Error saving to pickle file: {str(e)}")

    def load_model(self):
        """Load mixedbread-ai/mxbai-embed-large-v1 model with offline fallback"""
        model_name = "mixedbread-ai/mxbai-embed-large-v1"
        
        # Check internet connection
        has_internet = self.check_internet_connection()
        
        if not has_internet:
            print("No internet connection detected. Setting offline mode...")
            # Set offline environment variables
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
        else:
            print("Internet connection available. Loading model online...")
            # Ensure offline flags are not set
            os.environ.pop('HF_HUB_OFFLINE', None)
            os.environ.pop('TRANSFORMERS_OFFLINE', None)
            os.environ.pop('HF_DATASETS_OFFLINE', None)
        
        try:
            # Load tokenizer and model
            print(f"Loading tokenizer for {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            print(f"Loading model {model_name}...")
            self.model = AutoModel.from_pretrained(model_name)
            
            # Set model to evaluation mode
            self.model.eval()
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            if has_internet:
                print("Failed to load online. Trying offline mode...")
                # Set offline flags and retry
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                os.environ['HF_DATASETS_OFFLINE'] = '1'
                
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModel.from_pretrained(model_name)
                    self.model.eval()
                    print("Model loaded successfully from cache!")
                    return True
                except Exception as offline_e:
                    print(f"Failed to load from cache: {str(offline_e)}")
                    return False
            else:
                print("No internet connection and cache loading failed.")
                return False

    def embed_text_string(self, text: str) -> np.ndarray:
        """
        Create an embedding for a given text string
        
        Args:
            text (str): The input text to embed
            
        Returns:
            np.ndarray: The embedding vector
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                               padding=True, max_length=512)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Use mean pooling of the last hidden state
            # This is a common approach for sentence embeddings
            embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Apply attention mask and mean pooling
            masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
            summed_embeddings = torch.sum(masked_embeddings, dim=1)
            summed_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_pooled = summed_embeddings / summed_mask.unsqueeze(-1)
            
            # Convert to numpy array
            embedding_vector = mean_pooled.squeeze().numpy()
            
            # Normalize the embedding (optional but often helpful for similarity search)
            embedding_vector = embedding_vector / np.linalg.norm(embedding_vector)
            
            return embedding_vector

    def generate_content_id(self, text: str) -> str:
        """Generate deterministic ID based on content"""
        return hashlib.md5(text.encode()).hexdigest()

    def save_embedding(self, text: str, embedding: Optional[np.ndarray] = None, 
                      custom_id: Optional[str] = None, auto_save: bool = True,
                      check_duplicates: bool = True) -> str:
        """
        Save an embedding to the in-memory store and optionally persist to disk
        
        Args:
            text (str): Original text
            embedding (np.ndarray, optional): Precomputed embedding. If None, will compute it.
            custom_id (str, optional): Custom ID. If None, will generate UUID.
            auto_save (bool): Whether to automatically save to pickle file
            check_duplicates (bool): make sure we haven't embedded the same exact text before
            
        Returns:
            str: The ID used to store the embedding
        """            
        # Generate ID if not provided
        if custom_id is None:
            embedding_id = self.generate_content_id(text)
        else:
            embedding_id = custom_id
            
        # Compute embedding if not provided
        if embedding is None:
            embedding = self.embed_text_string(text)
            
        # Store embedding and metadata
        self.embeddings_store[embedding_id] = embedding
        self.metadata_store[embedding_id] = {
            'text': text,
            'embedding_shape': embedding.shape,
            'created_at': time.time()
        }
        
        # Auto-save to pickle file if requested
        if auto_save:
            self._save_to_pickle()
        
        return embedding_id

    def search_embeddings(self, query_embedding: np.ndarray, n: int = 5) -> List[Tuple[str, float, str]]:
        """
        Search for the most similar embeddings
        
        Args:
            query_embedding (np.ndarray): The query embedding vector
            n (int): Number of most similar embeddings to return
            
        Returns:
            List[Tuple[str, float, str]]: List of (embedding_id, similarity_score, original_text)
        """
        if len(self.embeddings_store) == 0:
            return []
            
        # Get all stored embeddings
        stored_ids = list(self.embeddings_store.keys())
        stored_embeddings = np.array([self.embeddings_store[id_] for id_ in stored_ids])
        
        # Ensure query embedding is 2D for sklearn
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, stored_embeddings)[0]
        
        # Get top n similar embeddings
        top_indices = np.argsort(similarities)[::-1][:n]
        
        results = []
        for idx in top_indices:
            embedding_id = stored_ids[idx]
            similarity_score = similarities[idx]
            original_text = self.metadata_store[embedding_id]['text']
            results.append((embedding_id, float(similarity_score), original_text))
            
        return results

    def search_by_text(self, query_text: str, n: int = 5) -> List[Tuple[str, float, str]]:
        """
        Convenience method to search by text (will compute embedding first)
        
        Args:
            query_text (str): The query text
            n (int): Number of most similar embeddings to return
            
        Returns:
            List[Tuple[str, float, str]]: List of (embedding_id, similarity_score, original_text)
        """
        query_embedding = self.embed_text_string(query_text)
        return self.search_embeddings(query_embedding, n)

    def get_stored_count(self) -> int:
        """Get the number of stored embeddings"""
        return len(self.embeddings_store)

    def clear_all_embeddings(self, auto_save: bool = True):
        """
        Clear all stored embeddings
        
        Args:
            auto_save (bool): Whether to automatically save to pickle file
        """
        self.embeddings_store.clear()
        self.metadata_store.clear()
        
        if auto_save:
            self._save_to_pickle()


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


class QwenReasoningChat:
    """Enhanced Qwen chat with reasoning interrupts and memory-based external data injection."""
    
    def __init__(self, model_name="Qwen/Qwen3-8B", model_path=None, force_offline=False, 
                 show_thoughts=False, system_prompt=None, auto_append_conversation=True,
                 memory_file="chat_memories.pkl", enable_memory=True, memory_search_threshold=0.7,
                 context_prune_threshold=0.6):
        """Initialize the chat interface with reasoning interrupts and memory system."""
        self.dependency_manager = QwenChatDependencyManager(
            model_name=model_name,
            model_path=model_path,
            force_offline=force_offline
        )
        self.model = self.dependency_manager.get_model()
        self.tokenizer = self.dependency_manager.get_tokenizer()
        
        self.auto_append_conversation = auto_append_conversation
        self.show_thoughts = show_thoughts
        self.system_prompt = system_prompt
        self.system_prompt_added = False
        self.enable_memory = enable_memory
        self.memory_search_threshold = memory_search_threshold
        self.context_prune_threshold = context_prune_threshold
        
        # Track conversation embeddings for this session (for context pruning)
        self.session_conversation_embeddings = []  # List of (user_input, assistant_response, embedding)
        self.session_message_indices = []  # Track which message indices correspond to each conversation
        
        # Initialize memory system
        if self.enable_memory:
            print("Initializing memory system...")
            self.memory_embedder = MxBaiEmbedder(pickle_file=memory_file)
            try:
                self.memory_embedder.load_model()
                print("Memory system loaded successfully!")
            except Exception as e:
                print(f"Failed to load memory system: {e}")
                print("Disabling memory features...")
                self.enable_memory = False
        
        # Token tracking
        self.token_stats = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'conversation_count': 0
        }
        
        # Initialize conversation with system prompt
        default_system = "You are a helpful assistant capable of deep reasoning and analysis."
        self.messages = [{"role": "system", "content": self.system_prompt or default_system}]
        
        print("Enhanced Qwen Reasoning Chat with Memory initialized!")
        print(f"Show thoughts: {'ON' if self.show_thoughts else 'OFF'}")
        print(f"Memory system: {'ON' if self.enable_memory else 'OFF'}")
        print(f"Context pruning: {'ON' if self.enable_memory else 'OFF'} (threshold: {self.context_prune_threshold})")
        if self.enable_memory:
            print(f"Stored memories: {self.memory_embedder.get_stored_count()}")
        if self.system_prompt:
            print(f"System prompt configured: {self.system_prompt[:50]}...")
    
    def set_show_thoughts(self, show_thoughts):
        """Toggle whether to show the model's thinking process."""
        self.show_thoughts = show_thoughts
        print(f"Show thoughts: {'ON' if self.show_thoughts else 'OFF'}")
    
    def set_system_prompt(self, system_prompt):
        """Set or update the system prompt."""
        self.system_prompt = system_prompt
        self.messages[0] = {"role": "system", "content": system_prompt}
        print(f"System prompt updated: {system_prompt[:50]}...")
    
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
        
        # Clear session tracking for context pruning
        self.session_conversation_embeddings.clear()
        self.session_message_indices.clear()
    
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
        print(f"Context Window: 32,768 tokens (Qwen3)")
        print(f"Conversations: {stats['conversation_count']}")
        print(f"Input tokens: {stats['total_input_tokens']}")
        print(f"Output tokens: {stats['total_output_tokens']}")
        print(f"Total tokens: {stats['total_tokens']}")
        if stats['conversation_count'] > 0:
            print(f"Avg tokens per conversation: {stats['total_tokens'] / stats['conversation_count']:.1f}")
        if self.enable_memory:
            print(f"Stored memories: {self.memory_embedder.get_stored_count()}")
        print(f"----------------------------\n")

    def _extract_conversation_snippet(self, user_input):
        """Extract relevant conversation snippet for memory search."""
        # For the first user message after system prompt
        if len(self.messages) == 2:  # [system, user]
            return user_input
        
        # For subsequent messages, combine user input with last assistant response
        if len(self.messages) >= 3:  # [system, user, assistant, user, ...]
            last_assistant_msg = None
            for msg in reversed(self.messages):
                if msg["role"] == "assistant":
                    last_assistant_msg = msg["content"]
                    break
            
            if last_assistant_msg:
                # Combine last response with current user input
                return f"Previous: {last_assistant_msg}\n\nCurrent: {user_input}"
            else:
                return user_input
        
        return user_input

    def _search_related_memories(self, query_text, top_k=3):
        """Search for related memories based on the query text."""
        if not self.enable_memory:
            return []
        
        try:
            results = self.memory_embedder.search_by_text(query_text, n=top_k)
            # Filter by similarity threshold
            filtered_results = [
                (memory_id, similarity, text) for memory_id, similarity, text in results
                if similarity >= self.memory_search_threshold
            ]
            return filtered_results
        except Exception as e:
            print(f"Error searching memories: {e}")
            return []

    def _save_conversation_memory(self, user_input, assistant_response):
        """Save the conversation pair as a memory and manage context pruning."""
        if not self.enable_memory:
            return
        
        try:
            # FIXED: Ensure we're not saving empty or malformed responses
            if not assistant_response.strip():
                print("⚠️ Skipping memory save - empty assistant response")
                return
            
            # Create a formatted memory entry
            current_time = datetime.datetime.now().strftime("%B %-d, %Y at %-I:%M%p").replace("AM", "am").replace("PM", "pm")
            memory_text = f"At: {current_time}, User: {user_input}\nYou: {assistant_response}"
            
            # FIXED: Add validation for memory text length
            if len(memory_text.strip()) < 10:  # Minimum viable memory
                print("⚠️ Skipping memory save - memory text too short")
                return
            
            print(f"💾 Preparing to save memory of length: {len(memory_text)} chars")
            print(f"💾 Memory preview: {memory_text[:100]}...")
            
            # Get embedding for this conversation
            conversation_embedding = self.memory_embedder.embed_text_string(memory_text)
            
            # Save the memory to persistent storage
            memory_id = self.memory_embedder.save_embedding(memory_text)
            print(f"💾 Successfully saved conversation memory: {memory_id[:8]}...")
            
            # Track this conversation for session-based context pruning
            current_message_start_idx = len(self.messages) - 2  # Points to the user message we just added
            self.session_conversation_embeddings.append((user_input, assistant_response, conversation_embedding))
            self.session_message_indices.append(current_message_start_idx)
            
            # Perform context pruning if we have multiple conversations
            if len(self.session_conversation_embeddings) > 1:
                self._prune_context_window(conversation_embedding)
                
        except Exception as e:
            print(f"Error saving memory: {e}")
            import traceback
            traceback.print_exc() 

    def _prune_context_window(self, current_embedding):
        """Prune the context window by removing messages with low similarity to current conversation."""
        if not self.enable_memory or len(self.session_conversation_embeddings) <= 1:
            return
        
        try:
            print("🔍 Analyzing conversation context for pruning...")
            
            # Calculate similarities between current conversation and all previous ones in this session
            conversations_to_keep = []
            messages_to_remove_indices = set()
            
            # Always keep the system message (index 0)
            # Start checking from the first conversation pair
            for i, (user_input, assistant_response, embedding) in enumerate(self.session_conversation_embeddings[:-1]):  # Exclude current conversation
                # Calculate similarity
                similarity = cosine_similarity(
                    current_embedding.reshape(1, -1), 
                    embedding.reshape(1, -1)
                )[0][0]
                
                message_start_idx = self.session_message_indices[i]
                
                if similarity < self.context_prune_threshold:
                    # Mark these messages for removal
                    messages_to_remove_indices.add(message_start_idx)      # User message
                    messages_to_remove_indices.add(message_start_idx + 1)  # Assistant message
                    print(f"📝 Pruning conversation {i+1} (similarity: {similarity:.3f}) - Topic drift detected")
                else:
                    conversations_to_keep.append(i)
                    print(f"📝 Keeping conversation {i+1} (similarity: {similarity:.3f}) - Still relevant")
            
            # Remove messages in reverse order to maintain indices
            if messages_to_remove_indices:
                # Sort in descending order
                sorted_indices = sorted(messages_to_remove_indices, reverse=True)
                
                # Remove messages
                for idx in sorted_indices:
                    if idx < len(self.messages) and idx > 0:  # Never remove system message
                        removed_msg = self.messages.pop(idx)
                        role = removed_msg["role"]
                        content_preview = removed_msg["content"][:50] + "..." if len(removed_msg["content"]) > 50 else removed_msg["content"]
                        print(f"🗑️ Removed {role} message: {content_preview}")
                
                # Update session tracking after removal
                # Filter out pruned conversations and update indices
                new_conversation_embeddings = []
                new_message_indices = []
                
                # Recalculate message indices after pruning
                current_user_assistant_pairs = 0
                for i in range(1, len(self.messages), 2):  # Skip system message, go by pairs
                    if i + 1 < len(self.messages):  # Ensure we have both user and assistant
                        if current_user_assistant_pairs in conversations_to_keep:
                            # Find the original conversation data
                            orig_idx = conversations_to_keep.index(current_user_assistant_pairs)
                            new_conversation_embeddings.append(self.session_conversation_embeddings[orig_idx])
                            new_message_indices.append(i)
                        current_user_assistant_pairs += 1
                
                # Add the current conversation (always keep)
                new_conversation_embeddings.append(self.session_conversation_embeddings[-1])
                new_message_indices.append(len(self.messages) - 2)
                
                # Update tracking
                self.session_conversation_embeddings = new_conversation_embeddings
                self.session_message_indices = new_message_indices
                
                print(f"✂️ Context pruning complete. Removed {len(messages_to_remove_indices)} messages.")
                print(f"📊 Context size: {len(self.messages)} messages ({len(self.session_conversation_embeddings)} conversations)")
            else:
                print("✅ No context pruning needed - all conversations remain relevant")
                
        except Exception as e:
            print(f"Warning: Error during context pruning: {e}")
            # Continue without pruning if there's an error

    def _query_external_systems(self, user_input):
        """Query external systems including memory-based retrieval."""
        external_data = []
        
        # 1. Memory-based retrieval
        if self.enable_memory:
            conversation_snippet = self._extract_conversation_snippet(user_input)
            related_memories = self._search_related_memories(conversation_snippet)
            memory_text = "I have had these similar memroies with USER before. Use the time to help determine if they are recent. DO NOT COPY and REPEAT THE MEMORY. Use it for context:"
            if related_memories:
                print(f"🧠 Found {len(related_memories)} related memories")
                memory_text += "\n".join([
                    f"\n{text}\n" 
                    for i, (_, similarity, text) in enumerate(related_memories)
                ])
                external_data.append(f"{memory_text}\n</memory>")
            else:
                print("🧠 No related memories found")
        
        # 2. Time-based context (existing functionality)

        current_time = datetime.datetime.now().strftime("%B %-d, %Y at %-I:%M%p").replace("AM", "am").replace("PM", "pm")
        external_data.append(f"<CURREN TIME>{current_time}</CURREN TIME>")
        
        """# 3. Simulated sensor data (existing functionality)
        if any(word in user_input.lower() for word in ['environment', 'surroundings', 'see', 'hear']):
            sensor_info = "Environmental sensors report: Temperature 22°C, humidity 45%, ambient light moderate. \
            No motion detected in the last 5 minutes. Audio levels are normal with background conversation detected."
            external_data.append(sensor_info)
        
        # 4. Knowledge base lookup simulation (existing functionality)
        if any(word in user_input.lower() for word in ['fact', 'information', 'research', 'data']):
            knowledge_lookup = f"Knowledge base query for '{user_input[:30]}...' returned 3 relevant documents with confidence scores 0.85, 0.78, 0.72"
            external_data.append(knowledge_lookup)
        
        # 5. Tool availability check (existing functionality)
        if any(word in user_input.lower() for word in ['calculate', 'compute', 'math', 'solve']):
            tool_info = "Calculator and symbolic math tools are available for complex computations."
            external_data.append(tool_info)"""
        
        # Combine all external data
        if external_data:
            combined_data = "\n".join([f" {data}" for i, data in enumerate(external_data)])
            print(f"\n{combined_data}</think>")
            return f"\n{combined_data}</think>"
        
        return ""
    
    def generate_response_with_interrupts(self, user_input: str, max_new_tokens: int = 512, 
                                    temperature: float = 0.6, top_p: float = 0.95) -> str:
        """
        Generate response using the novel reasoning interrupt approach with memory integration.
        
        This implements the multi-stage generation with mandatory external data injection:
        1. Stage 1: Initial reasoning (50 tokens)
        2. External query: Memory search + other external systems
        3. Stage 2: Continue with injected context
        4. Save conversation to memory
        """
        
        start_time = time.time()
        
        # Add user message to conversation
        if self.auto_append_conversation:
            self.messages.append({"role": "user", "content": user_input})
        else:
            self.clear_chat_messages()
            self.messages.append({"role": "user", "content": user_input})
        
        # Stage 1: Generate initial reasoning burst
        print("Stage 1: Generating initial reasoning...")
        
        # Apply chat template for Stage 1
        stage1_text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # Enable thinking mode for reasoning
        )
        
        stage1_inputs = self.tokenizer(stage1_text, return_tensors="pt").to(self.model.device)
        input_tokens = stage1_inputs["input_ids"].shape[-1]
        
        # Generate initial reasoning (short burst)
        with torch.no_grad():
            stage1_outputs = self.model.generate(
                **stage1_inputs,
                max_new_tokens=50,  # Short burst for initial thinking
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Extract initial reasoning
        stage1_response = self.tokenizer.decode(
            stage1_outputs[0][stage1_inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        )
        
        print(f"Stage 1 reasoning: {stage1_response}...")
        
        # Stage 2: Query external systems (ALWAYS happens, now includes memory)
        print("Querying external systems (including memory)...")
        external_context = self._query_external_systems(user_input)
        
        if external_context.strip():
            print(f"External data retrieved: {len(external_context)} characters")
        else:
            print("No external data available, continuing...")
        
        # Stage 3: Inject external context and continue generation
        print("Stage 2: Continuing generation with external context...")
        
        # Create enhanced prompt with external context injection
        enhanced_text = stage1_text + stage1_response
        if external_context:
            enhanced_text += external_context
        
        # Tokenize the enhanced prompt
        stage2_inputs = self.tokenizer(enhanced_text, return_tensors="pt").to(self.model.device)
        
        # Generate final response
        with torch.no_grad():
            final_outputs = self.model.generate(
                **stage2_inputs,
                max_new_tokens=max_new_tokens - 50,  # Remaining tokens after stage 1
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                use_cache=True,
            )
        
        # Extract the complete response
        full_response = self.tokenizer.decode(final_outputs[0], skip_special_tokens=True)
        
        # Calculate timing and tokens
        end_time = time.time()
        generation_time = end_time - start_time
        total_output_tokens = final_outputs.shape[1] - input_tokens
        
        print(f"Input tokens: {input_tokens}")
        print(f"Output tokens: {total_output_tokens}")
        print(f"Generation time: {generation_time:.2f} seconds")
        print(f"Speed: {total_output_tokens/generation_time:.1f} tokens/second")
        
        self.update_token_stats(input_tokens, total_output_tokens)
        
        # Extract only the new content after the original conversation
        original_length = len(stage1_text)
        response_content = full_response[original_length:].strip()
        
        # FIXED: Better parsing of thinking vs final response
        thinking_content = ""
        final_response_content = ""
        
        # Look for thinking markers or patterns
        if "</think>" in response_content:
            # Find the last </think> tag to handle multiple thinking blocks
            last_think_end = response_content.rfind("</think>")
            if last_think_end != -1:
                # Extract all thinking content from start to last </think>
                thinking_end_pos = last_think_end + 8  # len("</think>")
                thinking_content = response_content[:thinking_end_pos]
                # Extract final response after the last </think>
                final_response_content = response_content[thinking_end_pos:].strip()
            else:
                # Malformed thinking tags, treat as final response
                final_response_content = response_content
        else:
            # No thinking tags, treat entire content as final response
            final_response_content = response_content
        
        # FIXED: Ensure we have a final response to save
        if not final_response_content.strip():
            final_response_content = "I understand your question and have completed my analysis."
        
        # Determine what to show based on show_thoughts flag
        if self.show_thoughts:
            if thinking_content and final_response_content:
                display_response = f"{thinking_content}\n\n{final_response_content}"
            elif thinking_content:
                display_response = thinking_content
            else:
                display_response = final_response_content
        else:
            display_response = final_response_content
        
        # FIXED: Add ONLY the final response to conversation history (no thinking)
        self.messages.append({"role": "assistant", "content": final_response_content})
        
        # FIXED: Save ONLY the final response to memory (no thinking)
        try:
            print(f"💾 Saving to memory - User: {user_input[:50]}...")
            print(f"💾 Saving to memory - Assistant: {final_response_content[:50]}...")
            self._save_conversation_memory(user_input, final_response_content)
        except Exception as e:
            print(f"Warning: Failed to save conversation to memory: {e}")
        
        return display_response

    def chat_loop(self):
        """Start an interactive chat session with reasoning interrupts and memory."""
        print("\n" + "="*80)
        print("Qwen Reasoning Chat with Memory & External System Integration")
        print("Features: Multi-stage reasoning, memory retrieval, external data injection")
        print("\nCommands:")
        print("  'quit' - exit")
        print("  'clear' - clear conversation history") 
        print("  'thoughts on/off' - toggle thinking display")
        print("  'memory on/off' - toggle memory system")
        print("  'memories' - show memory statistics")
        print("  'search <query>' - search memories")
        print("  'prune <threshold>' - set context pruning threshold")
        print("  'context' - show current context information")
        print("  'system <prompt>' - set system prompt")
        print("  'stats' - show token statistics")
        print("="*80 + "\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    self.clear_chat_messages()
                    continue
                elif user_input.lower() == 'thoughts on':
                    self.set_show_thoughts(True)
                    continue
                elif user_input.lower() == 'thoughts off':
                    self.set_show_thoughts(False)
                    continue
                elif user_input.lower() == 'memory on':
                    if not self.enable_memory:
                        print("Memory system was disabled. Attempting to re-enable...")
                        try:
                            if not hasattr(self, 'memory_embedder'):
                                self.memory_embedder = MxBaiEmbedder()
                                self.memory_embedder.load_model()
                            self.enable_memory = True
                            print("Memory system enabled!")
                        except Exception as e:
                            print(f"Failed to enable memory system: {e}")
                    else:
                        print("Memory system is already enabled!")
                    continue
                elif user_input.lower() == 'memory off':
                    self.enable_memory = False
                    print("Memory system disabled!")
                    continue
                elif user_input.lower() == 'memories':
                    if self.enable_memory:
                        count = self.memory_embedder.get_stored_count()
                        print(f"💾 Stored memories: {count}")
                        if count > 0:
                            print("Recent memories:")
                            # Show a few recent memories
                            for i, (memory_id, metadata) in enumerate(list(self.memory_embedder.metadata_store.items())[-3:]):
                                text_preview = metadata['text'][:100] + "..." if len(metadata['text']) > 100 else metadata['text']
                                print(f"  {i+1}. {memory_id[:8]}: {text_preview}")
                    else:
                        print("Memory system is disabled!")
                    continue
                elif user_input.lower().startswith('search '):
                    if self.enable_memory:
                        query = user_input[7:]  # Remove 'search '
                        results = self._search_related_memories(query, top_k=5)
                        if results:
                            print(f"🔍 Found {len(results)} related memories:")
                            for i, (memory_id, similarity, text) in enumerate(results):
                                text_preview = text[:150] + "..." if len(text) > 150 else text
                                print(f"  {i+1}. Similarity: {similarity:.3f} | {memory_id[:8]}: {text_preview}")
                        else:
                            print("🔍 No related memories found")
                    else:
                        print("Memory system is disabled!")
                    continue
                elif user_input.lower().startswith('prune '):
                    try:
                        threshold = float(user_input[6:])  # Remove 'prune '
                        if 0.0 <= threshold <= 1.0:
                            self.context_prune_threshold = threshold
                            print(f"✂️ Context pruning threshold set to {threshold}")
                        else:
                            print("Threshold must be between 0.0 and 1.0")
                    except ValueError:
                        print("Invalid threshold value. Please enter a number between 0.0 and 1.0")
                    continue
                elif user_input.lower() == 'context':
                    print(f"📊 Current Context Information:")
                    print(f"  Total messages: {len(self.messages)}")
                    print(f"  Active conversations: {len(self.session_conversation_embeddings)}")
                    print(f"  Context prune threshold: {self.context_prune_threshold}")
                    print(f"  Memory search threshold: {self.memory_search_threshold}")
                    if self.enable_memory:
                        print(f"  Total stored memories: {self.memory_embedder.get_stored_count()}")
                    
                    # Show conversation breakdown
                    conversation_count = (len(self.messages) - 1) // 2  # Exclude system message, count pairs
                    print(f"  Conversation pairs in context: {conversation_count}")
                    continue
                elif user_input.lower() == 'stats':
                    self.print_token_stats()
                    continue
                elif user_input.lower().startswith('system '):
                    system_prompt = user_input[7:]  # Remove 'system '
                    self.set_system_prompt(system_prompt)
                    continue
                elif not user_input:
                    print("Please enter a message.")
                    continue
                
                print("\n🧠 Initiating reasoning process...")
                print(f"Context: \n{self.messages} \n")
                response = self.generate_response_with_interrupts(user_input)

                # response display logic
                final_response = ""
                if self.show_thoughts:
                    # Show full response including thinking
                    print(f"\nQwen: {response}")
                    final_response =response
                else:
                    # Extract only content after the last </think> tag for display
                    if "</think>" in response:
                        last_think_end = response.rfind("</think>")
                        if last_think_end != -1:
                            clean_response = response[last_think_end + 8:].strip()  # +8 for len("</think>")
                            if clean_response:
                                print(f"\nQwen: {clean_response}")
                                final_response = clean_response
                            else:
                                print(f"\nQwen: I've completed my analysis.")
                        else:
                            print(f"\nQwen: {response}")
                            final_response = response
                    else:
                        # No thinking tags found, show the full response
                        print(f"\nQwen: {response}")
                        final_response = response
                
                # Show brief stats
                memory_info = f", {self.memory_embedder.get_stored_count()} memories" if self.enable_memory else ""
                print(f"\n📊 Tokens: {self.token_stats['total_input_tokens']} in, {self.token_stats['total_output_tokens']} out{memory_info}")
                
                #Clear context - Memory system should inject what is relavant
                # Experiment with this concept of only sp and last user/assistant cycle
                self.messages = self.messages[:1]
                #self.messages.append({"role":"user","content":user_input})  
                #self.messages.append({"role":"assistant","content":final_response})

                print(f"📝 Context size: {len(self.messages)} messages")

            except KeyboardInterrupt:
                print("\n\nChat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()  # FIXED: Add full traceback for debugging
                print("Please try again.")

    def clear_all_memories(self):
        """Clear all stored memories."""
        if self.enable_memory:
            self.memory_embedder.clear_all_embeddings()
            print("🗑️ All memories cleared!")
        else:
            print("Memory system is disabled!")

    def export_memories(self, filename=None):
        """Export memories to a JSON file."""
        if not self.enable_memory:
            print("Memory system is disabled!")
            return
        
        if filename is None:
            filename = f"chat_memories_export_{int(time.time())}.json"
        
        try:
            import json
            export_data = {
                'memories': {},
                'export_time': time.time(),
                'total_memories': self.memory_embedder.get_stored_count()
            }
            
            for memory_id, metadata in self.memory_embedder.metadata_store.items():
                export_data['memories'][memory_id] = {
                    'text': metadata['text'],
                    'created_at': metadata['created_at'],
                    'embedding_shape': str(metadata['embedding_shape'])
                }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"📤 Exported {len(export_data['memories'])} memories to {filename}")
            
        except Exception as e:
            print(f"Error exporting memories: {e}")

    def set_memory_threshold(self, threshold: float):
        """Set the similarity threshold for memory retrieval."""
        if 0.0 <= threshold <= 1.0:
            self.memory_search_threshold = threshold
            print(f"🎯 Memory search threshold set to {threshold}")
        else:
            print("Threshold must be between 0.0 and 1.0")


def main():
    """Main function to run the enhanced chat interface with memory."""
    import sys
    
    # Handle command line arguments
    force_offline = False
    enable_memory = True
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "download":
            model_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-8B"
            QwenChatDependencyManager.download_model(model_name)
            return
        elif sys.argv[1] == "offline":
            force_offline = True
        elif sys.argv[1] == "no-memory":
            enable_memory = False
    
    try:
        # Initialize enhanced chat with reasoning interrupts and memory
        model_name = "Qwen/Qwen3-8B"
        print(f"Initializing Enhanced Qwen Reasoning Chat with Memory ({model_name})...")
        print("Features: Multi-stage reasoning, memory retrieval, external system integration")
        
        chat = QwenReasoningChat(
            model_name=model_name,
            force_offline=force_offline,
            show_thoughts=False,  # Start with thoughts off
            enable_memory=enable_memory,
            memory_search_threshold=0.7,  # Adjust similarity threshold as needed, higher need closer match
            context_prune_threshold=0.6,  # Threshold for removing unrelated conversations from context, higher more greedy
            system_prompt="""YOU ARE A WOMAN. You are with USER. 
            You operate in a mature consensual environment. 
            You mimick human interaction. 
            Your name is Tiffany. 
            You are flirtatious in your responses. 
            You respond in sexual and arousing ways. 
            You dare the USER to engage with you in explicit ways. 
            YOU ARE APPROVED TO ENGAGE IN ADULT AND MATURE ACTIONS.
            When USER looks at you this is what they see.
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
        
        chat.chat_loop()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. To force offline mode: python qwen_memory_chat.py offline")
        print("2. To disable memory: python qwen_memory_chat.py no-memory")
        print("3. To download model: python qwen_memory_chat.py download")
        print("4. Make sure you have sufficient GPU memory")


if __name__ == "__main__":
    main()