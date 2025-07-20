"""
Jarvis Voice Assistant with Memory Management 
Key Changes:
1. _create_memory_page() now combines prompt + response into single memory
2. Embeddings are generated on the combined prompt/response text
3. Added threshold filtering for similarity search results
4. Modified prompt processing to store complete conversation turns
"""

import time
import threading
import re
from enum import Enum
from typing import Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

# Import our custom modules (assuming they're in the same directory)
from whispervad_ import (
    VoiceAssistantSpeechProcessor, 
    VoiceAssistantCallback, 
    TranscriptSegment,
    SpeechEvent
)
from qwen_ import QwenChat
from kokoro_ import Kokoro
from text_embed import MxBaiEmbedder

class JarvisState(Enum):
    """States for the Jarvis assistant"""
    IDLE = "idle"                    # Waiting for wake word
    LISTENING = "listening"          # Actively listening for prompt after wake word
    THINKING = "thinking"            # Processing prompt with LLM
    RESPONDING = "responding"        # Speaking response (if TTS implemented)


@dataclass
class MemoryPage:
    """
    A memory page that stores conversation information with embeddings
    Now stores combined prompt/response pairs instead of separate entries
    """
    # Core conversation data - NOW COMBINED
    prompt_text: str = ""                       # The user's prompt
    response_text: str = ""                     # Jarvis's response
    combined_text: str = ""                     # Combined prompt + response for embedding
    text_embedding: Optional[np.ndarray] = None # Embedding of the combined text
    
    # Temporal information
    timestamp: datetime = field(default_factory=datetime.now)
    date_str: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    time_str: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    
    # Speaker information
    speaker_id: str = ""
    
    # Future multimedia support (not implemented yet)
    audio_clip: Optional[bytes] = None
    image: Optional[bytes] = None
    image_description: str = ""
    audio_description: str = ""
    combined_embedding: Optional[np.ndarray] = None  # Combined embedding of all modalities
    
    # Memory associations
    associated_memories: List[str] = field(default_factory=list)  # List of memory IDs
    
    # Unique identifier
    memory_id: str = field(default_factory=lambda: f"mem_{int(time.time() * 1000000)}")
    
    def get_full_context_text(self) -> str:
        """Get the full context text for embedding and retrieval"""
        context_parts = []
        
        # Add timestamp context
        context_parts.append(f"Date: {self.date_str} Time: {self.time_str}")
        
        # Add speaker context
        context_parts.append(f"Speaker: {self.speaker_id}")
        
        # Add the combined conversation text
        context_parts.append(f"Conversation: {self.combined_text}")
        
        # Add descriptions if available
        if self.image_description:
            context_parts.append(f"Image: {self.image_description}")
        if self.audio_description:
            context_parts.append(f"Audio: {self.audio_description}")
        
        return " | ".join(context_parts)


@dataclass
class JarvisConfig:
    """Configuration for Jarvis assistant"""
    wake_word: str = "jarvis"
    interrupt_phrase: str = "enough jarvis"
    wake_word_timeout: float = 30.0      # Seconds to wait for prompt after wake word
    response_timeout: float = 60.0       # Maximum thinking time
    conversation_timeout: float = 45.0   # Seconds of silence before requiring wake word again
    min_prompt_length: int = 2          # Minimum words in prompt - below this will not trigger response
    debug_mode: bool = True              # Print debug information
    continuous_conversation: bool = True  # Allow conversation without wake word after initial detection
    auto_append_messages: bool = False #setting to determine if conversation continues to append to messages or not
    # Memory system configuration
    memory_similarity_threshold: float = 0.7  # Minimum similarity for memory retrieval
    max_similar_memories: int = 2             # Maximum number of similar memories to include
    embeddings_file: str = "jarvis_memory.pkl"  # File to store memory embeddings


class JarvisCallback(VoiceAssistantCallback):
    """Enhanced callback that integrates with Jarvis assistant"""
    
    def __init__(self, jarvis_instance):
        super().__init__()
        self.jarvis = jarvis_instance
    
    def on_speech_start(self, event: SpeechEvent):
        """Handle speech start events"""
        if self.jarvis.config.debug_mode:
            print(f"🎤 Speech detected at {event.timestamp:.1f}")
        
        # Notify Jarvis of speech activity
        self.jarvis._on_speech_detected()
    
    def on_speech_end(self, event: SpeechEvent):
        """Handle speech end events"""
        if self.jarvis.config.debug_mode:
            print(f"🔇 Speech ended at {event.timestamp:.1f}")
    
    def on_transcript_update(self, segment: TranscriptSegment):
        """Handle real-time transcript updates"""
        if self.jarvis.config.debug_mode:
            print(f"📝 [{segment.speaker_id}] (live): {segment.text}")
        
        # Check for wake word or interrupt phrase in live transcript
        self.jarvis._process_live_transcript(segment.text, segment.speaker_id)
    
    def on_transcript_final(self, segment: TranscriptSegment):
        """Handle final transcript"""
        if self.jarvis.config.debug_mode:
            print(f"✅ [{segment.speaker_id}] (final): {segment.text}")
        
        self.current_conversation.append(segment)
        self.active_speakers.add(segment.speaker_id)
        
        # Process final transcript for prompts
        self.jarvis._process_final_transcript(segment)
    
    def on_speaker_change(self, old_speaker: Optional[str], new_speaker: str):
        """Handle speaker changes"""
        if ((old_speaker is not None) and (new_speaker != self.jarvis.jarvis_voice_id)):
            if self.jarvis.config.debug_mode:
                print(f"👥 Speaker change: {old_speaker} → {new_speaker}")
                self.jarvis.active_speakers = new_speaker
        else:
            if self.jarvis.config.debug_mode:
                print(f"👤 New speaker: {new_speaker}")


class Jarvis:
    """
    Main Jarvis Voice Assistant class
    Integrates speech processing with LLM and memory system for complete voice interaction
    """
    
    def __init__(self, config: Optional[JarvisConfig] = None):
        """Initialize Jarvis with configuration"""
        self.tts = Kokoro(
            lang_code='a', #American english
            voice="af_sky", #default female american voice
            save_audio_as_wav=False,  # Use temporary files
            play_audio_immediately=False  # Don't auto-play for demo control
        )
        self.tts.set_speech_audio_ready_callback(self.on_audio_ready)
        self.tts.set_speech_audio_playback_complete_callback(self.on_playback_complete)

        self.config = config or JarvisConfig()
        self.state = JarvisState.IDLE
        self.in_conversation = False     # Flag for continuous conversation mode
        self.primary_speaker = None #This is who we Jarvis is talking to
        self.jarvis_voice_id = None #this is set when program does a voice sample to start.
        self.prompt_buffer = []
        self.wake_word_detected_time = None
        self.thinking_start_time = None
        self.last_speech_time = None     # Track last speech activity for conversation timeout
        self._processed_segments = set() # Track processed final segments to avoid duplicates
        
        # Memory system
        self.memory_pages: List[MemoryPage] = []  # In-memory list of all memory pages
        self.embedder: Optional[MxBaiEmbedder] = None
        self.current_prompt = None  # Store current prompt being processed
        
        # Thread synchronization
        self.state_lock = threading.Lock()
        self.memory_lock = threading.Lock()  # Separate lock for memory operations
        self.should_stop = False
        
        # prompt processing queue to avoid blocking
        self.prompt_queue = []
        self.prompt_queue_lock = threading.Lock()
        
        # Initialize components
        print("Initializing Jarvis...")
        self._init_memory_system()
        self._init_speech_processor()
        self._init_llm()
    
    def _init_memory_system(self):
        """Initialize the memory embedding system with proper loading"""
        print("Loading memory embedding system...")
        try:
            self.embedder = MxBaiEmbedder(pickle_file=self.config.embeddings_file)
            if not self.embedder.load_model():
                print("Warning: Failed to load embedding model. Memory system will be disabled.")
                self.embedder = None
            else:
                stored_count = self.embedder.get_stored_count()
                print(f"Memory system loaded with {stored_count} existing memories")
                
                # Load memory pages from persistent storage
                self._load_memory_pages_from_storage()
                
        except Exception as e:
            print(f"Error initializing memory system: {e}")
            self.embedder = None

        """Initialize the memory embedding system"""
        print("Loading memory embedding system...")
        try:
            self.embedder = MxBaiEmbedder(pickle_file=self.config.embeddings_file)
            if not self.embedder.load_model():
                print("Warning: Failed to load embedding model. Memory system will be disabled.")
                self.embedder = None
            else:
                print(f"Memory system loaded with {self.embedder.get_stored_count()} existing memories")
        except Exception as e:
            print(f"Error initializing memory system: {e}")
            self.embedder = None
    
    def _load_memory_pages_from_storage(self):
        """Load memory pages from embedder storage"""
        if not self.embedder:
            return
        
        try:
            # Get all stored embeddings count
            stored_count = self.embedder.get_stored_count()
            print(f"Total memories in storage: {stored_count}")
            
            if stored_count == 0:
                print("No memories found in storage.")
                return
            
            # Access the internal stores directly
            embeddings_store = getattr(self.embedder, 'embeddings_store', {})
            metadata_store = getattr(self.embedder, 'metadata_store', {})
            
            print(f"Loading {len(embeddings_store)} memory pages from storage...")
            
            loaded_count = 0
            for memory_id in embeddings_store.keys():
                try:
                    # Get embedding from embeddings store
                    embedding = embeddings_store.get(memory_id)
                    
                    # Get metadata from metadata store
                    if memory_id in metadata_store:
                        metadata = metadata_store[memory_id]
                        stored_text = metadata.get('text', '')
                        
                        # Parse the stored full context text to reconstruct memory page
                        memory_page = self._reconstruct_memory_page_from_text(stored_text, memory_id, embedding)
                        
                        if memory_page:
                            self.memory_pages.append(memory_page)
                            loaded_count += 1
                            if self.config.debug_mode:
                                print(f"💾 Loaded memory: {memory_id[:12]}... - {memory_page.combined_text[:50]}...")
                        else:
                            if self.config.debug_mode:
                                print(f"❌ Failed to reconstruct memory page for ID: {memory_id[:12]}...")
                    else:
                        if self.config.debug_mode:
                            print(f"❌ No metadata found for memory ID: {memory_id[:12]}...")
                
                except Exception as e:
                    print(f"❌ Error loading memory {memory_id[:12]}...: {e}")
                    continue
            
            print(f"✅ Successfully loaded {loaded_count} memory pages from {stored_count} stored memories")
            
        except Exception as e:
            print(f"Error loading memory pages from storage: {e}")
            import traceback
            traceback.print_exc()

    def _reconstruct_memory_page_from_text(self, full_context_text: str, memory_id: str, embedding: np.ndarray) -> Optional[MemoryPage]:
        """Reconstruct a MemoryPage from stored full context text - SIMPLIFIED VERSION"""
        try:
            if self.config.debug_mode:
                print(f"🔧 Reconstructing memory from text: {full_context_text[:100]}...")
            
            # Initialize defaults
            date_str = datetime.now().strftime("%Y-%m-%d")
            time_str = datetime.now().strftime("%H:%M:%S")
            speaker_id = "unknown"
            combined_conversation = ""
            
            # Extract metadata from the full context text
            lines = full_context_text.split('\n') if '\n' in full_context_text else full_context_text.split(' | ')
            
            for line_or_part in lines:
                line_or_part = line_or_part.strip()
                if not line_or_part:
                    continue
                
                # Extract metadata
                if line_or_part.startswith('Date: '):
                    date_str = line_or_part.replace('Date: ', '').strip()
                elif line_or_part.startswith('Time: '):
                    time_str = line_or_part.replace('Time: ', '').strip()
                elif line_or_part.startswith('Speaker: '):
                    speaker_id = line_or_part.replace('Speaker: ', '').strip()
                elif line_or_part.startswith('Conversation: '):
                    combined_conversation = line_or_part.replace('Conversation: ', '').strip()
                    break  # Found the conversation, stop looking
            
            # If we didn't find a "Conversation:" prefix, use the whole text as conversation
            if not combined_conversation:
                # Remove metadata lines and use the rest as conversation
                conversation_lines = []
                for line_or_part in lines:
                    line_or_part = line_or_part.strip()
                    if (not line_or_part or 
                        line_or_part.startswith('Date: ') or 
                        line_or_part.startswith('Time: ') or 
                        line_or_part.startswith('Speaker: ')):
                        continue
                    conversation_lines.append(line_or_part)
                combined_conversation = ' '.join(conversation_lines).strip()
            
            # Create timestamp
            try:
                timestamp = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
            except:
                timestamp = datetime.now()
            
            # Debug output
            if self.config.debug_mode:
                print(f"🔧 Parsed - Date: {date_str}, Time: {time_str}, Speaker: {speaker_id}")
                print(f"🔧 Combined conversation: '{combined_conversation[:100]}...'")
            
            # Validate that we got something useful
            if not combined_conversation:
                print(f"❌ No conversation content found in memory {memory_id[:12]}...")
                print(f"❌ Raw text was: {full_context_text}")
                return None
            
            # Create memory page with simplified approach - don't try to separate prompt/response
            # Just store the combined conversation and let the memory system work with it
            memory_page = MemoryPage(
                prompt_text="",  # Leave empty for now - the combined_text has everything
                response_text="",  # Leave empty for now - the combined_text has everything  
                combined_text=combined_conversation,  # Use the full conversation as combined text
                text_embedding=embedding,
                timestamp=timestamp,
                date_str=date_str,
                time_str=time_str,
                speaker_id=speaker_id,
                memory_id=memory_id
            )
            
            return memory_page
            
        except Exception as e:
            print(f"Error reconstructing memory page {memory_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
                    
    def _init_speech_processor(self):
        """Initialize speech processing components"""
        print("Loading speech processor...")
        self.callback = JarvisCallback(self)
        self.speech_processor = VoiceAssistantSpeechProcessor(self.callback)
    
    def _init_llm(self):
        """Initialize the language model"""
        print(f"Loading language model..., auto_append_conversation={self.config.auto_append_messages}")
        self.llm = QwenChat(auto_append_conversation=self.config.auto_append_messages)
        
        # Add Jarvis-specific system prompt
        jarvis_prompt = f"""You are {self.config.wake_word}, an intelligent voice assistant with temporal and conversation aware memory."""
        
        # Update system prompt
        self.llm.messages[0]["content"] = jarvis_prompt
    
    def _create_memory_page(self, prompt_text: str, response_text: str, speaker_id: str) -> MemoryPage:
        """
        Create a new memory page with combined prompt/response and embedding
        """
        # Combine prompt and response for embedding
        combined_text = f"{speaker_id}: {prompt_text}\n{self.config.wake_word}: {response_text}"
        
        memory_page = MemoryPage(
            prompt_text=prompt_text,
            response_text=response_text,
            combined_text=combined_text,
            speaker_id=speaker_id
        )
        
        # Generate embedding if embedder is available
        if self.embedder:
            try:
                # Use the full context text for embedding (includes metadata + combined conversation)
                full_context = memory_page.get_full_context_text()
                memory_page.text_embedding = self.embedder.embed_text_string(full_context)
                
                # Save to persistent storage
                self.embedder.save_embedding(
                    text=full_context,
                    embedding=memory_page.text_embedding,
                    custom_id=memory_page.memory_id
                )
                
                if self.config.debug_mode:
                    print(f"💾 Created combined memory page: {memory_page.memory_id}")
                    print(f"💾 Combined text: {combined_text[:100]}...")
                    
            except Exception as e:
                print(f"Error creating embedding for memory page: {e}")
        
        return memory_page
    
    def _find_similar_memories(self, query_text: str, exclude_id: Optional[str] = None) -> List[MemoryPage]:
        """
        Find similar memories based on text embedding
        IMPROVED: More resilient handling of missing memory pages
        """
        if not self.embedder:
            return []
        
        try:
            # Enhance query for better semantic matching
            enhanced_query = f"User asks about: {query_text}. Conversation involving: {query_text}"
            
            if self.config.debug_mode:
                print(f"🔍 Searching memories with enhanced query: '{enhanced_query}'")
            
            # Search for similar embeddings
            search_limit = self.config.max_similar_memories * 3  # Get more to account for filtering
            similar_results = self.embedder.search_by_text(enhanced_query, n=search_limit)
            
            if self.config.debug_mode:
                print(f"🔍 Raw search returned {len(similar_results)} results")
            
            similar_memories = []
            for memory_id, similarity_score, stored_text in similar_results:
                if self.config.debug_mode:
                    print(f"🔍 Checking result: ID={memory_id[:12]}..., Score={similarity_score:.3f}")
                
                # Apply similarity threshold filtering
                if similarity_score < self.config.memory_similarity_threshold:
                    if self.config.debug_mode:
                        print(f"🧠 Filtered out low similarity: {similarity_score:.3f} < {self.config.memory_similarity_threshold}")
                    continue
                
                # Skip if this is the memory we want to exclude
                if exclude_id and memory_id == exclude_id:
                    continue
                
                # Find the corresponding memory page by ID
                matching_page = next(
                    (page for page in self.memory_pages if page.memory_id == memory_id), 
                    None
                )
                
                if matching_page:
                    similar_memories.append(matching_page)
                    if self.config.debug_mode:
                        print(f"🧠 ✅ Found similar memory: {similarity_score:.3f} - {matching_page.combined_text[:50]}...")
                else:
                    # If we can't find the memory page, create a temporary one from stored text
                    if self.config.debug_mode:
                        print(f"🧠 ⚠️  Memory page not found for ID: {memory_id[:12]}..., creating temporary from stored text")
                    
                    # Get the embedding from storage
                    embeddings_store = getattr(self.embedder, 'embeddings_store', {})
                    embedding = embeddings_store.get(memory_id)
                    
                    if embedding is not None:
                        # Create a temporary memory page from the stored text
                        temp_memory = self._reconstruct_memory_page_from_text(stored_text, memory_id, embedding)
                        if temp_memory:
                            similar_memories.append(temp_memory)
                            # Also add it to our memory_pages for future use
                            self.memory_pages.append(temp_memory)
                            if self.config.debug_mode:
                                print(f"🧠 ✅ Created temporary memory: {similarity_score:.3f} - {temp_memory.combined_text[:50]}...")
                        else:
                            if self.config.debug_mode:
                                print(f"🧠 ❌ Failed to reconstruct temporary memory for ID: {memory_id[:12]}...")
                    else:
                        if self.config.debug_mode:
                            print(f"🧠 ❌ No embedding found in storage for ID: {memory_id[:12]}...")
                
                # Stop when we have enough
                if len(similar_memories) >= self.config.max_similar_memories:
                    break
            
            if self.config.debug_mode:
                print(f"🧠 Final result: {len(similar_memories)} similar memories found")
            
            return similar_memories
            
        except Exception as e:
            print(f"Error finding similar memories: {e}")
            return []
        
    def _build_memory_context(self, current_prompt: str, speaker_id: str) -> str:
        """Build memory context for the system prompt"""
        context_parts = []
        
        # Always check for recent memories with this speaker first
        if self.memory_pages:
            # Find the most recent conversation with this speaker
            recent_memories = [
                page for page in self.memory_pages
                if page.speaker_id == speaker_id
            ]
            
            if recent_memories:
                # Sort by timestamp and get the most recent
                recent_memories.sort(key=lambda x: x.timestamp, reverse=True)
                latest_memory = recent_memories[0]
                
                context_parts.append("<most_recent_interaction>")
                context_parts.append(f"{latest_memory.combined_text}</most_recent_interaction>")
                
                if self.config.debug_mode:
                    print(f"🧠 Found recent memory with {speaker_id}: {latest_memory.combined_text[:50]}...")
        
        # Find and add similar memories based on current prompt
        similar_memories = self._find_similar_memories(current_prompt)
        print(similar_memories)
        if similar_memories:
            context_parts.append("\n<interaction_related_memories>")
            for memory in similar_memories:
                time_info = f"[{memory.date_str} {memory.time_str}]"
                context_parts.append(f"{time_info} {memory.combined_text}")
                
                if self.config.debug_mode:
                    print(f"🧠 Added similar memory: {memory.combined_text[:50]}...")
            context_parts.append("</interaction_related_memories>")
        
        result = "\n".join(context_parts) if context_parts else "You don't have relevant memories for the <most_recent_interaction>."
        
        if self.config.debug_mode:
            print(f"🧠 Built memory context ({len(context_parts)} parts):")
            print(result[:200] + "..." if len(result) > 200 else result)
        
        return result
    
    def start(self):
        """Start Jarvis voice assistant"""
        
        self.should_stop = False
        self.speech_processor.start()
        
        # Start monitoring thread
        monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        # Start prompt processing thread
        prompt_thread = threading.Thread(target=self._prompt_processing_loop, daemon=True)
        prompt_thread.start()

        #Voice Sample to get own voice:
        self.tts.generate_speech_async(self.self_voice_sample_text(), speed=1.0)
        
        try:
            while not self.should_stop:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop Jarvis voice assistant"""
        self.should_stop = True
        self.speech_processor.stop()
        self.tts.cleanup()

        print("system stopped.")
    
    def _monitoring_loop(self):
        """Background monitoring for timeouts and state management"""
        while not self.should_stop:
            try:
                current_time = time.time()
                
                with self.state_lock:
                    # Check for wake word timeout
                    if (self.state == JarvisState.LISTENING and 
                        self.wake_word_detected_time and
                        current_time - self.wake_word_detected_time > self.config.wake_word_timeout):
                        
                        print("Wake word timeout - returning to idle")
                        self._set_state_internal(JarvisState.IDLE)
                        self._reset_prompt_state_internal()
                    
                    # Check for thinking timeout
                    if (self.state == JarvisState.THINKING and
                        self.thinking_start_time and
                        current_time - self.thinking_start_time > self.config.response_timeout):
                        
                        print("Thinking timeout - returning to idle")
                        self._set_state_internal(JarvisState.IDLE)
                        self._reset_prompt_state_internal()
                    
                    # Check for conversation timeout (continuous conversation mode)
                    if (self.config.continuous_conversation and 
                        self.in_conversation and 
                        self.last_speech_time and
                        current_time - self.last_speech_time > self.config.conversation_timeout):
                        
                        print("Conversation timeout - requiring wake word again")
                        self._end_conversation_internal()
                
                time.sleep(1.0)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(1.0)
    
    def _prompt_processing_loop(self):
        """Background prompt processing to avoid blocking main thread"""
        while not self.should_stop:
            try:
                # Check for prompts to process
                prompt_to_process = None
                
                with self.prompt_queue_lock:
                    if self.prompt_queue:
                        prompt_to_process = self.prompt_queue.pop(0)
                
                if prompt_to_process:
                    self._execute_prompt_internal(prompt_to_process)
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Error in prompt processing loop: {e}")
                time.sleep(1.0)
    
    def _set_state(self, new_state: JarvisState):
        """Thread-safe state change"""
        with self.state_lock:
            self._set_state_internal(new_state)
    
    def _set_state_internal(self, new_state: JarvisState):
        """Internal state change (assumes lock is held)"""
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            if self.config.debug_mode:
                print(f"🔄 State: {old_state.value} → {new_state.value}")
    
    def _on_speech_detected(self):
        """Handle speech detection events"""
        # Update last speech time for conversation timeout tracking
        with self.state_lock:
            self.last_speech_time = time.time()
    
    def _process_live_transcript(self, text: str, speaker_id: str):
        """Process real-time transcript for wake words and interrupts"""
        text_lower = text.lower().strip()
        
        with self.state_lock:
            # Check for interrupt phrase during thinking
            if self.state == JarvisState.THINKING or self.state == JarvisState.RESPONDING:
                if self._contains_interrupt_phrase(text_lower):
                    print(f"Interrupt detected - stopping current processing")
                    self._set_state_internal(JarvisState.IDLE)
                    self._reset_prompt_state_internal()
                    self.tts.stop_playback()

                    # Clear any pending prompts
                    with self.prompt_queue_lock:
                        self.prompt_queue.clear()
                return
            
            # Check for wake word in idle state OR if not in conversation
            if self.state == JarvisState.IDLE and (not self.in_conversation or not self.config.continuous_conversation):
                if self._contains_wake_word(text_lower):
                    print(f"Wake word detected")
                    self._set_state_internal(JarvisState.LISTENING)
                    # Don't process prompts from live transcript - wait for final
                    return
            
            # NEW: Check for wake word during active conversation to reset primary speaker
            elif (self.in_conversation and 
                self._contains_wake_word(text_lower) and 
                speaker_id != self.primary_speaker):
                
                print(f"Wake word detected by different speaker - switching primary speaker")
                print(f"Previous: {self.primary_speaker} → New: {speaker_id}")
                
                # Reset primary speaker and restart conversation state
                self.primary_speaker = speaker_id
                self._set_state_internal(JarvisState.LISTENING)
                self.wake_word_detected_time = time.time()
                self.last_speech_time = time.time()
                self.prompt_buffer.clear()
                
                # Clear any pending prompts from previous speaker
                with self.prompt_queue_lock:
                    self.prompt_queue.clear()
                
                return
            
            # Handle prompts in continuous conversation mode (no wake word needed)
            elif (self.state == JarvisState.IDLE and 
                self.in_conversation and 
                self.config.continuous_conversation and
                speaker_id == self.primary_speaker):
                
                # Any speech from primary speaker is treated as a prompt
                prompt_clean = text.strip()
                if self._is_valid_prompt(prompt_clean):
                    print(f"Continuous conversation prompt: {prompt_clean}")
                    # Don't process from live transcript - wait for final
                    return
    
    
    def set_self_voice_id(self,speaker_id):
        print(f"set my voice ID as: {speaker_id}")
        self.jarvis_voice_id = speaker_id

    def self_voice_sample_text(self):
        return "I need to calibrate my voice. The quick brown fox jumps over the lazy dog"
                
    
    def _process_final_transcript(self, segment: TranscriptSegment):
        """Process final transcript for prompts - NON-BLOCKING VERSION"""
        # Create unique identifier for this segment to avoid duplicate processing
        segment_id = f"{segment.speaker_id}_{segment.start_time}_{segment.text}"

        #Ignore system voice
        if segment.speaker_id == self.jarvis_voice_id:
            print(f"🔄 Skipping that's just me saying: {segment.text}")
            return
        
        # Quick duplicate check
        if segment_id in self._processed_segments:
            if self.config.debug_mode:
                print(f"🔄 Skipping already processed segment: {segment.text}")
            return
        
        # Mark as processed
        self._processed_segments.add(segment_id)
        
        # Clean up old processed segments (keep only recent ones)
        if len(self._processed_segments) > 100:
            segments_list = list(self._processed_segments)
            self._processed_segments = set(segments_list[-50:])
        
        text_lower = segment.text.lower().strip()
        
        # Capture current state safely
        with self.state_lock:
            current_state = self.state
            current_in_conversation = self.in_conversation
            current_primary_speaker = self.primary_speaker
        

        # CASE 1: Wake word detected in IDLE state or After speaker switch (not in conversation or continuous mode disabled)
        if ((current_state == JarvisState.IDLE or current_state == JarvisState.LISTENING) and 
            (not current_in_conversation or not self.config.continuous_conversation) and
            self._contains_wake_word(text_lower)):
            
            print(f"Wake word detected in final transcript!")
            
            # Start conversation if not already started
            with self.state_lock:
                if not self.in_conversation:
                    self._start_conversation_internal(segment.speaker_id)
            
            # Extract and queue the prompt part
            prompt_clean = self._clean_prompt(segment.text)
            if self._is_valid_prompt(prompt_clean):
                print(f"Processing wake word + prompt: '{prompt_clean}'")
                with self.prompt_queue_lock:
                    self.prompt_queue.append(prompt_clean)
            else:
                print(f"Wake word detected, waiting for prompt...")
            return
        
        # CASE 2: Continuous conversation mode - any speech from primary speaker is a prompt
        elif (current_state == JarvisState.IDLE and 
              current_in_conversation and 
              self.config.continuous_conversation and
              segment.speaker_id == current_primary_speaker):
            
            prompt_clean = segment.text.strip()
            if self._is_valid_prompt(prompt_clean):
                print(f"Continuous conversation prompt: '{prompt_clean}'")
                with self.prompt_queue_lock:
                    self.prompt_queue.append(prompt_clean)
            else:
                print(f" Invalid prompt in continuous mode: '{prompt_clean}'")
            return
        
        # CASE 3: Already in LISTENING state - this is the most common case after wake word
        elif current_state == JarvisState.LISTENING and segment.speaker_id == current_primary_speaker:
            # For final transcript in listening state, use the full prompt directly
            prompt_clean = self._clean_prompt(segment.text)
            
            if self._is_valid_prompt(prompt_clean):
                print(f"Processing final prompt: '{prompt_clean}'")
                with self.prompt_queue_lock:
                    self.prompt_queue.append(prompt_clean)
        
        # CASE 4: Any other state or speaker - log and ignore
        else:
            if segment.text.lower() == self.self_voice_sample_text():
                print("that was calibration transcript")
                self.set_self_voice_id(segment.speaker_id)

            print(f"Ignoring transcript - State: {current_state.value}, Speaker: {segment.speaker_id}, Primary: {current_primary_speaker}")
    
    #Kokoro callbacks
    def on_audio_ready(self,audio_data):
            print(f" Audio ready callback: {audio_data.file_path}")
            self.tts.speak()
        
    def on_playback_complete(self):
            print("Playback complete callback triggered!")

    def _contains_wake_word(self, text: str) -> bool:
        """Check if text contains the wake word"""
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(self.config.wake_word.lower()) + r'\b'
        return bool(re.search(pattern, text))
    
    def _contains_interrupt_phrase(self, text: str) -> bool:
        """Check if text has interrupt phrase but does not start with the interrupt phrase"""
        if self.config.interrupt_phrase.lower() in text:
            if not text.lower().startswith(self.config.interrupt_phrase.lower()):
                return True
    
    def _clean_prompt(self, text: str) -> str:
        """Clean prompt text by removing wake word and extra whitespace"""
        # Remove wake word from the beginning
        text_lower = text.lower()
        wake_word_lower = self.config.wake_word.lower()
        
        # Find and remove wake word with potential punctuation
        if text_lower.startswith(wake_word_lower):
            text = text[len(wake_word_lower):].strip()
        else:
            # Remove wake word if it appears anywhere in the text
            pattern = r'\b' + re.escape(wake_word_lower) + r'\b'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
        
        # Remove leading punctuation that might be left after wake word removal
        text = re.sub(r'^[,.\s]+', '', text).strip()
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _is_valid_prompt(self, prompt: str) -> bool:
        """Check if prompt is valid for processing"""
        if not prompt:
            return False
        
        words = prompt.split()
        if len(words) < self.config.min_prompt_length:
            print("input too short to trigger a response")
            return False
        
        # Add other validation rules here if needed
        return True
    
    def _execute_prompt_internal(self, prompt: str):
        """
        Execute a voice prompt using the LLM with memory integration
        """
        # Set state to thinking
        with self.state_lock:
            if self.state == JarvisState.LISTENING or self.state == JarvisState.IDLE:  # Accept from both states
                self._set_state_internal(JarvisState.THINKING)
                self.thinking_start_time = time.time()
                print("Thinking...")
                # Clear prompt buffer since we're processing a prompt
                self.prompt_buffer.clear()
                # Store current prompt for memory creation
                self.current_prompt = prompt
            else:
                # prompt was queued but state changed inappropriately, ignore it
                print(f"Ignoring queued prompt due to inappropriate state {self.state.value}: {prompt}")
                return
        
        try:
            # Build memory context for system prompt
            memory_context = self._build_memory_context(prompt, self.primary_speaker or "unknown")
            
            # Update system prompt with memory context
            enhanced_system_prompt = f"""You are {self.config.wake_word}, an intelligent robotic system with temporal and conversation aware memory. 

                You are conversing with {self.primary_speaker or 'the user'}. Here is the most recent prompt and response you had with {self.primary_speaker or 'the user'}:

                <system_memory>
                {memory_context}
                </system_memory>

                Use this memory context to provide more relevant and personalized responses. Reference previous conversations when appropriate."""
            
            print("+"*50)
            print(enhanced_system_prompt)
            print("+"*50)

            self.llm._update_system_prompt(enhanced_system_prompt)
            
            if self.config.debug_mode:
                print(f"🧠 Updated system prompt with memory context")
            

            # Generate response from LLM
            response = self.llm.generate_response(prompt, max_new_tokens=512)
            
            # Check if we weren't interrupted
            with self.state_lock:
                if self.state == JarvisState.THINKING:
                    #Create single memory page for prompt+response pair
                    with self.memory_lock:
                        combined_memory = self._create_memory_page(
                            prompt_text=self.current_prompt,
                            response_text=response,
                            speaker_id=self.primary_speaker or "unknown"
                        )
                        self.memory_pages.append(combined_memory)
                        
                        if self.config.debug_mode:
                            print(f"💾 Stored combined memory: {combined_memory.memory_id}")
                            print(f"💾 Total memories: {len(self.memory_pages)}")
                    
                    #text to speech generation in kokoro 
                    self.tts.generate_speech_async(response, speed=1.0)        

                    print(f"\n{self.config.wake_word}: {response}\n")
                    
                    # Print token stats if in debug mode
                    if self.config.debug_mode:
                        self.llm.print_token_stats()
                    
                    # Return to idle state (conversation continues if enabled)
                    self._set_state_internal(JarvisState.IDLE)
                    self._reset_prompt_state_internal()
                    
                    # Keep conversation active if continuous mode is enabled
                    if self.config.continuous_conversation:
                        if self.config.debug_mode:
                            print(f"Conversation continues... (timeout in {self.config.conversation_timeout}s)")
                else:
                    print("Response cancelled due to interrupt")
            
        except Exception as e:
            print(f"Error processing prompt: {e}")
            with self.state_lock:
                self._set_state_internal(JarvisState.IDLE)
                self._reset_prompt_state_internal()
    
    def _start_conversation_internal(self, speaker_id: str):
        """Start a conversation with a specific speaker - internal version"""
        self.in_conversation = True
        self._set_state_internal(JarvisState.LISTENING)
        self.wake_word_detected_time = time.time()
        self.last_speech_time = time.time()
        self.primary_speaker = speaker_id
        self.prompt_buffer.clear()

        print(f"Conversation started with {speaker_id}")
    
    def _end_conversation_internal(self):
        """End the current conversation and return to wake word required mode - internal version"""
        self.in_conversation = False
        self.primary_speaker = None
        self._set_state_internal(JarvisState.IDLE)
        self._reset_prompt_state_internal()
        self.speech_processor.speaker_id._perform_global_clustering()
        print(f"Conversation ended - sleeping until wake word detected")
    
    def _reset_prompt_state_internal(self):
        """Reset prompt-related state variables - internal version"""
        
        self.prompt_buffer.clear()
        self.wake_word_detected_time = None
        self.thinking_start_time = None
        self.current_prompt = None
        # Note: in_conversation and last_speech_time are NOT reset here
        # They persist until conversation timeout
    
    def get_memory_stats(self) -> dict:
        """Get statistics about the memory system"""
        with self.memory_lock:
            total_memories = len(self.memory_pages)
            
            # Get embedding stats if available
            embedding_stats = {}
            if self.embedder:
                embedding_stats = {
                    'stored_embeddings': self.embedder.get_stored_count(),
                    'embeddings_file': self.embedder.get_pickle_file_path()
                }
            
            return {
                'total_memories': total_memories,
                'embedding_stats': embedding_stats,
                'recent_memories': [
                    {
                        'id': page.memory_id,
                        'prompt': page.prompt_text[:50] + '...' if len(page.prompt_text) > 50 else page.prompt_text,
                        'response': page.response_text[:50] + '...' if len(page.response_text) > 50 else page.response_text,
                        'speaker_id': page.speaker_id,
                        'timestamp': page.timestamp.isoformat()
                    }
                    for page in self.memory_pages[-5:]  # Last 5 memories
                ]
            }
    
    def search_memories(self, query: str, n: int = 5) -> List[MemoryPage]:
        """Search memories by text similarity"""
        return self._find_similar_memories(query, exclude_id=None)[:n]
    
    def clear_memory(self, confirm: bool = False):
        """Clear all memories (requires confirmation)"""
        if not confirm:
            print("⚠️  Memory clear requires confirmation. Call with confirm=True")
            return False
        
        with self.memory_lock:
            self.memory_pages.clear()
            if self.embedder:
                self.embedder.clear_all_embeddings()
            print("🗑️  All memories cleared")
            return True
    
    def export_memories(self, filename: str = None) -> str:
        """Export memories to a text file"""
        if filename is None:
            filename = f"jarvis_memories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with self.memory_lock:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Jarvis Memory Export - {datetime.now().isoformat()}\n")
                f.write("=" * 50 + "\n\n")
                
                for page in self.memory_pages:
                    f.write(f"[{page.timestamp.isoformat()}] Conversation with {page.speaker_id}:\n")
                    f.write(f"USER: {page.prompt_text}\n")
                    f.write(f"JARVIS: {page.response_text}\n")
                    if page.associated_memories:
                        f.write(f"Associated memories: {', '.join(page.associated_memories)}\n")
                    f.write("\n" + "-" * 40 + "\n\n")
        
        print(f"📄 Memories exported to: {filename}")
        return filename
    
    def get_status(self) -> dict:
        """Get current status of Jarvis including memory stats"""
        with self.state_lock:
            base_status = {
                'state': self.state.value,
                'in_conversation': self.in_conversation,
                'primary_speaker': self.primary_speaker,
                'active_speakers': list(self.callback.active_speakers),
                'conversation_count': len(self.callback.current_conversation),
                'token_stats': self.llm.token_stats,
                'last_speech_time': self.last_speech_time,
                'processed_segments_count': len(self._processed_segments),
                'prompt_queue_size': len(self.prompt_queue)
            }
            
            # Add memory stats
            base_status['memory_stats'] = self.get_memory_stats()
            
            return base_status


def main():
    """Main function to run Jarvis with memory system"""
    # Custom configuration example
    config = JarvisConfig(
        wake_word="jarvis",
        interrupt_phrase="jarvis",
        wake_word_timeout=30.0,
        debug_mode=True,
        # Memory system configuration
        memory_similarity_threshold=0.7,
        max_similar_memories=2,
        embeddings_file="jarvis_memory.pkl"
    )
    
    # Initialize and start Jarvis
    jarvis = Jarvis(config)
    
    try:
        print("\n" + "=" * 60)
        print("🤖 JARVIS VOICE ASSISTANT WITH MEMORY SYSTEM")
        print("=" * 60)
        print(f"Wake word: '{config.wake_word}'")
        print(f"Interrupt phrase: '{config.interrupt_phrase}'")
        print(f"Memory file: {config.embeddings_file}")
        print(f"Memory similarity threshold: {config.memory_similarity_threshold}")
        print("=" * 60)
        print("Say the wake word to start a conversation...")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")
        
        jarvis.start()
        
    except Exception as e:
        print(f"Error running Jarvis: {e}")
    finally:
        # Print final memory stats before shutdown
        try:
            memory_stats = jarvis.get_memory_stats()
            print(f"\n📊 Final Memory Stats:")
            print(f"   Total conversation memories: {memory_stats['total_memories']}")
            if memory_stats['embedding_stats']:
                print(f"   Stored embeddings: {memory_stats['embedding_stats']['stored_embeddings']}")
            print("   Recent conversations:")
            for mem in memory_stats['recent_memories']:
                print(f"     [{mem['timestamp']}] {mem['speaker_id']}: {mem['prompt']} -> {mem['response']}")
        except:
            pass
        
        jarvis.stop()


# Additional utility functions for memory management
class MemoryManager:
    """Utility class for advanced memory management operations"""
    
    def __init__(self, jarvis_instance):
        self.jarvis = jarvis_instance
    
    def find_conversations_with_speaker(self, speaker_id: str) -> List[MemoryPage]:
        """Find all conversations with a specific speaker"""
        with self.jarvis.memory_lock:
            return [page for page in self.jarvis.memory_pages if page.speaker_id == speaker_id]
    
    def get_conversation_timeline(self, days_back: int = 7) -> List[MemoryPage]:
        """Get conversation timeline for the last N days"""
        cutoff_time = datetime.now() - datetime.timedelta(days=days_back)
        with self.jarvis.memory_lock:
            return [
                page for page in self.jarvis.memory_pages 
                if page.timestamp >= cutoff_time
            ]
    
    def analyze_conversation_patterns(self) -> dict:
        """Analyze conversation patterns and return statistics"""
        with self.jarvis.memory_lock:
            if not self.jarvis.memory_pages:
                return {}
            
            # Analyze by speaker
            speaker_stats = {}
            for page in self.jarvis.memory_pages:
                if page.speaker_id not in speaker_stats:
                    speaker_stats[page.speaker_id] = {
                        'total_conversations': 0,
                        'avg_prompt_length': 0,
                        'avg_response_length': 0,
                        'first_interaction': page.timestamp,
                        'last_interaction': page.timestamp,
                        'topics': []  # Could be expanded with topic analysis
                    }
                
                stats = speaker_stats[page.speaker_id]
                stats['total_conversations'] += 1
                stats['last_interaction'] = max(stats['last_interaction'], page.timestamp)
                stats['first_interaction'] = min(stats['first_interaction'], page.timestamp)
            
            # Calculate average lengths
            for speaker_id, stats in speaker_stats.items():
                speaker_pages = [p for p in self.jarvis.memory_pages if p.speaker_id == speaker_id]
                if speaker_pages:
                    stats['avg_prompt_length'] = sum(len(p.prompt_text.split()) for p in speaker_pages) / len(speaker_pages)
                    stats['avg_response_length'] = sum(len(p.response_text.split()) for p in speaker_pages) / len(speaker_pages)
            
            return {
                'total_memories': len(self.jarvis.memory_pages),
                'unique_speakers': len(speaker_stats),
                'speaker_stats': speaker_stats,
                'conversation_span': {
                    'start': min(p.timestamp for p in self.jarvis.memory_pages),
                    'end': max(p.timestamp for p in self.jarvis.memory_pages)
                }
            }


if __name__ == "__main__":
    main()