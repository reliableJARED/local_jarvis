"""
Jarvis Voice Assistant - Fixed thread management version
Combines speech processing with LLM capabilities for natural conversation
"""

import time
import threading
import re
from enum import Enum
from typing import Optional, List
from dataclasses import dataclass

# Import our custom modules (assuming they're in the same directory)
from whispervad_ import (
    VoiceAssistantSpeechProcessor, 
    VoiceAssistantCallback, 
    TranscriptSegment,
    SpeechEvent
)
from qwen_ import QwenChat
from kokoro_ import Kokoro

class JarvisState(Enum):
    """States for the Jarvis assistant"""
    IDLE = "idle"                    # Waiting for wake word
    LISTENING = "listening"          # Actively listening for command after wake word
    THINKING = "thinking"            # Processing command with LLM
    RESPONDING = "responding"        # Speaking response (if TTS implemented)


@dataclass
class JarvisConfig:
    """Configuration for Jarvis assistant"""
    wake_word: str = "jarvis"
    interrupt_phrase: str = "enough jarvis"
    wake_word_timeout: float = 30.0      # Seconds to wait for command after wake word
    response_timeout: float = 60.0       # Maximum thinking time
    conversation_timeout: float = 45.0   # Seconds of silence before requiring wake word again
    min_command_length: int = 2          # Minimum words in command - below this will not trigger response
    debug_mode: bool = True              # Print debug information
    continuous_conversation: bool = True  # Allow conversation without wake word after initial detection


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
        
        # Process final transcript for commands
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
    Integrates speech processing with LLM for complete voice interaction
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
        self.command_buffer = []
        self.wake_word_detected_time = None
        self.thinking_start_time = None
        self.last_speech_time = None     # Track last speech activity for conversation timeout
        self._processed_segments = set() # Track processed final segments to avoid duplicates
        
        # Thread synchronization
        self.state_lock = threading.Lock()
        self.should_stop = False
        
        # Command processing queue to avoid blocking
        self.command_queue = []
        self.command_queue_lock = threading.Lock()
        
        # Initialize components
        print("Initializing Jarvis...")
        self._init_speech_processor()
        self._init_llm()
    
    def _init_speech_processor(self):
        """Initialize speech processing components"""
        print("Loading speech processor...")
        self.callback = JarvisCallback(self)
        self.speech_processor = VoiceAssistantSpeechProcessor(self.callback)
    
    def _init_llm(self):
        """Initialize the language model"""
        print("Loading language model...")
        self.llm = QwenChat()
        
        # Add Jarvis-specific system prompt
        jarvis_prompt = """You are Jarvis, an intelligent voice assistant."""
        
        # Update system prompt
        self.llm.messages[0]["content"] = jarvis_prompt
    
    def start(self):
        """Start Jarvis voice assistant"""
        
        self.should_stop = False
        self.speech_processor.start()
        
        # Start monitoring thread
        monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        # Start command processing thread
        command_thread = threading.Thread(target=self._command_processing_loop, daemon=True)
        command_thread.start()

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
                        self._reset_command_state_internal()
                    
                    # Check for thinking timeout
                    if (self.state == JarvisState.THINKING and
                        self.thinking_start_time and
                        current_time - self.thinking_start_time > self.config.response_timeout):
                        
                        print("Thinking timeout - returning to idle")
                        self._set_state_internal(JarvisState.IDLE)
                        self._reset_command_state_internal()
                    
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
    
    def _command_processing_loop(self):
        """Background command processing to avoid blocking main thread"""
        while not self.should_stop:
            try:
                # Check for commands to process
                command_to_process = None
                
                with self.command_queue_lock:
                    if self.command_queue:
                        command_to_process = self.command_queue.pop(0)
                
                if command_to_process:
                    self._execute_command_internal(command_to_process)
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Error in command processing loop: {e}")
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
            if self.state == JarvisState.THINKING:
                if self._contains_interrupt_phrase(text_lower):
                    print(f"Interrupt detected - stopping current processing")
                    self._set_state_internal(JarvisState.IDLE)
                    self._reset_command_state_internal()
                    self.tts.stop_playback()

                    # Clear any pending commands
                    with self.command_queue_lock:
                        self.command_queue.clear()
                return
            
            # Check for wake word in idle state OR if not in conversation
            if self.state == JarvisState.IDLE and (not self.in_conversation or not self.config.continuous_conversation):
                if self._contains_wake_word(text_lower):
                    print(f"Wake word detected")
                    self._set_state_internal(JarvisState.LISTENING)
                    # Don't process commands from live transcript - wait for final
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
                self.command_buffer.clear()
                
                # Clear any pending commands from previous speaker
                with self.command_queue_lock:
                    self.command_queue.clear()
                
                return
            
            # Handle commands in continuous conversation mode (no wake word needed)
            elif (self.state == JarvisState.IDLE and 
                self.in_conversation and 
                self.config.continuous_conversation and
                speaker_id == self.primary_speaker):
                
                # Any speech from primary speaker is treated as a command
                command_clean = text.strip()
                if self._is_valid_command(command_clean):
                    print(f"Continuous conversation command: {command_clean}")
                    # Don't process from live transcript - wait for final
                    return
    
    
    def set_self_voice_id(self,speaker_id):
        print(f"set my voice ID as: {speaker_id}")
        self.jarvis_voice_id = speaker_id

    def self_voice_sample_text(self):
        return "I need to calibrate my voice. The quick brown fox jumps over the lazy dog"
                
    
    def _process_final_transcript(self, segment: TranscriptSegment):
        """Process final transcript for commands - NON-BLOCKING VERSION"""
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
            
            # Extract and queue the command part
            command_clean = self._clean_command(segment.text)
            if self._is_valid_command(command_clean):
                print(f"Processing wake word + command: '{command_clean}'")
                with self.command_queue_lock:
                    self.command_queue.append(command_clean)
            else:
                print(f"Wake word detected, waiting for command...")
            return
        
        # CASE 2: Continuous conversation mode - any speech from primary speaker is a command
        elif (current_state == JarvisState.IDLE and 
              current_in_conversation and 
              self.config.continuous_conversation and
              segment.speaker_id == current_primary_speaker):
            
            command_clean = segment.text.strip()
            if self._is_valid_command(command_clean):
                print(f"Continuous conversation command: '{command_clean}'")
                with self.command_queue_lock:
                    self.command_queue.append(command_clean)
            else:
                print(f" Invalid command in continuous mode: '{command_clean}'")
            return
        
        # CASE 3: Already in LISTENING state - this is the most common case after wake word
        elif current_state == JarvisState.LISTENING and segment.speaker_id == current_primary_speaker:
            # For final transcript in listening state, use the full command directly
            command_clean = self._clean_command(segment.text)
            
            if self._is_valid_command(command_clean):
                print(f"Processing final command: '{command_clean}'")
                with self.command_queue_lock:
                    self.command_queue.append(command_clean)
        
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
    
    def _clean_command(self, text: str) -> str:
        """Clean command text by removing wake word and extra whitespace"""
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
    
    def _is_valid_command(self, command: str) -> bool:
        """Check if command is valid for processing"""
        if not command:
            return False
        
        words = command.split()
        if len(words) < self.config.min_command_length:
            print("input too short to trigger a response")
            return False
        
        # Add other validation rules here if needed
        return True
    
    def _execute_command_internal(self, command: str):
        """Execute a voice command using the LLM - internal version"""
        # Set state to thinking
        with self.state_lock:
            if self.state == JarvisState.LISTENING or self.state == JarvisState.IDLE:  # Accept from both states
                self._set_state_internal(JarvisState.THINKING)
                self.thinking_start_time = time.time()
                print("Thinking...")
                # Clear command buffer since we're processing a command
                self.command_buffer.clear()
            else:
                # Command was queued but state changed inappropriately, ignore it
                print(f"Ignoring queued command due to inappropriate state {self.state.value}: {command}")
                return
        
        try:
            # Get response from LLM.
            #self.llm.messages is handled inside 
            response = self.llm.generate_response(command,max_new_tokens=512)
            
            # Check if we weren't interrupted
            with self.state_lock:
                if self.state == JarvisState.THINKING:
                    
                    #text to speech generation in kokoro 
                    self.tts.generate_speech_async(response, speed=1.0)        

                    print(f"\nJarvis: {response}\n")
                    
                    # Print token stats if in debug mode
                    if self.config.debug_mode:
                        self.llm.print_token_stats()
                    
                    # Return to idle state (conversation continues if enabled)
                    self._set_state_internal(JarvisState.IDLE)
                    self._reset_command_state_internal()
                    
                    # Keep conversation active if continuous mode is enabled
                    if self.config.continuous_conversation:
                        if self.config.debug_mode:
                            print(f"Conversation continues... (timeout in {self.config.conversation_timeout}s)")
                else:
                    print("Response cancelled due to interrupt")
            
        except Exception as e:
            print(f"Error processing command: {e}")
            with self.state_lock:
                self._set_state_internal(JarvisState.IDLE)
                self._reset_command_state_internal()
    
    def _start_conversation_internal(self, speaker_id: str):
        """Start a conversation with a specific speaker - internal version"""
        self.in_conversation = True
        self._set_state_internal(JarvisState.LISTENING)
        self.wake_word_detected_time = time.time()
        self.last_speech_time = time.time()
        self.primary_speaker = speaker_id
        self.command_buffer.clear()

        print(f"Conversation started with {speaker_id}")
    
    def _end_conversation_internal(self):
        """End the current conversation and return to wake word required mode - internal version"""
        self.in_conversation = False
        self.primary_speaker = None
        self._set_state_internal(JarvisState.IDLE)
        self._reset_command_state_internal()

        print(f"Conversation ended - sleeping until wake word detected")
    
    def _reset_command_state_internal(self):
        """Reset command-related state variables - internal version"""
        
        self.command_buffer.clear()
        self.wake_word_detected_time = None
        self.thinking_start_time = None
        # Note: in_conversation and last_speech_time are NOT reset here
        # They persist until conversation timeout
    
    def get_status(self) -> dict:
        """Get current status of Jarvis"""
        with self.state_lock:
            return {
                'state': self.state.value,
                'in_conversation': self.in_conversation,
                'primary_speaker': self.primary_speaker,
                'active_speakers': list(self.callback.active_speakers),
                'conversation_count': len(self.callback.current_conversation),
                'token_stats': self.llm.token_stats,
                'last_speech_time': self.last_speech_time,
                'processed_segments_count': len(self._processed_segments),
                'command_queue_size': len(self.command_queue)
            }


def main():
    """Main function to run Jarvis"""
    # Custom configuration example
    config = JarvisConfig(
        wake_word="jarvis",
        interrupt_phrase="jarvis",
        wake_word_timeout=30.0,
        debug_mode=True
    )
    
    # Initialize and start Jarvis
    jarvis = Jarvis(config)
    
    try:
        jarvis.start()
    except Exception as e:
        print(f"Error running Jarvis: {e}")
    finally:
        jarvis.stop()
        


if __name__ == "__main__":
    main()