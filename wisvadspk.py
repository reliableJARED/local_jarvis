"""
Integrated Speech Processing System
Combines speaker recognition with VAD and transcription
"""

import time
import sys
import ctypes
import threading
from typing import Optional, List
import datetime

# Import the modular components
from speaker_recognition import SpeakerIdentifier
from whispervad_ import VADTranscriptionProcessor, SpeechCallback, TranscriptSegment, SpeechEvent


class IntegratedSpeechCallback(SpeechCallback):
    """Integrated callback that handles both speaker recognition and transcription"""
    
    def __init__(self, speaker_identifier: SpeakerIdentifier):
        self.speaker_identifier = speaker_identifier
        self.current_conversation = []
        self.active_speakers = set()
        self.primary_speaker = None  # Track primary human speaker
        
        # For integration with external systems (like Jarvis)
        self.jarvis = None
        self.jarvis_voice_id = None
    
    def set_jarvis_integration(self, jarvis_instance, jarvis_voice_id: str = None):
        """Set up integration with Jarvis voice assistant"""
        self.jarvis = jarvis_instance
        self.jarvis_voice_id = jarvis_voice_id or "JARVIS"
    
    def on_speech_start(self, event: SpeechEvent):
        print(f"🎤 Speech detected at {event.timestamp:.1f}")
    
    def on_speech_end(self, event: SpeechEvent):
        print(f"🔇 Speech ended at {event.timestamp:.1f}")
        
        # Identify speaker using the audio data
        if event.audio_data is not None and len(event.audio_data) > 8000:  # ~0.5 seconds
            speaker_id = self.speaker_identifier.identify_speaker(event.audio_data)
            
            # Update primary speaker if this is a human speaker
            if (speaker_id != "USER_00" and 
                speaker_id != self.jarvis_voice_id and
                not speaker_id.startswith('SYSTEM_')):
                self.primary_speaker = speaker_id
    
    def on_transcript_update(self, segment: TranscriptSegment):
        # Identify speaker using the audio data if available
        if hasattr(segment, 'audio_data') and segment.audio_data is not None:
            if len(segment.audio_data) > 8000:  # ~0.5 seconds
                identified_speaker = self.speaker_identifier.identify_speaker(segment.audio_data)
                segment.speaker_id = identified_speaker
                
                # Update primary speaker
                if (identified_speaker != "USER_00" and 
                    identified_speaker != self.jarvis_voice_id and
                    not identified_speaker.startswith('SYSTEM_')):
                    self.primary_speaker = identified_speaker
        
        print(f"📝 [{segment.speaker_id}] (live): {segment.text}")
    
    def on_transcript_final(self, segment: TranscriptSegment):
        # For final transcripts, we might want to re-identify the speaker more accurately
        if hasattr(segment, 'audio_data') and segment.audio_data is not None:
            if len(segment.audio_data) > 8000:  # ~0.5 seconds
                # Re-identify speaker with full audio for better accuracy
                identified_speaker = self.speaker_identifier.identify_speaker(segment.audio_data)
                segment.speaker_id = identified_speaker
                
                # Update primary speaker
                if (identified_speaker != "USER_00" and 
                    identified_speaker != self.jarvis_voice_id and
                    not identified_speaker.startswith('SYSTEM_')):
                    self.primary_speaker = identified_speaker
        
        print(f"✅ [{segment.speaker_id}] (final): {segment.text}")
        self.current_conversation.append(segment)
        self.active_speakers.add(segment.speaker_id)
        
        # If we have Jarvis integration, pass the transcript to Jarvis
        if self.jarvis and hasattr(self.jarvis, 'process_transcript'):
            try:
                self.jarvis.process_transcript(segment)
            except Exception as e:
                print(f"Error passing transcript to Jarvis: {e}")
    
    def on_speaker_change(self, old_speaker: Optional[str], new_speaker: str):
        if old_speaker is not None:
            print(f"👥 Speaker change: {old_speaker} → {new_speaker}")
        else:
            print(f"👤 New speaker: {new_speaker}")
    
    def get_conversation_history(self) -> List[TranscriptSegment]:
        """Get full conversation history"""
        return self.current_conversation.copy()
    
    def get_recent_transcript(self, seconds: float = 30.0) -> str:
        """Get recent transcript as formatted string"""
        cutoff_time = time.time() - seconds
        recent_segments = [s for s in self.current_conversation if s.start_time >= cutoff_time]
        
        result = []
        for segment in recent_segments:
            result.append(f"[{segment.speaker_id}]: {segment.text}")
        
        return "\n".join(result)
    
    def get_primary_speaker(self) -> Optional[str]:
        """Get primary human speaker"""
        return self.primary_speaker
    
    def get_active_speakers(self) -> List[str]:
        """Get list of active speakers"""
        return list(self.active_speakers)
    
    def rename_speaker(self, speaker_id: str, new_name: str) -> Optional[str]:
        """
        Rename a speaker with underscore-based collision detection
        
        Args:
            speaker_id: Current speaker ID to rename (e.g., 'USER_00')
            new_name: New name for the speaker (e.g., 'Joe')
        
        Returns:
            str: The final speaker_id that was assigned, or None if operation failed
            
        Examples:
            - First 'Joe' → 'Joe'
            - Second 'Joe' → 'Joe_'  
            - Third 'Joe' → 'Joe__'
            - Fourth 'Joe' → 'Joe___'
        """
        # Check if the speaker exists
        if speaker_id not in self.speaker_profiles:
            print(f"Error: Speaker '{speaker_id}' not found")
            return None
        
        # Clean and format the new name
        formatted_name = new_name.strip().replace(' ', '_')
        
        # Remove any characters that might cause issues (keep alphanumeric, underscore, hyphen)
        import re
        formatted_name = re.sub(r'[^a-zA-Z0-9_\-]', '', formatted_name)
        
        if not formatted_name:
            print("Error: Invalid name provided")
            return None
        
        # Handle collision detection with underscores
        final_speaker_id = self._get_unique_speaker_id_with_underscores(formatted_name)
        
        # If the final ID is the same as current, no change needed
        if final_speaker_id == speaker_id:
            print(f"Speaker '{speaker_id}' name unchanged")
            return speaker_id
        
        # Perform the rename by moving the profile to the new key
        try:
            # Get the existing profile
            profile = self.speaker_profiles[speaker_id]
            
            # Update the speaker_id within the profile
            profile.speaker_id = final_speaker_id
            profile.last_updated = datetime.now()
            
            # Move to new key in the dictionary
            self.speaker_profiles[final_speaker_id] = profile
            
            # Remove the old key
            del self.speaker_profiles[speaker_id]
            
            # Update current_speaker_id if it was pointing to the renamed speaker
            if self.current_speaker_id == speaker_id:
                self.current_speaker_id = final_speaker_id
            
            print(f"Successfully renamed speaker '{speaker_id}' to '{final_speaker_id}'")
            
            # Schedule a save to persist the change
            self._schedule_save()
            
            return final_speaker_id
            
        except Exception as e:
            print(f"Error renaming speaker: {e}")
            return None
        
    def get_speaker_stats(self):
        """Get speaker statistics"""
        return self.speaker_identifier.get_speaker_stats()
    
    def get_clustering_stats(self):
        """Get clustering statistics"""
        return self.speaker_identifier.get_clustering_stats()


class IntegratedSpeechProcessor:
    """
    Main integrated speech processing system that combines:
    - Speaker recognition and profiling
    - Voice activity detection  
    - Speech transcription
    - Text input handling
    """
    
    def __init__(self, 
                 speaker_db_path: str = "integrated_speaker_profiles.pkl",
                 auto_clustering: bool = False,
                 jarvis_integration=None):
        
        # Initialize speaker recognition
        self.speaker_identifier = SpeakerIdentifier(
            speaker_db_path=speaker_db_path,
            auto_clustering=auto_clustering
        )
        
        # Initialize integrated callback
        self.callback = IntegratedSpeechCallback(self.speaker_identifier)
        
        # Set up Jarvis integration if provided
        if jarvis_integration:
            self.callback.set_jarvis_integration(jarvis_integration)
        
        # Initialize VAD and transcription processor
        self.vad_transcription = VADTranscriptionProcessor(self.callback)
        
        # State tracking
        self.is_running = False
    
    def start(self):
        """Start the integrated speech processing system"""
        print("Starting integrated speech processing system...")
        print("Features:")
        print("  - Speaker recognition and clustering")
        print("  - Voice activity detection")
        print("  - Real-time speech transcription")
        print("  - Text input support")
        print("  - Word boundary detection")
        
        self.is_running = True
        self.vad_transcription.start()
        
        print("System ready! Speak into the microphone or type messages.")
        print("Press Ctrl+C to stop.")
    
    def stop(self, force_kill=False, timeout=2.0):
        """Stop the integrated system"""
        if not self.is_running:
            return
            
        print("Stopping integrated speech processing system...")
        
        # Stop VAD and transcription first
        self.vad_transcription.stop(timeout=timeout)
        
        # Clean up speaker identifier (saves profiles)
        try:
            self.speaker_identifier.cleanup()
        except Exception as e:
            print(f"Error cleaning up speaker identifier: {e}")
        
        # Handle speaker identifier threads more gracefully
        self._graceful_speaker_cleanup(timeout=timeout)
        
        # Only use force_kill as absolute last resort
        if force_kill:
            print("Warning: Using force kill - this may cause instability")
            self._force_cleanup()
        
        self.is_running = False
        print("Integrated speech processing system stopped.")
    
    def _graceful_speaker_cleanup(self, timeout=2.0):
        """Attempt graceful cleanup of speaker identifier threads"""
        speaker_threads = []
        
        if hasattr(self.speaker_identifier, '_auto_save_thread') and self.speaker_identifier._auto_save_thread.is_alive():
            speaker_threads.append(("speaker_auto_save_thread", self.speaker_identifier._auto_save_thread))
        if hasattr(self.speaker_identifier, '_clustering_thread') and self.speaker_identifier._clustering_thread.is_alive():
            speaker_threads.append(("speaker_clustering_thread", self.speaker_identifier._clustering_thread))
        
        if not speaker_threads:
            print("No speaker threads to clean up")
            return
        
        print(f"Waiting for {len(speaker_threads)} speaker threads to finish gracefully...")
        
        for thread_name, thread in speaker_threads:
            print(f"Waiting for {thread_name}...")
            thread.join(timeout=timeout)
            
            if thread.is_alive():
                print(f"Warning: {thread_name} did not stop gracefully within {timeout}s")
            else:
                print(f"✓ {thread_name} stopped gracefully")
    
    def _force_cleanup(self):
        """Force cleanup of any remaining threads"""
        # This would handle any stubborn threads from the speaker identifier
        speaker_threads = []
        if hasattr(self.speaker_identifier, '_auto_save_thread') and self.speaker_identifier._auto_save_thread.is_alive():
            speaker_threads.append(("speaker_auto_save_thread", self.speaker_identifier._auto_save_thread))
        if hasattr(self.speaker_identifier, '_clustering_thread') and self.speaker_identifier._clustering_thread.is_alive():
            speaker_threads.append(("speaker_clustering_thread", self.speaker_identifier._clustering_thread))
        
        for thread_name, thread in speaker_threads:
            try:
                thread.join(timeout=1.0)
                if thread.is_alive():
                    print(f"Force terminating {thread_name}")
                    self._force_kill_thread(thread, thread_name)
            except Exception as e:
                print(f"Error handling {thread_name}: {e}")
    
    def _force_kill_thread(self, thread, thread_name="unknown"):
        """Force kill a thread (platform dependent - use with caution)"""
        if not thread.is_alive():
            print(f"Thread {thread_name} is already dead")
            return
        
        print(f"Force killing thread: {thread_name}")
        
        try:
            thread_id = thread.ident
            if thread_id is None:
                print(f"Could not get thread ID for {thread_name}")
                return
            
            if sys.platform == "win32":
                # Windows
                try:
                    import ctypes.wintypes
                    kernel32 = ctypes.windll.kernel32
                    THREAD_TERMINATE = 0x0001
                    thread_handle = kernel32.OpenThread(THREAD_TERMINATE, False, thread_id)
                    
                    if thread_handle:
                        result = kernel32.TerminateThread(thread_handle, 0)
                        kernel32.CloseHandle(thread_handle)
                        if result:
                            print(f"Thread {thread_name} terminated (Windows)")
                        else:
                            print(f"Failed to terminate thread {thread_name}")
                    else:
                        print(f"Could not open thread handle for {thread_name}")
                except Exception as win_error:
                    print(f"Windows thread termination failed for {thread_name}: {win_error}")
            
            elif sys.platform.startswith("linux") or sys.platform == "darwin":
                # Linux/macOS
                try:
                    import ctypes.util
                    pthread_lib_name = ctypes.util.find_library("pthread")
                    if pthread_lib_name:
                        pthread = ctypes.CDLL(pthread_lib_name)
                        result = pthread.pthread_cancel(ctypes.c_ulong(thread_id))
                        if result == 0:
                            print(f"Thread {thread_name} cancelled (Unix)")
                        else:
                            print(f"Failed to cancel thread {thread_name}: error code {result}")
                    else:
                        print(f"Could not find pthread library for {thread_name}")
                except Exception as unix_error:
                    print(f"Unix thread cancellation failed for {thread_name}: {unix_error}")
            
            else:
                print(f"Unsupported platform for force killing: {sys.platform}")
        
        except Exception as e:
            print(f"Error in force kill for {thread_name}: {e}")
        
        # Give a moment for cleanup
        time.sleep(0.1)
        
        # Check if thread is still alive after attempt
        if thread.is_alive():
            print(f"Warning: Thread {thread_name} may still be running after force kill attempt")
    
    def enable_text_input(self):
        """Enable text input mode"""
        self.vad_transcription.enable_text_input()
    
    def disable_text_input(self):
        """Disable text input mode"""
        self.vad_transcription.disable_text_input()
    
    def rename_speaker(self, speaker_id: str, new_name: str) -> Optional[str]:
        """Rename a speaker"""
        return self.callback.rename_speaker(speaker_id, new_name)
    
    def _get_unique_speaker_id_with_underscores(self, base_name: str) -> str:
        """
        Generate a unique speaker ID using underscore collision handling
        Args:
            base_name: The desired base name (e.g., 'Joe')
        Returns:
            str: A unique speaker ID (e.g., 'Joe', 'Joe_', 'Joe__', 'Joe___')
        """
        # Check if the base name is already unique
        if base_name not in self.speaker_profiles:
            return base_name
        
        # Keep adding underscores until we find a unique name
        candidate_name = base_name
        while candidate_name in self.speaker_profiles:
            candidate_name += "_"
            
            # Safety check to prevent infinite loops (shouldn't happen in practice)
            if len(candidate_name) > len(base_name) + 100:
                # Fallback to timestamp-based uniqueness
                timestamp_suffix = str(int(time.time()))[-6:]  # Last 6 digits of timestamp
                return f"{base_name}_{timestamp_suffix}"
        
        return candidate_name

    def rename_speaker_with_underscores(self, speaker_id: str, new_name: str) -> Optional[str]:
        """Rename a speaker using underscore collision handling"""
        return self.speaker_identifier.rename_speaker_with_underscores(speaker_id, new_name)

    def get_speaker_stats(self):
        """Get speaker statistics"""
        return self.callback.get_speaker_stats()
    
    def get_clustering_stats(self):
        """Get clustering statistics"""  
        return self.callback.get_clustering_stats()
    
    def get_conversation_history(self) -> List[TranscriptSegment]:
        """Get conversation history"""
        return self.callback.get_conversation_history()
    
    def get_recent_transcript(self, seconds: float = 30.0) -> str:
        """Get recent transcript"""
        return self.callback.get_recent_transcript(seconds)
    
    def merge_speakers(self, speaker_id1: str, speaker_id2: str, keep_id: str = None) -> str:
        """Merge two speaker profiles"""
        return self.speaker_identifier.merge_speakers(speaker_id1, speaker_id2, keep_id)
    
    def set_jarvis_integration(self, jarvis_instance, jarvis_voice_id: str = None):
        """Set up Jarvis integration"""
        self.callback.set_jarvis_integration(jarvis_instance, jarvis_voice_id)


# Example usage and testing
if __name__ == "__main__":
    # Create the integrated processor
    processor = IntegratedSpeechProcessor(
        speaker_db_path="integrated_test_speakers.pkl",
        auto_clustering=True  # Enable automatic clustering
    )
    
    try:
        processor.start()
        
        # Main loop - demonstrate functionality
        last_stats_time = time.time()
        
        while True:
            time.sleep(5)
            
            # Show recent conversation
            recent = processor.get_recent_transcript(30.0)
            if recent:
                print(f"\n--- Recent conversation (30s) ---")
                print(recent)
                print("="*50)
            
            # Show speaker stats every 30 seconds
            current_time = time.time()
            if current_time - last_stats_time > 30:
                stats = processor.get_speaker_stats()
                if stats:
                    print(f"\n--- Speaker Statistics ---")
                    for speaker_id, data in stats.items():
                        print(f"{speaker_id}: {data['total_samples']} samples, "
                              f"confidence: {data['avg_confidence']:.2f}")
                    print("="*50)
                last_stats_time = current_time
    
    except KeyboardInterrupt:
        print("\nShutting down...")
        processor.stop(force_kill=True)
        print("Goodbye!")
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        processor.stop(force_kill=True)