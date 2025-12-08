

import queue
import time

from datetime import datetime

from collections import deque

import threading
import logging
import sqlite3


logging.basicConfig(level=logging.INFO) #ignore everything use (level=logging.CRITICAL + 1)

# Set the Werkzeug logger level to ERROR to ignore INFO messages
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


"""
TEMPORAL LOBE
"""
class TemporalLobe:
    """
    Combines visual and auditory cortex data into unified temporal awareness.
    Buffers data for a specified time window and creates unified updates by rolling up
    non-default values from the most recent to oldest frames.
    """
    
    def __init__(self, visual_cortex=None, auditory_cortex=None,mpm=None, database_path=":memory:"):
        """
        Initialize TemporalLobe with visual and auditory cortex instances.
        
        Args:
            visual_cortex: VisualCortex instance
            auditory_cortex: AuditoryCortex instance
            mpm: Multiprocessing manager for creating queues
        """
        self.running = False
        self.collection_thread = None
        if not mpm:
            logging.warning("You should pass a multiprocessing.Manager() instance for real-time queues")
        #VISUAL    
        self.visual_cortex = visual_cortex#running instance of VisualCortex
        self.visual_data = None #latest visual data frame
        #AUDITORY
        self.auditory_cortex = auditory_cortex#running instance of AuditoryCortex
        self.auditory_data = None #latest auditory data frame
        #UI INPUTS
        self.user_input_dict = mpm.dict({'text':""}) #latest user text input from UI

        self.external_temporallobe_to_prefrontalcortex = mpm.Queue(maxsize=1) #real-time unified state queue to Cerebrum

        #last unified state
        self.last_unified_state = self._create_empty_unified_state()

        # Connect to database with sqlite-vec extension
        self.db = sqlite3.connect(database_path)

        logging.info(f"TemporalLobe initialized ")

    def speak(self,text):
        self.auditory_cortex.brocas_area.synthesize_speech(text, auto_play=True)


    def _create_empty_unified_state(self):
        """Create an empty unified state with default values"""
        return {
            # Timing
            'timestamp': time.time(),
            'formatted_time': datetime.now().strftime('%H:%M:%S'),
            
            # Visual data
            'person_detected': False,
            'person_match': False,
            'person_match_probability': 0.0,
            'caption': "",
            'vlm_timestamp': None,
            'visual_camera_index': None,
            
            # Audio data  
            'speech_detected': False,
            'transcription': "",
            'final_transcript': False,
            'voice_id': False,
            'voice_probability': 0.0,
            'audio_device_index': None,
            'transcription_timestamp': None,
            'audio_capture_timestamp': None,
            
            # Active speaker tracking
            'locked_speaker_id': None,
            'locked_speaker_timestamp': None,
            'is_locked_speaker': False,

            #Speech Output
            'actively_speaking':False,
            
            # UI Inputs
            'user_text_input': "",#string input from user
            #TODO: user file inputs will need OCR string extraction before going to context for AI since this is NOT a multi-modal model
            #'user_file_input': [],#list of file blobs
            #TODO: update the visual/audio cortexes to handle user specific inputs like pictures/screen shots/audio clips
        }

    def _is_default_value(self, key, value):
        """Check if a value is considered 'default' and should be replaced.
        This is for user inputs and audio/visual"""
        defaults = self._create_empty_unified_state()
        return key in defaults and value == defaults[key]
        
    
    ########
    #PRIMARY METHOD - runs in a parallel process threaded loop
    ########
    def _temporalLobe_State_Loop(self):
        """triggered by a Background thread to continuously collect frames from both cortexes and relay to real-time queues"""
        logging.info("\nTemporalLobe frame collection loop started\n")
        
        while self.running:
            current_time = time.time()
            user_text = False
            # Collect visual frames
            if self.visual_cortex:
                try:
                    visual_data = self.visual_cortex.external_cortex_queue.get_nowait()
                    visual_data['collection_timestamp'] = current_time
                    self.visual_data = visual_data
                                    
                except queue.Empty:
                    pass
            else:
                logging.error("Can't Collect visual data, No visual cortex available in Temporal Lobe")
            
            # Collect audio frames  
            if self.auditory_cortex:
                try:
                    audio_data = self.auditory_cortex.external_cortex_queue.get_nowait()
                    self.auditory_data = audio_data
                    audio_data['collection_timestamp'] = current_time
                    
                except queue.Empty:
                    pass
            else:
                logging.error("Can't Collect auditory data, No auditory cortex available in Temporal Lobe")

            # collect User Inputs from UI
            try:
                user_text = self.user_input_dict.get('text',False)
                if user_text:
                    self.user_input_dict.update({'text':""}) #clear after reading
            except Exception as e:
                logging.error(f"Error reading user input dict: {e}")


            #check if we have a final transcript OR user text input.
            #WE HAVE NEW DATA TO SEND TO CEREBRUM
            if self.auditory_data.get('final_transcript',False) or len(user_text.strip())>0:
                #Create unified state
                unified_state = self._create_empty_unified_state()
                
                #If Auditory Data:
                if self.auditory_data.get('final_transcript',False):
                    #Merge auditory data first to prioritize speech
                    for key, value in self.auditory_data.items():
                        if not self._is_default_value(key, value):
                            unified_state[key] = value
                #Else user text input only if we didn't have audio
                else:
                    #Merge user text input
                    unified_state['user_text_input'] = user_text.strip()

                #Merge visual data in both cases
                for key, value in self.visual_data.items():
                        if not self._is_default_value(key, value):
                            unified_state[key] = value
                    
                #Update timestamp
                unified_state['timestamp'] = current_time
                unified_state['formatted_time'] = datetime.now().strftime('%H:%M:%S')
                
                #Send to real-time queue

                if self.external_temporallobe_to_prefrontalcortex.full():
                    logging.warning("TemporalLobe to Cerebrum queue is full, not ready to process new data")
                else:
                    self.external_temporallobe_to_prefrontalcortex.put_nowait(unified_state)
                    #Also store last unified state
                    self.last_unified_state = unified_state

                    
            # Small delay to prevent excessive CPU usage
            time.sleep(0.001)
        
        logging.info("TemporalLobe frame collection stopped")


    def start_collection(self):
        """Start background frame collection"""
        if self.running:
            logging.warning("TemporalLobe collection already running")
            return
            
        self.running = True
        self.collection_thread = threading.Thread(target=self._temporalLobe_State_Loop, daemon=True)
        self.collection_thread.start()
        logging.info("TemporalLobe collection started")

    def stop_collection(self):
        """Stop background frame collection"""
        if not self.running:
            logging.warning("TemporalLobe collection not running")
            return
            
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=2)
        logging.info("TemporalLobe collection stopped")

    def start_visual_nerve(self, device_index=0):
        """Start visual nerve for specified camera"""
        if self.visual_cortex:
            self.visual_cortex.start_nerve(device_index)
            logging.info(f"Started visual nerve for camera {device_index}")
        else:
            logging.error("No visual cortex available")

    def stop_visual_nerve(self, device_index=0):
        """Stop visual nerve for specified camera"""
        if self.visual_cortex:
            success = self.visual_cortex.stop_nerve(device_index)
            logging.info(f"Stopped visual nerve for camera {device_index}: {success}")
            return success
        return False

    def start_audio_nerve(self, device_index=0):
        """Start auditory nerve for specified audio device"""  
        if self.auditory_cortex:
            self.auditory_cortex.start_nerve(device_index)
            logging.info(f"Started auditory nerve for device {device_index}")
        else:
            logging.error("No auditory cortex available")

    def stop_audio_nerve(self, device_index=0):
        """Stop auditory nerve for specified audio device"""
        if self.auditory_cortex:
            success = self.auditory_cortex.stop_nerve(device_index)
            logging.info(f"Stopped auditory nerve for device {device_index}: {success}")
            return success
        return False

    def stop_visual_system(self):
        """Stop complete visual system"""
        if self.visual_cortex:
            success = self.visual_cortex.shutdown()
            if success:
                logging.info(f"Temporal Lobe Visual system stopped")
                return success
        return False

    def stop_audio_system(self):
        """Stop complete audio system"""  
        if self.auditory_cortex:
            success = self.auditory_cortex.shutdown()
            if success:
                logging.info(f"Temporal Lobe Auditory system stopped")
                return success
        return False

    def shutdown_all(self):
        """Complete system shutdown"""
        logging.info("Starting Temporal Lobe complete shutdown")
        
        # Stop collection first
        self.stop_collection()
        
        had_failure = False
        
        # Shutdown visual system
        if self.visual_cortex:
            visual_success = self.stop_visual_system()
            if not visual_success:
                logging.error("Visual cortex shutdown had failures")
                had_failure = True
        
        # Shutdown audio system  
        if self.auditory_cortex:
            audio_success = self.stop_audio_system()
            if not audio_success:
                logging.error("Auditory cortex shutdown had failures") 
                had_failure = True
        
        # Clear buffers
        self.visual_buffer.clear()
        self.audio_buffer.clear()
        
        # Clear speaker lock
        self.locked_speaker_id = None
        self.locked_speaker_timestamp = None
        
        if had_failure:
            logging.error("Temporal Lobe shutdown completed with some failures")
        else:
            logging.info("Temporal Lobe shutdown completed successfully")
            
        return not had_failure

    def get_last_unified_state(self):
        """Get the last created unified state without creating a new one"""
        return self.last_unified_state
    