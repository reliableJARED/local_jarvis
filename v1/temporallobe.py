

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
    
    def __init__(self, visual_cortex=None, auditory_cortex=None,mpm=None, database_path=":memory:",status_dict=None,PrefrontalCortex_interrupt_dict=None):
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
        #INTERUPT DICT
        self.prefrontal_cortex_interrupt_dict = PrefrontalCortex_interrupt_dict

        self.external_temporallobe_to_prefrontalcortex = mpm.Queue(maxsize=30) #real-time unified state queue to Cerebrum

        #last unified state
        self.last_unified_state = self._create_empty_unified_state()

        #status dict for UI monitoring
        self.status_dict = status_dict
        self.status_dict.update({"running":self.running,
                                 "visual_data":None,
                                 "auditory_data":None,
                                 "user_input_text":self.user_input_dict.get('text',""),
                                 "last_unified_state":self.last_unified_state,
                                 })

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

            #Interruption Attempt
            'is_interrupt_attempt':False,
            
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
            #data schema for the unified state
            unified_state = None
            received_audio_transcript = False
            # Collect VISUAL Data
            if self.visual_cortex:
                try:
                    visual_data = self.visual_cortex.external_cortex_queue.get_nowait()
                    visual_data['collection_timestamp'] = current_time
                    #if we have visual data results store it - not everything has as caption
                    if visual_data['caption'] != "":
                        logging.debug("Visual data received in Temporal Lobe\n")
                        #logging.debug(visual_data)
                        #logging.debug("\n","="*50,"\n")
                        self.visual_data = visual_data
                                    
                except queue.Empty:
                    pass
            else:
                logging.error("Can't Collect visual data, No visual cortex available in Temporal Lobe")
            
            # Collect AUDIO Data  
            if self.auditory_cortex:
                try:
                    audio_data = self.auditory_cortex.external_cortex_queue.get_nowait()
                    self.auditory_data = audio_data
                    audio_data['collection_timestamp'] = current_time

                    #check for interruption attempt
                    if audio_data.get('is_interrupt_attempt',False):
                        #Set interrupt flag for Prefrontal Cortex
                        self.prefrontal_cortex_interrupt_dict.update({'interrupt':True})
                        logging.debug("Interruption attempt detected from Auditory Cortex - setting interrupt flag for Prefrontal Cortex")

                    #If Auditory Data was final transcript from our locked speaker, we prioritize collecting over User Text Input
                    if audio_data.get('final_transcript',False) and audio_data.get('is_locked_speaker',False):
                        received_audio_transcript = True
                        logging.debug("Audio data received in Temporal Lobe\n")
                        logging.debug(audio_data)
                        logging.debug("\n","+"*50,"\n")
                        #Create unified state
                        unified_state = self._create_empty_unified_state()

                        #Merge auditory data first to prioritize speech
                        for key, value in audio_data.items():
                            if not self._is_default_value(key, value):
                                unified_state[key] = value
                    
                except queue.Empty:
                    pass
            else:
                logging.error("Can't Collect auditory data, No auditory cortex available in Temporal Lobe")

            # collect USER INPUT from UI
            try:
                if not received_audio_transcript:
                    user_text = self.user_input_dict.get('text',"")
                    #If we have user text input and no unified state yet, create one
                    if unified_state is None and user_text.strip() != "":
                        #we only add user text input if we don't already have final transcript data. Audio takes priority
                        unified_state = self._create_empty_unified_state()
                        unified_state['user_text_input'] = user_text.strip()

                        #Clear after setting
                        self.user_input_dict.update({'text':""}) 

            except Exception as e:
                logging.error(f"Error reading user input dict: {e}")
                pass


            #We have unified_state data ready (audio or text), now add visual data if available
            if unified_state is not None:
                
                if self.visual_data is not None:
                    #Merge visual data in both cases (we either had final audio transcript or user text input)
                    for key, value in self.visual_data.items():
                            if not self._is_default_value(key, value):
                                unified_state[key] = value
                        
                #Update timestamp
                unified_state['timestamp'] = current_time
                unified_state['formatted_time'] = datetime.now().strftime('%H:%M:%S')
                #reset
                received_audio_transcript = False
                # Send to real-time queue
                try:
                    self.external_temporallobe_to_prefrontalcortex.put_nowait(unified_state)
                    self.last_unified_state = unified_state
                    self.status_dict.update({
                                 "visual_data":self.visual_data,
                                 "auditory_data":self.auditory_data,
                                 "user_input_text":self.user_input_dict.get('text',""),
                                 "last_unified_state":self.last_unified_state,
                                 })
                except queue.Full:
                    # --- QUEUE FULL HANDLER ---
                    try:
                        # Remove ONLY the oldest item to make room for the new one
                        self.external_temporallobe_to_prefrontalcortex.get_nowait()
                    except queue.Empty:
                        pass # Queue was emptied by consumer just now
                    
                    # Try inserting again
                    #this is the only source of adding unified state to the queue so should be safe
                    try:
                        self.external_temporallobe_to_prefrontalcortex.put_nowait(unified_state)
                        self.last_unified_state = unified_state
                        self.status_dict.update({
                                 "visual_data":self.visual_data,
                                 "auditory_data":self.auditory_data,
                                 "user_input_text":self.user_input_dict.get('text',""),
                                 "last_unified_state":self.last_unified_state,
                                 })
                    except queue.Full:
                        pass # Still full (shouldn't be possible), skip this frame

            # Small delay to prevent excessive CPU usage if we didn't create unified state data this iteration
            if unified_state is None:
                time.sleep(0.005)
        
        logging.info("TemporalLobe frame collection stopped")

    def get_last_unified_state(self):
        """Get the last created unified state without creating a new one"""
        return self.last_unified_state
    
    def get_status(self):
        """Get the current status dictionary."""
        return dict(self.status_dict)
    
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
        
        
        # Clear speaker lock
        self.locked_speaker_id = None
        self.locked_speaker_timestamp = None
        
        if had_failure:
            logging.error("Temporal Lobe shutdown completed with some failures")
        else:
            logging.info("Temporal Lobe shutdown completed successfully")
            
        return not had_failure

    

if __name__ == "__main__":
    import multiprocessing as mp
    import cv2
    from visualcortex import VisualCortex
    from audiocortex import AuditoryCortex

    # Create multiprocessing manager
    manager = mp.Manager()
    # Create the Status Dictionary
    TemporalLobe_status_dict = manager.dict()

    # Initialize AuditoryCortex
    ac = AuditoryCortex(
        mpm=manager,
        wakeword_name='jarvis',
        database_path=":memory:",
        gpu_device=0
    )

    # Initialize VisualCortex
    vc = VisualCortex(mpm=manager)

    # Initialize TemporalLobe
    tl = TemporalLobe(
        visual_cortex=vc,
        auditory_cortex=ac,
        mpm=manager,
        database_path=":memory:",
        status_dict=TemporalLobe_status_dict,
        PrefrontalCortex_interrupt_dict=manager.dict({'interrupt':False})
    )

    try:
        # Start the systems
        logging.info("initialize visual nerve")
        tl.start_visual_nerve(device_index=0)
        logging.info("initialize audio nerve")
        tl.start_audio_nerve(device_index=0)
        logging.info("Starting TemporalLobe data collection...")
        tl.start_collection()

        logging.info("TemporalLobe running. Press Ctrl+C to stop.")
        logging.info("Monitoring data on the external_temporallobe_to_prefrontalcortex queue...\n")

        # Display feed from the queue
        # Initialize the timer with the current time for User Input testing
        last_user_input_time = time.time()
        while True:
            has_activity = False # Flag to track if we did work this iteration

            # --- HANDLE IMAGE FEED ---
            try:
                # Get the latest frame from the queue (non-blocking)
                img_data = tl.visual_cortex.external_img_queue.get_nowait()
                frame = img_data.get('frame')
                if frame is not None:
                    cv2.imshow("Visual Cortex Feed", frame)
                    has_activity = True
            except queue.Empty:
                pass

            # --- HANDLE UNIFIED STATE FEED ---
            try:
                unified_state = tl.external_temporallobe_to_prefrontalcortex.get_nowait()
                has_activity = True # We found data

                # Display relevant fields
                # Check if we have specific data to show to reduce console spam
                #either a final transcript from audio or user text input
                has_content = (
                    (unified_state.get('final_transcript', False) and unified_state.get('is_locked_speaker',False)) or 
                    unified_state.get('user_text_input') != "" 
                )

                if has_content:
                    print("-" * 50)
                    if unified_state.get('transcription'):
                        logging.info(f"Speech Detected: {unified_state['speech_detected']}")
                        logging.info(f"Transcription: {unified_state['transcription']}")

                    if unified_state.get('user_text_input'):
                        logging.info(f"User Input: {unified_state['user_text_input']}")

                    if unified_state.get('caption'):
                        logging.info(f"Caption: {unified_state['caption']}")

                    if unified_state.get('person_detected'):
                        logging.info(f"Person Detected: {unified_state['person_detected']}")
                        logging.info(f"Person Match: {unified_state['person_match']} (prob: {unified_state['person_match_probability']:.2f})")
                    print("-" * 50)

            except queue.Empty:
                # Do NOT sleep here blindly, or you slow down the video feed processing
                pass
            except Exception as e:
                logging.error(f"Error retrieving data: {e}")

            # --- OPENCV GUI & TIMING ---
            # cv2.waitKey(1) handles GUI events, MUST have this to work right
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                raise KeyboardInterrupt

            # Only sleep if NEITHER queue had data, to prevent CPU spinning at 100%
            if not has_activity:
                time.sleep(0.005)

            # --- USER INPUT TESTING ---
            current_time = time.time()
            # Check if 15 seconds have passed since the last speech
            if current_time - last_user_input_time >= 15:
                last_user_input_time = time.time()#reset timer
                #create test user input
                tl.user_input_dict.update({'text':"THIS IS TEST INPUT FROM THE UI"})  

    except KeyboardInterrupt:
        logging.info("\nCtrl+C detected. Shutting down...")
        tl.shutdown_all()
        # Ensure opencv windows close
        cv2.destroyAllWindows() 
        logging.info("Shutdown complete.")