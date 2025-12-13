
import multiprocessing as mp
import logging
import sys
import queue
import time

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CEREBRUM")

# Import Brain Components
try:
    from audiocortex import AuditoryCortex
except ImportError:
    logger.error("audiocortex.py not found.")
    AuditoryCortex = None

try:
    from temporallobe import TemporalLobe
except ImportError:
    logger.error("temporallobe.py not found.")
    TemporalLobe = None

try:
    from prefrontalcortex import PrefrontalCortex
except ImportError:
    logger.error("prefrontalcortex.py not found.")
    PrefrontalCortex = None

try:
    from visualcortex import VisualCortex
except ImportError:
    logger.warning("visualcortex.py not found. System will run without Visual Cortex.")
    VisualCortex = None


class Cerebrum:
    """
    The Cerebrum is the central controller (High-Level API). 
    It orchestrates the initialization, connection, and shutdown of the 
    Auditory Cortex, Visual Cortex, Temporal Lobe, and Prefrontal Cortex.
    
    It serves as the backend controller for the Flask UI.
    """
    def __init__(self, wakeword='jarvis', model_name="Qwen/Qwen2.5-Coder-7B-Instruct", db_path=":memory:"):
        print("----------------------------------------------------------------")
        print(" INITIALIZING CEREBRUM... ")
        print("----------------------------------------------------------------")

        logger.info("Initializing Cerebrum...")
        
        # Initialize Multiprocessing Manager (The central nervous system)
        self.manager = mp.Manager()
        # Create the Status Dictionary
        self.PrefrontalCortex_status_dict = self.manager.dict()
        PrefrontalCortex_interrupt_dict = self.manager.dict()
        PrefrontalCortex_interrupt_dict.update({'should_interrupt': False})
        # create transient data queue
        self.temporal_lobe_external_sensory_queue = self.manager.Queue(maxsize=30)

        self.TemporalLobe_status_dict = self.manager.dict()

        self.active = False
        self.wakeword = wakeword
        breakword=f"enough {self.wakeword}"
        exitword=f"goodbye {self.wakeword}"
        


        # Initialize Auditory Cortex
        if AuditoryCortex:
            logger.info("Spinning up Auditory Cortex...")
            audio_cortex = AuditoryCortex(
                mpm=self.manager,
                wakeword_name=wakeword,
                breakword=breakword,
                exitword=exitword,
                database_path=db_path
            )
        else:
            logger.critical("Auditory Cortex failed to load.")
            sys.exit(1)

        #  Initialize Visual Cortex
        if VisualCortex:
            logger.info("Spinning up Visual Cortex...")
            visual_cortex = VisualCortex(mpm=self.manager)
        else:
            logger.info("Visual Cortex disabled or missing.")

        #  Initialize Temporal Lobe (The Bridge to Visual and Auditory)
        # Connects both Audio and Visual Cortex via
        # self.temporal_lobe.auditory_cortex and
        # self.temporal_lobe.visual_cortex
        if TemporalLobe:
            logger.info("Connecting Temporal Lobe...")
            self.temporal_lobe = TemporalLobe(
                visual_cortex=visual_cortex,
                auditory_cortex=audio_cortex,
                mpm=self.manager,
                database_path=db_path,
                status_dict=self.TemporalLobe_status_dict,
                PrefrontalCortex_interrupt_dict=PrefrontalCortex_interrupt_dict,
                external_sensory_queue=self.temporal_lobe_external_sensory_queue
            )
        else:
            logger.critical("Temporal Lobe failed to load.")
            sys.exit(1)

        # 5. Initialize Prefrontal Cortex (The Logic/LLM Output)
        if PrefrontalCortex:
            logger.info("Awakening Prefrontal Cortex...")
                
            self.prefrontal_cortex = PrefrontalCortex(
                    model_name=model_name,
                    external_temporallobe_to_prefrontalcortex=self.temporal_lobe.external_temporallobe_to_prefrontalcortex,
                    interrupt_dict=PrefrontalCortex_interrupt_dict,
                    audio_cortex=audio_cortex,
                    status_dict=self.PrefrontalCortex_status_dict
                )
        else:
            logger.critical("Prefrontal Cortex failed to load.")
            sys.exit(1)

        
        logger.info("Cerebrum Initialization Complete.")

    def start_systems(self, start_mic=True, start_cam=True):
        """
        Begins the processing loops for input and data aggregation.
        """
        logger.info("Starting System Processes...")
        self.active = True

        # Start Input Nerves
        if start_mic:
            self.temporal_lobe.auditory_cortex.start_nerve(device_index=0)
        
        if start_cam:
            self.temporal_lobe.visual_cortex.start_nerve(device_index=0)

        # Start Temporal Lobe Aggregation Loop
        self.temporal_lobe.start_collection()

        # Play startup sound/greeting
        self.temporal_lobe.speak(f"Hello, my name is {self.wakeword}. All my systems are now online.")

    def stop_systems(self):
        """
        Graceful shutdown of all subsystems.
        """
        logger.info("Initiating System Shutdown...")
        self.active = False

        # Shutdown PFC (Logic)
        if self.prefrontal_cortex:
            self.prefrontal_cortex.shutdown()
        
        # Shutdown Temporal Lobe (Bridge)
        if self.temporal_lobe:
            self.temporal_lobe.shutdown_all()
        

        logger.info("System Shutting down.")

    # ============================================
    # UI CONTROLLER METHODS (For Flask App)
    # ============================================
    def ui_get_transient_sensory_data(self):
        """
        Returns a dictionary of the most recent sensory data from the Temporal Lobe.
        Maintains state to prevent UI flickering for captions and handles streaming text.
        """
        #  Initialize persistent state variables
        if not hasattr(self, '_ui_last_caption'):
            self._ui_last_caption = ""
        if not hasattr(self, '_ui_current_transcription'):
            self._ui_current_transcription = ""
        if not hasattr(self, '_ui_speech_detected'):
            self._ui_speech_detected = False
        if not hasattr(self, '_ui_person_detected'):
            self._ui_person_detected = False
        if not hasattr(self, 'last_live_transcription_time'):
            self.last_live_transcription_time = time.time()
        if not hasattr(self,'transcription_delay_time'):
            self.transcription_delay_time = 2.0 #seconds to wait before clearing transcription after speech ends

        #Used to Drain the queue
        data_item = False
        
        while not self.temporal_lobe_external_sensory_queue.empty():

            try:
                # We explicitly isolate the fetch operation.
                # If THIS fails, the queue is likely empty/closed, so we BREAK.
                data_item = self.temporal_lobe_external_sensory_queue.get_nowait()
            except Exception:
                break 

            #  Process the data (Flag Checks)
            # We do this OUTSIDE the try/except for the queue. 
            # If there is a logic error here, we technically just skip this item 
            # naturally and loop again to the next item.
            if data_item:
                if data_item.get('visual_data',False):
                    self._ui_last_caption = data_item.get('caption', "")
                    self._ui_person_detected = data_item.get('person_detected', False)
                
                if data_item.get('audio_data', False):
                    #blank transcripts with poitive speech are received between transcription chunks
                    if self.TemporalLobe_status_dict.get("speech_detected",False) and data_item.get('transcription', "") != "":
                        self._ui_current_transcription = data_item.get('transcription', "")
                        self.last_live_transcription_time = time.time()

                self._ui_speech_detected = data_item.get('speech_detected', False)

        # Implement 'sticky' transcription logic
        if not self._ui_speech_detected:
            #reset transcription after short delay
            if time.time() - self.last_live_transcription_time > self.transcription_delay_time:
                self._ui_current_transcription = ""
                

        return {
            'timestamp': time.time(),
            'transcription': self._ui_current_transcription,
            'caption': self._ui_last_caption,
            'speech_detected': self._ui_speech_detected,
            'person_detected': self._ui_person_detected,
        }

    def ui_get_unified_state(self):
        """
        Returns a dictionary containing the real-time state of the entire system.
        Useful for polling via AJAX in the Flask UI.
        """
        #TODO: CRASHHES IF NON-JSON Serializable DATA in Combined State sent to server for jsonify
        # Get current sensory state from Temporal Lobe
        tl_state = self.temporal_lobe.get_status()
        """
        tl_state['last_unified_state'] = {
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
            'unlock_speaker':False,

            #Speech Output
            'actively_speaking':False,
            
            # UI Inputs
            'user_text_input': ""}
            """
        
        # Get cognitive state from Prefrontal Cortex (Thinking status, current response)
        pfc_state = self.prefrontal_cortex.get_status()
        """
        pfc_state = {
            'thinking': False,
            'model': self.model_name,
            'temperature':0.7,
            'max_new_tokens':4500,
            'current_token_count':0,
            'system_prompt_base': self.system_prompt_base,
            'system_prompt_tools': self.prompt_for_tool_use,
            'system_prompt_visual':self.system_prompt_visual,#contains scene description of visual
            'system_prompt_audio':"",#contains scene description of audio, sounds detected not text
            'messages': self.messages,
            'last_input':"",
            'last_response':""
        }
        """
        
        # Combine them
        combined_state = {
            "sensory": tl_state['last_unified_state'],#extract the templobe unified state,
            "cognitive": pfc_state,
            "system": {
                "state": self.active,
                "name": self.wakeword,
                "locked_speaker_id": tl_state['locked_speaker_id'],
                "speech_detected": tl_state['speech_detected'],
                "actively_speaking": tl_state['actively_speaking'],#this is the system outputting speech audio
                "person_detected": tl_state['person_detected']
            }
        }
        return combined_state

    def ui_send_text_input(self, text_input):
        """
        Injects text from the UI directly into the Temporal Lobe stream,
        bypassing the microphone.
        """
        logger.info(f"UI Input Received: {text_input}")
        self.temporal_lobe.user_input_dict.update({'text':text_input})  

    def ui_toggle_microphone(self, active: bool, device_index=0):
        """Turn microphone processing on/off"""
        if active:
            self.temporal_lobe.start_audio_nerve(device_index)
        else:
            self.temporal_lobe.stop_audio_nerve(device_index)

    def ui_toggle_camera(self, active: bool, device_index=0):
        """Turn camera processing on/off"""

        if active:
            self.temporal_lobe.start_visual_nerve(device_index)
        else:
            self.temporal_lobe.stop_visual_nerve(device_index)
     
    def ui_get_prefrontal_cortex_config(self):
        full_config = self.prefrontal_cortex.status_dict.copy()
        print(full_config)
        #must remove messages because not json serializable
        del full_config['messages']
        del full_config['last_input']
        del full_config['last_response']
        return full_config
     
    def update_prefrontal_cortex_config(self,data):
        print("TODO: Make this update endpoint work")
        print(data)
        
        return True

if __name__ == "__main__":
    
    brain = Cerebrum(
        wakeword='jarvis',
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        db_path=":memory:"
    )
    brain.start_systems(start_mic=True, start_cam=True)

    last_print = time.time()
    while True:
        current_time = time.time()
        try:
            if current_time - last_print > 10:
                last_print = current_time
                print("-"*40)
                print(brain.ui_get_unified_state())
                print("-"*40)
            trans = brain.ui_get_transient_sensory_data()
            #if trans['transcription'] != "":
            #print("#"*40)
            #print(trans)
            #print("#"*40)

            time.sleep(0.05)#small delay to reduce print spam
        except KeyboardInterrupt:
            brain.stop_systems()
            break