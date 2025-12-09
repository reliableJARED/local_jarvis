
import multiprocessing as mp
import logging
import sys


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
        logger.info("Initializing Cerebrum...")
        
        # 1. Initialize Multiprocessing Manager (The central nervous system)
        self.manager = mp.Manager()
        # 3. Create the Status Dictionary
        self.PrefrontalCortex_status_dict = self.manager.dict()
        PrefrontalCortex_interrupt_dict = self.manager.dict()
        PrefrontalCortex_interrupt_dict.update({'should_interrupt': False})

        self.TemporalLobe_status_dict = self.manager.dict()

        self.active = False
        self.wakeword = wakeword
        breakword=f"enough {self.wakeword}"
        exitword=f"goodbye {self.wakeword}"


        # 2. Initialize Auditory Cortex
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

        # 3. Initialize Visual Cortex
        if VisualCortex:
            logger.info("Spinning up Visual Cortex...")
            visual_cortex = VisualCortex(mpm=self.manager)
        else:
            logger.info("Visual Cortex disabled or missing.")

        # 4. Initialize Temporal Lobe (The Bridge to Visual and Auditory)
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
                PrefrontalCortex_interrupt_dict=PrefrontalCortex_interrupt_dict
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

    def ui_get_unified_state(self):
        """
        Returns a dictionary containing the real-time state of the entire system.
        Useful for polling via AJAX in the Flask UI.
        """
        # Get current sensory state from Temporal Lobe
        tl_state = self.temporal_lobe.get_status()
        """
        tl_state = {
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
            'user_text_input': ""}
            """
        
        # Get cognitive state from Prefrontal Cortex (Thinking status, current response)
        pfc_state = self.prefrontal_cortex.get_status()
        """
        pfc_state = {
            'thinking': False,
            'model': self.model_name,
            'system_prompt_base': self.system_prompt_base,
            'system_prompt_tools': self.prompt_for_tool_use,
            'system_prompt_visual':"",#contains scene description of visual
            'system_prompt_audio':"",#contains scene description of audio, sounds detected not text
            'messages': self.messages,
            'last_input':"",
            'last_response':""}
        """
        
        # Combine them
        combined_state = {
            "sensory": tl_state['last_unified_state'],#don't need all the visual/auditory data here only the unified state,
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


if __name__ == "__main__":
    import time
    brain = Cerebrum(
        wakeword='jarvis',
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        db_path=":memory:"
    )
    brain.start_systems(start_mic=True, start_cam=False)

    while True:
        try:
            time.sleep(10)
            print(brain.ui_get_unified_state())
        except KeyboardInterrupt:
            brain.stop_systems()
            break