
import multiprocessing as mp
import logging
import time
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

# Attempt to import VisualCortex, handle gracefully if missing
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
        self.running = False
        self.wakeword = wakeword
        breakword=f"enough {self.wakeword}"
        exitword=f"goodbye {self.wakeword}"

        # 2. Initialize Auditory Cortex
        if AuditoryCortex:
            logger.info("Spinning up Auditory Cortex...")
            self.audio_cortex = AuditoryCortex(
                mpm=self.manager,
                wakeword_name=wakeword,
                breakword=breakword,
                exitword=exitword,
                database_path=db_path
            )
        else:
            logger.critical("Auditory Cortex failed to load.")
            sys.exit(1)

        # 3. Initialize Visual Cortex (Optional/Placeholder)
        self.visual_cortex = None
        if VisualCortex:
            logger.info("Spinning up Visual Cortex...")
            self.visual_cortex = VisualCortex(mpm=self.manager)
        else:
            logger.info("Visual Cortex disabled or missing.")

        # 4. Initialize Temporal Lobe (The Bridge)
        if TemporalLobe:
            logger.info("Connecting Temporal Lobe...")
            self.temporal_lobe = TemporalLobe(
                visual_cortex=self.visual_cortex,
                auditory_cortex=self.audio_cortex,
                mpm=self.manager,
                database_path=db_path
            )
            # 5. Initialize Prefrontal Cortex (The Logic/LLM)
            if PrefrontalCortex:
                logger.info("Awakening Prefrontal Cortex...")
                # Create shared status dict for PFC
                self.pfc_status = self.manager.dict()
                
                self.prefrontal_cortex = PrefrontalCortex(
                    model_name=model_name,
                    external_temporallobe_to_prefrontalcortex=self.temporal_lobe.external_temporallobe_to_prefrontalcortex,
                    audio_cortex=self.audio_cortex,
                    wakeword_name=wakeword,
                    status_dict=self.pfc_status
                )
            else:
                logger.critical("Prefrontal Cortex failed to load.")
                sys.exit(1)

        else:
            logger.critical("Temporal Lobe failed to load.")
            sys.exit(1)

       

        logger.info("Cerebrum Initialization Complete.")

    def start_systems(self, start_mic=True, start_cam=True):
        """
        Begins the processing loops for input and data aggregation.
        """
        logger.info("Starting System Processes...")
        self.running = True

        # Start Input Nerves
        if start_mic:
            self.audio_cortex.start_nerve(device_index=0)
        
        if self.visual_cortex and start_cam:
            self.visual_cortex.start_nerve(device_index=0)

        # Start Temporal Lobe Aggregation Loop
        self.temporal_lobe.start_collection()

        # Play startup sound/greeting
        self.audio_cortex.brocas_area.synthesize_speech(f"Systems online. I am now is listening for my name.", auto_play=True)

    def stop_systems(self):
        """
        Graceful shutdown of all subsystems.
        """
        logger.info("Initiating System Shutdown...")
        self.running = False

        # Shutdown PFC (Logic)
        if self.prefrontal_cortex:
            self.prefrontal_cortex.shutdown()
        
        # Shutdown Temporal Lobe (Bridge)
        if self.temporal_lobe:
            self.temporal_lobe.shutdown_all()
        
        # Shutdown Audio/Visual (Sensors) handled by temporal lobe shutdown usually, 
        # but calling explicit shutdowns to be safe
        if self.audio_cortex:
            self.audio_cortex.shutdown()
        
        if self.visual_cortex:
            self.visual_cortex.shutdown()

        logger.info("System Shutdown Complete.")

    # ============================================
    # UI CONTROLLER METHODS (For Flask App)
    # ============================================

    def ui_get_unified_state(self):
        """
        Returns a dictionary containing the real-time state of the entire system.
        Useful for polling via AJAX in the Flask UI.
        """
        # Get immediate sensory state from Temporal Lobe
        tl_state = self.temporal_lobe.get_last_unified_state()
        
        # Get cognitive state from Prefrontal Cortex (Thinking status, current response)
        pfc_state = dict(self.pfc_status)
        
        # Combine them
        combined_state = {
            "sensory": tl_state,
            "cognitive": pfc_state,
            "system": {
                "running": self.running,
                "wakeword": self.wakeword
            }
        }
        return combined_state

    def ui_send_text_input(self, text_input):
        """
        Injects text from the UI directly into the Temporal Lobe stream,
        bypassing the microphone.
        """
        logger.info(f"UI Input Received: {text_input}")
        self.temporal_lobe.user_input_dict['text'] = text_input

    def ui_toggle_microphone(self, active: bool, device_index=0):
        """Turn microphone processing on/off"""
        if active:
            self.temporal_lobe.start_audio_nerve(device_index)
        else:
            self.temporal_lobe.stop_audio_nerve(device_index)

    def ui_toggle_camera(self, active: bool, device_index=0):
        """Turn camera processing on/off"""
        if not self.visual_cortex:
            return False
            
        if active:
            self.temporal_lobe.start_visual_nerve(device_index)
        else:
            self.temporal_lobe.stop_visual_nerve(device_index)


if __name__ == "__main__":
    # Example standalone usage (if not running via Flask)
    brain = Cerebrum(wakeword="Computer")
    
    try:
        brain.start_systems()
        
        print("\n *** BRAIN RUNNING. Press Ctrl+C to stop. *** \n")
        
        while True:
            # Simulate a "keep-alive" loop or debug print
            time.sleep(5)
            # state = brain.ui_get_unified_state()
            # print(f"Thinking: {state['cognitive'].get('thinking')} | Last Input: {state['cognitive'].get('last_input')}")

    except KeyboardInterrupt:
        brain.stop_systems()