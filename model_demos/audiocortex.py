
import multiprocessing as mp
import queue
import time
import numpy as np
from datetime import datetime
import sounddevice as sd
from collections import deque
import torch
import gc
from voicerecognition import VoiceRecognitionSystem
from dataclasses import dataclass, field
import logging
from typing import Optional
logging.basicConfig(level=logging.INFO) #ignore everything use (level=logging.CRITICAL + 1)


#Dict Struct on the audio_cortex.nerve_from_input_to_cortex
#This is raw input audio from microphone
AudioCortexNervelDataTemplate = {
    'device_index': None,  # int: Audio input device identifier, default 0
    'audio_frame': None,   # np.ndarray: float32 audio samples, shape (n_samples,)
    'capture_timestamp': None,  # float: Unix timestamp when frame was captured
    'sample_rate': None    # int: Sample rate in Hz (default, 16000)
}




#Tuple sent on the audio_cortex.nerve_from_cortex_to_stt
# This takes VAD processed audio and sends it for transcription 
"""nerve_from_cortex_to_stt.put_nowait((
                                full_speech.copy(), Data
                                speech_active,  bool
                                device_index, int
                                sample_rate, int
                                capture_timestamp, float
                                is_playback_context bool
                            ))
"""

#Dict Struct on the audio_cortex.nerve_from_stt_to_cortex
#This is post VAD, Recognition, Transcription data processed one last time in cortex for interruption detection prior to sending externally
"""cortex_output = {
                        'transcription': "",
                        'final_transcript': False,
                        'voice_match': None,
                        'voice_probability': 0.0,
                        'device_index': device_index,
                        'speech_detected': speech_active,
                        'capture_timestamp': capture_timestamp,
                        'transcription_timestamp': capture_timestamp,
                        'formatted_time': datetime.fromtimestamp(capture_timestamp).strftime('%H:%M:%S'),
                        'sample_rate': sample_rate,
                        'duration': duration,
                        'hear_self_speaking': False,
                    }

"""

#dataclass structure o
#This will have the state bool if speaking, transcript of speech and other info about active speaking from the BrocasArea
"""
BROCAS_AUDIO_TEMPLATE = {
    'transcript': "",  # what is being spoken
    'audio_data': None,  # np.ndarray - raw data of the audio
    'samplerate': 24000,  # hz of sample - smaller, slower/lower - bigger faster/higher
    'num_channels': 1  # Mono audio from Kokoro
}
"""

"""
AUDIO
"""
def auditory_nerve_connection(device_index, nerve_from_input_to_cortex, external_stats_queue,sample_rate=16000, chunk_size=512):
    """Worker process for capturing audio frames from a specific device (auditory nerve function).
    Optimized chunk size: 512 samples = ~32ms at 16kHz for optimal Silero VAD performance.
    """
    
    print(f"Started auditory nerve process for device {device_index}")
    frame_count = 0
    start_time = time.time()
    
    def audio_callback(indata, frames, time_info, status):
        nonlocal frame_count
        if status:
            print(f"Audio callback status: {status}")
        
        # Create audio frame data - keep as float32 for Silero VAD
        timestamp = time.time()
        audio_data = {
            'device_index': device_index,
            'audio_frame': indata.copy().flatten().astype(np.float32),  # Already float32, optimal for VAD
            'capture_timestamp': timestamp,
            'sample_rate': sample_rate
        }
        
        # Non-blocking queue put - drop frame if queue is full
        try:
            nerve_from_input_to_cortex.put_nowait(audio_data)
        except:
            logging.warning(f"Audio nerve queue {device_index} data not being consumed fast enough!")
            try:
                # Queue is full, delete last frame to make space for this one to maintain real-time performance
                _ = nerve_from_input_to_cortex.get_nowait()
                nerve_from_input_to_cortex.put_nowait(audio_data)
            except queue.Empty:
                nerve_from_input_to_cortex.put_nowait(audio_data)
            pass
        
        
        # Calculate and update auditory nerve stats
        frame_count += 1
        if frame_count % 100 == 0:  # Update every 100 frames
            current_time = time.time()
            fps = frame_count / (current_time - start_time)
            stats_data = {f'auditory_nerve_fps_{device_index}':fps,
                          f'last_auditory_nerve_{device_index}':current_time}
            try:
                external_stats_queue.put_nowait(stats_data)# Will drop and lose this data if no put
            except:
                logging.warning(f"Audio nerve {device_index} unable to send stats data to external_stats_queue because it's full")
    
    try:
        # Start audio stream with optimized settings
        with sd.InputStream(
            device=device_index,
            channels=1,
            samplerate=sample_rate,
            blocksize=chunk_size,  # 512 samples = ~32ms at 16kHz
            dtype=np.float32,      # Native format for Silero VAD
            callback=audio_callback,
            latency='low'          # Request low latency from sounddevice
        ):
            print(f"Audio nerve stream started for device {device_index} with {chunk_size} sample chunks")
            
            # Keep the stream alive
            while True:
                time.sleep(0.001)
                
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Audio stream error for device {device_index}: {e}")
    finally:
        print(f"Auditory nerve process for device {device_index} stopped")

"""
#VOICE RECOGNITION
                        # send data to voice recognition in a separate worker via queue
                        try:
                            vr_data = {'audio':chunk_to_process,'my_voice':my_voice}
                            nerve_from_cortex_to_vr.put_nowait(vr_data)
                        except:
                            logging.error("Error sending data on nerve_from_cortex_to_vr")


                        # Set recognition results from the container
                        voice_match_result = False
                        voice_probability = 0.0
                        try:
                            data = nerve_from_vr_to_cortex.get(timeout=1)#{'human_id':human_id,'new_person':new_person, 'similarity':similarity}
                            voice_match_result = data['human_id']
                            voice_probability =  data['similarity']
                        except:
                            logging.error("Error getting data on nerve_from_vr_to_cortex")
                        
                        
                        logging.debug(f"Voice recognition result: {voice_match_result}, probability: {voice_probability:.3f}")
"""

def auditory_cortex_core(nerve_from_input_to_cortex, external_cortex_queue, external_stats_queue,
                        nerve_from_cortex_to_stt, nerve_from_stt_to_cortex,
                        detected_name_wakeword_queue, wakeword_name, 
                        external_brocas_state_queue,nerve_from_cortex_to_vr,nerve_from_vr_to_cortex):
    """
    Enhanced auditory cortex with dynamic background noise monitoring.
    
    
    """
    print("Started auditory cortex process")
    frame_count = 0
    start_time = time.time()

    # Pre-roll buffer for capturing speech beginnings
    pre_speech_buffer = deque(maxlen=10)
    speech_active = False
    speech_buffer = []
    full_speech = []
    min_silence_duration_ms = 1000
    
    # Voice lock management
    non_match_duration = 0.0
    RELEASE_THRESHOLD = 2.5
    SILENCE_RELEASE_THRESHOLD = 4.0
    locked_speaker_id = None
    
    # === ENHANCED NOISE MONITORING ===
    # Track different noise profiles
    ambient_noise_samples = deque(maxlen=100)  # ~3.2 seconds of ambient noise
    playback_noise_samples = deque(maxlen=100)  # Noise during playback/assistant speech
    
    # Current noise estimates
    ambient_noise_floor = 0.0
    playback_noise_floor = 0.0
    
    # Adaptive thresholds
    ambient_noise_percentile = 75  # Use 75th percentile to avoid outliers
    playback_noise_percentile = 90  # Higher percentile during playback (more conservative)
    
    # Energy tracking
    energy_history = deque(maxlen=50)
    
    # Noise update counters
    NOISE_UPDATE_INTERVAL = 10  # Update noise floor every N samples
    
    #Indicator if system is playing auido
    is_playback_context = False#playback_state.get('is_playing', False)

    #Need voice to calibrate first
    assistant_voice_id = None
    
    # Initialize Silero VAD
    try:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                    model='silero_vad',
                                    force_reload=False,
                                    onnx=False)
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
        
        vad_iterator = VADIterator(
            model,
            threshold=0.5,
            sampling_rate=16000,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=30
        )
        
        print("Silero VAD with VADIterator loaded successfully")
        
    except Exception as e:
        print(f"Error loading Silero VAD: {e}")
        return
    
    try:
        SAMPLE_SIZE_REQUIREMENT = 512#DO NOT CHANGE VALUE required for VAD
        while True:
            try:
                audio_data = nerve_from_input_to_cortex.get_nowait()
                
                audio_chunk = audio_data['audio_frame']
                capture_timestamp = audio_data['capture_timestamp']
                device_index = audio_data['device_index']
                sample_rate = audio_data['sample_rate']
                duration = SAMPLE_SIZE_REQUIREMENT / sample_rate
                
                if len(audio_chunk) != SAMPLE_SIZE_REQUIREMENT:
                    logging.error(f'received audio chunk from nerve {device_index} not equal to 512 samples')
                    continue
                
                # Calculate energy of this chunk
                chunk_energy = np.sqrt(np.mean(audio_chunk ** 2))  # RMS energy
                energy_history.append(chunk_energy)
                
                # Convert to tensor for Silero VAD
                audio_tensor = torch.from_numpy(audio_chunk)
                
                # Add audio to pre-speech buffer
                pre_speech_buffer.append(audio_chunk)
                
                # Run VADIterator
                try:
                    #check for speech
                    speech_dict = vad_iterator(audio_tensor, return_seconds=True)
                   
                    # Process VAD results
                    if speech_dict:
                        if 'start' in speech_dict:
                            logging.info(f"Speech START detected at {speech_dict['start']:.3f}s")
                            speech_active = True
                            speech_buffer = list(pre_speech_buffer)
                                                
                        if 'end' in speech_dict:
                            logging.info(f"Speech END detected at {speech_dict['end']:.3f}s")
                            speech_active = False
                           
                    # === NOISE FLOOR & PLAYBACK UPDATES ===
                    # Determine current context (is assistant playing audio) and update appropriate noise floor, shared dict with Speaker Output
                    #TODO speech
                    try:
                        brocas_state = external_brocas_state_queue.get_nowait()
                        is_playback_context = brocas_state.get('is_playing',False)
                    except Exception as e:
                        #print(e)
                        pass
                    
                    
                    if not speech_active:
                        #could be non speech audio, like music, or user has headphones and mic can't hear speech
                        if is_playback_context:
                            
                            playback_noise_samples.append(chunk_energy)
                            
                            # Update every NOISE_UPDATE_INTERVAL samples if we have enough history
                            if len(playback_noise_samples) >= 20 and len(playback_noise_samples) % NOISE_UPDATE_INTERVAL == 0:
                                playback_noise_floor = np.percentile(
                                    list(playback_noise_samples), 
                                    playback_noise_percentile
                                )
                                
                                logging.debug(f"Updated playback noise floor: {playback_noise_floor:.6f}")

                        #general abient noise
                        else:
                            
                            ambient_noise_samples.append(chunk_energy)
                            
                            # Update every NOISE_UPDATE_INTERVAL samples if we have enough history
                            if len(ambient_noise_samples) >= 20 and len(ambient_noise_samples) % NOISE_UPDATE_INTERVAL == 0:
                                ambient_noise_floor = np.percentile(
                                    list(ambient_noise_samples), 
                                    ambient_noise_percentile
                                )
                                logging.debug(f"Updated ambient noise floor: {ambient_noise_floor:.6f}")
                                
                    if speech_active or (not speech_active and len(full_speech) > 0):
                        speech_buffer.append(audio_chunk)
                        full_speech = np.concatenate(speech_buffer)
                        
                        # send data to voice recognition
                        try:
                            nerve_from_cortex_to_vr.put_nowait({'audio':full_speech.copy(),'my_voice':is_playback_context})
                        except:
                            logging.error("Error sending data on nerve_from_cortex_to_vr")

                        try:
                            #send data out for transcription
                            nerve_from_cortex_to_stt.put_nowait((
                                full_speech.copy(), 
                                speech_active, 
                                device_index, 
                                sample_rate, 
                                capture_timestamp
                            ))
                            
                        except queue.Full:
                            logging.debug("nerve_from_cortex_to_stt FULL - clearing and retrying")
                            try:
                                while True:
                                    try:
                                        trash = nerve_from_cortex_to_stt.get_nowait()
                                    except queue.Empty:
                                        break
                                nerve_from_cortex_to_stt.put_nowait((
                                    full_speech.copy(), 
                                    speech_active, 
                                    device_index, 
                                    sample_rate, 
                                    capture_timestamp,
                                    is_playback_context
                                ))
                            except queue.Full:
                                logging.debug("nerve_from_cortex_to_stt still FULL")

                        if not speech_active:
                            speech_buffer = []
                            full_speech = []
                    
                    # Prepare cortex output
                    cortex_output = {
                        'transcription': "",
                        'final_transcript': False,
                        'voice_match': None,
                        'voice_probability': 0.0,
                        'device_index': device_index,
                        'speech_detected': speech_active,
                        'capture_timestamp': capture_timestamp,
                        'transcription_timestamp': capture_timestamp,
                        'formatted_time': datetime.fromtimestamp(capture_timestamp).strftime('%H:%M:%S'),
                        'sample_rate': sample_rate,
                        'duration': duration,
                        'hear_self_speaking': False,
                    }
                    
                    try:
                        # Get transcription and voice recognition if available
                        stt_output = nerve_from_stt_to_cortex.get_nowait()
                        cortex_output.update(stt_output)

                        vr_output = nerve_from_vr_to_cortex.get_nowait()
                        cortex_output.update(vr_output)

                        #TODO:  - - Need to Merge via timestamp VR and STT for now don't just merge as is - - 

                        voice_match_id = cortex_output.get('voice_match')
                        transcription = cortex_output.get('transcription', '').lower().strip()

                        #TODO speech
                        if is_playback_context:
                            assistant_voice_id = voice_match_id

                        is_interrupt_attempt = False

                        # Determine if this is speech from our locked speaker
                        is_locked_speaker = (voice_match_id == locked_speaker_id)

                        # Check if this is the assistant speaking
                        if voice_match_id == assistant_voice_id:
                            cortex_output['hear_self_speaking'] = True
                            logging.debug(f"Detected my own voice: '{transcription}' - skipping")
                        
                        # Check if system is playing audio but we are not detecting assistant voice as source of speech
                        elif is_playback_context and voice_match_id != assistant_voice_id:
                            # Use playback-specific noise floor for comparison
                            applicable_noise_floor = playback_noise_floor if playback_noise_floor > 0 else ambient_noise_floor
                            
                            # Calculate SNR (Signal-to-Noise Ratio)
                            snr = chunk_energy / (applicable_noise_floor + 1e-10)
                            
                            # During playback, also check against expected playback energy
                            expected_energy = 0#playback_state.get('expected_energy', 0.0)

                            energy_ratio = chunk_energy / (expected_energy + 1e-10)
                            
                            logging.debug(f"During playback - SNR: {snr:.2f}, Energy ratio: {energy_ratio:.2f}, Voice ID: {voice_match_id}")
                            
                            
                            # Adaptive interrupt criteria based on noise floor
                            # Higher SNR required = speech must be significantly above background
                            min_snr = 3.0  # At least 3x above noise floor
                            min_energy_ratio = 2.0  # At least 2x above playback energy
                            
                            if snr > min_snr and energy_ratio > min_energy_ratio:
                                if is_locked_speaker :
                                    # Locked user is interrupting - high priority
                                    is_interrupt_attempt = True
                                    logging.info(f"INTERRUPT detected from locked user: '{transcription}' "
                                               f"(SNR: {snr:.2f}, energy ratio: {energy_ratio:.2f})")
                                elif snr > 4.0 and energy_ratio > 3.0:
                                    # Very strong signal from someone - allow interrupt even if not locked
                                    logging.info(f"STRONG INTERRUPT detected: '{transcription}' "
                                               f"(SNR: {snr:.2f}, energy ratio: {energy_ratio:.2f})")
                                
                            else:
                                # Below threshold - likely echo or background
                                logging.debug(f"noise or speech Below SNR/energy threshold during playback - ignoring: '{transcription}' "
                                            f"(SNR: {snr:.2f}, min: {min_snr})")
                        
                        # Handle voice lock management (when not playing)
                        else:
                            # Check for wake word and if we should lock on to new voice
                            if wakeword_name.lower() in transcription.lower():
                                if transcription.lower().startswith(wakeword_name.lower()):
                                    print(f"\n\nWake word detected: {transcription}\n\n")
                                    # Lock to this speaker
                                    if voice_match_id:
                                        locked_speaker_id = voice_match_id
                                        logging.info(f"Voice lock acquired by speaker: {voice_match_id}")
                                    detected_name_wakeword_queue.put({'spoken_by': voice_match_id, 'is_interrupt_attempt':is_interrupt_attempt})
                                else:
                                    print(f"\n\nWake word in middle of speech: {transcription}\n\n")
                                    print(f"\nSPOKEN BY: {voice_match_id} -Should Interrupt? {is_interrupt_attempt}\n")
                                    detected_name_wakeword_queue.put({'spoken_by': voice_match_id, 'is_interrupt_attempt':is_interrupt_attempt})
                            
                            if locked_speaker_id == voice_match_id and transcription != '':
                                #we want to ignore this speech since it's not the voice we are locked on to
                                print("\nSending Transcript - Use some type of time out for speech not from Locked User to indicate final.\n")
                                pass
                            

                    except queue.Empty:
                        logging.debug("No data in nerve_from_stt_to_cortex")
                        
                    # Put processed audio in output queue
                    try:
                        external_cortex_queue.put_nowait(cortex_output)
                    except:
                        logging.debug("external_cortex_queue FULL - CRITICAL ERROR")

                except Exception as vad_error:
                    print(f"VAD processing error: {vad_error}")
                    continue
                
                # Calculate stats
                frame_count += 1
                if frame_count % 50 == 0:
                    now = time.time()
                    fps = frame_count / (now - start_time)
                    
                    # Calculate average SNR
                    current_snr = 0.0
                    if ambient_noise_floor > 0:
                        current_snr = chunk_energy / ambient_noise_floor
                    
                    stats_dict = {
                        'auditory_cortex_fps': fps,
                        'last_auditory_cortex': now,
                        'ambient_noise_floor': ambient_noise_floor.item(),# Convert the NumPy float to a Python float during dictionary creation
                        'playback_noise_floor': playback_noise_floor,#TODO: When this is implemented will need the .item() convert also
                        'current_energy': chunk_energy.item(),# Convert the NumPy float to a Python float during dictionary creation
                        'current_snr': current_snr.item(),# Convert the NumPy float to a Python float during dictionary creation
                        'locked_speaker': locked_speaker_id,
                        'ambient_samples': len(ambient_noise_samples),
                        'playback_samples': len(playback_noise_samples)
                    }
                    try:
                        external_stats_queue.put_nowait(stats_dict)
                    except:
                        pass
                    
            except queue.Empty:
                continue

            time.sleep(0.001)
                
    except KeyboardInterrupt:
        pass
    finally:
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        print("Auditory cortex process stopped")

def auditory_cortex_worker_speechToText(nerve_from_cortex_to_stt,nerve_from_stt_to_cortex):
    """
    This is a multiprocess worker that will process audio chunks known to have speech.
    Uses LIFO - only processes the most recent audio data, skip and remove older data tasks from queue.
    Uses sliding window approach to build up transcript incrementally.
    Includes voice recognition functionality that runs in a separate thread.

    IMPORTANT - Currently does NOT actually support multi device because of how the transcript splicing data works, there is no device transcript tracking

    """
    from stt import SpeechTranscriber
    import logging
    import queue
    import gc

   
    def find_overlap_position(old_transcript, new_raw_transcript):
        """
        Find where the new raw transcript overlaps with the existing working transcript.
        Returns the position in the working transcript where we should splice.
        """
        if not old_transcript or not new_raw_transcript:
            return len(old_transcript)
        
        # Limit search to last ~100 characters for efficiency
        max_lookback_chars = 100
        search_start_pos = max(0, len(old_transcript) - max_lookback_chars)
        search_portion = old_transcript[search_start_pos:]
        
        old_words = search_portion.lower().split()
        new_words = new_raw_transcript.lower().split()
        
        best_overlap_pos = len(old_transcript)  # Default: append to end
        max_overlap_len = 0
        
        original_old_words = search_portion.split()
        
        # Search backwards from the end
        for i in range(len(old_words) - 1, -1, -1):
            match_length = 0
            for j in range(min(len(old_words) - i, len(new_words))):
                if old_words[i + j] == new_words[j]:
                    match_length += 1
                else:
                    break
            
            if match_length >= 2 and match_length > max_overlap_len:
                max_overlap_len = match_length
                chars_before_match = len(' '.join(original_old_words[:i]))
                if i > 0:
                    chars_before_match += 1
                best_overlap_pos = search_start_pos + chars_before_match
                break
        
        # Fallback for single word match if no good overlap found
        if max_overlap_len < 2 and new_words:
            new_first_word = new_words[0]
            last_words = old_words[-10:] if len(old_words) > 10 else old_words
            for i in range(len(last_words) - 1, -1, -1):
                if last_words[i] == new_first_word:
                    words_from_end = len(last_words) - i
                    total_old_words = len(old_transcript.split())
                    splice_word_index = total_old_words - words_from_end
                    original_words = old_transcript.split()
                    best_overlap_pos = len(' '.join(original_words[:splice_word_index]))
                    if splice_word_index > 0:
                        best_overlap_pos += 1
                    break
        
        return best_overlap_pos

    # Initialize Speech Transcriber
    try:
        speech_transcriber = SpeechTranscriber()
        print("Speech transcriber loaded successfully")
    except Exception as e:
        print(f"Failed to load speech transcriber: {e}")
        speech_transcriber = None
        return
    
    # State tracking
    working_transcript = ""
    last_processed_length = 0  # Track how much of the current speech we've processed
    
    # Parameters (at 16kHz sample rate)
    min_chunk_size = 80000  # 5 seconds minimum before starting transcription (if no final audio flag received)
    overlap_samples = 24000  # 1.5 second overlap for word boundary detection
    incremental_threshold = 48000  # 3 seconds of new audio before processing again
    
    try:
        while True:
            try:
                # Implementation, Get most recent data
                latest_task = None
                items_skipped = 0
                
                # Get first item (blocking)
                try:
                    latest_task = nerve_from_cortex_to_stt.get_nowait()#.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if latest_task is None:  # Shutdown signal
                    break
                
                # Drain queue to get the most recent item (LIFO behavior)
                while True:
                    try:
                        newer_task = nerve_from_cortex_to_stt.get_nowait()
                        if newer_task is None:
                            break
                        latest_task = newer_task
                        items_skipped += 1
                    except queue.Empty:
                        break
                
                if items_skipped > 0:
                    logging.debug(f"Audio STT skipped {items_skipped} older audio chunks (LIFO mode)")
                
                # Unpack the latest task
                full_audio_data, more_speech_coming, device_index, sample_rate, capture_timestamp = latest_task
                current_audio_length = len(full_audio_data)
                
                logging.debug(f"Audio length: {current_audio_length}, More coming: {more_speech_coming}, Last processed: {last_processed_length}")
                
                # Determine if we should process
                should_process = False #toggle if we have enough audio for a transcription run, if no final speech bool received
                chunk_to_process = None
                
                # Case 1: Final chunk (speech ended)
                if not more_speech_coming:
                    should_process = True
                    if last_processed_length == 0:
                        # Short utterance - process everything
                        chunk_to_process = full_audio_data
                        logging.debug("Processing final chunk (short utterance)")
                    else:
                        # Long utterance ending - process remaining unprocessed audio with overlap
                        start_pos = max(0, last_processed_length - overlap_samples)
                        chunk_to_process = full_audio_data[start_pos:]
                        logging.debug(f"Processing final chunk (long utterance) from {start_pos}")
                    
                    # Reset for next speech session
                    last_processed_length = 0
                    
                # Case 2: Ongoing speech - check if we have enough new audio
                elif current_audio_length >= min_chunk_size:
                    new_audio_length = current_audio_length - last_processed_length
                    
                    # First chunk of long speech
                    if last_processed_length == 0:
                        should_process = True
                        chunk_to_process = full_audio_data
                        last_processed_length = current_audio_length
                        logging.debug(f"Processing first chunk of long speech: {current_audio_length} samples")
                        
                    # Incremental processing for long speech
                    elif new_audio_length >= incremental_threshold:
                        should_process = True
                        # Process from overlap point to get better word boundaries
                        start_pos = max(0, last_processed_length - overlap_samples)
                        chunk_to_process = full_audio_data[start_pos:]
                        last_processed_length = current_audio_length
                        logging.debug(f"Processing incremental chunk from {start_pos}, new audio: {new_audio_length}")
                    else:
                        logging.debug(f"Waiting for more audio: {new_audio_length} < {incremental_threshold}")
                else:
                    logging.debug(f"Not enough audio yet: {current_audio_length} < {min_chunk_size}")
                
                # Process the audio if we should
                if should_process and chunk_to_process is not None and len(chunk_to_process) > 0:
                    try:
                        

                        # Transcribe the audio chunk (runs in parallel with voice recognition)
                        raw_transcription = speech_transcriber.transcribe(chunk_to_process)
                        
                        
                        # Update working transcript
                        if working_transcript and more_speech_coming:
                            # Find overlap and merge for ongoing speech
                            overlap_pos = find_overlap_position(working_transcript, raw_transcription)
                            
                            if overlap_pos < len(working_transcript):
                                working_transcript = working_transcript[:overlap_pos] + " " + raw_transcription
                            else:
                                working_transcript = working_transcript + " " + raw_transcription
                            
                            working_transcript = working_transcript.strip()
                        else:
                            # First transcription or final independent chunk
                            if not more_speech_coming and working_transcript:
                                # Final chunk of long speech - merge
                                overlap_pos = find_overlap_position(working_transcript, raw_transcription)
                                if overlap_pos < len(working_transcript):
                                    working_transcript = working_transcript[:overlap_pos] + " " + raw_transcription
                                else:
                                    working_transcript = working_transcript + " " + raw_transcription
                                working_transcript = working_transcript.strip()
                            else:
                                # New transcription
                                working_transcript = raw_transcription
                        
                        # Prepare and send transcript data
                        analysis_timestamp = time.time()
                        transcript_data = {
                            'transcription': working_transcript,
                            'final_transcript': not more_speech_coming,
                            #'voice_match': voice_match_result,
                            #'voice_probability': voice_probability,
                            'device_index': device_index,
                            'speech_detected': True,
                            'capture_timestamp': capture_timestamp,
                            'transcription_timestamp': analysis_timestamp,
                            'formatted_time': datetime.fromtimestamp(capture_timestamp).strftime('%H:%M:%S'),
                            'sample_rate': sample_rate,
                            'duration':current_audio_length/sample_rate
                        }
                        
                      
                        try:
                            nerve_from_stt_to_cortex.put_nowait(transcript_data)
                            logging.debug(f"Sent {'final' if not more_speech_coming else 'interim'} transcript_data: {transcript_data} chars")
                        except queue.Full:
                            logging.debug("Transcription queue Full - consume faster - will start clearing!")
                            # Clear the transcription queue before putting new data
                            while True:
                                try:
                                    _ = nerve_from_stt_to_cortex.get_nowait()
                                except queue.Empty:
                                    logging.debug("Cleared old transcription from queue")
                                    nerve_from_stt_to_cortex.put_nowait(transcript_data)
                                    logging.debug(f"Sent {'final' if not more_speech_coming else 'interim'} transcript_data: {transcript_data} chars")
                                    break
                        
                        # Reset working transcript if this was final
                        if not more_speech_coming:
                            working_transcript = ""
                            logging.debug("Reset working transcript for next speech session")
                            
                    except Exception as e:
                        logging.error(f"Transcription error: {e}")
                        # Reset on error to avoid getting stuck
                        if not more_speech_coming:
                            working_transcript = ""
                            last_processed_length = 0
                
            except Exception as e:
                logging.error(f"STT worker error: {e}")
                # Reset state on error
                working_transcript = ""
                last_processed_length = 0
            
            finally:
                # Clean up
                gc.collect()

    except Exception as e:
        print(f"auditory_cortex_worker_speechToText Primary Loop Error: {e}")

    finally:
        # No cleanup needed
        pass

def auditory_cortex_worker_voiceRecognition(nerve_from_cortex_to_vr,nerve_from_vr_to_cortex,database_path,gpu_device):
    import uuid
    import os
    #Set which GPU we run the voice recognition on on
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)

    #Create instance of the voice recognizer
    vr = VoiceRecognitionSystem(db_path=database_path)
    while True:
        voice_data = None
        try:
            #voice_data: in np data array of Audio sample (should be 16kHz mono)
            data = nerve_from_cortex_to_vr.get_nowait()
            voice_data = data['audio']
            is_my_voice_id = data['my_voice']#bool is system is speaking now
        except queue.Empty:
            continue

        if voice_data is not None:
            human_id, similarity = vr.recognize_speaker(voice_data)
            print(f"VR Similarity: {similarity}")
            print(f"VR Expect My Voice: {is_my_voice_id}")
            new_person = False
            if not human_id:
                #we dont have a profile for this voice, so we create a new one
                #if this is 'my_voice' it's the system talking
                
                #TODO: speech
                human_id = uuid.uuid4().hex
                if is_my_voice_id:
                    human_id = "my_voice_id"
                
                success = vr.add_speaker_profile(human_id,[voice_data])
                new_person = True
                similarity = 1
            else:
                success = vr.add_voice_sample(human_id,voice_data)
            
            #now put the update in queue
            try:
                results = {'human_id':human_id,'new_person':new_person, 'similarity':similarity}
                nerve_from_vr_to_cortex.put(results,timeout=1)
            except queue.Full:
                logging.error("nerve_from_vr_to_cortex is FULL - consume faster")
                 

        time.sleep(0.001)#small delay to help cpu
    

class AuditoryCortex():
    """
    manage all running audio functions. on init will start the auditory cortex, speech to text processes and default to connection
    with sound input device 0
    """
    def __init__(self,cortex=auditory_cortex_core,stt=auditory_cortex_worker_speechToText,nerve=auditory_nerve_connection,vr=auditory_cortex_worker_voiceRecognition,mpm=False,wakeword_name='jarvis',database_path=":memory:",gpu_device=0):
        logging.info("Starting Visual Cortex. This will run at minimum 3 separte processes via multiprocess (nerve,cortex,stt)")
        if not mpm:
            logging.warning("You MUST pass a multi processing manager instance: multiprocessing.Manager(), using arg: AuditoryCortex(mpm= multiprocessing.Manager()), to initiate the AuditoryCortex")
        #processes
        self.auditory_processes = {}
        self.auditory_processes['nerve'] = {}

        self.database_path = database_path
        self.wakeword_name = wakeword_name

        #Data queues
        self.external_cortex_queue = mpm.Queue(maxsize=30)
        #stat tracker
        self.external_stats_queue = mpm.Queue(maxsize=5)
        #raw sound capture carried to audio cortex
        self.nerve_from_input_to_cortex = mpm.Queue(maxsize=20) #holds small buffer because we always want most recent anyway, queue is constantly drained if full
        # internally used by Audio Cortex to hold speech audio that needs transcription
        self.nerve_from_cortex_to_stt = mpm.Queue(maxsize=100) #100 chunks of 32ms = ~3200ms (3.2 second) buffer
        # internally used by speech to text to send data back to Audio Cortex(transcriptions)
        self.nerve_from_stt_to_cortex = mpm.Queue(maxsize=5) 
        #internally used to send speech audio clips to regognizer (voice recognition)
        self.nerve_from_cortex_to_vr = mpm.Queue(maxsize=1) #nerve_from_cortex_to_vr
        #internally used to get speech recognition results
        self.nerve_from_vr_to_cortex = mpm.Queue(maxsize=1)
        
        #name/wakeword indicator to determine if data should activly be acted on.
        self.detected_name_wakeword_queue = mpm.Queue(maxsize=1)#data schema in queue {'name_detected':bool, 'active_speaker':bool, 'recognized_speaker':{} or False}

        #TODO speech
        #This will have the state bool if speaking, transcript of speech and expected energy from the BrocasArea
        self.external_brocas_state_queue =  mpm.Queue(maxsize=1)
        #playback_state['is_playing'] = False
        #playback_state['expected_energy'] = 0.0


        #nerve controller function
        self.nerve = nerve

        #Primary hub process
        auditory_cortex_process = mp.Process(
            target=cortex,
            args=(self.nerve_from_input_to_cortex, 
                  self.external_cortex_queue,
                  self.external_stats_queue, 
                  self.nerve_from_cortex_to_stt,
                  self.nerve_from_stt_to_cortex,
                  self.detected_name_wakeword_queue,
                  self.wakeword_name,
                  self.external_brocas_state_queue,
                  self.nerve_from_cortex_to_vr,
                  self.nerve_from_vr_to_cortex)
        )
        auditory_cortex_process.start()
        self.auditory_processes['core'] = auditory_cortex_process

        #CPU intense speech to text
        stt_worker = mp.Process(
            target=stt,
            args=(self.nerve_from_cortex_to_stt,
                  self.nerve_from_stt_to_cortex)
        )
        stt_worker.start()
        self.auditory_processes['stt'] = stt_worker

        #CPU/GPU voice recognition process
        vr_worker = mp.Process(
            target=vr,
            args=(self.nerve_from_cortex_to_vr,
                  self.nerve_from_vr_to_cortex,
                  self.database_path,
                  gpu_device)
        )
        vr_worker.start()
        self.auditory_processes['vr'] = vr_worker

        

    def start_nerve(self,device_index=0):
        #I/O raw audio data capture
        auditory_nerve_process = mp.Process(
            target=self.nerve,
            args=(device_index, 
                  self.nerve_from_input_to_cortex, 
                  self.external_stats_queue)
        )
        auditory_nerve_process.start()
        self.auditory_processes['nerve'][device_index] = auditory_nerve_process


    def stop_nerve(self,device_index=0):
        # Terminate auditory nerve process, default to 0
        success = self.process_kill(self.auditory_processes['nerve'][device_index])
        return success
    
    def stop_stt(self):
        # Terminate speech to text
        success = self.process_kill(self.auditory_processes['stt'])
        return success
    
    def stop_vr(self):
        # Terminate voice recognition
        success = self.process_kill(self.auditory_processes['vr'])
        return success
    
    def stop_cortex(self):
        # Terminate cortex process
        success = self.process_kill(self.auditory_processes['core'])
        return success
    
    def shutdown(self):
        #shut down all auditory functions
        had_failure = False
        #Cortex
        success = self.stop_cortex()
        if not success:
            logging.error("Error shutting down Auditory Cortex process")
            had_failure = True
        #Speech To Text
        success = self.stop_stt()
        if not success:
            logging.error("Error shutting down Auditory Speech to Text process")
            had_failure = True
        #voice recognition
        success = self.stop_vr()
        if not success:
            logging.error("Error shutting down Auditory voice recognition process")
            had_failure = True
        #Input Devices
        for i,nerve in enumerate(self.auditory_processes['nerve']):
            success = self.process_kill(nerve)
            if not success:
                logging.error(f"Error shutting down Auditory Nerve device index: {i}")
                had_failure = True
        return had_failure
            
    def process_kill(self,process):
        try:
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
            del process
            return True
        except Exception as e:
            logging.error(f"Auditory Process Stop Error: {e}")
            return False

