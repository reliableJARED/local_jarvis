
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

from brocasArea import BrocasArea

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
            except queue.Full:
                pass
                #logging.warning(f"Audio nerve {device_index} unable to send stats data to external_stats_queue because it's full")
            except Exception as e:
                logging.error(f"Error in Audio nerve {device_index} putting data in external_stats_queue: {e}")
    
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
                            nerve_from_stt_to_vr.put_nowait(vr_data)
                        except:
                            logging.error("Error sending data on nerve_from_stt_to_vr")


                        # Set recognition results from the container
                        voice_match_result = False
                        voice_probability = 0.0
                        try:
                            data = nerve_from_vr_to_stt.get(timeout=1)#{'human_id':human_id,'new_person':new_person, 'similarity':similarity}
                            voice_match_result = data['human_id']
                            voice_probability =  data['similarity']
                        except:
                            logging.error("Error getting data on nerve_from_vr_to_stt")
                        
                        
                        logging.debug(f"Voice recognition result: {voice_match_result}, probability: {voice_probability:.3f}")
"""

def auditory_cortex_core(nerve_from_input_to_cortex, external_cortex_queue, external_stats_queue,
                        nerve_from_cortex_to_stt, nerve_from_stt_to_cortex,
                        detected_name_wakeword_queue, wakeword_name, 
                        external_brocas_state_dict):
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
                            is_playback_context = external_brocas_state_dict.get('is_playing',False)
                                                
                        if 'end' in speech_dict:
                            logging.info(f"Speech END detected at {speech_dict['end']:.3f}s")
                            speech_active = False
                           
                    # === NOISE FLOOR & PLAYBACK UPDATES ===
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
                        
                        try:
                            #send data out for transcription and recognition
                            nerve_from_cortex_to_stt.put_nowait((
                                full_speech.copy(), 
                                speech_active, 
                                device_index, 
                                sample_rate, 
                                capture_timestamp,
                                is_playback_context
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
                            logging.debug(f"Detected my own voice: '{transcription}' ")
                        
                        # Check if system is playing audio but we are not detecting assistant voice as source of speech
                        elif is_playback_context and voice_match_id != assistant_voice_id:
                            if wakeword_name.lower() in transcription.lower():
                                if voice_match_id:
                                    locked_speaker_id = voice_match_id
                                    is_interrupt_attempt = True
                                    print("INTERRUPT!!!!!!!!!!!!!!!!!!!!!!")

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
                            min_snr = 1.0  # At least 3x above noise floor
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
                        
                        # Handle voice lock management
                        else:
                            # Check for wake word and if we should lock on to new voice
                            if wakeword_name.lower() in transcription.lower():
                                if transcription.lower().startswith(wakeword_name.lower()):
                                    print(f"\n\nWake word detected: {transcription}\n\n")
                                    # Lock to this speaker
                                    if voice_match_id:
                                        locked_speaker_id = voice_match_id
                                        logging.info(f"Voice lock acquired by speaker: {voice_match_id}")
                                    try:
                                        detected_name_wakeword_queue.put({'spoken_by': voice_match_id, 'is_interrupt_attempt':is_interrupt_attempt})
                                    except queue.Full:
                                        logging.error("detected_name_wakeword_queue FULL - consume faster")
                                    except Exception as e:
                                        logging.error(f"Error: detected_name_wakeword_queue: {e}")

                                else:
                                    print(f"\n\nWake word in middle of speech: {transcription}\n\n")
                                    print(f"\nSPOKEN BY: {voice_match_id} -Should Interrupt? {is_interrupt_attempt}\n")
                                    """try:
                                        detected_name_wakeword_queue.put({'spoken_by': voice_match_id, 'is_interrupt_attempt':is_interrupt_attempt})
                                    except queue.Full:
                                        logging.error("detected_name_wakeword_queue FULL - consume faster")
                                    except Exception as e:
                                        logging.error(f"Error: detected_name_wakeword_queue: {e}")"""
                            
                            if locked_speaker_id != voice_match_id and transcription != '' and not is_playback_context:
                                #we want to ignore this speech since it's not the voice we are locked on to
                                print("\nI hear speech, but not from a voice I am locked on to.. Ignore?\n")
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

def auditory_cortex_worker_speechToText(nerve_from_cortex_to_stt,nerve_from_stt_to_cortex,nerve_from_stt_to_vr,nerve_from_vr_to_stt):
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
                
                # Drain queue to get the most recent item 
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
                full_audio_data, more_speech_coming, device_index, sample_rate, capture_timestamp, my_voice = latest_task
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
                         
                        try:
                            #voice recognition in a separate worker via queue
                            vr_data = {'audio':chunk_to_process,'my_voice':my_voice}
                            nerve_from_stt_to_vr.put_nowait(vr_data)
                        except:
                            logging.error("Error sending data on nerve_from_stt_to_vr")

                        # Transcribe the audio chunk (runs in parallel with voice recognition)
                        raw_transcription = speech_transcriber.transcribe(chunk_to_process)
                        
                        # Set recognition results from the container
                        voice_match_result = False
                        voice_probability = 0.0
                        try:
                            data = nerve_from_vr_to_stt.get(timeout=1)#{'human_id':human_id,'new_person':new_person, 'similarity':similarity}
                            voice_match_result = data['human_id']
                            voice_probability =  data['similarity']
                        except:
                            logging.error("Error getting data on nerve_from_vr_to_stt")
                        
                        
                        logging.debug(f"Voice recognition result: {voice_match_result}, probability: {voice_probability:.3f}")
                        
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
                            'voice_match': voice_match_result,
                            'voice_probability': voice_probability,
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

def auditory_cortex_worker_voiceRecognition(nerve_from_stt_to_vr,nerve_from_vr_to_stt,database_path,gpu_device,my_voice_id):
    import uuid
    import os
    #Set which GPU we run the voice recognition on on
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)
    
    #Create instance of the voice recognizer
    vr = VoiceRecognitionSystem(db_path=database_path,my_voice_id=my_voice_id)

    while True:
        voice_data = None
        try:
            data = nerve_from_stt_to_vr.get_nowait()
            voice_data = data['audio'] #data in np data array of Audio sample (should be 16kHz mono)
            is_my_voice_id = data['my_voice']#bool
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"error getting data from nerve_from_stt_to_vr")

        if voice_data is not None:
            
            human_id, similarity = vr.recognize_speaker(voice_data)
            print(f"VR Similarity: {similarity}")
            print(f"VR Expect My Voice: {is_my_voice_id}")
            

            new_person = False
            if not human_id:
                #we dont have a profile for this voice, so we create a new one
                #if is_my_voice_id is true it's 'my_voice' it's the system talking
                human_id = uuid.uuid4().hex
                if is_my_voice_id:
                    human_id = my_voice_id
                    
                #print(f"add_speaker_profile({human_id},[{voice_data}])")
                success = vr.add_speaker_profile(human_id,[voice_data])
                new_person = True
                similarity = 1
            else:
                success = vr.add_voice_sample(human_id,voice_data)
            
            #TODO: This doesn't work - investigate more
            if is_my_voice_id:
                #remove the assistant voice audio from the sample
                voice_data = vr.removeVoiceAudio(voice_data)
                print(type(voice_data))
                human_id, similarity = vr.recognize_speaker(voice_data)
                print(f"2 - VR Similarity: {similarity}")
                print(f"2 - VR Expect My Voice: {is_my_voice_id}")

            #now put the update in queue
            try:
                results = {'human_id':human_id,'new_person':new_person, 'similarity':similarity}
                nerve_from_vr_to_stt.put_nowait(results) 
            except queue.Full:
                logging.debug("nerve_from_vr_to_stt queue Full - consume faster - will start clearing!")
                # Clear the transcription queue before putting new data
                while True:
                    try:
                        _ = nerve_from_vr_to_stt.get_nowait()
                    except queue.Empty:
                        logging.debug("Cleared old transcription from queue")
                        nerve_from_vr_to_stt.put_nowait(results)
                        break

        time.sleep(0.001)#small delay to help cpu
    

class AuditoryCortex():
    """
    manage all running audio functions. on init will start the auditory cortex, speech to text processes and default to connection
    with sound input device 0
    """
    def __init__(self,cortex=auditory_cortex_core,stt=auditory_cortex_worker_speechToText,nerve=auditory_nerve_connection,vr=auditory_cortex_worker_voiceRecognition,ba=BrocasArea,mpm=False,wakeword_name='jarvis',database_path=":memory:",gpu_device=0):
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
        self.nerve_from_stt_to_vr = mpm.Queue(maxsize=1) #MUST be 1 result, used for matching
        #internally used to get speech recognition results
        self.nerve_from_vr_to_stt = mpm.Queue(maxsize=1)
        
        #name/wakeword indicator to determine if data should activly be acted on.
        self.detected_name_wakeword_queue = mpm.Queue(maxsize=1)#data schema in queue {'name_detected':bool, 'active_speaker':bool, 'recognized_speaker':{} or False}

        #TODO speech
        #This will have the state bool if speaking, transcript of speech and expected energy from the BrocasArea
        self.my_voice_id = 'my_voice_id'
        self.brocas_area = ba()
        """self.external_brocas_state_dict = mpm.dict({'is_playing':False,
                                                    'expected_energy':0.0,
                                                    'transcript':"",
                                                    'interrupt':False})"""

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
                  self.brocas_area.status)
        )
        auditory_cortex_process.start()
        self.auditory_processes['core'] = auditory_cortex_process

        #CPU intense speech to text
        stt_worker = mp.Process(
            target=stt,
            args=(self.nerve_from_cortex_to_stt,
                  self.nerve_from_stt_to_cortex,
                  self.nerve_from_stt_to_vr,
                  self.nerve_from_vr_to_stt)
        )
        stt_worker.start()
        self.auditory_processes['stt'] = stt_worker

        #CPU/GPU voice recognition process
        vr_worker = mp.Process(
            target=vr,
            args=(self.nerve_from_stt_to_vr,
                  self.nerve_from_vr_to_stt,
                  self.database_path,
                  gpu_device,
                  self.my_voice_id)
        )
        vr_worker.start()
        self.auditory_processes['vr'] = vr_worker

        #calibration of system voice
        self.run_calibration()

    def resample_audio(self,audio_data: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
        from scipy import signal
        """
        Resample audio from one sample rate to another.
        
        Args:
            audio_data: Input audio data
            orig_rate: Original sample rate (e.g., 24000)
            target_rate: Target sample rate (e.g., 16000)
            
        Returns:
            Resampled audio data
        """
        # Calculate the number of samples in the resampled audio
        num_samples = int(len(audio_data) * target_rate / orig_rate)
        
        # Use scipy's resample function for high-quality resampling
        resampled = signal.resample(audio_data, num_samples)
        
        return resampled.astype(np.float32)


    def run_calibration(self):
        """
        Run the voice calibration process.
        """
        logging.info("Starting voice calibration process...")
        
        # 10 diverse phrases covering different phonetic patterns
        calibration_phrases = [
            "The quick brown fox jumps over the lazy dog.",
            "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
            "She sells seashells by the seashore.",
            "Peter Piper picked a peck of pickled peppers.",
            "I scream, you scream, we all scream for ice cream!",
            "Unique New York, you need New York, you know you need unique New York.",
            "Red leather, yellow leather, red leather, yellow leather.",
            "The sixth sick sheikh's sixth sheep's sick.",
            "Can you can a can as a canner can can a can?",
            "How can a clam cram in a clean cream can?"
        ]
        
        logging.info(f"Prepared {len(calibration_phrases)} calibration phrases")
        
        logging.info("\n" + "="*50)
        logging.info("SYNTHESIZING CALIBRATION PHRASES")
        logging.info("="*50)
        
        for i, phrase in enumerate(calibration_phrases, 1):
            logging.info(f"\nPhrase {i}/{len(calibration_phrases)}: {phrase[:50]}...")
            
            # Synthesize speech (DO NOT PLAY - auto_play=False)
            logging.info("  → Synthesizing speech via BrocasArea...")
            audio_dict = self.brocas_area.synthesize_speech(phrase, auto_play=False)
            
            if audio_dict is None:
                logging.error(f"  ✗ Failed to synthesize phrase {i}")
                continue
            
            # Extract audio data
            audio_data_24k = audio_dict['audio_data']
            orig_rate = audio_dict['samplerate']  # Should be 24000 Hz
            
            # Convert from 24kHz to 16kHz for voice recognition
            logging.info(f"  → Converting from {orig_rate}Hz to 16kHz...")
            audio_data_16k = self.resample_audio(audio_data_24k, orig_rate, 16000)
            
            #put the data into the recognition worker
            try:
                self.nerve_from_stt_to_vr.put({'audio':audio_data_16k,'my_voice':True})
            except queue.Full:
                logging.error("calibration error: cant put data on nerve_from_stt_to_vr its FULL")
            except Exception as e:
                logging.error(f"Calibartion error nerve_from_stt_to_vr: {e}")

            try:
                #wait for result of embedding and saving to memory
                _ = self.nerve_from_vr_to_stt.get(timeout=20)
            except queue.Empty:
                logging.error("timed out waiting for data on nerve_from_vr_to_stt during calibration")
            except Exception as e:
                logging.error(f"Calibartion error nerve_from_vr_to_stt: {e}")
            
            logging.info(f"  ✓ Phrase {i} processed successfully")
            logging.info(f"    Original length: {len(audio_data_24k)} samples @ {orig_rate}Hz")
            logging.info(f"    Resampled length: {len(audio_data_16k)} samples @ 16000Hz")

        logging.info("\n" + "="*50)
        logging.info("CALIBRATION COMPLETE")
        logging.info("="*50)
        
        return None


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

##############################
"""
def subtract_average_playback_energy(mic_chunk, playback_avg_energy, subtraction_factor=0.7):

    Ultra-simple energy-based subtraction.
    
    Args:
        mic_chunk: Current microphone chunk (np.ndarray)
        playback_avg_energy: Average RMS energy of the playback audio
        subtraction_factor: How much to subtract (0.5-0.9 recommended)
    
    Returns:
        np.ndarray: Cleaned audio

    if playback_avg_energy is None or playback_avg_energy == 0:
        return mic_chunk
    
    # Calculate current mic energy
    mic_energy = np.sqrt(np.mean(mic_chunk ** 2))
    
    # If mic energy is similar to or less than playback energy, it's likely just echo
    if mic_energy <= playback_avg_energy * 1.2:
        # Strongly attenuate - likely just echo
        return mic_chunk * 0.2  # Keep 20% to detect very strong interrupts
    
    # Mic has significantly more energy than expected playback
    # Subtract the playback energy contribution
    energy_to_subtract = playback_avg_energy * subtraction_factor
    
    # Scale down the chunk proportionally
    if mic_energy > energy_to_subtract:
        scaling_factor = (mic_energy - energy_to_subtract) / mic_energy
        return mic_chunk * scaling_factor
    else:
        # Not enough energy to be real speech over playback
        return mic_chunk * 0.2
"""

"""
# Even simpler - just gate based on energy threshold:
def energy_gate_during_playback(mic_chunk, playback_avg_energy, threshold_multiplier=2.5):

    Simplest approach: only allow audio through if significantly above playback energy.
    
    Args:
        mic_chunk: Current microphone chunk
        playback_avg_energy: Average energy of playback
        threshold_multiplier: How much louder mic must be (2-3 recommended)
    
    Returns:
        np.ndarray: Original or heavily attenuated audio

    if playback_avg_energy is None or playback_avg_energy == 0:
        return mic_chunk
    
    mic_energy = np.sqrt(np.mean(mic_chunk ** 2))
    
    # Must be significantly louder than playback to pass through
    if mic_energy > playback_avg_energy * threshold_multiplier:
        return mic_chunk  # Clear interrupt
    else:
        return mic_chunk * 0.1  # Heavily attenuate likely echo
"""

"""
Why this works so well:

Dead simple - just compare energies
No timing issues - don't need to align audio samples
Adaptive - updates with each new playback chunk
Fast - minimal computation (one RMS calculation)
Tunable - adjust threshold_multiplier easily

Tuning guide:

threshold_multiplier = 2.0: More sensitive, easier to interrupt (may get false positives)
threshold_multiplier = 2.5: Balanced (recommended starting point)
threshold_multiplier = 3.0: More strict, harder to interrupt (fewer false positives)

To test and calibrate:

Have assistant speak
Try interrupting with wake word at normal volume
If it doesn't detect → lower threshold_multiplier
If it false-triggers on echo → raise threshold_multiplier
"""
##############################





if __name__ == "__main__":

    import soundfile as sf

    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create multiprocessing manager
    manager = mp.Manager()
    
    # Initialize the Auditory Cortex
    logging.info("Initializing Auditory Cortex...")
    cortex = AuditoryCortex(
        mpm=manager,
        wakeword_name='jarvis',
        database_path=":memory:",
        gpu_device=0
    )
    
    # Start the nerve (microphone input) on device 0
    logging.info("Starting auditory nerve on device 0...")
    cortex.start_nerve(device_index=0)
    
    # Sleep for a few seconds to let everything initialize
    logging.info("Waiting for initialization...")
    time.sleep(8)
    
    # Signal that we're about to play audio
    logging.info("Speech to Text Test...")
    cortex.brocas_area.synthesize_speech("This is a demonstration of the Brocas Area text to speech system inside of the Audio Cortex. This is what my voice sounds like.",auto_play=True)

    logging.info("\nTo activate voice lock say: Jarvis \n")
    
    # Now listen and display transcriptions
    logging.info("\n" + "="*50)
    logging.info("Listening for speech... (Press Ctrl+C to stop)")
    logging.info("="*50 + "\n")
    
    try:
        tic = 0
        while True:
            # Check if there's data in the cortex queue - populated when text is being transcribed
            if not cortex.external_cortex_queue.empty():
                cortex_data = cortex.external_cortex_queue.get()
                
                # Display the transcription
                if cortex_data['transcription']:
                    status = "FINAL" if cortex_data['final_transcript'] else "INTERIM"
                    speaker_info = ""
                    
                    if cortex_data['voice_match']:
                        speaker_info = f" [Speaker: {cortex_data['voice_match']} " \
                                     f"({cortex_data['voice_probability']:.2%})]"
                    
                    print(f"[{cortex_data['formatted_time']}] [{status}]{speaker_info}: "
                          f"{cortex_data['transcription']}")
                    
                    # Add extra newline for final transcripts for readability
                    if cortex_data['final_transcript']:
                        print()
                        
            else:
                # Small sleep to prevent busy-waiting
                time.sleep(0.1)
            tic += 1

            #10 Second Loop
            if (tic % 1000) == 0:
                print("10 seconds")
                logging.info("Speech to Text Test...")
                cortex.brocas_area.synthesize_speech("This is me talking for a little while so that you can try and interrupt me. If you say my name I should stop talking. if you say nothing I'll just finish this rant.",auto_play=True)
        
                
    except KeyboardInterrupt:
        logging.info("\n\nShutting down Auditory Cortex...")
        cortex.shutdown()
        logging.info("Shutdown complete")