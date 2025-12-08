
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

import logging

from brocasArea import BrocasArea

logging.basicConfig(level=logging.DEBUG) #ignore everything use (level=logging.CRITICAL + 1)



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

def auditory_cortex_core(nerve_from_input_to_cortex, external_cortex_queue, external_stats_queue,
                        nerve_from_cortex_to_stt, nerve_from_stt_to_cortex,
                        state_of_cortex_speakerID_wakeword, wakeword_name, breakword, exitword,
                        external_brocas_state_dict,
                        my_voice_id,
                        brocas_area_interrupt_dict):
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
    locked_speaker_id = None
    locked_speaker_speaking = False #flag to trigger stop on transcription even if speech still detected by non-locked speaker
    
    #Indicator if system is playing auido
    system_actively_speaking = False #playback_state.get('is_playing', False)

    #ID of the system voice
    system_voice_id = my_voice_id
    
    # Initialize Silero VAD to find audio with speech
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
        SAMPLE_SIZE_REQUIREMENT = 512 #DO NOT CHANGE VALUE required for VAD
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
                            system_actively_speaking = external_brocas_state_dict.get('is_playing',False)
                                                
                        if 'end' in speech_dict:
                            logging.info(f"Speech END detected at {speech_dict['end']:.3f}s")
                            speech_active = False
                           
                    #Either active speech, or final chunk after speech stopped to prevent final word clipping     
                    if speech_active or (not speech_active and len(full_speech) > 0):
                        speech_buffer.append(audio_chunk)
                        full_speech = np.concatenate(speech_buffer)
                        #continue to check brocas TTS state because of delays
                        system_actively_speaking = external_brocas_state_dict.get('is_playing',False)
                        try:
                            #send data out for transcription and recognition
                            nerve_from_cortex_to_stt.put_nowait({
                                'full_speech': full_speech.copy(),# entire Data buffer of np.float32 audio samples containing speech audio
                                'speech_active': speech_active,# bool VAD detected speech
                                'device_index':device_index,# int of device index
                                'sample_rate':sample_rate,#sample rate int
                                'capture_timestamp':capture_timestamp,#time stamp float
                                'system_actively_speaking':system_actively_speaking#is the AI speaking bool
                                })
                            
                        except queue.Full:
                            logging.debug("nerve_from_cortex_to_stt FULL - clearing and retrying")
                            try:
                                while True:
                                    try:
                                        _ = nerve_from_cortex_to_stt.get_nowait()
                                    except queue.Empty:
                                        break
                                nerve_from_cortex_to_stt.put_nowait({
                                'full_speech': full_speech.copy(),# entire Data buffer of np.float32 audio samples containing speech audio
                                'speech_active': speech_active,# bool VAD detected speech
                                'device_index':device_index,# int of device index
                                'sample_rate':sample_rate,#sample rate int
                                'capture_timestamp':capture_timestamp,#time stamp float
                                'system_actively_speaking':system_actively_speaking#is the AI speaking bool
                                })
                            except queue.Full:
                                logging.debug("nerve_from_cortex_to_stt still FULL")

                        #Reset after speaking done
                        if not speech_active:
                            speech_buffer = []
                            full_speech = []
                    
                    # Prepare cortex output
                    cortex_output = {
                        'transcription': "",
                        'final_transcript': False,
                        'voice_id': None,
                        'voice_probability': 0.0,
                        'device_index': device_index,
                        'speech_detected': speech_active,
                        'audio_capture_timestamp': capture_timestamp,
                        'transcription_timestamp': capture_timestamp,
                        'formatted_time': datetime.fromtimestamp(capture_timestamp).strftime('%H:%M:%S'),
                        'audio_data': None,# NOT adding data here - design is only passing data if activly listening 
                        'sample_rate': sample_rate,
                        'duration': duration,
                        'hear_self_speaking': False,
                        'is_interrupt_attempt':False,
                        'is_locked_speaker': False,
                        'unlock_speaker':False
                    }
                   

                    try:
                        # Get transcription and voice recognition results if available
                        stt_output = nerve_from_stt_to_cortex.get_nowait()
                        cortex_output.update(stt_output)
                        if cortex_output['final_transcript']:
                            logging.debug("Received final transcript in cortex core\n")

                        voice_match_id = cortex_output.get('voice_id', None)
                        transcription = cortex_output.get('transcription', '').lower().strip()


                        # Determine if this is speech from our current locked speaker
                        if voice_match_id == locked_speaker_id:
                            cortex_output['is_locked_speaker'] = True
                            logging.debug("LOCKED SPEAKER:", transcription)
                            #Our locked speaker is speaking
                            locked_speaker_speaking = True
                        
                        # Check if this is the assistant speaking
                        if voice_match_id == system_voice_id:
                            cortex_output['hear_self_speaking'] = True
                            logging.debug(f"Detected my own voice: '{transcription}' ")

                        #Check for the breakword interruption phrase in the transcript
                        if breakword.lower().replace('.', '').replace(',', '') in transcription.replace('.', '').replace(',', '') and system_actively_speaking:
                            #placeholder to a User feedback audio sound like a beep. just play sys sound for now
                            logging.debug('\a')
                            
                            #Check if system actually in playback
                            if external_brocas_state_dict.get('is_playing',True):
                                #system is playing AND we have an interruption breakword detected
                                logging.debug("INTERRUPTION - set True 1")
                                cortex_output['is_interrupt_attempt'] = True
                                cortex_output['transcription'] = ""
                                brocas_area_interrupt_dict.update({'interrupt':True})
                        
                        #Check for the exitword release locked speaker
                        if exitword.lower().lower().replace('.', '').replace(',', '') in transcription.lower():
                            cortex_output['unlock_speaker'] = True
                        
                        #Flag our locked speaker is done speaking
                        if cortex_output['final_transcript'] and cortex_output['is_locked_speaker']:
                            logging.debug("Locked speaker stopped talking")
                            locked_speaker_speaking = False
                            cortex_output['audio_data'] = full_speech.copy() #add the audio data
                            logging.debug("INTERRUPTION - set False 1")
                            #establish that TTS can process system response to user input
                            #brocas_area_interrupt_dict.update({'interrupt':False})

                        # Handle voice lock management
                        # Check for wake word and if we should lock on to new voice
                        if wakeword_name.lower() in transcription.lower():
                                if transcription.lower().startswith(wakeword_name.lower()):
                                    logging.debug(f"\n\nWake word detected: {transcription}\n\n")
                                    logging.debug("INTERRUPTION - set False 2")
                                    #brocas_area_interrupt_dict.update({'interrupt':False})
                                    # Lock to this speaker
                                    if voice_match_id:
                                        locked_speaker_id = voice_match_id
                                        cortex_output['is_locked_speaker'] = True
                                        cortex_output['audio_data'] = full_speech.copy() #provide the audio data
                                        logging.info(f"Voice lock acquired by speaker: {locked_speaker_id}")
                                    try:
                                        state_of_cortex_speakerID_wakeword.update({'spoken_by': voice_match_id})
                                    except queue.Full:
                                        logging.error("state_of_cortex_speakerID_wakeword FULL - consume faster")
                                    except Exception as e:
                                        logging.error(f"Error: state_of_cortex_speakerID_wakeword: {e}")

                        #Handle ignore non-locked speaker still talking causing VAD speech detection true    
                        if locked_speaker_id != voice_match_id and transcription != '' and not system_actively_speaking:
                                #we want to ignore this speech since it's not the voice we are locked on to
                                logging.debug("\nI hear speech, but not from a voice I am locked on to.. Ignoring\n")
                                logging.debug(cortex_output['transcription'])
                                logging.debug("\n")
                                #If a non-locked speaker is speaking in the background VAD will still feed speech, need to flag our locked speaker is done speaking so even though VAD still true, indicate final transcript for locked speaker
                                if locked_speaker_speaking:
                                    logging.debug("locked speaker was speaking, but stopped. There is still talking from others though - will ignore\n")
                                    #our locked speaker WAS speaking, but no longer is
                                    locked_speaker_speaking = False
                                    cortex_output['final_transcript'] = True
                                    cortex_output['is_locked_speaker'] = True# Change this so down stream the final transcript AND locked are joined on this frame
                                    cortex_output['audio_data'] = full_speech.copy() #provide the audio data
                                    logging.debug("INTERRUPTION - set False 3")
                                    #brocas_area_interrupt_dict.update({'interrupt':False})
                        
                            

                    except queue.Empty:
                        logging.debug("No data in nerve_from_stt_to_cortex")
                        
                    # Put processed audio in output queue
                    try:
                        if cortex_output['final_transcript']:
                            logging.debug("Final transcript sent to external cortex queue\n")
                        external_cortex_queue.put_nowait(cortex_output)
                        #If our speaker said the exitword, reset
                        if cortex_output['unlock_speaker']:
                            locked_speaker_id = None
                    except:
                        logging.debug("external_cortex_queue FULL - CRITICAL ERROR")

                except Exception as vad_error:
                    logging.debug(f"VAD processing error: {vad_error}")
                    continue
                
                # Calculate stats
                frame_count += 1
                if frame_count % 50 == 0:
                    now = time.time()
                    fps = frame_count / (now - start_time)
                    

                    stats_dict = {
                        'auditory_cortex_fps': fps,
                        'last_auditory_cortex': now,
                    }

                    try:
                        external_stats_queue.put_nowait(stats_dict)
                    except:
                        pass
                    
            except queue.Empty:
                continue

            #delay to prevent CPU overload
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
    min_chunk_size = 64000  # 4 seconds minimum before starting transcription (if no final audio flag received)
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
                full_audio_data = latest_task['full_speech']
                more_speech_coming = latest_task['speech_active']
                device_index = latest_task['device_index']
                sample_rate = latest_task['sample_rate']
                capture_timestamp = latest_task['capture_timestamp']
                my_voice = latest_task['system_actively_speaking']

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
                        #PARALLEL VOICE RECOGNITION and TRANSCRIPTION 
                        try:
                            #voice recognition in a separate worker via queue
                            vr_data = {'audio':chunk_to_process,'my_voice':my_voice}
                            nerve_from_stt_to_vr.put_nowait(vr_data)
                        except:
                            logging.error("Error sending data on nerve_from_stt_to_vr")

                        # Transcribe the audio chunk
                        raw_transcription = speech_transcriber.transcribe(chunk_to_process)
                        
                        # Set recognition results from the container
                        voice_match_result = False
                        voice_probability = 0.0

                        #Combine transcription and voice recognition waiting for VR result
                        try:
                            data = nerve_from_vr_to_stt.get(timeout=1)#{'human_id':human_id,'new_person':new_person, 'similarity':similarity}
                            voice_match_result = data['human_id']
                            voice_probability =  data['similarity']
                        except:
                            logging.error("Error getting data on nerve_from_vr_to_stt")
                        
                        
                        logging.debug(f"Voice recognition result: {voice_match_result}, probability: {voice_probability:.3f}")
                        
                        # Update working transcript
                        if working_transcript and more_speech_coming:
                            # Find overlap and merge for ongoing real time speech transcript
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
                        logging.debug("FINAL TRANSCRIPT:" if not more_speech_coming else "INTERIM TRANSCRIPT:", working_transcript)
                        transcript_data = {
                            'transcription': working_transcript,
                            'final_transcript': not more_speech_coming,
                            'voice_id': voice_match_result,
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
            logging.debug(f"VR Similarity: {similarity}")
            logging.debug(f"VR Expect My Voice: {is_my_voice_id}")
            logging.debug(f"human_id recalled: {human_id}")
            
                
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

            #now put the recognized voice in queue
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
    def __init__(self,cortex=auditory_cortex_core,stt=auditory_cortex_worker_speechToText,nerve=auditory_nerve_connection,vr=auditory_cortex_worker_voiceRecognition,ba=BrocasArea,mpm=False,wakeword_name='jarvis',breakword="enough jarvis",exitword="goodbye jarvis",database_path=":memory:",gpu_device=0):
        logging.info("Starting Auditory Cortex. This will run at separte processes via multiprocess (nerve,cortex,stt,vr)")
        if not mpm:
            logging.warning("You MUST pass a multi processing manager instance: multiprocessing.Manager(), using arg: AuditoryCortex(mpm= multiprocessing.Manager()), to initiate the AuditoryCortex")
        #processes
        self.auditory_processes = {}
        self.auditory_processes['nerve'] = {}

        self.database_path = database_path
        self.wakeword_name = wakeword_name
        self.breakword = breakword
        self.exitword = exitword
        
        #Data queues
        self.external_cortex_queue = mpm.Queue(maxsize=30)
        #This is data struct of the post VAD, Recognition, Transcription data processed one last time in cortex for interruption detection prior to sending externally
        self.external_cortex_queue_data_struct = {
                        'transcription': "",
                        'final_transcript': False,
                        'voice_id': None,
                        'voice_probability': 0.0,
                        'device_index': 0,#device_index,
                        'speech_detected': False,#speech_active,
                        'audio_capture_timestamp': 0,#capture_timestamp,
                        'transcription_timestamp': 0,#capture_timestamp,
                        'formatted_time': 0,#datetime.fromtimestamp(capture_timestamp).strftime('%H:%M:%S'),
                        'audio_data': [],#Data array of np.float32 audio chunks
                        'sample_rate': 16000,#sample_rate,
                        'duration': 0,#duration,
                        'hear_self_speaking': False,
                        'is_interrupt_attempt':False,
                        'is_locked_speaker': False,
                        'unlock_speaker':False
                    }

        
        #stat tracker
        self.external_stats_queue = mpm.Queue(maxsize=5)

        #raw sound capture carried to audio cortex
        self.nerve_from_input_to_cortex = mpm.Queue(maxsize=20) #holds small buffer because we always want most recent anyway, queue is constantly drained if full
        self.nerve_from_input_to_cortex_data_struct = {
                            'device_index': None,  # int: Audio input device identifier, default 0
                            'audio_frame': None,   # np.ndarray: float32 audio samples, shape (n_samples,)
                            'capture_timestamp': None,  # float: Unix timestamp when frame was captured
                            'sample_rate': None    # int: Sample rate in Hz (default, 16000)
                        }
        
        # internally used by Audio Cortex to hold speech audio that needs transcription
        self.nerve_from_cortex_to_stt = mpm.Queue(maxsize=100) #100 chunks of 32ms = ~3200ms (3.2 second) buffer
        self.nerve_from_cortex_to_stt_data_struct = {
                                'full_speech': [],# entire Data buffer of np.float32 audio samples containing speech audio
                                'speech_active': False,# bool VAD detected speech
                                'device_index':0,# int of device index
                                'sample_rate':16000,#sample rate int
                                'capture_timestamp':0,#time stamp float
                                'system_actively_speaking':False#is the AI speaking bool
                                }
        
        # internally used by speech to text to send data back to Audio Cortex(transcriptions)
        self.nerve_from_stt_to_cortex = mpm.Queue(maxsize=5)
        self.nerve_from_stt_to_cortex_data_struct = {
                            'transcription': "",# working_transcript,
                            'final_transcript': False,#bool more_speech_coming,
                            'voice_id': 0,#UUID of voice_match_result,
                            'voice_probability': 0,#voice_probability for the voice_id,
                            'device_index': 0,#device_index,
                            'speech_detected': True,#bool speech detected
                            'capture_timestamp': 0,#capture_timestamp,
                            'transcription_timestamp': 0,#analysis_timestamp,
                            'formatted_time': 0,#datetime.fromtimestamp(capture_timestamp).strftime('%H:%M:%S'),
                            'sample_rate': 16000,#sample_rate,
                            'duration': 0#time of current_audio_length/sample_rate
                        } 
        
        #internally used to send speech audio clips to regognizer (voice recognition)
        self.nerve_from_stt_to_vr = mpm.Queue(maxsize=1) #MUST be 1 result, used for matching
        self.nerve_from_stt_to_vr_data_struct = {'audio':[],#chunk_to_process,
                                                 'my_voice':0#bool my_voice
                                                 }
        
        #internally used to get speech recognition results
        self.nerve_from_vr_to_stt = mpm.Queue(maxsize=1)
        self.nerve_from_vr_to_stt_data_struct = {'human_id':None,#human_id is the UUID of recognized speaker,
                                                 'new_person':False, #new_person bool, if true a new profile was created
                                                 'similarity':0.0 #similarity of match to human_id
                                                 }
        
        #name/wakeword indicator to determine ID of who said the wakeword/name
        self.state_of_cortex_speakerID_wakeword = mpm.dict(
            {'spoken_by': 0#speaker ID who said the wakeword/name last
        })

        # Create shared dict for current status
        self.brocas_area_interrupt_dict = mpm.dict({
            'interrupt': False
        })

        #Flag to connect assistant speaker embedding to audio output
        self.my_voice_id = 'my_voice_id'
        self.brocas_area = ba(self.brocas_area_interrupt_dict)
        """self.brocas_area.status = {
                'is_playing': True,
                'transcript': audio_template['transcript'],
                'samplerate': samplerate,
                'num_channels': audio_template['num_channels'],
                'audio_length': len(audio),
                'avg_energy': avg_energy,  # for Echo Cancellation,
                'audo_data' : audio
            }"""

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
                  self.state_of_cortex_speakerID_wakeword,
                  self.wakeword_name,
                  self.breakword,
                  self.exitword,
                  self.brocas_area.status,
                  self.my_voice_id,
                  self.brocas_area_interrupt_dict)
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
    
    def stop_brocas_area(self):
        # Terminate brocas area
        success = self.brocas_area.shutdown()
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
        success = self.stop_brocas_area()
        if not success:
            logging.error("Error shutting down Brocas Area process")
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





if __name__ == "__main__":


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
        # Initialize the timer with the current time
        last_speech_time = time.time()
        while True:
            # Check if there's data in the cortex queue - populated when text is being transcribed
            if not cortex.external_cortex_queue.empty():
                cortex_data = cortex.external_cortex_queue.get()
                """ try:
                    _  = cortex.state_of_cortex_speakerID_wakeword.get()
                except:
                    pass"""
                #Do we hear the locked speaker speaking?
                if cortex_data['is_locked_speaker']: 
                    print("Speaker is locked?",cortex_data['voice_id'] == cortex.state_of_cortex_speakerID_wakeword['spoken_by'])
                    print("Locked Speaker is talking")

                # Display the transcription
                if cortex_data['transcription']:
                    status = "FINAL" if cortex_data['final_transcript'] else "INTERIM"
                    speaker_info = ""
                    
                    if cortex_data['voice_id']:
                        speaker_info = f" [Speaker: {cortex_data['voice_id']} " \
                                     f"({cortex_data['voice_probability']:.2%})]"
                    
                    print(f"[{cortex_data['formatted_time']}] [{status}]{speaker_info}: "
                          f"{cortex_data['transcription']}")
                    
                    # Add extra newline for final transcripts for readability
                    if cortex_data['final_transcript']:
                        print("Final Transcript Received\n")
                
                #Interrupt
                if cortex_data['is_interrupt_attempt']:
                    cortex.brocas_area.stop_playback()
                        
            else:
                # Small sleep to prevent CPU overuse
                time.sleep(0.01)

 
            current_time = time.time()
            # Check if 20 seconds have passed since the last speech
            if current_time - last_speech_time >= 20:
                print("10 seconds passed")
                logging.info("Speech to Text Test...")
                
                cortex.brocas_area.synthesize_speech(
                    "This is me talking for a little while so that you can try and interrupt me. "
                    "If you say enough and then my name I should stop talking. "
                    "if you say nothing I'll just finish talking.",
                    auto_play=True
                )
                
                # Reset the timer
                last_speech_time = current_time
                
    except KeyboardInterrupt:
        logging.info("\n\nShutting down Auditory Cortex...")
        cortex.shutdown()
        logging.info("Shutdown complete")