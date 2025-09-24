from flask import Flask, Response, jsonify, request
import cv2
import multiprocessing as mp
import queue
import time
import numpy as np
from datetime import datetime
import sounddevice as sd
from collections import deque
import sys
from PIL import Image
import torch
import gc
import threading
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO) #ignore everything use (level=logging.CRITICAL + 1)

# Set the Werkzeug logger level to ERROR to ignore INFO messages
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

# Configuration
QUEUE_SIZE = 5   # Maximum number of frames in queue
FPS_TARGET = 30   # Target FPS for camera capture, Moondream alone will make 1fps a stretch

# ===== GLOBAL PROCESS TRACKING  =====
# Visual process tracking
optic_nerve_processes = {}
visual_cortex_processes = {}
# Audio process tracking
auditory_nerve_processes = {}
auditory_cortex_processes = {}
#=====================================

# Global VLM (moondream) analysis state
VLM_availible = True
VLM_thread = None

#Audio Analyis reference
AUDIO_SCENE = {'device_index': None,
                    'audio_data': None,
                    'speech_detected': None,
                    'final_transcript':False,
                    'transcription': None, 
                    'capture_timestamp': None,
                    'formatted_time':None,
                    'auditory_cortex_timestamp': None,
                    'sample_rate':None,
                    'duration': None}

"""
VISUAL
"""
def optic_nerve_worker(camera_index, optic_nerve_queue, stats_dict, fps_target):
    """Worker process for capturing frames from a specific camera (optic nerve function).
    Will drop frames if the queue is full to maintain real-time performance.
    will also update stats_dict with fps and last frame time.
    will stream to optic_nerve_queue."""
    cap = cv2.VideoCapture(camera_index)
    
    # Camera setup
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, fps_target)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return
    
    print(f"Started optic nerve process for camera {camera_index}")
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to capture frame from camera {camera_index}")
                time.sleep(0.1)
                continue
                
            # Create frame data with camera index and timestamp
            timestamp = time.time()
            frame_data = {
                'camera_index': camera_index,
                'frame': frame,
                'capture_timestamp': timestamp
            }
            
            # Non-blocking queue put - drop frame if queue is full
            try:
                optic_nerve_queue.put_nowait(frame_data)
            except:
                # Queue is full, skip this frame
                pass
            
            # Calculate and update optic nerve FPS
            frame_count += 1
            if frame_count % 30 == 0:  # Update every 30 frames
                current_time = time.time()
                fps = frame_count / (current_time - start_time)
                stats_dict[f'optic_nerve_fps_{camera_index}'] = fps
                stats_dict[f'last_optic_nerve_{camera_index}'] = current_time
                
            # Target frame rate control
            time.sleep(1.0 / fps_target)
            
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print(f"Optic nerve process for camera {camera_index} stopped")

def visual_cortex_worker(optic_nerve_queue, visual_cortex_queue, stats_dict, visual_scene_queue):
    """Worker process for processing image frames from all cameras (visual cortex function)
    LIFO - will always process the most recent frame in the queue, dropping older frames.
    If person is detected a more detailed VLM is run in a thread. """
    print("Started visual cortex process")
    frame_count = 0
    start_time = time.time()

    # Global VLM state for this process
    VLM_availible_local = True

    #Visual Cortex Processing Core CNN (yolo)
    from yolo_ import YOLOhf
    

    #reuse single instance
    IMG_DETECTION_CNN = YOLOhf()

    try:
        while True:
            try:
                # Sync local VLM busy state with global state
                VLM_availible_local = stats_dict.get('VLM_availible', True)
                
                # Get the most recent frame by draining the queue
                frame_data = None
                frames_discarded = 0
                
                # Keep getting frames until queue is empty, keeping only the last one
                while True:
                    try:
                        frame_data = optic_nerve_queue.get_nowait()
                        frames_discarded += 1
                    except queue.Empty:
                        break
                
                # If we got at least one frame, process the most recent one
                if frame_data is not None:
                    frames_discarded -= 1  # Don't count the frame we're actually processing
                    if frames_discarded > 0:
                        print(f"Visual cortex: Discarded {frames_discarded} older frames, processing most recent")
                    
                    # Process the frame
                    processed_frame, person_detected = visual_cortex_process_img(frame_data['frame'], 
                                                  frame_data['camera_index'],
                                                  frame_data['capture_timestamp'],
                                                  IMG_DETECTION_CNN)
                    
                    # If person detected and VLM not busy, start async analysis
                    if person_detected and VLM_availible_local:
                        print(f"Starting VLM analysis - person detected and VLM not busy")
                        
                        # Set busy states
                        VLM_availible_local = False
                        stats_dict['VLM_availible'] = False

                        # Start VLM Moondream analysis in separate thread
                        VLM_thread = threading.Thread(
                            target=visual_cortex_process_vlm_analysis_thread,
                            args=(frame_data['frame'], frame_data['camera_index'], 
                                frame_data['capture_timestamp'], visual_scene_queue, stats_dict),
                                daemon = True
                            )  

                        VLM_thread.start()

                    elif person_detected and not VLM_availible_local:
                        print(f"Person detected but VLM busy - skipping analysis")
                    
                    # Create processed frame data
                    process_timestamp = time.time()
                    processed_data = {
                        'camera_index': frame_data['camera_index'],
                        'frame': processed_frame,
                        'capture_timestamp': frame_data['capture_timestamp'],
                        'visual_cortex_timestamp': process_timestamp
                    }
                    
                    # Put processed frame in output queue
                    try:
                        visual_cortex_queue.put_nowait(processed_data)
                    except:
                        # Queue is full, try to remove oldest and add new
                        try:
                            visual_cortex_queue.get_nowait()
                            visual_cortex_queue.put_nowait(processed_data)
                        except:
                            pass
                    
                    # Calculate visual cortex processing FPS
                    frame_count += 1
                    if frame_count % 30 == 0:
                        fps = frame_count / (time.time() - start_time)
                        stats_dict['visual_cortex_fps'] = fps
                        stats_dict['last_visual_cortex'] = time.time()
                else:
                    # No frames available, wait a bit before trying again
                    time.sleep(0.01)
                    
            except queue.Empty:
                continue
                
    except KeyboardInterrupt:
        pass
    finally:
        print("Visual cortex process stopped")

def visual_cortex_process_img(frame, camera_index, timestamp, IMG_DETECTION_CNN):
    """
    Process frame from specific camera (visual cortex processing).
    - YOLO runs at full speed
    - Returns processed frame and person_detected flag

    """
    results = IMG_DETECTION_CNN.detect(frame, confidence_threshold=0.5)

    # Check for person detections
    person_detected = False
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        class_name: str = IMG_DETECTION_CNN.class_names[label.item()]
        if class_name.lower() == 'person':  # Only print person detections
            person_detected = True
            box_rounded = [round(i, 2) for i in box.tolist()]
            print(f"Detected {class_name} with confidence {round(score.item(), 3)} at location {box_rounded}")

    # Draw annotations (filter for person only)
    frame = IMG_DETECTION_CNN.draw_detections(frame, results, filter_person_only=True)

    # Explicit cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Force GPU memory cleanup
        print(f"GPU memory after cleanup: {torch.cuda.memory_allocated()/1024**2:.1f}MB")

    return frame, person_detected

def visual_cortex_process_vlm_analysis_thread(frame, camera_index, timestamp, visual_scene_queue, stats_dict):
    """Thread function to run VLM (Moondream) analysis without blocking main processing"""
    # Import and initialize VLM Moondream (done in thread to avoid blocking memory if it stays loaded)
    from moondream_ import MoondreamWrapper
    IMG_PROCESSING_VLM = MoondreamWrapper(local_files_only=True)

    # Convert frame to RGB for Moondream
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Wait for sufficient GPU memory (gpu_min_gb threshold)
    gpu_min_gb = 3000
    max_wait_time = 30  # Maximum wait time in seconds
    wait_start = time.time()
    
    while torch.cuda.is_available():
        available_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**2
        if available_memory >= gpu_min_gb:  # min GPU MB threshold becore starting
            break
        if time.time() - wait_start > max_wait_time:
            print(f"Timeout waiting for GPU memory, aborting VLM analysis")
            stats_dict['VLM_availible'] = True
            return
        time.sleep(0.1)  # Check every 100ms
        
    try:
        print(f"Starting VLM (moondream) analysis for camera {camera_index}")
        if torch.cuda.is_available():
            print(f"GPU memory available: {available_memory:.1f}MB")
        
        # Convert to PIL Image
        pil_frame = Image.fromarray(frame_rgb)
        
        # Generate caption
        caption = IMG_PROCESSING_VLM.caption_image(pil_frame, length="normal")
        
        # Add to visual scene queue
        visual_scene_data = {
            'camera_index': camera_index,
            'timestamp': timestamp,
            'caption': caption,
            'analysis_time': time.time(),
            'formatted_time': datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
        }
        
        try:
            visual_scene_queue.put_nowait(visual_scene_data)
            print(f"Added VLM analysis to queue: {caption}")
        except:
            # Queue full, remove oldest and add new
            try:
                visual_scene_queue.get_nowait()
                visual_scene_queue.put_nowait(visual_scene_data)
                print(f"Queue full, replaced oldest analysis: {caption}")
            except:
                print(f"Failed to add analysis to queue: {caption}")
        
        print(f"VLM analysis complete: {caption}")
        
    except Exception as e:
        print(f"VLM analysis error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up and ensure VLM_availible is reset
        try:
            del IMG_PROCESSING_VLM
            # Explicit cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Force GPU memory cleanup
                print(f"GPU memory after cleanup: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        except:
            pass

        stats_dict['VLM_availible'] = True
        print(f"VLM analysis thread finished for camera {camera_index}")

def generate_img_frames(camera_index=None):
    """Generator function for video streaming from specific camera or all cameras"""
    frame_count = 0
    start_time = time.time()
    
    while True:
        try:
            # Get processed frame (blocking with timeout)
            frame_data = visual_cortex_queue.get(timeout=1.0)
            
            # Filter by camera index if specified
            if camera_index is not None and frame_data['camera_index'] != camera_index:
                continue
            
            processed_frame = frame_data['frame']
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            
            # Calculate streaming FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - start_time)
                if camera_index is not None:
                    stats_dict[f'stream_fps_{camera_index}'] = fps
                else:
                    stats_dict['stream_fps_all'] = fps
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except queue.Empty:
            # Send a placeholder frame if no processed frames available
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cam_text = f"Camera {camera_index}" if camera_index is not None else "All Cameras"
            cv2.putText(placeholder, f"No Signal - {cam_text}", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

"""
AUDIO
"""
def auditory_nerve_connection(device_index, audio_nerve_queue, stats_dict,sample_rate=16000, chunk_size=512):
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
            audio_nerve_queue.put_nowait(audio_data)
        except:
            # Queue is full, skip this frame to maintain real-time performance
            pass
        
        # Calculate and update auditory nerve stats
        frame_count += 1
        if frame_count % 100 == 0:  # Update every 100 frames
            current_time = time.time()
            fps = frame_count / (current_time - start_time)
            stats_dict[f'auditory_nerve_fps_{device_index}'] = fps
            stats_dict[f'last_auditory_nerve_{device_index}'] = current_time
    
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
            print(f"Audio stream started for device {device_index} with {chunk_size} sample chunks")
            # Keep the stream alive
            while True:
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Audio stream error for device {device_index}: {e}")
    finally:
        print(f"Auditory nerve process for device {device_index} stopped")

def auditory_cortex(audio_nerve_queue, audio_cortex_queue, stats_dict,audio_cortex_internal_queue_speech_audio,audio_cortex_internal_queue_transcription):
    """
    Worker process for processing audio frames (auditory cortex function)
    Optimized for real-time streaming with VADIterator and minimal buffering.
    Now includes speech-to-text transcription.
    """
    print("Started auditory cortex process")
    frame_count = 0
    start_time = time.time()
    process_timestamp = time.time()
    # Pre-roll buffer for capturing speech beginnings (audio right before the speech was detected)
    pre_speech_buffer = deque(maxlen=10)  # ~10 chunks of 32ms = ~320ms buffer
    speech_active = False
    speech_buffer = []
    full_speech = [] #holds numpy array of speech data
    min_silence_duration_ms = 1000  # Minimum silence to end speech, 1000 = 1 second
    
    # Initialize Silero VAD with VADIterator for streaming
    try:
        # Load Silero VAD model
        #torch.set_num_threads(1)  # Optimize for single-thread performance
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                    model='silero_vad',
                                    force_reload=False,
                                    onnx=False)
        
        # Extract VADIterator from utils
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
        
        # Initialize VADIterator for streaming
        vad_iterator = VADIterator(
            model,
            threshold=0.5,           # Speech probability threshold
            sampling_rate=16000,     # Must match audio sample rate 16hz
            min_silence_duration_ms=min_silence_duration_ms,  # Minimum silence to end speech
            speech_pad_ms=30         # Padding around speech segments
        )
        
        print("Silero VAD with VADIterator loaded successfully")
        
    except Exception as e:
        print(f"Error loading Silero VAD: {e}")
        return
    
    try:
        while True:
            
            try:
                # Get audio frame (blocking with timeout) from the nerve
                audio_data = audio_nerve_queue.get(timeout=1.0)
                
                # Process immediately - no accumulation needed for 512-sample chunks
                audio_chunk = audio_data['audio_frame']
                
                # Ensure we have exactly 512 samples for consistent processing
                if len(audio_chunk) != 512:
                    continue
                
                # Convert to tensor for Silero VAD
                audio_tensor = torch.from_numpy(audio_chunk)
                
                # Add to pre-speech buffer
                pre_speech_buffer.append(audio_chunk)
                
                # Run VADIterator - this handles streaming state internally
                try:
                    speech_dict = vad_iterator(audio_tensor, return_seconds=True)
                    
                    # Process VAD results
                    if speech_dict:
                        # Speech event detected (start or end)
                        if 'start' in speech_dict:
                            # Speech start detected
                            logging.info(f"Speech START detected at {speech_dict['start']:.3f}s")
                            speech_active = True
                            # Add pre-roll buffer to speech
                            speech_buffer = list(pre_speech_buffer)
                                                        
                        if 'end' in speech_dict:
                            # Speech end detected
                            logging.info(f"Speech END detected at {speech_dict['end']:.3f}s")
                            speech_active = False                               

                    if speech_active or (not speech_active and len(full_speech)>0):
                        #add to our audio with speech buffer
                        speech_buffer.append(audio_chunk)
                        # Concatenate all speech audio chunks
                        full_speech = np.concatenate(speech_buffer)
                        try :
                            #when speech_active is False, that will indicate it's our final chunk with speech audio
                            audio_cortex_internal_queue_speech_audio.put_nowait((full_speech.copy(),speech_active))
                        except queue.Full:
                            print("internal_queue_speech_audio FULL - will clear queue and put data.")
                            try :
                                # Drain the queue
                                while True:
                                    try:
                                        trash = audio_cortex_internal_queue_speech_audio.get_nowait()
                                    except queue.Empty:
                                        break
                                #try again to put our data in
                                audio_cortex_internal_queue_speech_audio.put_nowait((full_speech.copy(),speech_active))
                            except queue.Full:
                                print("Tried making space but internal_queue_speech_audio still FULL")

                        if not speech_active:
                                # Clear speech buffers after sending final audio for transcription
                                speech_buffer = []
                                full_speech = []

                     

                    transcription = {'transcription':"",'final_transcript':False}
                    try:
                        transcription = audio_cortex_internal_queue_transcription.get_nowait()#don't block
                    except queue.Empty:
                        logging.debug("No data in Transcription")

                    #Update The Audio Cortex Output Queue
                    processed_data = {'device_index': audio_data['device_index'],
                            'audio_data': full_speech,
                            'speech_detected': speech_active,
                            'final_transcript': transcription.get('final_transcript',""),
                            'transcription': transcription.get('transcription',""),
                            'capture_timestamp': audio_data['capture_timestamp'],
                            'formatted_time': datetime.fromtimestamp(audio_data['capture_timestamp']).strftime('%H:%M:%S'),
                            'auditory_cortex_timestamp': process_timestamp,
                            'sample_rate': audio_data['sample_rate'],
                            'duration': len(full_speech) / audio_data['sample_rate']}
                    
                    # Put processed audio in output queue
                    try:
                        audio_cortex_queue.put_nowait(processed_data)#-This will drop frame if queue is full
                    except:
                        # Queue is full, try to remove oldest and add new
                        try:
                            logging.debug("audio_cortex_queue FULL - Try and consume faster")
                        except:
                            pass


                except Exception as vad_error:
                    print(f"VAD processing error: {vad_error}")
                    continue
                
                # Calculate auditory cortex processing stats
                frame_count += 1
                if frame_count % 50 == 0:  # Update every 50 frames (~1.6 seconds)
                    fps = frame_count / (time.time() - start_time)
                    stats_dict['auditory_cortex_fps'] = fps
                    stats_dict['last_auditory_cortex'] = time.time()
                    
            except queue.Empty:
                continue
                
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

def auditory_cortex_worker_stt(internal_queue_speech_audio, internal_queue_transcription):
    """
    This is a multiprocess worker that will process audio chunks known to have speech.
    Uses LIFO - only processes the most recent audio data, skip and remove older data tasks from queue.
    Uses sliding window approach to build up transcript incrementally.
    """
    from stt import SpeechTranscriber
    import logging
    import queue
    import gc
    
    # Initialize Speech Transcriber
    try:
        speech_transcriber = SpeechTranscriber()
        print("Speech transcriber loaded successfully")
    except Exception as e:
        print(f"Failed to load speech transcriber: {e}")
        speech_transcriber = None
        return

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

    # State tracking
    working_transcript = ""
    last_processed_length = 0  # Track how much of the current speech we've processed
    current_speech_id = 0  # Track speech session changes
    
    # Parameters (at 16kHz sample rate)
    min_chunk_size = 80000  # 5 seconds minimum before starting transcription
    overlap_samples = 16000  # 1 second overlap for word boundary detection
    incremental_threshold = 48000  # 3 seconds of new audio before processing again
    
    while True:
        try:
            # LIFO Implementation: Get most recent data
            latest_task = None
            items_skipped = 0
            
            # Get first item (blocking)
            try:
                latest_task = internal_queue_speech_audio.get(timeout=1.0)
            except queue.Empty:
                continue
            
            if latest_task is None:  # Shutdown signal
                break
            
            # Drain queue to get the most recent item (LIFO behavior)
            while True:
                try:
                    newer_task = internal_queue_speech_audio.get_nowait()
                    if newer_task is None:
                        break
                    latest_task = newer_task
                    items_skipped += 1
                except queue.Empty:
                    break
            
            if items_skipped > 0:
                logging.debug(f"Skipped {items_skipped} older audio chunks (LIFO mode)")
            
            # Unpack the latest task
            full_audio_data, more_speech_coming = latest_task
            current_audio_length = len(full_audio_data)
            
            logging.debug(f"Audio length: {current_audio_length}, More coming: {more_speech_coming}, Last processed: {last_processed_length}")
            
            # Determine if we should process
            should_process = False
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
                    # Transcribe the audio chunk
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
                    transcript_data = {
                        'transcription': working_transcript,
                        'final_transcript': not more_speech_coming
                    }
                    
                    # Clear the transcription queue before putting new data (LIFO behavior)
                    while True:
                        try:
                            _ = internal_queue_transcription.get_nowait()
                            logging.debug("Cleared old transcription from queue")
                        except queue.Empty:
                            break
                    
                    # Send the latest transcription
                    try:
                        internal_queue_transcription.put_nowait(transcript_data)
                        logging.debug(f"Sent {'final' if not more_speech_coming else 'interim'} transcript: {len(working_transcript)} chars")
                    except queue.Full:
                        logging.error("Transcription queue still full after clearing!")
                    
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


"""
TEMPORAL LOBE
"""

"""
HTML Interface - UPDATED with visual scene display functionality
"""

@app.route('/')
def index():
    """Updated HTML page with audio controls and display"""
    return '''
    <!DOCTYPE html>
<html>
<head>
    <title>Temporal Lobe</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        
        /* Audio Section Styles */
        .audio-section { 
            margin: 20px 0; 
            padding: 20px; 
            border: 2px solid #ddd; 
            border-radius: 10px; 
            background: #f9f9f9;
        }
        .audio-controls { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px; 
            margin: 15px 0; 
        }
        .audio-device { 
            padding: 15px; 
            border-radius: 10px; 
            text-align: center; 
            border: 2px solid #ddd;
            transition: all 0.3s ease;
        }
        .audio-device.connected { 
            border-color: #4CAF50; 
            background: #e8f5e8; 
        }
        .audio-device.disconnected { 
            border-color: #f44336; 
            background: #ffebee; 
        }
        .speech-indicator {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 10px;
            transition: all 0.3s ease;
        }
        .speech-detected { background-color: #2196F3; }
        .speech-idle { background-color: white; border: 2px solid #ddd; }
        .audio-status {
            font-weight: bold;
            margin-top: 10px;
        }
        
        /* Speech-to-text styles */
        .speech-to-text-section {
            margin: 20px 0;
            padding: 20px;
            border: 2px solid #2196F3;
            border-radius: 10px;
            background: #f0f8ff;
        }
        .transcription-item {
            padding: 15px;
            margin: 10px 0;
            background: #ffffff;
            border-left: 4px solid #2196F3;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .transcription-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .transcription-time {
            color: #666;
            font-size: 0.9em;
            font-weight: bold;
        }
        .transcription-device {
            color: #2196F3;
            font-weight: bold;
            font-size: 0.9em;
        }
        .transcription-text {
            font-size: 1.1em;
            line-height: 1.4;
            color: #333;
            font-style: italic;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
        }
        .transcription-duration {
            color: #666;
            font-size: 0.8em;
            margin-top: 5px;
        }
        .no-transcription {
            text-align: center;
            color: #666;
            font-style: italic;
            padding: 20px;
        }
        
        /* Existing Camera Styles */
        .camera-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; margin: 20px 0; }
        .camera-box { border: 2px solid #ddd; border-radius: 10px; padding: 10px; text-align: center; }
        .camera-box.active { border-color: #4CAF50; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 20px 0; }
        .stat-box { padding: 10px; background: #f0f0f0; border-radius: 5px; text-align: center; }
        .controls { margin: 20px 0; text-align: center; }
        button { margin: 5px; padding: 10px 20px; font-size: 14px; cursor: pointer; }
        .start-btn { background: #4CAF50; color: white; border: none; border-radius: 5px; }
        .stop-btn { background: #f44336; color: white; border: none; border-radius: 5px; }
        .visual-scene { margin: 20px 0; padding: 20px; border: 2px solid #ddd; border-radius: 10px; }
        .scene-item { padding: 10px; margin: 10px 0; background: #f9f9f9; border-left: 4px solid #4CAF50; }
        .scene-time { color: #666; font-size: 0.9em; }
        .scene-camera { color: #333; font-weight: bold; }
        .scene-caption { margin-top: 5px; font-style: italic; }
        .vlm-status { padding: 10px; margin: 10px 0; text-align: center; border-radius: 5px; }
        .vlm-busy { background: #ffeb3b; }
        .vlm-idle { background: #e8f5e8; }
    </style>
</head>
<body>
    <div class="container">
        <h1>TEMPORAL LOBE</h1>

        <!-- Stats -->
        <div class="stats" id="stats-container">
            <!-- Stats will be populated by JavaScript -->
        </div>

        <!-- Audio Section -->
        <div class="audio-section">
            <h3>Auditory Nerves</h3>
            <div class="audio-controls">
                <div class="audio-device" id="audio-device-0">
                    <h4>Microphone 0</h4>
                    <div>
                        <span class="speech-indicator speech-idle" id="speech-indicator-0"></span>
                        <span id="speech-status-0">No Speech</span>
                    </div>
                    <div class="audio-status" id="audio-connection-0">Disconnected</div>
                    <button class="start-btn" onclick="startAudioDevice(0)">Start Audio</button>
                    <button class="stop-btn" onclick="stopAudioDevice(0)">Stop Audio</button>
                </div>
                <div class="audio-device" id="audio-device-1">
                    <h4>Microphone 1</h4>
                    <div>
                        <span class="speech-indicator speech-idle" id="speech-indicator-1"></span>
                        <span id="speech-status-1">No Speech</span>
                    </div>
                    <div class="audio-status" id="audio-connection-1">Disconnected</div>
                    <button class="start-btn" onclick="startAudioDevice(1)">Start Audio</button>
                    <button class="stop-btn" onclick="stopAudioDevice(1)">Stop Audio</button>
                </div>
            </div>
        </div>

        
        
        <h3>Optic Nerves</h3>
        <!-- Camera Controls -->
        <div class="controls">
            <button class="start-btn" onclick="startCamera(0)">Start Optic Nerve 0</button>
            <button class="start-btn" onclick="startCamera(1)">Start Optic Nerve 1</button>
            <button class="start-btn" onclick="startCamera(2)">Start Optic Nerve 2</button>
            <button class="stop-btn" onclick="stopCamera(0)">Stop Optic Nerve 0</button>
            <button class="stop-btn" onclick="stopCamera(1)">Stop Optic Nerve 1</button>
            <button class="stop-btn" onclick="stopCamera(2)">Stop Optic Nerve 2</button>
            <button class="stop-btn" onclick="stopAll()">Stop All</button>
        </div>
        
        <!-- Camera Grid -->
        <div class="camera-grid">
            <div class="camera-box" id="camera-0">
                <h3>Camera 0</h3>
                <img src="/video_feed/0" width="320" height="240" onerror="this.style.display='none'" onload="this.style.display='block'">
                <div id="status-0">Stopped</div>
            </div>
            <div class="camera-box" id="camera-1">
                <h3>Camera 1</h3>
                <img src="/video_feed/1" width="320" height="240" onerror="this.style.display='none'" onload="this.style.display='block'">
                <div id="status-1">Stopped</div>
            </div>
            <div class="camera-box" id="camera-2">
                <h3>Camera 2</h3>
                <img src="/video_feed/2" width="320" height="240" onerror="this.style.display='none'" onload="this.style.display='block'">
                <div id="status-2">Stopped</div>
            </div>
        </div>
        
        <!-- Speech-to-Text Section -->
        <div class="speech-to-text-section">
            <h3>Speech-to-Text Transcriptions</h3>
            <div id="transcription-display">
                <div class="no-transcription">
                    No speech transcriptions yet. Start an audio device and speak to see transcriptions appear here.
                </div>
            </div>
        </div>

        

        <!-- Visual Scene Analysis -->
        <div class="visual-scene">
            <h3>Visual Scene Analysis - VLM Output</h3>
            <div id="vlm-status" class="vlm-status vlm-idle">VLM Status: Idle</div>
            <div id="scene-analysis">
                <p>No scene analysis data available. Start a camera and detect a person to begin analysis.</p>
            </div>
        </div>

        <!-- Audio Scene Analysis -->
        <div class="visual-scene">
            <h3>Audio Scene Analysis</h3>
            <div id="audio-scene-analysis">
                <p>Not connected yet, will populate with audio description</p>
            </div>
        </div>
        
    </div>
    
    <script>
        // Audio control functions
        function startAudioDevice(index) {
            fetch(`/start_audio/${index}`)
                .then(response => response.json())
                .then(data => {
                    console.log(data.message);
                    const deviceElement = document.getElementById(`audio-device-${index}`);
                    deviceElement.classList.remove('disconnected');
                    deviceElement.classList.add('connected');
                    document.getElementById(`audio-connection-${index}`).textContent = 'Connected';
                })
                .catch(err => console.error('Audio start failed:', err));
        }
        
        function stopAudioDevice(index) {
            fetch(`/stop_audio/${index}`)
                .then(response => response.json())
                .then(data => {
                    console.log(data.message);
                    const deviceElement = document.getElementById(`audio-device-${index}`);
                    deviceElement.classList.remove('connected');
                    deviceElement.classList.add('disconnected');
                    document.getElementById(`audio-connection-${index}`).textContent = 'Disconnected';
                    // Reset speech indicator
                    document.getElementById(`speech-indicator-${index}`).className = 'speech-indicator speech-idle';
                    document.getElementById(`speech-status-${index}`).textContent = 'No Speech';
                })
                .catch(err => console.error('Audio stop failed:', err));
        }
        
        function updateAudioStatus() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    // Check for human speech 
                    const speechDetected = data.audio_cortex && data.audio_cortex.speech_detected;
                    
                    const speechDeviceIndex = data.audio_cortex && 
                                            data.audio_cortex.device_index !== undefined ? 
                                            data.audio_cortex.device_index : -1;
                    
                    // Update speech detection indicators
                    for (let i = 0; i < 2; i++) {
                        const indicator = document.getElementById(`speech-indicator-${i}`);
                        const status = document.getElementById(`speech-status-${i}`);
                        
                        // Only show speech detected for the specific device that detected it
                        if (speechDetected && speechDeviceIndex === i) {
                            indicator.className = 'speech-indicator speech-detected';
                            status.textContent = 'Speech Detected';
                        } else {
                            indicator.className = 'speech-indicator speech-idle';
                            status.textContent = 'No Speech';
                        }
                    }
                })
                .catch(err => console.error('Audio stats update failed:', err));
        }
        
        function updateTranscriptions() {
            fetch('/audio_scenes')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('transcription-display');
                    
                    if (data.audio_scenes && data.audio_scenes.length > 0) {
                        container.innerHTML = '';
                        
                        // Show most recent transcriptions first
                        data.audio_scenes.slice().reverse().forEach(scene => {
                            if (scene.transcription && scene.transcription.trim() !== '' && 
                                !scene.transcription.includes('[Audio too short') && 
                                !scene.transcription.includes('[Transcription failed]')) {
                                
                                const transcriptionDiv = document.createElement('div');
                                transcriptionDiv.className = 'transcription-item';
                                
                                const duration = scene.duration ? scene.duration.toFixed(1) : '0.0';
                                
                                transcriptionDiv.innerHTML = `
                                    <div class="transcription-header">
                                        <div class="transcription-time">${scene.formatted_time}</div>
                                        <div class="transcription-device">Microphone ${scene.device_index}</div>
                                    </div>
                                    <div class="transcription-text">"${scene.transcription}"</div>
                                    <div class="transcription-duration">Duration: ${duration}s</div>
                                `;
                                container.appendChild(transcriptionDiv);
                            }
                        });
                        
                        // If no valid transcriptions, show placeholder
                        if (container.children.length === 0) {
                            container.innerHTML = '<div class="no-transcription">Waiting for speech to transcribe...</div>';
                        }
                    } else {
                        container.innerHTML = '<div class="no-transcription">No speech transcriptions yet. Start an audio device and speak to see transcriptions appear here.</div>';
                    }
                })
                .catch(err => {
                    console.error('Transcription update failed:', err);
                    document.getElementById('transcription-display').innerHTML = 
                        '<div class="no-transcription">Error loading transcriptions.</div>';
                });
        }
        
        // Existing camera control functions
        function startCamera(index) {
            fetch(`/start/${index}`)
                .then(response => response.json())
                .then(data => {
                    console.log(data.message);
                    document.getElementById(`camera-${index}`).classList.add('active');
                    document.getElementById(`status-${index}`).textContent = 'Running';
                });
        }
        
        function stopCamera(index) {
            fetch(`/stop/${index}`)
                .then(response => response.json())
                .then(data => {
                    console.log(data.message);
                    document.getElementById(`camera-${index}`).classList.remove('active');
                    document.getElementById(`status-${index}`).textContent = 'Stopped';
                });
        }
        
        function stopAll() {
            fetch('/stop_all')
                .then(response => response.json())
                .then(data => {
                    console.log(data.message);
                    for (let i = 0; i < 3; i++) {
                        document.getElementById(`camera-${i}`).classList.remove('active');
                        document.getElementById(`status-${i}`).textContent = 'Stopped';
                    }
                    // Also stop audio devices
                    for (let i = 0; i < 2; i++) {
                        const deviceElement = document.getElementById(`audio-device-${i}`);
                        deviceElement.classList.remove('connected');
                        deviceElement.classList.add('disconnected');
                        document.getElementById(`audio-connection-${i}`).textContent = 'Disconnected';
                        document.getElementById(`speech-indicator-${i}`).className = 'speech-indicator speech-idle';
                        document.getElementById(`speech-status-${i}`).textContent = 'No Speech';
                    }
                });
        }
        
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('stats-container');
                    container.innerHTML = '';
                    
                    Object.keys(data).forEach(key => {
                        if (key !== 'timestamp') {
                            const statBox = document.createElement('div');
                            statBox.className = 'stat-box';
                            const value = typeof data[key] === 'number' ? data[key].toFixed(1) : data[key];
                            statBox.innerHTML = `<strong>${key.replace(/_/g, ' ').toUpperCase()}:</strong><br>${value}`;
                            container.appendChild(statBox);
                        }
                        
                    });
                })
                .catch(err => console.error('Stats update failed:', err));
        }
        
        function updateVisualScenes() {
            fetch('/visual_scenes')
                .then(response => response.json())
                .then(data => {
                    // Update VLM status
                    const statusDiv = document.getElementById('vlm-status');
                    if (data.VLM_availible) {
                        statusDiv.textContent = 'VLM Status: Analyzing...';
                        statusDiv.className = 'vlm-status vlm-busy';
                    } else {
                        statusDiv.textContent = 'VLM Status: Idle';
                        statusDiv.className = 'vlm-status vlm-idle';
                    }
                    
                    // Update scene analysis
                    const container = document.getElementById('scene-analysis');
                    if (data.scenes && data.scenes.length > 0) {
                        container.innerHTML = '';
                        // Show most recent scenes first
                        data.scenes.slice().reverse().forEach(scene => {
                            const sceneDiv = document.createElement('div');
                            sceneDiv.className = 'scene-item';
                            sceneDiv.innerHTML = `
                                <div class="scene-time">${scene.formatted_time}</div>
                                <div class="scene-camera">Camera ${scene.camera_index}</div>
                                <div class="scene-caption">"${scene.caption}"</div>
                            `;
                            container.appendChild(sceneDiv);
                        });
                    } else {
                        container.innerHTML = '<p>No scene analysis data available. Start a camera and detect a person to begin analysis.</p>';
                    }
                })
                .catch(err => console.error('Visual scenes update failed:', err));
        }
        
        function updateAudioScenes() {
            fetch('/audio_scenes')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('audio-scene-analysis');
                    if (data.audio_scenes && data.audio_scenes.length > 0) {
                        container.innerHTML = '';
                        data.audio_scenes.slice().reverse().forEach(scene => {
                            const sceneDiv = document.createElement('div');
                            sceneDiv.className = 'scene-item';
                            sceneDiv.innerHTML = `
                                <div class="scene-time">${scene.formatted_time}</div>
                                <div class="scene-camera">Microphone ${scene.device_index}</div>
                            `;
                            container.appendChild(sceneDiv);
                        });
                    } else {
                        container.innerHTML = '<p>Not connected yet, will populate with audio description</p>';
                    }
                })
                .catch(err => console.error('Audio scenes update failed:', err));
        }
        
        // Update intervals
        setInterval(updateStats, 1000);
        setInterval(updateVisualScenes, 2000);
        setInterval(updateAudioStatus, 500);  // Update audio status more frequently for real-time speech detection
        setInterval(updateTranscriptions, 1000);  // Update transcriptions every second
        setInterval(updateAudioScenes, 500);  // Update audio scenes
        
        // Initial updates
        updateStats();
        updateVisualScenes();
        updateAudioStatus();
        updateTranscriptions();
        updateAudioScenes();
    </script>
</body>
</html>
    '''

@app.route('/stats')
def stats():
    """API endpoint for performance statistics"""
    global AUDIO_SCENE
    current_stats = dict(stats_dict)
    current_stats['timestamp'] = datetime.now().isoformat()
    current_stats['optic_nerve_queue_size'] = optic_nerve_queue.qsize()

    #Visual queue stats
    current_stats['processed_queue_size'] = visual_cortex_queue.qsize()
    current_stats['visual_scene_queue_size'] = visual_scene_queue.qsize()
    current_stats['active_optic_nerves'] = list(optic_nerve_processes.keys())
    
    #Audio queue stats
    current_stats['audio_nerve_queue_size'] = audio_nerve_queue.qsize()
    current_stats['audio_cortex_queue_size'] = audio_cortex_queue.qsize()
    current_stats['active_auditory_nerves'] = list(auditory_nerve_processes.keys())

    #Object/Person awareness
    current_stats['audio_cortex'] = {}

    #We want to make sure we remove the raw data, only take fields that UI needs
    try:
        current_stats['audio_cortex'] = {
                                    'device_index': AUDIO_SCENE['device_index'],
                                    'speech_detected': AUDIO_SCENE['speech_detected']}
    except queue.Empty:
        print("The audio cortex queue is empty. Waiting for new data...")
        current_stats['audio_cortex'] = {
                                    'device_index': False,
                                    'speech_detected': False}

    

    return jsonify(current_stats)

#Visual and Audio full Stop
@app.route('/stop_all')
def stop_all():
    """Stop all optic nerves, visual cortex, auditory nerves, and auditory cortex processing"""
    global optic_nerve_processes, visual_cortex_processes, auditory_nerve_processes, auditory_cortex_processes
    
    # Stop all optic nerve processes
    for camera_index in list(optic_nerve_processes.keys()):
        stop_camera(camera_index)
    
    # Stop all auditory nerve processes
    for device_index in list(auditory_nerve_processes.keys()):
        stop_audio_device(device_index)
    
    # Stop visual cortex processes
    for process in visual_cortex_processes.values():
        process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
    
    # Stop auditory cortex processes
    for process in auditory_cortex_processes.values():
        process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
    
    visual_cortex_processes.clear()
    auditory_cortex_processes.clear()
    stats_dict.clear()
    
    return jsonify({
        'status': 'stopped', 
        'message': 'All optic nerves, visual cortex, auditory nerves, and auditory cortex stopped'
    })


"""
THESE ROUTES PROVIDE THE API ENDPOINTS FOR CONTROLLING THE OPTIC NERVES AND VISUAL CORTEX
"""
# ===== API ROUTES FOR VIDEO =====
@app.route('/video_feed')
@app.route('/video_feed/<int:camera_index>')
def video_feed(camera_index=None):
    """Video streaming route for specific camera or all cameras"""
    return Response(generate_img_frames(camera_index),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/visual_scenes')
def visual_scenes():
    """API endpoint for visual scene analysis data"""
    scenes = []
    
    # Get all available scenes from the queue without blocking
    while True:
        try:
            scene_data = visual_scene_queue.get_nowait()
            scenes.append(scene_data)
        except:
            break
    
    # Put them back in the queue (keeping last 3)
    scenes = scenes[-3:]  # Keep only last 3 analyses
    for scene_data in scenes:
        try:
            visual_scene_queue.put_nowait(scene_data)
        except:
            # Queue full, skip older ones
            break
    
    return jsonify({
        'scenes': scenes,
        'count': len(scenes),
        'VLM_availible': stats_dict.get('VLM_availible', False)
    })

@app.route('/start/<int:camera_index>')
def start_camera(camera_index):
    """Start optic nerve and visual cortex processing for specific camera"""
    global optic_nerve_processes, visual_cortex_processes
    
    if camera_index in optic_nerve_processes:
        return jsonify({
            'status': 'already_running', 
            'message': f'Optic nerve {camera_index} is already running'
        })
    
    # Start optic nerve process for this camera
    optic_nerve_process = mp.Process(
        target=optic_nerve_worker,
        args=(camera_index, optic_nerve_queue, stats_dict, FPS_TARGET)
    )
    optic_nerve_process.start()
    optic_nerve_processes[camera_index] = optic_nerve_process
    
    # Start visual cortex if not already running
    if not visual_cortex_processes:
        visual_cortex_process = mp.Process(
            target=visual_cortex_worker,
            args=(optic_nerve_queue, visual_cortex_queue, stats_dict, visual_scene_queue)
        )
        visual_cortex_process.start()
        visual_cortex_processes['main'] = visual_cortex_process

        
    
    return jsonify({
        'status': 'started', 
        'message': f'Optic nerve {camera_index} started'
    })

@app.route('/stop/<int:camera_index>')
def stop_camera(camera_index):
    """Stop optic nerve for specific camera"""
    global optic_nerve_processes
    
    if camera_index not in optic_nerve_processes:
        return jsonify({
            'status': 'not_running', 
            'message': f'Optic nerve {camera_index} is not running'
        })
    
    # Terminate optic nerve process
    process = optic_nerve_processes[camera_index]
    process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()
    
    del optic_nerve_processes[camera_index]
    
    # Clean up stats
    stats_keys_to_remove = [key for key in stats_dict.keys() 
                           if key.endswith(f'_{camera_index}')]
    for key in stats_keys_to_remove:
        del stats_dict[key]
    
    return jsonify({
        'status': 'stopped', 
        'message': f'Optic nerve {camera_index} stopped'
    })


"""
THESE ROUTES PROVIDE THE API ENDPOINTS FOR CONTROLLING THE AUDITORY NERVES AND AUDIO CORTEX
"""
# ===== API ROUTES FOR AUDIO =====
@app.route('/start_audio/<int:device_index>')
def start_audio_device(device_index):
    """Start auditory nerve and auditory cortex processing for specific audio device"""
    global auditory_nerve_processes, auditory_cortex_processes
    
    if device_index in auditory_nerve_processes:
        return jsonify({
            'status': 'already_running', 
            'message': f'Auditory nerve {device_index} is already running'
        })
    
    #Start auditory nerve process for this device. This will capture and supply raw audio data to audio_nerve_queue
    auditory_nerve_process = mp.Process(
        target=auditory_nerve_connection,
        args=(device_index, audio_nerve_queue, stats_dict)
    )
    auditory_nerve_process.start()
    auditory_nerve_processes[device_index] = auditory_nerve_process
    
    #Start auditory cortex if not already running, will read from audio_nerve_queue and provide analysis to audio_cortex_queue.
    #Single instance shared by all nerves
    if not auditory_cortex_processes:
        #Primary
        auditory_cortex_process = mp.Process(
            target=auditory_cortex,
            args=(audio_nerve_queue, audio_cortex_queue, stats_dict,audio_cortex_internal_queue_speech_audio,audio_cortex_internal_queue_transcription)
        )
        auditory_cortex_process.start()
        auditory_cortex_processes['main'] = auditory_cortex_process

        #Specific speech to text process
        stt_worker = mp.Process(
            target=auditory_cortex_worker_stt,
            args=(audio_cortex_internal_queue_speech_audio,audio_cortex_internal_queue_transcription)
        )
        stt_worker.start()
        auditory_cortex_processes['stt'] = stt_worker

    
    return jsonify({
        'status': 'started', 
        'message': f'Auditory nerve {device_index} started'
    })

@app.route('/stop_audio/<int:device_index>')
def stop_audio_device(device_index):
    """Stop auditory nerve for specific audio device"""
    global auditory_nerve_processes
    
    if device_index not in auditory_nerve_processes:
        return jsonify({
            'status': 'not_running', 
            'message': f'Auditory nerve {device_index} is not running'
        })
    
    # Terminate auditory nerve process
    process = auditory_nerve_processes[device_index]
    process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()
    
    del auditory_nerve_processes[device_index]
    
    # Clean up stats
    stats_keys_to_remove = [key for key in stats_dict.keys() 
                           if key.endswith(f'_audio_{device_index}')]
    for key in stats_keys_to_remove:
        del stats_dict[key]
    
    return jsonify({
        'status': 'stopped', 
        'message': f'Auditory nerve {device_index} stopped'
    })


#global variable for transcription history
TRANSCRIPTION_HISTORY = []
MAX_TRANSCRIPTION_HISTORY = 3
@app.route('/audio_scenes')
def audio_scenes():
    """API endpoint for audio scene analysis data with transcriptions"""
    global AUDIO_SCENE, TRANSCRIPTION_HISTORY
    while True:
        #get all of the audio data frames from the audio cortex
        try:
            current_scene = audio_cortex_queue.get_nowait()#get no wait to drain the queue
            # Only add to history if we have a valid transcription
            if (current_scene.get('transcription', "").strip() != ""):
                
                scene_data = {
                    'device_index': current_scene['device_index'],
                    'timestamp': current_scene['auditory_cortex_timestamp'],
                    'transcription': current_scene.get('transcription', ''),
                    'formatted_time': current_scene['formatted_time'],
                    'duration': current_scene.get('duration', 0)
                }
                
                # Add to history and keep only recent ones
                TRANSCRIPTION_HISTORY.append(scene_data)
                TRANSCRIPTION_HISTORY = TRANSCRIPTION_HISTORY[-MAX_TRANSCRIPTION_HISTORY:]
                
            AUDIO_SCENE = current_scene
        except Exception as e:
            logging.debug(e)#should be No more data exception since we got it all from the queue
            break

    return jsonify({
        'audio_scenes': TRANSCRIPTION_HISTORY,
        'count': len(TRANSCRIPTION_HISTORY)
    })

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print("\nShutting down...")
    stop_all()
    sys.exit(0)

if __name__ == '__main__':
    import signal

    # Global multiprocessing queues and managers MUST go here because of issues with Windows
    manager = mp.Manager()
    
    # Existing visual queues
    optic_nerve_queue = manager.Queue(maxsize=QUEUE_SIZE * 4)
    visual_cortex_queue = manager.Queue(maxsize=QUEUE_SIZE * 4)
    visual_scene_queue = manager.Queue(maxsize=10)
    
    #Audio queues
    audio_nerve_queue = manager.Queue(maxsize=QUEUE_SIZE * 4)
    audio_cortex_queue = manager.Queue(maxsize=QUEUE_SIZE * 4)
    # internally used by Audio Cortex to hold speech audio that needs transcription
    audio_cortex_internal_queue_speech_audio = manager.Queue(maxsize=100)  #100 chunks of 32ms = ~3200ms (3.2 second) buffer
    # internally used by Audio Cortex to hold speech to text results (transcriptions)
    audio_cortex_internal_queue_transcription = manager.Queue(maxsize=5)
     
    stats_dict = manager.dict()

   
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    # Handle shutdown signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("Starting Flask server with multi-camera OpenCV streaming...")
        print("Visit http://localhost:5000 to view the streams")
        print("API endpoints:")
        print("  GET /start/<camera_index> - Start specific optic nerve")
        print("  GET /stop/<camera_index>  - Stop specific optic nerve")
        print("  GET /stop_all             - Stop all optic nerves and visual cortex")
        print("  GET /stats                - Get performance statistics")
        print("  GET /visual_scenes        - Get VLM analysis results")
        print("  GET /video_feed/<camera_index> - Stream from specific camera")
        print("  GET /video_feed           - Stream from visual cortex (all cameras)")
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_all()
        print("Cleanup complete")

