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
#Temporal Lobe
temporal_lobe_processes = {}
#=====================================


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

def visual_cortex_worker(optic_nerve_queue, visual_cortex_queue, stats_dict, visual_cortex_queue_img_display,visual_cortex_internal_queue_vlm):
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
                        logging.debug(f"Visual cortex: Discarded {frames_discarded} older frames, processing most recent")
                    
                    # Process the frame
                    processed_frame, person_detected = visual_cortex_process_img(frame_data['frame'], 
                                                  frame_data['camera_index'],
                                                  frame_data['capture_timestamp'],
                                                  IMG_DETECTION_CNN)
                    
                    # If person detected and VLM not busy, start async analysis
                    if person_detected and VLM_availible_local:
                        logging.debug(f"Starting VLM analysis - person detected and VLM not busy")
                        
                        # Set busy states
                        VLM_availible_local = False
                        stats_dict['VLM_availible'] = VLM_availible_local

                        """
                        Why Threading not Multiprocessing
                        Although MP would be faster, it leaves the VLM loaded in the GPU. Using threading so that we can more easily 
                        repurpose gpu memeory as needed. The VLM uses a lot of GPU space ~4GB.  
                        This is different than the Audio cortex speech to text since that is a much smaller model
                        """
                        # Start VLM Moondream analysis in separate thread
                        VLM_thread = threading.Thread(
                            target=visual_cortex_process_vlm_analysis_thread,
                            args=(frame_data['frame'], 
                                  frame_data['camera_index'], 
                                  frame_data['capture_timestamp'],
                                  person_detected, 
                                  visual_cortex_internal_queue_vlm),
                                  daemon = True)  

                        VLM_thread.start()

                    
                    # Create processed frame data
                    process_timestamp = time.time()
                    vlm_data = {
                            'vlm_ready':VLM_availible_local,
                            'camera_index': frame_data['camera_index'],
                            'timestamp': process_timestamp,
                            'caption': None,
                            'analysis_time': process_timestamp,
                            'formatted_time': datetime.fromtimestamp(process_timestamp).strftime('%H:%M:%S')
                        }
                    try:
                        vlm_data = visual_cortex_internal_queue_vlm.get_nowait()#don't block
                    except queue.Empty:
                        logging.debug("No data in vlm")

                    #Update our VLM Ready flag
                    VLM_availible_local = vlm_data['vlm_ready']

                    processed_data = {
                        'camera_index': frame_data['camera_index'],
                        'formatted_time': datetime.fromtimestamp(process_timestamp).strftime('%H:%M:%S'),
                        'person_detected':person_detected,
                        'frame':processed_frame,  
                        'capture_timestamp': frame_data['capture_timestamp'],
                        'visual_cortex_timestamp': process_timestamp,
                        'vlm':vlm_data
                    }
                    
                    #Queue used for the GUI
                    try:
                        visual_cortex_queue_img_display.put_nowait(processed_data.copy())
                    except:
                        print("\nvisual_cortex_queue_img_display queue full - Consume faster\n")
                        pass
                    
                    # Put processed frame in output queue
                    try:
                        visual_cortex_queue.put_nowait(processed_data.copy())
                    except:
                        # Queue is full, try to remove oldest and add new
                        try:
                            visual_cortex_queue.get_nowait()
                            visual_cortex_queue.put_nowait(processed_data.copy())
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
        
            #slow the loop a bit to help CPU
            time.sleep(0.001)
                
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
            logging.debug(f"Detected {class_name} with confidence {round(score.item(), 3)} at location {box_rounded}")

    # Draw annotations (filter for person only)
    frame = IMG_DETECTION_CNN.draw_detections(frame, results, filter_person_only=True)

    # Explicit cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Force GPU memory cleanup
        logging.debug(f"GPU memory after cleanup: {torch.cuda.memory_allocated()/1024**2:.1f}MB")

    return frame, person_detected

def visual_cortex_process_vlm_analysis_thread(frame, camera_index, timestamp, person_detected,visual_cortex_internal_queue_vlm):
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
    
    #push empty data to queue to indicate VLM is about to go to work 
    visual_scene_data = {
            'vlm_ready':False,
            'person_detected':person_detected,
            'camera_index': camera_index,
            'timestamp': timestamp,
            'caption': None,
            'analysis_time': time.time(),
            'formatted_time': datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
        }
    
    try:
        visual_cortex_internal_queue_vlm.put_nowait(visual_scene_data)
    except:
        # Queue full, don't bother
        logging.error(f"visual_cortex_internal_queue_vlm FULL, skipping blank data")

    try:
        logging.debug(f"Starting VLM (moondream) analysis for camera {camera_index}")
        if torch.cuda.is_available():
            logging.debug(f"GPU memory available: {available_memory:.1f}MB")
        
        # Convert to PIL Image
        pil_frame = Image.fromarray(frame_rgb)
        
        # Generate caption
        caption = IMG_PROCESSING_VLM.caption_image(pil_frame, length="normal")
        
        # Add to visual scene queue
        visual_scene_data = {
            'vlm_ready':True,
            'camera_index': camera_index,
            'timestamp': timestamp,
            'caption': caption,
            'analysis_time': time.time(),
            'formatted_time': datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
        }
        
        
        
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
            
            #Send Update
            try:
                visual_cortex_internal_queue_vlm.put_nowait(visual_scene_data)
                print(f"Added VLM analysis to queue: {caption}")
            except:
                # Queue full, remove oldest and add new. Here it is important to put our analysis
                try:
                    visual_cortex_internal_queue_vlm.get_nowait()
                    visual_cortex_internal_queue_vlm.put_nowait(visual_scene_data)
                    print(f"Queue full, replaced oldest analysis: {caption}")
                except:
                    print(f"Failed to add analysis to queue: {caption}")
            
            print(f"VLM analysis complete: {caption}")

        except:
            print(f"VLM analysis thread error for camera {camera_index}")
            pass

def generate_img_frames(camera_index=None):
    """Generator function for video streaming from specific camera or all cameras"""
    frame_count = 0
    start_time = time.time()
    global visual_cortex_queue_img_display
    while True:
        try:
            # Get processed frame (blocking with timeout)
            frame_data = visual_cortex_queue_img_display.get(timeout=1)
            
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
                time.sleep(0.001)
                
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
                        

                    print(f"\n\n{transcription}\n\n")
                    process_timestamp = time.time()
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

            # Small delay to prevent excessive CPU usage
            time.sleep(0.001)  # 1ms delay
                
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
                    
                    # Send the latest transcription
                    print(transcript_data['transcription'])
                    try:
                        internal_queue_transcription.put_nowait(transcript_data)
                        logging.debug(f"Sent {'final' if not more_speech_coming else 'interim'} transcript: {len(working_transcript)} chars")
                    except queue.Full:
                        logging.debug("Transcription queue still full start clearing!")
                        # Clear the transcription queue before putting new data (LIFO behavior)
                        while True:
                            try:
                                _ = internal_queue_transcription.get_nowait()
                                logging.debug("Cleared old transcription from queue")
                            except queue.Empty:
                                internal_queue_transcription.put_nowait(transcript_data)
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


"""
TEMPORAL LOBE
"""
def temporal_lobe(audio_cortex_queue, visual_cortex_queue, temporal_lobe_queue):
    """
    Gets Data from the audio and visual cortex and puts on the temporal_lobe_queue for digestion
    """
    process_timestamp = time.time()
    temporal_lobe_data_template = {
        'formatted_time': datetime.fromtimestamp(process_timestamp).strftime('%H:%M:%S'),
        'temporal_lobe_timestamp': process_timestamp,
        'audio': {
            'device_index': None,
            'audio_data': None,
            'speech_detected': None,
            'final_transcript': None,
            'transcription': None,
            'capture_timestamp': None,
            'formatted_time': None,
            'auditory_cortex_timestamp': None,
            'sample_rate': None,
            'duration': None
        },
        'visual': {
            'camera_index': None,
            'timestamp': None,
            'person_detected': None,
            'analysis_time': None,
            'formatted_time': None,
            'vlm': {
                'vlm_ready': True,
                'camera_index': None,
                'timestamp': process_timestamp,
                'caption': None,
                'analysis_time': process_timestamp,
                'formatted_time': datetime.fromtimestamp(process_timestamp).strftime('%H:%M:%S')
            }
        }
    }
    
    while True:
        tl_timestamp = time.time()
        
        try:
            # Get latest audio data (LIFO Implementation)
            latest_audio_cortex_data = temporal_lobe_data_template['audio'].copy()
            audio_cortex_items_skipped = 0
            
            try:
                latest_audio_cortex_data = audio_cortex_queue.get_nowait()
                
                # Drain queue to get the most recent item (LIFO behavior)
                while True:
                    try:
                        newer_task = audio_cortex_queue.get_nowait()
                        if newer_task is None:
                            break
                        latest_audio_cortex_data = newer_task
                        audio_cortex_items_skipped += 1
                    except queue.Empty:
                        break
                        
                logging.debug(f"Temporal Lobe: Got audio data, skipped {audio_cortex_items_skipped} items")
                
            except queue.Empty:
                logging.debug("Temporal Lobe: audio_cortex_queue EMPTY")
                # Don't continue - process with None audio data
                pass
            
            # Get latest visual data (LIFO Implementation) 
            latest_visual_cortex_data = temporal_lobe_data_template['visual'].copy()
            visual_cortex_items_skipped = 0
            
            try:
                latest_visual_cortex_data = visual_cortex_queue.get_nowait() 
                
                # Drain queue to get the most recent item (LIFO behavior)
                while True:
                    try:
                        newer_task = visual_cortex_queue.get_nowait()  # FIXED LINE
                        if newer_task is None:
                            break
                        latest_visual_cortex_data = newer_task
                        visual_cortex_items_skipped += 1
                    except queue.Empty:
                        break
                        
                logging.debug(f"Temporal Lobe: Got visual data, skipped {visual_cortex_items_skipped} items")
                
            except queue.Empty:
                logging.debug("Temporal Lobe: visual_cortex_queue EMPTY")
                # Don't continue - process with None visual data
                pass
            
            # Create output data (process even if one or both inputs are None)
            tl_out = temporal_lobe_data_template.copy()  # Make a copy to avoid modifying template
            tl_out.update({
                'formatted_time': datetime.fromtimestamp(tl_timestamp).strftime('%H:%M:%S'),
                'temporal_lobe_timestamp': tl_timestamp,
                'audio': latest_audio_cortex_data,
                'visual': latest_visual_cortex_data
            })
            
            # Put data on temporal lobe queue
            try:
                temporal_lobe_queue.put_nowait(tl_out)  # This will drop frame if queue is full
                logging.debug("Temporal Lobe Data Added")
                logging.debug(f"Audio items skipped: {audio_cortex_items_skipped}, Visual items skipped: {visual_cortex_items_skipped}")
                
            except queue.Full:
                logging.debug("temporal_lobe_queue FULL - Consumer needs to process faster")
                
                
        except Exception as e:
            logging.error(f"Temporal Lobe Data Queue Read Error: {e}")
            # Continue processing instead of crashing
            
        #print(tl_out)
        # Small delay to prevent excessive CPU usage
        time.sleep(0.001)  # 1ms delay
 
# Global state to hold the latest temporal lobe data
TEMPORAL_LOBE_STATE = {
    'current_data': None,
    'visual_scenes': [],
    'audio_scenes': [],
    'last_update': None
}
MAX_TRANSCRIPTION_HISTORY = 10
def temporal_lobe_state():
    """
    Middleware function that reads from temporal_lobe_queue and maintains state
    for multiple consumers to access the same data
    """
    global TEMPORAL_LOBE_STATE, MAX_TRANSCRIPTION_HISTORY, temporal_lobe_queue
    
    # First, collect all frames from the queue
    frames = []
    while True:
        try:
            scene_data = temporal_lobe_queue.get_nowait()  # Non-blocking get
            frames.append(scene_data)
        except queue.Empty:
            break
        except Exception as e:
            logging.error(f"Error reading from temporal lobe queue: {e}")
            break
    
    if not frames:
        return False  # No new data processed
    
    # Process frames in LIFO order (most recent first)
    frames.reverse()  # Now most recent frame is at index 0
    
    new_data_processed = False
    visual_updated = False
    audio_updated = False
    
    try:
        # Update current state with the most recent frame
        TEMPORAL_LOBE_STATE['current_data'] = frames[0]
        TEMPORAL_LOBE_STATE['last_update'] = time.time()
        
        # Process visual data - find most recent frame with visual data
        for scene_data in frames:
            if not visual_updated and scene_data.get('visual', {}).get('vlm', {}).get('caption') is not None:
                TEMPORAL_LOBE_STATE['visual_scenes'].append(scene_data['visual']['vlm'])
                # Keep only recent visual scenes
                TEMPORAL_LOBE_STATE['visual_scenes'] = TEMPORAL_LOBE_STATE['visual_scenes'][-MAX_TRANSCRIPTION_HISTORY:]
                visual_updated = True
                new_data_processed = True
        
        # Process audio data - find most recent frame with audio data
        for scene_data in frames:
            if not audio_updated and scene_data.get('audio', {}).get('transcription'):
                transcription = scene_data.get('audio', {}).get('transcription', "").strip()
                if transcription:
                    audio_scene = {
                        'device_index': scene_data['audio']['device_index'],
                        'speech_detected': scene_data['audio']['speech_detected'],
                        'timestamp': scene_data['audio']['auditory_cortex_timestamp'],
                        'transcription': transcription,
                        'formatted_time': scene_data['audio']['formatted_time'],
                        'duration': scene_data['audio'].get('duration', 0)
                    }
                    TEMPORAL_LOBE_STATE['audio_scenes'].append(audio_scene)
                    # Keep only recent audio scenes
                    TEMPORAL_LOBE_STATE['audio_scenes'] = TEMPORAL_LOBE_STATE['audio_scenes'][-MAX_TRANSCRIPTION_HISTORY:]
                    audio_updated = True
                    new_data_processed = True
        
        #print(f"\n\n{TEMPORAL_LOBE_STATE}\n\n")
        
    except Exception as e:
        logging.error(f"Error processing temporal lobe data: {e}")
    
    return new_data_processed

#Background thread to continuously update state
def temporal_lobe_state_updater():
    """Background thread function to continuously update temporal lobe state"""
    while True:
        try:
            success = temporal_lobe_state()
            if not success:
                logging.debug("Failed to update temoral lobe state")
            time.sleep(0.01)  # 10ms update interval
        except Exception as e:
            logging.error(f"Temporal lobe state updater error: {e}")
            time.sleep(0.1)  # Longer sleep on error

"""
HTML Interface - UPDATED with visual scene display functionality
"""

@app.route('/')
def index():
    """Updated HTML page with audio controls and display"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Processing Interface</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Crimson+Text:ital,wght@0,400;0,600;1,400&display=swap');
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            background: linear-gradient(135deg, #2c1810 0%, #1a0f0a 50%, #0f0805 100%);
            min-height: 100vh;
            font-family: 'Crimson Text', serif;
            color: #d4af37;
            overflow-x: auto;
        }

        .main-container {
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }

        .section {
            background: linear-gradient(145deg, rgba(139, 69, 19, 0.3), rgba(101, 67, 33, 0.2));
            border: 3px solid #8b4513;
            border-radius: 15px;
            margin: 20px 0;
            padding: 20px;
            box-shadow: 
                0 0 20px rgba(212, 175, 55, 0.3),
                inset 0 0 15px rgba(139, 69, 19, 0.2);
            position: relative;
        }

        .section::before {
            content: '';
            position: absolute;
            top: -5px;
            left: -5px;
            right: -5px;
            bottom: -5px;
            background: linear-gradient(45deg, #d4af37, #8b4513, #d4af37, #8b4513);
            border-radius: 20px;
            z-index: -1;
            opacity: 0.7;
        }

        .section-title {
            font-family: 'Cinzel', serif;
            font-size: 1.8rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 20px;
            color: #d4af37;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
            letter-spacing: 2px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .stat-card {
            background: rgba(139, 69, 19, 0.4);
            border: 2px solid #8b4513;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.5);
        }

        .stat-label {
            font-family: 'Cinzel', serif;
            font-size: 0.9rem;
            color: #cd853f;
            margin-bottom: 5px;
        }

        .stat-value {
            font-size: 1.2rem;
            font-weight: 600;
            color: #d4af37;
        }

        .device-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }

        .device-card {
            background: linear-gradient(145deg, rgba(101, 67, 33, 0.6), rgba(139, 69, 19, 0.4));
            border: 2px solid #8b4513;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 
                0 5px 15px rgba(0, 0, 0, 0.4),
                inset 0 0 10px rgba(212, 175, 55, 0.1);
        }

        .device-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .device-title {
            font-family: 'Cinzel', serif;
            font-size: 1.2rem;
            font-weight: 600;
            color: #d4af37;
        }

        .status-indicator {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            border: 2px solid #8b4513;
            transition: all 0.3s ease;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
        }

        .status-active { background: radial-gradient(circle, #00ff00, #008000); box-shadow: 0 0 15px #00ff00; }
        .status-inactive { background: radial-gradient(circle, #666, #333); }
        .status-speech { background: radial-gradient(circle, #ff4500, #ff6347); box-shadow: 0 0 15px #ff4500; }
        .status-person { background: radial-gradient(circle, #1e90ff, #4169e1); box-shadow: 0 0 15px #1e90ff; }

        .steam-button {
            background: linear-gradient(145deg, #8b4513, #654321);
            border: 2px solid #d4af37;
            color: #d4af37;
            padding: 10px 20px;
            border-radius: 8px;
            font-family: 'Cinzel', serif;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .steam-button:hover {
            background: linear-gradient(145deg, #a0522d, #8b4513);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
            transform: translateY(-2px);
        }

        .steam-button:active {
            transform: translateY(0);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }

        .steam-button.active {
            background: linear-gradient(145deg, #228b22, #006400);
            border-color: #00ff00;
            color: #fff;
            box-shadow: 0 0 15px rgba(0, 255, 0, 0.3);
        }

        .transcript-container, .vlm-container {
            max-height: 200px;
            overflow-y: auto;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid #8b4513;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
        }

        .transcript-item, .vlm-item {
            background: rgba(139, 69, 19, 0.2);
            border-left: 4px solid #d4af37;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
        }

        .timestamp {
            font-size: 0.8rem;
            color: #cd853f;
            font-style: italic;
        }

        .transcript-text, .vlm-text {
            margin-top: 5px;
            line-height: 1.4;
        }

        .camera-feed {
            width: 100%;
            max-height: 200px;
            border: 2px solid #8b4513;
            border-radius: 8px;
            background: #000;
            object-fit: cover;
        }

        .legend {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 15px;
            font-size: 0.9rem;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(139, 69, 19, 0.3);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb {
            background: #8b4513;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #d4af37;
        }

        @media (max-width: 768px) {
            .device-grid {
                grid-template-columns: 1fr;
            }
            
            .stats-grid {
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            }
        }
    </style>
</head>
<body>
    <div class="main-container">
        <h1 style="text-align: center; font-family: 'Cinzel', serif; font-size: 2.5rem; margin-bottom: 30px; color: #d4af37; text-shadow: 3px 3px 6px rgba(0,0,0,0.8); letter-spacing: 3px;">
            ⚙️ NEURAL PROCESSING INTERFACE ⚙️
        </h1>

        <!-- Performance Statistics -->
        <div class="section">
            <div class="section-title">⚡ SYSTEM PERFORMANCE METRICS ⚡</div>
            <div class="stats-grid" id="statsGrid">
                <!-- Stats will be populated by JavaScript -->
            </div>
        </div>

        <!-- Audio Processing Section -->
        <div class="section">
            <div class="section-title">AUDITORY CORTEX CONTROL</div>
            <div class="device-grid">
                <div class="device-card" id="audio-0">
                    <div class="device-header">
                        <div class="device-title">Microphone 0</div>
                        <div class="status-indicator status-inactive" id="audio-status-0"></div>
                    </div>
                    <button class="steam-button" onclick="toggleAudio(0)" id="audio-btn-0">Activate</button>
                    <div class="legend">
                        <div class="legend-item">
                            <div class="status-indicator status-active"></div>
                            <span>Active</span>
                        </div>
                        <div class="legend-item">
                            <div class="status-indicator status-speech"></div>
                            <span>Speech Detected</span>
                        </div>
                    </div>
                </div>
                
                <div class="device-card" id="audio-1">
                    <div class="device-header">
                        <div class="device-title">Microphone 1</div>
                        <div class="status-indicator status-inactive" id="audio-status-1"></div>
                    </div>
                    <button class="steam-button" onclick="toggleAudio(1)" id="audio-btn-1">Activate</button>
                </div>
            </div>
            
            <div class="transcript-container" id="transcriptContainer">
                <h3 style="color: #d4af37; margin-bottom: 10px; font-family: 'Cinzel', serif;">Recent Transcriptions:</h3>
                <div id="transcriptsList">
                    <div class="transcript-item">
                        <div class="timestamp">Waiting for audio activation...</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Visual Processing Section -->
        <div class="section">
            <div class="section-title">VISUAL CORTEX CONTROL</div>
            <div class="device-grid">
                <div class="device-card" id="camera-0">
                    <div class="device-header">
                        <div class="device-title">Camera 0</div>
                        <div class="status-indicator status-inactive" id="camera-status-0"></div>
                    </div>
                    <button class="steam-button" onclick="toggleCamera(0)" id="camera-btn-0">Activate</button>
                    <img class="camera-feed" id="camera-feed-0" src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMjIyIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNiIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkNhbWVyYSBPZmZsaW5lPC90ZXh0Pjwvc3ZnPg==" alt="Camera 0">
                    <div class="legend">
                        <div class="legend-item">
                            <div class="status-indicator status-active"></div>
                            <span>Active</span>
                        </div>
                        <div class="legend-item">
                            <div class="status-indicator status-person"></div>
                            <span>Person Detected</span>
                        </div>
                    </div>
                </div>
                
                <div class="device-card" id="camera-1">
                    <div class="device-header">
                        <div class="device-title">Camera 1</div>
                        <div class="status-indicator status-inactive" id="camera-status-1"></div>
                    </div>
                    <button class="steam-button" onclick="toggleCamera(1)" id="camera-btn-1">Activate</button>
                    <img class="camera-feed" id="camera-feed-1" src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMjIyIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNiIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkNhbWVyYSBPZmZsaW5lPC90ZXh0Pjwvc3ZnPg==" alt="Camera 1">
                </div>
                
                <div class="device-card" id="camera-2">
                    <div class="device-header">
                        <div class="device-title">Camera 2</div>
                        <div class="status-indicator status-inactive" id="camera-status-2"></div>
                    </div>
                    <button class="steam-button" onclick="toggleCamera(2)" id="camera-btn-2">Activate</button>
                    <img class="camera-feed" id="camera-feed-2" src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMjIyIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNiIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkNhbWVyYSBPZmZsaW5lPC90ZXh0Pjwvc3ZnPg==" alt="Camera 2">
                </div>
            </div>
            
            <div class="vlm-container" id="vlmContainer">
                <h3 style="color: #d4af37; margin-bottom: 10px; font-family: 'Cinzel', serif;">Recent Visual Analysis:</h3>
                <div id="vlmList">
                    <div class="vlm-item">
                        <div class="timestamp">Waiting for visual activation...</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Global state tracking
        var deviceStates = {
            audio: {0: false, 1: false},
            camera: {0: false, 1: false, 2: false}
        };

        // XMLHttpRequest helper function
        function makeRequest(method, url, callback) {
            var xhr = new XMLHttpRequest();
            xhr.open(method, url, true);
            xhr.onreadystatechange = function() {
                if (xhr.readyState === 4) {
                    if (xhr.status === 200) {
                        try {
                            var data = JSON.parse(xhr.responseText);
                            callback(null, data);
                        } catch (e) {
                            callback(e, null);
                        }
                    } else {
                        callback(new Error('Request failed'), null);
                    }
                }
            };
            xhr.send();
        }

        // Audio control functions
        function toggleAudio(deviceIndex) {
            var isActive = deviceStates.audio[deviceIndex];
            var endpoint = isActive ? '/stop_audio/' + deviceIndex : '/start_audio/' + deviceIndex;
            
            makeRequest('GET', endpoint, function(error, result) {
                if (error) {
                    console.error('Audio toggle error:', error);
                    return;
                }
                
                console.log('Audio response:', result); // Debug logging
                
                if (result.status === 'started' || result.status === 'already_running') {
                    deviceStates.audio[deviceIndex] = true; // UPDATE STATE HERE
                    updateAudioUI(deviceIndex, true);
                } else if (result.status === 'stopped' || result.status === 'not_running') {
                    deviceStates.audio[deviceIndex] = false; // UPDATE STATE HERE
                    updateAudioUI(deviceIndex, false);
                }
            });
        }

        // Camera control functions
        function toggleCamera(deviceIndex) {
            var isActive = deviceStates.camera[deviceIndex];
            var endpoint = isActive ? '/stop/' + deviceIndex : '/start/' + deviceIndex;
            
            makeRequest('GET', endpoint, function(error, result) {
                if (error) {
                    console.error('Camera toggle error:', error);
                    return;
                }
                
                console.log('Camera response:', result); // Debug logging
                
                if (result.status === 'started' || result.status === 'already_running') {
                    deviceStates.camera[deviceIndex] = true; // UPDATE STATE HERE
                    updateCameraUI(deviceIndex, true);
                } else if (result.status === 'stopped' || result.status === 'not_running') {
                    deviceStates.camera[deviceIndex] = false; // UPDATE STATE HERE
                    updateCameraUI(deviceIndex, false);
                }
            });
        }

        // UI update functions
        function updateAudioUI(deviceIndex, isActive) {
            var statusEl = document.getElementById('audio-status-' + deviceIndex);
            var buttonEl = document.getElementById('audio-btn-' + deviceIndex);
            
            if (isActive) {
                statusEl.className = 'status-indicator status-active';
                buttonEl.textContent = 'Deactivate';
                buttonEl.classList.add('active');
            } else {
                statusEl.className = 'status-indicator status-inactive';
                buttonEl.textContent = 'Activate';
                buttonEl.classList.remove('active');
            }
        }

        function updateCameraUI(deviceIndex, isActive) {
            var statusEl = document.getElementById('camera-status-' + deviceIndex);
            var buttonEl = document.getElementById('camera-btn-' + deviceIndex);
            var feedEl = document.getElementById('camera-feed-' + deviceIndex);
            
            if (isActive) {
                statusEl.className = 'status-indicator status-active';
                buttonEl.textContent = 'Deactivate';
                buttonEl.classList.add('active');
                feedEl.src = '/video_feed/' + deviceIndex;
            } else {
                statusEl.className = 'status-indicator status-inactive';
                buttonEl.textContent = 'Activate';
                buttonEl.classList.remove('active');
                feedEl.src = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMjIyIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNiIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkNhbWVyYSBPZmZsaW5lPC90ZXh0Pjwvc3ZnPg==";
            }
        }

        // Data fetching and display functions
        function updateStats() {
            makeRequest('GET', '/stats', function(error, stats) {
                if (error) {
                    console.error('Stats update error:', error);
                    return;
                }
                
                var statsGrid = document.getElementById('statsGrid');
                statsGrid.innerHTML = '';
                
                // Key metrics to display
                var metricsToShow = [
                    { key: 'visual_cortex_fps', label: 'Visual Cortex FPS', format: function(val) { return val ? val.toFixed(1) : '0.0'; } },
                    { key: 'auditory_cortex_fps', label: 'Auditory Cortex FPS', format: function(val) { return val ? val.toFixed(1) : '0.0'; } },
                    { key: 'optic_nerve_queue_size', label: 'Optic Nerve Queue', format: function(val) { return val || '0'; } },
                    { key: 'audio_nerve_queue_size', label: 'Audio Nerve Queue', format: function(val) { return val || '0'; } },
                    { key: 'processed_queue_size', label: 'Processed Queue', format: function(val) { return val || '0'; } },
                    { key: 'VLM_availible', label: 'VLM Status', format: function(val) { return val ? 'Ready' : 'Busy'; } }
                ];
                
                for (var i = 0; i < metricsToShow.length; i++) {
                    var metric = metricsToShow[i];
                    var card = document.createElement('div');
                    card.className = 'stat-card';
                    card.innerHTML = 
                        '<div class="stat-label">' + metric.label + '</div>' +
                        '<div class="stat-value">' + metric.format(stats[metric.key]) + '</div>';
                    statsGrid.appendChild(card);
                }

                // Update device status based on stats
                updateDeviceStates(stats);
            });
        }

        function updateTranscripts() {
            makeRequest('GET', '/audio_scenes', function(error, data) {
                if (error) {
                    console.error('Transcripts update error:', error);
                    return;
                }
                
                var transcriptsList = document.getElementById('transcriptsList');
                
                if (data.audio_scenes && data.audio_scenes.length > 0) {
                    var html = '';
                    var scenes = data.audio_scenes.slice(-3).reverse();
                    for (var i = 0; i < scenes.length; i++) {
                        var scene = scenes[i];
                        html += '<div class="transcript-item">' +
                               '<div class="timestamp">' + scene.formatted_time + ' - Device ' + scene.device_index + '</div>' +
                               '<div class="transcript-text">' + (scene.transcription || 'Processing...') + '</div>' +
                               '</div>';
                    }
                    transcriptsList.innerHTML = html;
                }
            });
        }

        function updateVLM() {
            makeRequest('GET', '/visual_scenes', function(error, data) {
                if (error) {
                    console.error('VLM update error:', error);
                    return;
                }
                
                var vlmList = document.getElementById('vlmList');
                
                if (data.scenes && data.scenes.length > 0) {
                    var html = '';
                    var scenes = data.scenes.slice(-3).reverse();
                    for (var i = 0; i < scenes.length; i++) {
                        var scene = scenes[i];
                        html += '<div class="vlm-item">' +
                               '<div class="timestamp">' + scene.formatted_time + ' - Camera ' + scene.camera_index + '</div>' +
                               '<div class="vlm-text">' + (scene.caption || 'Processing...') + '</div>' +
                               '</div>';
                    }
                    vlmList.innerHTML = html;
                }
            });
        }

        function updateDeviceStates(stats) {
            // Update audio device states based on speech detection
            if (stats.audio_cortex) {
                var speechDetected = stats.audio_cortex.speech_detected;
                var deviceIndex = stats.audio_cortex.device_index;
                
                if (deviceIndex !== null && deviceIndex !== false) {
                    var statusEl = document.getElementById('audio-status-' + deviceIndex);
                    if (statusEl && deviceStates.audio[deviceIndex]) {
                        statusEl.className = speechDetected ? 
                            'status-indicator status-speech' : 
                            'status-indicator status-active';
                    }
                }
            }
            
            // Update camera device states based on person detection
            if (stats.visual_cortex) {
                var personDetected = stats.visual_cortex.person_detected;
                var deviceIndex = stats.visual_cortex.camera_index;
                
                if (deviceIndex !== null && deviceIndex !== false) {
                    var statusEl = document.getElementById('camera-status-' + deviceIndex);
                    if (statusEl && deviceStates.camera[deviceIndex]) {
                        statusEl.className = personDetected ? 
                            'status-indicator status-person' : 
                            'status-indicator status-active';
                    }
                }
            }
        }

        // Initialize and start periodic updates
        function startPeriodicUpdates() {
            updateStats();
            updateTranscripts();
            updateVLM();
            
            // Update every 100ms
            setInterval(function() {
                updateStats();
                updateTranscripts();
                updateVLM();
            }, 100);
        }

        // Start everything when page loads
        document.addEventListener('DOMContentLoaded', startPeriodicUpdates);
    </script>
</body>
</html>
    '''

@app.route('/stats')
def stats():
    """API endpoint for performance statistics"""
    global AUDIO_SCENE, TEMPORAL_LOBE_STATE
    current_stats = dict(stats_dict)
    current_stats['timestamp'] = datetime.now().isoformat()
    current_stats['optic_nerve_queue_size'] = optic_nerve_queue.qsize()

    # Visual queue stats
    current_stats['processed_queue_size'] = visual_cortex_queue.qsize()
    current_stats['visual_cortex_internal_queue_vlm_size'] = visual_cortex_internal_queue_vlm.qsize()
    current_stats['active_optic_nerves'] = list(optic_nerve_processes.keys())
    
    # Audio queue stats
    current_stats['audio_nerve_queue_size'] = audio_nerve_queue.qsize()
    current_stats['audio_cortex_queue_size'] = audio_cortex_queue.qsize()
    current_stats['active_auditory_nerves'] = list(auditory_nerve_processes.keys())

    temporal_lobe_state()  # Update state
    current_data = TEMPORAL_LOBE_STATE.get('current_data')
    # Audio cortex data
    try:
        if current_data and current_data.get('audio'):
            current_stats['audio_cortex'] = {
                'device_index': current_data['audio_scenes']['audio']['device_index'],
                'speech_detected': current_data['audio_scenes']['audio']['speech_detected']
            }
    except:
        current_stats['audio_cortex'] = {
            'device_index': False,
            'speech_detected': False
        }

    #Visual cortex data for person detection
    try:
        
        if current_data and current_data.get('visual'):
            current_stats['visual_cortex'] = {
                'camera_index': current_data['visual'].get('camera_index'),
                'person_detected': current_data['visual'].get('person_detected', False)
            }
        else:
            current_stats['visual_cortex'] = {
                'camera_index': None,
                'person_detected': False
            }
    except:
        current_stats['visual_cortex'] = {
            'camera_index': None,
            'person_detected': False
        }

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

    """temporal_lobe_updater_thread.terminate()
    temporal_lobe_updater_thread.join(timeout=2)
    if temporal_lobe_updater_thread.is_alive():
            temporal_lobe_updater_thread.kill()"""
    
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
    global TEMPORAL_LOBE_STATE, stats_dict
    
    # Update state from temporal lobe queue
    temporal_lobe_state()
    
    return jsonify({
        'scenes': TEMPORAL_LOBE_STATE['visual_scenes'].copy(),  # Return copy to avoid modification
        'count': len(TEMPORAL_LOBE_STATE['visual_scenes']),
        'VLM_availible': stats_dict['VLM_availible'],
        'last_update': TEMPORAL_LOBE_STATE['last_update']
    })

@app.route('/start/<int:camera_index>')
def start_camera(camera_index):
    """Start optic nerve and visual cortex processing for specific camera"""
    global optic_nerve_processes, visual_cortex_processes, visual_cortex_internal_queue_vlm
    
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
            args=(optic_nerve_queue, visual_cortex_queue, stats_dict,visual_cortex_queue_img_display, visual_cortex_internal_queue_vlm)
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


@app.route('/audio_scenes')
def audio_scenes():
    """API endpoint for audio scene analysis data with transcriptions"""
    global TEMPORAL_LOBE_STATE
    
    # Update state from temporal lobe queue
    temporal_lobe_state()
    
    return jsonify({
        'audio_scenes': TEMPORAL_LOBE_STATE['audio_scenes'].copy(),  # Return copy to avoid modification
        'count': len(TEMPORAL_LOBE_STATE['audio_scenes']),
        'last_update': TEMPORAL_LOBE_STATE['last_update']
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
    
    # Visual queues
    optic_nerve_queue = manager.Queue(maxsize=QUEUE_SIZE * 4)
    visual_cortex_queue = manager.Queue(maxsize=QUEUE_SIZE * 4)
    visual_cortex_queue_img_display = manager.Queue(maxsize=QUEUE_SIZE * 4)
    visual_cortex_internal_queue_vlm = manager.Queue(maxsize=10)
    
    #Audio queues
    audio_nerve_queue = manager.Queue(maxsize=QUEUE_SIZE * 4)
    audio_cortex_queue = manager.Queue(maxsize=QUEUE_SIZE * 4)
    # internally used by Audio Cortex to hold speech audio that needs transcription
    audio_cortex_internal_queue_speech_audio = manager.Queue(maxsize=100)  #100 chunks of 32ms = ~3200ms (3.2 second) buffer
    # internally used by Audio Cortex to hold speech to text results (transcriptions)
    audio_cortex_internal_queue_transcription = manager.Queue(maxsize=5)

    #Temporal Lobe - holds outputs of final combined data from AC and VC
    temporal_lobe_queue = manager.Queue(maxsize=5)

    #Start temporal lobe
    tl = mp.Process(
        target=temporal_lobe,
        args=(audio_cortex_queue,visual_cortex_queue,temporal_lobe_queue))
    
    tl.start()
    temporal_lobe_processes['main'] = tl

    """temporal_lobe_updater_thread = threading.Thread(target=temporal_lobe_state_updater, daemon=True)
    temporal_lobe_updater_thread.start()"""

     
    stats_dict = manager.dict()
    stats_dict['VLM_availible'] = True

   
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

