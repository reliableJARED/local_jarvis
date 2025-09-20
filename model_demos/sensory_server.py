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
VLM_availible = False
VLM_thread = None
"""
CENTRAL NERVOUS SYSTEM
"""

class CentralNervousSystem:
    """
    Central queue manager for all sensory processing pipelines.
    Provides organized access to all inter-process communication queues.
    """
    
    def __init__(self, queue_size: int = 5):
        self.queue_size = queue_size
        
        # Visual processing queues
        self.optic_nerve_queue = mp.Queue(maxsize=queue_size)
        self.visual_cortex_queue = mp.Queue(maxsize=queue_size)
        self.visual_scene_queue = mp.Queue(maxsize=queue_size)
        
        # Audio processing queues  
        self.audio_nerve_queue = mp.Queue(maxsize=queue_size)
        self.audio_cortex_queue = mp.Queue(maxsize=queue_size)
        self.audio_scene_queue = mp.Queue(maxsize=queue_size)
        
        # Shared state management
        self.manager = mp.Manager()
        self.stats_dict = self.manager.dict()
        self.global_awareness = self.manager.dict()
        
        # Process tracking
        self.optic_nerve_processes: Dict[int, mp.Process] = {}
        self.visual_cortex_processes: Dict[int, mp.Process] = {}
        self.auditory_nerve_processes: Dict[int, mp.Process] = {}
        self.auditory_cortex_processes: Dict[int, mp.Process] = {}
        
        # Initialize shared state
        self._initialize_shared_state()
    
    def _initialize_shared_state(self):
        """Initialize shared state dictionaries with default values."""
        self.stats_dict.update({
            'VLM_availible': False,
            'visual_cortex_fps': 0,
            'auditory_cortex_fps': 0,
            'last_visual_cortex': 0,
            'last_auditory_cortex': 0
        })
        
        self.global_awareness.update({
            'human': None,
            'detection_type': {'speech': False, 'visual': False},
            'detection_timestamp': 0,
            'device_index': None
        })
    
    def start_optic_nerve(self, camera_index: int, fps_target: int = 30):
        """Start optic nerve process for a specific camera."""
        if camera_index in self.optic_nerve_processes:
            print(f"Optic nerve process for camera {camera_index} already running")
            return
            
        process = mp.Process(
            target=optic_nerve_worker,
            args=(camera_index, self.optic_nerve_queue, self.stats_dict, fps_target)
        )
        process.start()
        self.optic_nerve_processes[camera_index] = process
        print(f"Started optic nerve process for camera {camera_index}")
    
    def start_visual_cortex(self):
        """Start visual cortex process."""
        if hasattr(self, 'visual_cortex_process') and self.visual_cortex_process.is_alive():
            print("Visual cortex process already running")
            return
            
        self.visual_cortex_process = mp.Process(
            target=visual_cortex_worker,
            args=(self.optic_nerve_queue, self.visual_cortex_queue, 
                  self.stats_dict, self.visual_scene_queue)
        )
        self.visual_cortex_process.start()
        print("Started visual cortex process")
    
    def start_auditory_nerve(self, device_index: int, sample_rate: int = 16000, chunk_size: int = 512):
        """Start auditory nerve process for a specific audio device."""
        if device_index in self.auditory_nerve_processes:
            print(f"Auditory nerve process for device {device_index} already running")
            return
            
        process = mp.Process(
            target=auditory_nerve_worker,
            args=(device_index, self.audio_nerve_queue, self.stats_dict, 
                  self.global_awareness, sample_rate, chunk_size)
        )
        process.start()
        self.auditory_nerve_processes[device_index] = process
        print(f"Started auditory nerve process for device {device_index}")
    
    def start_auditory_cortex(self, device_index: int):
        """Start auditory cortex process."""
        if hasattr(self, 'auditory_cortex_process') and self.auditory_cortex_process.is_alive():
            print("Auditory cortex process already running")
            return
            
        self.auditory_cortex_process = mp.Process(
            target=auditory_cortex_worker,
            args=(self.audio_nerve_queue, self.audio_cortex_queue, self.stats_dict,
                  self.audio_scene_queue, self.global_awareness, device_index)
        )
        self.auditory_cortex_process.start()
        print("Started auditory cortex process")
    
    def stop_all_processes(self):
        """Stop all running processes gracefully."""
        print("Stopping all nervous system processes...")
        
        # Stop optic nerve processes
        for camera_index, process in self.optic_nerve_processes.items():
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
                print(f"Stopped optic nerve process for camera {camera_index}")
        
        # Stop visual cortex
        if hasattr(self, 'visual_cortex_process') and self.visual_cortex_process.is_alive():
            self.visual_cortex_process.terminate()
            self.visual_cortex_process.join(timeout=2)
            print("Stopped visual cortex process")
        
        # Stop auditory nerve processes
        for device_index, process in self.auditory_nerve_processes.items():
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
                print(f"Stopped auditory nerve process for device {device_index}")
        
        # Stop auditory cortex
        if hasattr(self, 'auditory_cortex_process') and self.auditory_cortex_process.is_alive():
            self.auditory_cortex_process.terminate()
            self.auditory_cortex_process.join(timeout=2)
            print("Stopped auditory cortex process")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        return dict(self.stats_dict)
    
    def get_awareness(self) -> Dict[str, Any]:
        """Get current global awareness state."""
        return dict(self.global_awareness)
    
    def __del__(self):
        """Cleanup on destruction."""
        self.stop_all_processes()

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
    VLM_availible_local = False

    #Visual Cortex Processing Core CNN (yolo)
    from yolo_ import YOLOhf
    

    #reuse single instance
    IMG_DETECTION_CNN = YOLOhf()

    try:
        while True:
            try:
                # Sync local VLM busy state with global state
                VLM_availible_local = stats_dict.get('VLM_availible', False)
                
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
                    if person_detected and not VLM_availible_local:
                        print(f"Starting VLM analysis - person detected and VLM not busy")
                        
                        # Set busy states
                        VLM_availible_local = True
                        stats_dict['VLM_availible'] = True

                        
                        # Start VLM Moondream analysis in separate thread
                        VLM_thread = threading.Thread(
                            target=visual_cortex_process_vlm_analysis_thread,
                            args=(frame_data['frame'], frame_data['camera_index'], 
                                frame_data['capture_timestamp'], visual_scene_queue, stats_dict)
                            )  
                         
                        VLM_thread.daemon = True
                        VLM_thread.start()

                    elif person_detected and VLM_availible_local:
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
            stats_dict['VLM_availible'] = False
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

        stats_dict['VLM_availible'] = False
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

def auditory_nerve_worker(device_index, audio_nerve_queue, stats_dict, global_awareness,sample_rate=16000, chunk_size=512):
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

def auditory_cortex_worker(audio_nerve_queue, audio_cortex_queue, stats_dict, audio_scene_queue,global_awareness,device_index):
    """Worker process for processing audio frames (auditory cortex function)
    Optimized for real-time streaming with VADIterator and minimal buffering.
    """
    print("Started auditory cortex process")
    frame_count = 0
    start_time = time.time()
    
    # Pre-roll buffer for capturing speech beginnings (100ms worth of audio)
    pre_roll_buffer = deque(maxlen=5)  # ~5 chunks of 32ms = ~160ms buffer
    speech_active = False
    speech_buffer = []
    min_silence_duration_ms=1000  # Minimum silence to end speech
    
    # Initialize Silero VAD with VADIterator for streaming
    try:
        # Load Silero VAD model
        torch.set_num_threads(1)  # Optimize for single-thread performance
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
            sampling_rate=16000,     # Must match audio sample rate
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
                # Get audio frame (blocking with timeout)
                audio_data = audio_nerve_queue.get(timeout=1.0)
                
                # Process immediately - no accumulation needed for 512-sample chunks
                audio_chunk = audio_data['audio_frame']
                
                # Ensure we have exactly 512 samples for consistent processing
                if len(audio_chunk) != 512:
                    continue
                
                # Convert to tensor for Silero VAD
                audio_tensor = torch.from_numpy(audio_chunk)
                
                # Add to pre-roll buffer
                pre_roll_buffer.append(audio_chunk)
                
                # Run VADIterator - this handles streaming state internally
                try:
                    speech_dict = vad_iterator(audio_tensor, return_seconds=True)
                    
                    # Process VAD results
                    if speech_dict:
                        # Speech event detected (start or end)
                        if 'start' in speech_dict:
                            # Speech start detected
                            print(f"Speech START detected at {speech_dict['start']:.3f}s")
                            speech_active = True
                            # Add pre-roll buffer to speech
                            speech_buffer = list(pre_roll_buffer)

                            #speech is from humans, so system is aware there is human
                            human_time = time.time()
                            global_awareness.update({
                                'human': 'unknown name', 
                                'detection_type': {'speech': True}, 
                                'detection_timestamp': human_time,
                                'device_index':device_index
                            })
                                                        
                        if 'end' in speech_dict:
                            # Speech end detected
                            print(f"Speech END detected at {speech_dict['end']:.3f}s")
                            speech_active = False

                            #TODO: perhaps decay, for now no speech means no human awareness
                            global_awareness.update({
                                'human': 'unknown name', 
                                'detection_type': {'speech': False}, 
                                'detection_timestamp': human_time,
                                'device_index':device_index
                            })
                            
                            # Process accumulated speech
                            if speech_buffer:
                                # Concatenate all speech chunks
                                full_speech = np.concatenate(speech_buffer)
                                
                                # Create processed audio data
                                process_timestamp = time.time()
                                processed_data = {
                                    'device_index': audio_data['device_index'],
                                    'audio_data': full_speech,
                                    'speech_probability': 1.0,  # Confirmed speech segment
                                    'speech_detected': True,
                                    'capture_timestamp': audio_data['capture_timestamp'],
                                    'auditory_cortex_timestamp': process_timestamp,
                                    'sample_rate': audio_data['sample_rate'],
                                    'duration': len(full_speech) / audio_data['sample_rate']
                                }
                                
                                # Put processed audio in output queue
                                try:
                                    audio_cortex_queue.put_nowait(processed_data)
                                except:
                                    # Queue is full, try to remove oldest and add new
                                    try:
                                        audio_cortex_queue.get_nowait()
                                        audio_cortex_queue.put_nowait(processed_data)
                                    except:
                                        pass
                                
                                # Add to audio scene queue for further analysis
                                audio_scene_data = {
                                    'device_index': audio_data['device_index'],
                                    'timestamp': audio_data['capture_timestamp'],
                                    'speech_probability': 1.0,
                                    'analysis_time': process_timestamp,
                                    'formatted_time': datetime.fromtimestamp(audio_data['capture_timestamp']).strftime('%H:%M:%S'),
                                    'audio_data': full_speech,
                                    'duration': len(full_speech) / audio_data['sample_rate']
                                }
                                
                                try:
                                    audio_scene_queue.put_nowait(audio_scene_data)
                                except:
                                    # Queue full, remove oldest and add new
                                    try:
                                        audio_scene_queue.get_nowait()
                                        audio_scene_queue.put_nowait(audio_scene_data)
                                    except:
                                        pass
                                
                                # Clear speech buffer
                                speech_buffer = []
                    
                    # If speech is active, add current chunk to buffer
                    if speech_active:
                        speech_buffer.append(audio_chunk)

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
        print("Auditory cortex process stopped")

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
        
        <!-- Stats -->
        <div class="stats" id="stats-container">
            <!-- Stats will be populated by JavaScript -->

            <!-- Audio Scene Analysis -->
        </div>
                <div class="visual-scene">
                <h3>Audio Speech to Text</h3>
                <div id="audio-scene-analysis">
                    <p>not connected yet, will stream detected speech-to-text</p>
                </div>

                <h3>Audio Scene Analysis</h3>
                <div id="audio-scene-analysis">
                    <p>Not connected yet, will populate with audio description</p>
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
                    // Check global_awareness for human speech detection
                    const speechDetected = data.present_subjects && 
                                        data.present_subjects.detection_type && 
                                        data.present_subjects.detection_type.speech;
                    
                    const speechDeviceIndex = data.present_subjects && 
                                            data.present_subjects.device_index !== undefined ? 
                                            data.present_subjects.device_index : -1;
                    
                    // Update speech detection indicators for both microphones
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
                    
                    // Update audio scene analysis if available
                    if (data.audio_scenes && data.audio_scenes.length > 0) {
                        const container = document.getElementById('audio-scene-analysis');
                        container.innerHTML = '';
                        data.audio_scenes.slice().reverse().forEach(scene => {
                            const sceneDiv = document.createElement('div');
                            sceneDiv.className = 'scene-item';
                            sceneDiv.innerHTML = `
                                <div class="scene-time">${scene.formatted_time}</div>
                                <div class="scene-camera">Microphone ${scene.device_index}</div>
                                <div class="scene-caption">Speech probability: ${scene.speech_probability}</div>
                            `;
                            container.appendChild(sceneDiv);
                        });
                    }
                })
                .catch(err => console.error('Audio stats update failed:', err));
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
        
        // Update intervals
        setInterval(updateStats, 1000);
        setInterval(updateVisualScenes, 2000);
        setInterval(updateAudioStatus, 500);  // Update audio status more frequently for real-time speech detection
        
        // Initial updates
        updateStats();
        updateVisualScenes();
        updateAudioStatus();
    </script>
</body>
</html>
    '''

@app.route('/stats')
def stats():
    """API endpoint for performance statistics"""
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
    current_stats['audio_scene_queue_size'] = audio_scene_queue.qsize()
    current_stats['active_auditory_nerves'] = list(auditory_nerve_processes.keys())

    #Object/Person awareness
    current_stats['present_subjects'] = dict(global_awareness)

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
    
    # Put them back in the queue (keeping last 10)
    scenes = scenes[-10:]  # Keep only last 10 analyses
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
# ===== NEW API ROUTES FOR AUDIO =====

@app.route('/start_audio/<int:device_index>')
def start_audio_device(device_index):
    """Start auditory nerve and auditory cortex processing for specific audio device"""
    global auditory_nerve_processes, auditory_cortex_processes
    
    if device_index in auditory_nerve_processes:
        return jsonify({
            'status': 'already_running', 
            'message': f'Auditory nerve {device_index} is already running'
        })
    
    # Start auditory nerve process for this device
    auditory_nerve_process = mp.Process(
        target=auditory_nerve_worker,
        args=(device_index, audio_nerve_queue, stats_dict,global_awareness)
    )
    auditory_nerve_process.start()
    auditory_nerve_processes[device_index] = auditory_nerve_process
    
    # Start auditory cortex if not already running
    if not auditory_cortex_processes:
        auditory_cortex_process = mp.Process(
            target=auditory_cortex_worker,
            args=(audio_nerve_queue, audio_cortex_queue, stats_dict, audio_scene_queue,global_awareness,device_index)
        )
        auditory_cortex_process.start()
        auditory_cortex_processes['main'] = auditory_cortex_process
    
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
    """API endpoint for audio scene analysis data"""
    scenes = []
    
    # Get all available scenes from the queue without blocking
    while True:
        try:
            scene_data = audio_scene_queue.get_nowait()
            scenes.append({
                'device_index': scene_data['device_index'],
                'timestamp': scene_data['timestamp'],
                'speech_probability': scene_data['speech_probability'],
                'analysis_time': scene_data['analysis_time'],
                'formatted_time': scene_data['formatted_time']
            })
        except:
            break
    
    # Put them back in the queue (keeping last 10)
    scenes = scenes[-10:]  # Keep only last 10 analyses
    for scene_data in scenes:
        try:
            audio_scene_queue.put_nowait(scene_data)
        except:
            # Queue full, skip older ones
            break
    
    return jsonify({
        'audio_scenes': scenes,
        'count': len(scenes)
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
    audio_scene_queue = manager.Queue(maxsize=10)
    
    stats_dict = manager.dict()

    # Initialize global awareness structure
    global_awareness = manager.dict()  # Global awareness tracking


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


