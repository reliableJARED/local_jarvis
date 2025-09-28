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


"""
VISUAL
"""
def optic_nerve_connection(camera_index, internal_nerve_queue, external_stats_queue, fps_capture_target):
    """Worker process for capturing frames from a specific camera (optic nerve function).
    Will drop frames if the queue is full to maintain real-time performance.
    will also update stats_dict with fps and last frame time.
    will stream to optic_nerve_queue."""
    cap = cv2.VideoCapture(camera_index)
    
    # Camera setup
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, fps_capture_target)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
    
    if not cap.isOpened():
        logging.error(f"Error: Could not open camera {camera_index}")
        return
    
    logging.info(f"Started optic nerve process for camera {camera_index}")
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error(f"Failed to capture frame from camera {camera_index}")
                time.sleep(0.1)
                continue
                
            # Create frame data with camera index and timestamp
            timestamp = time.time()
            frame_data = {
                'camera_index': camera_index,
                'frame': frame,
                'capture_timestamp': timestamp
            }
            
            # Non-blocking queue put - replace frame if queue is full
            try:
                internal_nerve_queue.put_nowait(frame_data)
            except queue.Full:
                #allow time to avoid race condition for visual cortex to grab frame if it was just in waiting loop
                time.sleep(0.1)
                # then take old frame out, put new one in
                try:
                    _ = internal_nerve_queue.get_nowait()
                    internal_nerve_queue.put_nowait(frame_data)
                except queue.Empty:
                    #the cortex got the last frame before sleep finished, DO NOT put a new frame in
                    #We want to go back through the loop again and give the most updated frame
                    continue
                
            
            # Calculate and update optic nerve FPS
            frame_count += 1
            if frame_count % 30 == 0:  # Update every 30 frames
                current_time = time.time()
                fps = frame_count / (current_time - start_time)
                stats_data = {f'optic_nerve_fps_{camera_index}':fps,
                            f'last_optic_nerve_{camera_index}': current_time}
                try:
                    external_stats_queue.put_nowait(stats_data)# Will drop and lose this data if no put
                except:
                    logging.warning(f"Optic nerve {camera_index} unable to send stats data to external_stats_queue because it's full")
                
            # Target frame rate control
            time.sleep(1.0 / fps_capture_target)
            
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        logging.info(f"Optic nerve process for camera {camera_index} stopped")

def visual_cortex_core(optic_nerve_queue, external_cortex_queue, external_stats_queue, external_img_queue, internal_from_vlm_queue, internal_to_vlm_queue,internal_cortex_process_control_queue,control_msg_struct,vlm,gpu_to_use):
    """Worker process for processing image frames from all cameras (visual cortex function)"""

    logging.info("Started visual cortex process")
    
    # Visual Cortex Processing Core CNN (yolo)
    from yolo_ import YOLOhf
    IMG_DETECTION_CNN = YOLOhf()
    

    try:
        #counters and timer used in loop
        frame_count = 0
        start_time = time.time()

        #initial tool instructions
        use_ = control_msg_struct
        """schema:
        use_person_detect = control_msg_struct['person_detection']
        use_person_recognize = control_msg_struct['person_recognition']
        use_vlm_caption = control_msg_struct['vlm']['caption']
        use_vlm_query = control_msg_struct['vlm']['query']
        use_vlm_point = control_msg_struct['vlm']['point']
        use_vlm_detect = control_msg_struct['vlm']['detect']"""
        limitedGPUenv = False
        # VLM thread worker for this process
        vlm_process = None

        if not limitedGPUenv:
            #this will create a persistant VLM, loaded in memory which is reused, else in loop for analysis vlm is created/used/destroyed so gpu ram is released
            logging.debug("starting a long running VLM process")
            vlm_gpu = str(gpu_to_use)
            vlm_process = mp.Process(target=vlm, args=(limitedGPUenv,internal_to_vlm_queue, internal_from_vlm_queue,external_stats_queue,vlm_gpu),daemon=True)
            vlm_process.start()



        while True:
            frame_data = None
            try:
                #check for control instructions
                try:
                    use_ = internal_cortex_process_control_queue.get_nowait()
                    logging.debug(f"visual_cortex_core received new operation instructions: {use_}")
                except queue.Empty:
                    #no new instructions
                    pass
                
                # Queue only holds one frame, the nerve is managing putting frames in as (LIFO) at target frame rate of 30 fps ~0.033, so we wait if empty for one frame rotation
                frame_data = optic_nerve_queue.get_nowait()

                # If we got at least one frame, there is a connected active nerve
                if frame_data is not None:
                    
                    device_index = frame_data['camera_index']
                    capture_timestamp = frame_data['capture_timestamp']
                    raw_frame = frame_data['frame']
                    
                    processed_frame=raw_frame
                    person_detected = False

                    if use_['person_detection']:
                        # Process the frame (blocking)
                        processed_frame, person_detected = visual_cortex_process_img(raw_frame,IMG_DETECTION_CNN)
                    
                    # Push processed frame out for viewing 
                    try:
                        external_img_queue.put_nowait({
                            'frame': processed_frame,
                            'camera_index': device_index,  
                            'capture_timestamp': capture_timestamp
                        })
                    except queue.Full:
                        # Drop oldest frame and add new one
                        try:
                            _ = external_img_queue.get_nowait()
                        except queue.Empty:
                            #race condition protection incase external consumer got the frame before this loop did
                            pass
                        external_img_queue.put_nowait({
                                'frame': processed_frame,
                                'camera_index': device_index,
                                'capture_timestamp': capture_timestamp
                            })
                        

                    # VLM processing logic
                    if (use_['vlm']['caption'] or use_['vlm']['detect'] != '' or use_['vlm']['point'] != '' or use_['vlm']['query']  != ''):
                        logging.debug(f"Starting VLM analysis - VLM not busy")
                        
                        #If we are in a limited GPU situation, we load, use, and destroy the VLM. Else we are are using the one loaded in mem
                        if limitedGPUenv:
                            vlm_gpu = str(0)#default, assume there is only 1 gpu in a situation like this
                            vlm_process = mp.Process(
                                target=vlm,
                                args=(limitedGPUenv,internal_to_vlm_queue, internal_from_vlm_queue,external_stats_queue,external_stats_queue,vlm_gpu),
                                daemon=True)
                            vlm_process.start()
                        
                        try:
                            #put the frame in queue for the VLM
                            internal_to_vlm_queue.put_nowait({'frame':raw_frame,'device_index':device_index,'capture_timestamp':capture_timestamp, 'person_detected':person_detected,'instructions':use_})
                        except queue.Full:
                            #Do not get and replace the frame. We let the VLM finish what it was working on, this frame is just skipped
                            logging.debug("VLM - Queue FULL - It is still working on previous")
                            pass

                    # Create processed frame data
                    processed_data = {
                            'capture_timestamp':capture_timestamp,
                            'person_detected':person_detected,
                            'person_match': False,
                            'person_match_probability': 0.0,
                            'camera_index': device_index,
                            'caption': "",
                            'query': "",
                            'obj_detect': [],
                            'points': [],
                            'vlm_timestamp': capture_timestamp,
                            'formatted_time': datetime.fromtimestamp(capture_timestamp).strftime('%H:%M:%S')
                        }
                    

                    # Check for VLM results
                    try:
                        vlm_data = internal_from_vlm_queue.get_nowait()
                        processed_data.update(vlm_data)  # Merge VLM results

                        if vlm_process:
                            if limitedGPUenv:
                                #end the vlm in limited gpu env so we can give back GPU memory
                                vlm_process.join()

                    except queue.Empty:
                        logging.debug("No VLM data availible")

                    # Put processed frame in output queue with better error handling
                    try:
                        external_cortex_queue.put_nowait(processed_data)
                    except queue.Full:
                        # Drop oldest frame and add new one
                        try:
                            _ = external_cortex_queue.get_nowait()
                            external_cortex_queue.put_nowait(processed_data)
                        except queue.Empty:
                            pass
                    
                    # Calculate visual cortex processing FPS
                    frame_count += 1
                    if frame_count % 30 == 0:
                        fps = frame_count / (time.time() - start_time)
                        stats_data = {
                            'visual_cortex_fps': fps,
                            'last_visual_cortex': time.time()
                        }
                        try:
                            external_stats_queue.put_nowait(stats_data)
                        except queue.Full:
                            try:
                                _ = external_stats_queue.get_nowait()
                                external_stats_queue.put_nowait(stats_data)
                            except queue.Empty:
                                pass
                    
            except queue.Empty:
                logging.debug("visual cortex not receving image frames")
                continue
        
            # Slow the loop a bit to help CPU while waiting for a nerve to connect
            if frame_data is None:
                time.sleep(0.001)
                
    except KeyboardInterrupt:
        pass
    finally:
        print("Visual cortex process stopped")

def visual_cortex_process_img(frame, IMG_DETECTION_CNN):
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

def visual_cortex_worker_vlm(limitedGPUenv,internal_to_vlm_queue,visual_cortex_internal_queue_vlm,external_stats_queue,gpu_device):
    """Thread function to run VLM (Moondream) analysis without blocking main processing"""
    # Import and initialize VLM Moondream (done in thread to avoid blocking memory if it stays loaded)
    import os
    import logging
    import concurrent.futures
    #Set which GPU we run the VLM on
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device

    log = logging.getLogger('pyvips')
    log.setLevel(logging.ERROR)

    from moondream_ import MoondreamWrapper
    IMG_PROCESSING_VLM = MoondreamWrapper(local_files_only=True)

    # At module level:
    person_recognition_worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)


    """
    #TODO: Integrate the use of 'instructions' arg
    instructions = {'vlm': {'caption': True, 
                        'detect': '', 
                        'point': '', 
                        'query': ''},
                'person_detection': True,
                'person_recognition': True}

    The VLM has the following skills/methods.

    IMG_PROCESSING_VLM.caption_image()
    IMG_PROCESSING_VLM.ask_question()
    IMG_PROCESSING_VLM.detect_objects()
    IMG_PROCESSING_VLM.point_to_objects()
    IMG_PROCESSING_VLM.analyze_image_complete()
    """
    
    def person_recognition(frame, result_container):
        """
        Placeholder function for facial recognition.

        #TODO: Facial Recognition should probably run as a process itself, not inside the VLM.  Leave it here for now since it's not actually working
        
        Args:
            frame: raw image frame with a person
            result_container: List to store results [match_result, probability]
        """
        #TODO: these are just placeholders
        logging.debug("RUN PERSON RECOGNITION")
        if frame is not None:
            logging.debug("got PERSON RECOGNITION image")
        try:
            # Placeholder return values
            # For now, simulate no match
            result_container[0] = False
            result_container[1] = 0.0
        except Exception as e:
            logging.error(f"facial recognition error: {e}")
            result_container[0] = False
            result_container[1] = 0.0
        finally:
            # Explicit cleanup
            # del embedding, similarities, etc.
            gc.collect()  # force garbage collection

    

    def update_availiblity(external_stats_queue,status_boolean):
        #Used to tell outside if the VLM can be used or if it's currently in use
        try:
            external_stats_queue.put_nowait({'VLM_availible': status_boolean})
        except queue.Full:
            try:
                _ = external_stats_queue.get_nowait()
                external_stats_queue.put_nowait({'VLM_availible': status_boolean})
            except queue.Empty:
                pass

    single_run = False
    got_a_frame = False
    while not single_run:
        update_availiblity(external_stats_queue,False)
        frame = False
        camera_index = False
        capture_timestamp = False
        person_detected = False
        instructions = False
        while True:
            try:
                #{'frame':raw_frame,'device_index':device_index,'capture_timestamp':capture_timestamp, 'person_detected':person_detected,'instructions':use_}
                data = internal_to_vlm_queue.get(timeout=0.1) #slow blocking delay
                frame = data['frame']
                camera_index = data['device_index']
                capture_timestamp = data['capture_timestamp']
                person_detected = data['person_detected']
                instructions = data['instructions']
                got_a_frame = True

            except queue.Empty:
                #if we didn't get a frame, flag as false
                if not got_a_frame:
                    got_a_frame = False
                break

        if got_a_frame:

            # Convert frame to RGB for Moondream
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Wait for sufficient GPU memory (gpu_min_gb threshold)
            gpu_min_gb = 3000
            max_wait_time = 30  # Maximum wait time in seconds
            wait_start = time.time()

            #Mem allocation checker if this is not being used in a persistant thread on a different gpu
            # Inside the frame processing loop, fix the memory check:
            if limitedGPUenv:
                print("Checking GPU memory availability...")
                while torch.cuda.is_available():
                    try:
                        available_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**2
                        print(f"Available GPU memory: {available_memory:.1f}MB (need {gpu_min_gb}MB)")
                        
                        if available_memory >= gpu_min_gb:
                            print("Sufficient GPU memory available")
                            break
                            
                        if time.time() - wait_start > max_wait_time:
                            print(f"Timeout waiting for GPU memory after {max_wait_time}s")
                            # Send error response back
                            visual_scene_data['caption'] = "ERROR: GPU memory timeout"
                            try:
                                visual_cortex_internal_queue_vlm.put_nowait(visual_scene_data)
                            except:
                                pass
                            return
                            
                        time.sleep(0.1)
                    except Exception as e:
                        print(f"Error checking GPU memory: {e}")
                        break
            
            visual_scene_data = {
                                    'capture_timestamp':capture_timestamp,
                                    'person_detected':person_detected,
                                    'person_match': False,
                                    'person_match_probability': 0.0,
                                    'camera_index': camera_index,
                                    'caption': "",
                                    'query': "",
                                    'obj_detect': [],
                                    'points': [],
                                    'vlm_timestamp': capture_timestamp,
                                    'formatted_time': datetime.fromtimestamp(capture_timestamp).strftime('%H:%M:%S')
                                }
            
                
            try:
                logging.debug(f"Starting VLM analysis for camera {camera_index}")
                
                # Convert to PIL Image
                pil_frame = Image.fromarray(frame_rgb)

                #run person recognition
                person_recognition_result_container = [False, 0.0]  # [match_result, probability]

                if instructions['person_recognition']:
                    pr_future_result = person_recognition_worker.submit(person_recognition, frame, person_recognition_result_container)

                # Generate caption
                caption = ""
                if instructions['vlm']['caption']:
                    caption = IMG_PROCESSING_VLM.caption_image(pil_frame, length="normal")

                ###
                #TODO: Implement Points, Detection, Query - note when doing multi will want to use
                #IMG_PROCESSING_VLM.encode_image(image) first
                ###

                # If we are using person recognition, get (blocking) facial recognition results from our thread
                if instructions['person_recognition'] and pr_future_result:
                    pr_future_result.result()  # Still blocking
                
                # Set recognition results from the container
                person_match_result = person_recognition_result_container[0]
                match_probability = person_recognition_result_container[1]
                
                vlm_timestamp = time.time()

                # Add to visual scene analysis data
                visual_scene_data['caption'] = caption
                visual_scene_data['vlm_timestamp']=vlm_timestamp
                visual_scene_data['formatted_time']=datetime.fromtimestamp(vlm_timestamp).strftime('%H:%M:%S')
                visual_scene_data['person_match'] = person_match_result,
                visual_scene_data['person_match_probability'] = match_probability
                
                try:
                    visual_cortex_internal_queue_vlm.put_nowait(visual_scene_data)
                    logging.debug(f"Added VLM analysis to queue: {caption}")
                except:
                        # Queue full, remove oldest and add new. Here it is important to put our analysis so system knows it can use VLM again
                        try:
                            visual_cortex_internal_queue_vlm.get_nowait()
                            visual_cortex_internal_queue_vlm.put_nowait(visual_scene_data)
                            logging.debug(f"Queue full, replaced oldest analysis")
                        except:
                            logging.debug(f"Failed to add VLM analysis to queue")

                #flags to control if we keep looping or if this thread is one and done
                single_run = limitedGPUenv
                got_a_frame = False
            except Exception as e:
                single_run = limitedGPUenv
                got_a_frame = False
                logging.error(f"VLM analysis error: {e}")
                import traceback
                traceback.print_exc()
                update_availiblity(external_stats_queue,True)
        update_availiblity(external_stats_queue,True)

    
class VisualCortex():
    def __init__(self,cortex=visual_cortex_core,vlm=visual_cortex_worker_vlm,nerve=optic_nerve_connection,device_index=0,mpm=False,gpu_to_use=0):
        logging.info("Starting Visual Cortex. This will run at minimum 3 separte processes via multiprocess (nerve,cortex,vlm)")
        if not mpm:
            logging.warning("You MUST pass a multi processing manager instance: multiprocessing.Manager(), using arg: VisualCortex(mpm= multiprocessing.Manager()), to initiate the VisualCortex")
        #processes
        self.visual_processes = {}
        # Initialize nerve processes dictionary
        self.visual_processes['nerve'] = {}

        # Visual queues
        self.external_cortex_queue = mpm.Queue(maxsize=30)
        #raw images from camera
        self.internal_nerve_queue = mpm.Queue(maxsize=1)
        #images shown by generate_image_frame
        self.external_img_queue = mpm.Queue(maxsize=3)
        #holds the output of the vlm
        self.internal_from_vlm_queue = mpm.Queue(maxsize=1)#only hold a single result, this is used to determine if it's ready or not
        self.internal_to_vlm_queue = mpm.Queue(maxsize=1)#only hold a single result, this is used to determine if it's ready or not
        #used to signal on/off VLM, Person Detection, Face Detection
        self.internal_cortex_process_control_queue = mpm.Queue(maxsize=1)#only hold a single dict of what the cortex should be running
        #stat tracker
        self.external_stats_queue = mpm.Queue(maxsize=5)
        #nerve control function
        self.nerve = nerve
        self.fps_capture_target = 30
        #get a copy of default cortex control message (setting)
        ctrl_msg_struct = self._cortex_control_msg()
        #set the GPU to use for the VLM model
        gpu_to_use = gpu_to_use
        # Start visual cortex
        visual_cortex_process = mp.Process(
            target=cortex,
            args=(self.internal_nerve_queue, self.external_cortex_queue, self.external_stats_queue,self.external_img_queue, self.internal_from_vlm_queue, self.internal_to_vlm_queue,self.internal_cortex_process_control_queue,ctrl_msg_struct,vlm,gpu_to_use)
        )
        visual_cortex_process.start()
        self.visual_processes['core'] = visual_cortex_process

        #set the initial visual cortex operational instructions
        _ = self.send_cortex_instructions(ctrl_msg_struct)

        try:
            self.internal_cortex_process_control_queue.put_nowait()
        except:
            logging.error("Unable to send control message to visual cortex, will try one more time")
            try:
                self.internal_cortex_process_control_queue.put(timeout=1)#block for 1 second
            except:
                logging.error("Still Unable to send control message to visual cortex, Visual Cortex may not function correctly")

    def send_cortex_instructions(self, vlm={}, person_detection=True, person_recognition=True):
        """
        Send cortex instructions, validate instructions. These instructions are used to set what the visual cortex is doing
        """
        # Quick type checks
        if not isinstance(vlm, dict):
            logging.error(f"vlm must be dict, got {type(vlm).__name__}")
            return False
        
        if not isinstance(person_detection, bool):
            logging.error(f"person_detection must be bool, got {type(person_detection).__name__}")
            return False
            
        if not isinstance(person_recognition, bool):
            logging.error(f"person_recognition must be bool, got {type(person_recognition).__name__}")
            return False
        
        # Check vlm contents
        valid_keys = {'caption', 'detect', 'point', 'query'}
        for key, value in vlm.items():
            if key not in valid_keys:
                logging.error(f"Invalid vlm key: {key}")
                return False
            
            if key == 'caption' and not isinstance(value, bool):
                logging.error(f"vlm['caption'] must be bool, got {type(value).__name__}")
                return False
                
            if key in {'detect', 'point', 'query'} and not isinstance(value, str):
                logging.error(f"vlm['{key}'] must be str, got {type(value).__name__}")
                return False
        
        # Create message payload
        msg = {
            'vlm': vlm,
            'person_detection': person_detection,
            'person_recognition': person_recognition
        }
        
        # Send instructions with timeout handling
        logging.info("Sending cortex instructions")
        try:
            # Try immediate send first
            self.internal_cortex_process_control_queue.put_nowait(msg)
            logging.info("Cortex instructions sent successfully")
            return True
        except queue.Full:
            logging.warning("Queue full, attempting with timeout...")
            try:
                # Fallback with timeout
                self.internal_cortex_process_control_queue.put(msg, timeout=1)
                logging.info("Cortex instructions sent successfully (with timeout)")
                return True
            except queue.Full:
                logging.error("Failed to send cortex instructions - queue remains full after timeout")
                return False

    def _cortex_control_msg(self):
        """Define the default structure for cortex control messages. Controls what processes it runs on image analysis"""
        return {'vlm': {'caption': True, 
                        'detect': '', 
                        'point': '', 
                        'query': ''},
                'person_detection': True,
                'person_recognition': True}

    def generate_img_frames(self, camera_index=None):
        """Generator function for video streaming from specific camera to web"""
        frame_count = 0
        start_time = time.time()
        last_frame = None  # Cache last frame to prevent blank screens
        
        while True:
            try:
                # Get processed frame (non-blocking to prevent hanging)
                frame_data = self.external_img_queue.get_nowait()#.get(timeout=0.1)
                
                # Filter by camera index if specified
                if camera_index is not None:
                    # Only process frames from the requested camera
                    if frame_data.get('camera_index') != camera_index:
                        continue
                    
                    last_frame = frame_data['frame']  # Cache the frame
                else:
                    last_frame = frame_data['frame']
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', last_frame, 
                                        [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret:
                    continue
                    
                frame_bytes = buffer.tobytes()
                
                # Calculate streaming FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = frame_count / (time.time() - start_time)
                    stats_data = {}
                    if camera_index is not None:
                        stats_data[f'stream_fps_{camera_index}'] = fps
                    else:
                        stats_data['stream_fps_all'] = fps
                    
                    # Update stats with queue management
                    try:
                        self.external_stats_queue.put_nowait(stats_data)
                    except queue.Full:
                        try:
                            _ = self.external_stats_queue.get_nowait()
                            self.external_stats_queue.put_nowait(stats_data)
                        except queue.Empty:
                            pass
                
                # Yield frame in multipart format
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
            except queue.Empty:
                # If we have a cached frame, keep showing it
                if last_frame is not None:
                    ret, buffer = cv2.imencode('.jpg', last_frame, 
                                            [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # Send a placeholder frame if no frames available
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cam_text = f"Camera {camera_index}" if camera_index is not None else "All Cameras"
                    cv2.putText(placeholder, f"No Signal - {cam_text}", (150, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', placeholder)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Small delay to prevent excessive CPU usage
            time.sleep(1/self.fps_capture_target)  # ~30 FPS max if self.fps_capture_target=30, 0.033ms sleep
                
    def start_nerve(self,device_index=0):
        # Start optic nerve process for this camera
        optic_nerve_process = mp.Process(
            target=self.nerve,
            args=(device_index, self.internal_nerve_queue, self.external_stats_queue, self.fps_capture_target)
        )
        optic_nerve_process.start()
        self.visual_processes['nerve'][device_index] = optic_nerve_process

    def stop_nerve(self, device_index=0):
        # Terminate optic nerve process, default to 0
        success = self.process_kill(self.visual_processes['nerve'][device_index])
        return success
    
    def stop_cortex(self):
        # Terminate cortex process
        success = self.process_kill(self.visual_processes['core'])
        return success
    
    def shutdown(self):
        # Shut down all visual functions
        had_failure = False
        # Cortex
        success = self.stop_cortex()
        if not success:
            logging.error("Error shutting down Visual Cortex process")
            had_failure = True
        # Input Devices
        for i, nerve in enumerate(self.visual_processes['nerve']):
            success = self.process_kill(self.visual_processes['nerve'][i])
            if not success:
                logging.error(f"Error shutting down Optic Nerve device index: {i}")
                had_failure = True
        return had_failure
            
    def process_kill(self, process):
        try:
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
            del process
            return True
        except Exception as e:
            logging.error(f"Visual Process Stop Error: {e}")
            return False

        


"""
AUDIO
"""
def auditory_nerve_connection(device_index, internal_nerve_queue, external_stats_queue,sample_rate=16000, chunk_size=512):
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
            internal_nerve_queue.put_nowait(audio_data)
        except:
            logging.warning(f"Audio nerve queue {device_index} data not being consumed fast enough!")
            try:
                # Queue is full, delete last frame to make space for this one to maintain real-time performance
                _ = internal_nerve_queue.get_nowait()
                internal_nerve_queue.put_nowait(audio_data)
            except queue.Empty:
                internal_nerve_queue.put_nowait(audio_data)
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

def auditory_cortex_core(internal_nerve_queue, external_cortex_queue, external_stats_queue,internal_speech_audio_queue,internal_transcription_queue):
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
        SAMPLE_SIZE_REQUIREMENT = 512
        while True:
            
            try:
                # Get audio frame (blocking with timeout) from the nerve
                audio_data = internal_nerve_queue.get_nowait()#.get(timeout=1.0)
                
                # Process immediately - no accumulation needed for 512-sample chunks
                audio_chunk = audio_data['audio_frame']
                capture_timestamp = audio_data['capture_timestamp']
                device_index = audio_data['device_index']
                sample_rate = audio_data['sample_rate']
                duration = SAMPLE_SIZE_REQUIREMENT /sample_rate
                
                # Ensure we have exactly 512 samples for consistent processing
                if len(audio_chunk) != SAMPLE_SIZE_REQUIREMENT:
                    logging.error(f'received audio chunk from nerve {device_index} not equal to 512 samples in auditory_cortex_core')
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

                        #internal_speech_audio_queue is consumed by speech to text and voice recognition process
                        try :
                            #when speech_active is False, that will indicate it's our final chunk with speech audio
                            internal_speech_audio_queue.put_nowait((full_speech.copy(),speech_active,device_index,sample_rate,capture_timestamp))
                        except queue.Full:
                            logging.debug("internal_speech_audio_queue FULL - will clear queue and put data. - consume it faster")
                            try :
                                # Drain the queue
                                while True:
                                    try:
                                        trash = internal_speech_audio_queue.get_nowait()
                                    except queue.Empty:
                                        break
                                #try again to put our data in
                                internal_speech_audio_queue.put_nowait((full_speech.copy(),speech_active,device_index,sample_rate,capture_timestamp))
                            except queue.Full:
                                logging.debug("Tried making space but internal_queue_speech_audio still FULL")

                        if not speech_active:
                                # Clear speech buffers after sending final audio for transcription
                                speech_buffer = []
                                full_speech = []

                     
                    cortex_output = {
                            'transcription': "",
                            'final_transcript': False,
                            'voice_match': False,
                            'voice_probability': 0.0,
                            'device_index': device_index,
                            'speech_detected': speech_active,
                            'capture_timestamp': capture_timestamp,
                            'transcription_timestamp': capture_timestamp,
                            'formatted_time': datetime.fromtimestamp(capture_timestamp).strftime('%H:%M:%S'),
                            'sample_rate': sample_rate,
                            'duration':duration
                        }
                    try:
                        #Replace with transcription data if any exists -note the timestamp will be the past
                        cortex_output = internal_transcription_queue.get_nowait()#don't block
                    except queue.Empty:
                        logging.debug("No data in internal_transcription_queue")
                        
                    # Put processed audio in output queue
                    try:
                        external_cortex_queue.put_nowait(cortex_output)#-This data will be lost if queue is full
                    except:
                        try:
                            #TODO: consider blocking request with .put(timeout=1), note that means for raw frames consumed at the start we would have to catch up and use LIFO 
                            logging.debug("external_cortex_queue FULL - CRITICAL ERROR - important result data could be lost")
                        except:
                            pass


                except Exception as vad_error:
                    print(f"VAD processing error: {vad_error}")
                    continue
                
                # Calculate auditory cortex processing stats
                frame_count += 1
                if frame_count % 50 == 0:  # Update every 50 frames (~1.6 seconds)
                    now = time.time()
                    fps = frame_count / ( now - start_time)
                    stats_dict={'auditory_cortex_fps': fps,
                                'last_auditory_cortex': now}
                    # Update Stats
                    try:
                        external_stats_queue.put_nowait(stats_dict)#-This data will be lost if queue is full
                    except:
                        try:
                            logging.debug("external_stats_queue FULL - Try and consume faster")
                        except:
                            pass
                    
                    
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

def auditory_cortex_worker_stt(internal_speech_audio_queue,internal_transcription_queue):
    """
    This is a multiprocess worker that will process audio chunks known to have speech.
    Uses LIFO - only processes the most recent audio data, skip and remove older data tasks from queue.
    Uses sliding window approach to build up transcript incrementally.
    Includes voice recognition functionality that runs in a separate thread.

    IMPORTANT - Currently does NOT turely support multi device because of how the LIFO data queue works and there is no device transcript tracking

    """
    from stt import SpeechTranscriber
    import logging
    import queue
    import gc
    import threading
    import numpy as np
    import concurrent.futures
    # At module level:
    voice_recognition_worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    
    def voice_recognition(chunk_to_process: np.ndarray, result_container: list) -> None:
        """
        Placeholder function for voice recognition.
        
        Args:
            chunk_to_process: Audio chunk as numpy array
            result_container: List to store results [match_result, probability]
        """
        import gc
        try:
            # TODO: Implement actual voice recognition logic
            # This is a placeholder implementation
            
            # Step 1: Extract audio embedding from chunk_to_process
            # embedding = extract_audio_embedding(chunk_to_process)
            
            # Step 2: Compare with embedding database
            # similarities = compare_with_database(embedding)
            
            # Step 3: Find best match above threshold
            # threshold = 0.85  # Adjust as needed
            # best_match_id, best_similarity = find_best_match(similarities, threshold)
            
            # Placeholder return values
            # For now, simulate no match
            result_container[0] = False
            result_container[1] = 0.0
            
            # Example of what a match would look like:
            # if best_similarity > threshold:
            #     result_container[0] = best_match_id
            #     result_container[1] = best_similarity
            # else:
            #     result_container[0] = False
            #     result_container[1] = best_similarity
            
        except Exception as e:
            logging.error(f"Voice recognition error: {e}")
            result_container[0] = False
            result_container[1] = 0.0
        finally:
            # Explicit cleanup
            # del embedding, similarities, etc.
            gc.collect()  # force garbage collection

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
    
    
    # Voice recognition will be handled per-chunk with individual threads
    
    # Parameters (at 16kHz sample rate)
    min_chunk_size = 80000  # 5 seconds minimum before starting transcription (if no final audio flag received)
    overlap_samples = 16000  # 1 second overlap for word boundary detection
    incremental_threshold = 48000  # 3 seconds of new audio before processing again
    
    try:
        while True:
            try:
                # LIFO Implementation: Get most recent data
                latest_task = None
                items_skipped = 0
                
                # Get first item (blocking)
                try:
                    latest_task = internal_speech_audio_queue.get_nowait()#.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if latest_task is None:  # Shutdown signal
                    break
                
                # Drain queue to get the most recent item (LIFO behavior)
                while True:
                    try:
                        newer_task = internal_speech_audio_queue.get_nowait()
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
                        # Store result in a shared container
                        voice_result_container = [False, 0.0]  # [match_result, probability]
                        
                        # Start voice recognition in a separate thread worker
                        vr_future_result =voice_recognition_worker.submit(voice_recognition, chunk_to_process, voice_result_container)
                        
                        # Transcribe the audio chunk (runs in parallel with voice recognition)
                        raw_transcription = speech_transcriber.transcribe(chunk_to_process)
                        
                        # Get (blocking) voice recognition results
                        vr_future_result.result()
                        
                        # Set recognition results from the container
                        voice_match_result = voice_result_container[0]
                        voice_probability = voice_result_container[1]
                        
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
                            internal_transcription_queue.put_nowait(transcript_data)
                            logging.debug(f"Sent {'final' if not more_speech_coming else 'interim'} transcript_data: {transcript_data} chars")
                        except queue.Full:
                            logging.debug("Transcription queue Full - consume faster - will start clearing!")
                            # Clear the transcription queue before putting new data (LIFO behavior)
                            while True:
                                try:
                                    _ = internal_transcription_queue.get_nowait()
                                except queue.Empty:
                                    logging.debug("Cleared old transcription from queue")
                                    internal_transcription_queue.put_nowait(transcript_data)
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
        print(f"auditory_cortex_worker_stt Primary Loop Error: {e}")

    finally:
        # No cleanup needed
        pass

class AuditoryCortex():
    """
    manage all running audio functions. on init will start the auditory cortex, speech to text processes and default to connection
    with sound input device 0
    """
    def __init__(self,cortex=auditory_cortex_core,stt=auditory_cortex_worker_stt,nerve=auditory_nerve_connection,device_index=0,mpm=False):
        logging.info("Starting Visual Cortex. This will run at minimum 3 separte processes via multiprocess (nerve,cortex,stt)")
        if not mpm:
            logging.warning("You MUST pass a multi processing manager instance: multiprocessing.Manager(), using arg: AuditoryCortex(mpm= multiprocessing.Manager()), to initiate the AuditoryCortex")
        #processes
        self.auditory_processes = {}
        self.auditory_processes['nerve'] = {}

        #Data queues
        self.external_cortex_queue = mpm.Queue(maxsize=30)
        #stat tracker
        self.external_stats_queue = mpm.Queue(maxsize=5)
        #raw sound capture
        self.internal_nerve_queue = mpm.Queue(maxsize=20)
        # internally used by Audio Cortex to hold speech audio that needs transcription
        self.internal_speech_audio_queue = mpm.Queue(maxsize=100)  #100 chunks of 32ms = ~3200ms (3.2 second) buffer
        # internally used by Audio Cortex to hold speech to text results (transcriptions)
        self.internal_transcription_queue = mpm.Queue(maxsize=5)
        #nerve controller function
        self.nerve = nerve
        
        #Primary hub process
        auditory_cortex_process = mp.Process(
            target=cortex,
            args=(self.internal_nerve_queue, 
                  self.external_cortex_queue,
                  self.external_stats_queue, 
                  self.internal_speech_audio_queue,
                  self.internal_transcription_queue)
        )
        auditory_cortex_process.start()
        self.auditory_processes['core'] = auditory_cortex_process

        #CPU intense speech to text and voice recognition process
        stt_worker = mp.Process(
            target=stt,
            args=(self.internal_speech_audio_queue,
                  self.internal_transcription_queue)
        )
        stt_worker.start()
        self.auditory_processes['stt'] = stt_worker

        

    def start_nerve(self,device_index=0):
        #I/O raw audio data capture
        auditory_nerve_process = mp.Process(
            target=self.nerve,
            args=(device_index, 
                  self.internal_nerve_queue, 
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

    
  
"""
TEMPORAL LOBE
"""

class TemporalLobe:
    """
    Combines visual and auditory cortex data into unified temporal awareness.
    Buffers data for a specified time window and creates unified updates by rolling up
    non-default values from the most recent to oldest frames.
    """
    
    def __init__(self, visual_cortex=None, auditory_cortex=None, buffer_duration=1.0, update_interval=1.0, mpm=None):
        """
        Initialize TemporalLobe with visual and auditory cortex instances.
        
        Args:
            visual_cortex: VisualCortex instance
            auditory_cortex: AuditoryCortex instance  
            buffer_duration: How long to buffer data in seconds (default 1.0)
            update_interval: How often to create unified updates in seconds (default 1.0)
            mpm: Multiprocessing manager for creating queues
        """
        if not mpm:
            logging.warning("You should pass a multiprocessing.Manager() instance for real-time queues")
            
        self.visual_cortex = visual_cortex
        self.auditory_cortex = auditory_cortex
        self.buffer_duration = buffer_duration
        self.update_interval = update_interval

        #UI Helper
        self.persistent_transcripts = []
        
        # Data buffers
        self.visual_buffer = deque()
        self.audio_buffer = deque()
        
        # Real-time pass-through queues
        if mpm:
            self.audio_realtime_queue = mpm.Queue(maxsize=30)
            self.visual_realtime_queue = mpm.Queue(maxsize=30)
        else:
            self.audio_realtime_queue = None
            self.visual_realtime_queue = None
        
        # Unified state tracking
        self.last_unified_state = self._create_empty_unified_state()
        
        # Control flags
        self.running = False
        self.collection_thread = None
        
        # Statistics - Updated to include all FPS stats
        self.stats = {
            # TemporalLobe processing stats
            'visual_frames_processed': 0,
            'audio_frames_processed': 0,
            'unified_updates_created': 0,
            'last_update_time': None,
            'audio_realtime_frames_relayed': 0,
            'visual_realtime_frames_relayed': 0,
            
            # Visual system FPS stats (from external_stats_queue)
            'visual_cortex_fps': 0.0,
            'last_visual_cortex': None,
            'VLM_availible': True,  # VLM availability status
            
            # Audio system FPS stats (from external_stats_queue)  
            'auditory_cortex_fps': 0.0,
            'last_auditory_cortex': None,
            
            # Per-device nerve FPS stats (dynamic based on active devices)
            # Format: 'optic_nerve_fps_0': fps_value, 'last_optic_nerve_0': timestamp
            # Format: 'auditory_nerve_fps_0': fps_value, 'last_auditory_nerve_0': timestamp
            # Format: 'stream_fps_0': fps_value (for video streaming)
        }
        
        logging.info(f"TemporalLobe initialized with {buffer_duration}s buffer, {update_interval}s updates")

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
            'voice_match': False,
            'voice_probability': 0.0,
            'audio_device_index': None,
            'transcription_timestamp': None,
            
            # Frame counts for this update
            'visual_frames_in_update': 0,
            'audio_frames_in_update': 0,
            
            # Processing stats
            'buffer_visual_frames': 0,
            'buffer_audio_frames': 0
        }

    def _is_default_value(self, key, value):
        """Check if a value is considered 'default' and should be replaced"""
        defaults = {
            'person_detected': False,
            'person_match': False, 
            'person_match_probability': 0.0,
            'caption': "",
            'vlm_timestamp': None,
            'speech_detected': False,
            'transcription': "",
            'final_transcript': False,
            'voice_match': False,
            'voice_probability': 0.0,
            'transcription_timestamp': None
        }
        
        return key in defaults and value == defaults[key]
    
    def _collect_frames(self):
        """Background thread to continuously collect frames from both cortexes and relay to real-time queues"""
        logging.info("TemporalLobe frame collection started")
        
        while self.running:
            current_time = time.time()
            
            # Collect visual frames
            if self.visual_cortex:
                try:
                    visual_data = self.visual_cortex.external_cortex_queue.get_nowait()
                    visual_data['collection_timestamp'] = current_time
                    
                    # Add to buffer for temporal processing
                    self.visual_buffer.append(visual_data.copy())
                    self.stats['visual_frames_processed'] += 1
                    
                    # Relay to real-time queue for immediate consumption
                    if self.visual_realtime_queue:
                        try:
                            self.visual_realtime_queue.put_nowait(visual_data)
                            self.stats['visual_realtime_frames_relayed'] += 1
                        except queue.Full:
                            # If real-time queue is full, remove oldest and add new (LIFO-like behavior)
                            try:
                                _ = self.visual_realtime_queue.get_nowait()
                                self.visual_realtime_queue.put_nowait(visual_data)
                                self.stats['visual_realtime_frames_relayed'] += 1
                            except queue.Empty:
                                # Queue was just emptied by consumer, try again
                                try:
                                    self.visual_realtime_queue.put_nowait(visual_data)
                                    self.stats['visual_realtime_frames_relayed'] += 1
                                except queue.Full:
                                    logging.debug("Visual realtime queue full - frame dropped")
                                    
                except queue.Empty:
                    pass
            
            # Collect audio frames  
            if self.auditory_cortex:
                try:
                    audio_data = self.auditory_cortex.external_cortex_queue.get_nowait()
                    audio_data['collection_timestamp'] = current_time
                    
                    # Add to buffer for temporal processing
                    self.audio_buffer.append(audio_data.copy())
                    self.stats['audio_frames_processed'] += 1
                    
                    # Relay to real-time queue for immediate consumption
                    if self.audio_realtime_queue:
                        try:
                            self.audio_realtime_queue.put_nowait(audio_data)
                            self.stats['audio_realtime_frames_relayed'] += 1
                        except queue.Full:
                            # If real-time queue is full, remove oldest and add new (LIFO-like behavior)
                            try:
                                _ = self.audio_realtime_queue.get_nowait()
                                self.audio_realtime_queue.put_nowait(audio_data)
                                self.stats['audio_realtime_frames_relayed'] += 1
                            except queue.Empty:
                                # Queue was just emptied by consumer, try again
                                try:
                                    self.audio_realtime_queue.put_nowait(audio_data)
                                    self.stats['audio_realtime_frames_relayed'] += 1
                                except queue.Full:
                                    logging.debug("Audio realtime queue full - frame dropped")
                                    
                except queue.Empty:
                    pass
            
            # Collect visual stats
            if self.visual_cortex:
                try:
                    visual_stats = self.visual_cortex.external_stats_queue.get_nowait()
                    # Update our stats dictionary with visual system stats
                    self.stats.update(visual_stats)
                    logging.debug(f"Updated visual stats: {visual_stats}")
                except queue.Empty:
                    pass
            
            # Collect audio stats  
            if self.auditory_cortex:
                try:
                    audio_stats = self.auditory_cortex.external_stats_queue.get_nowait()
                    # Update our stats dictionary with audio system stats
                    self.stats.update(audio_stats)
                    logging.debug(f"Updated audio stats: {audio_stats}")
                except queue.Empty:
                    pass
            
            # Clean old frames from buffers
            self._clean_old_frames()
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.001)
        
        logging.info("TemporalLobe frame collection stopped")

    def _clean_old_frames(self):
        """Remove frames older than buffer_duration from both buffers"""
        current_time = time.time()
        cutoff_time = current_time - self.buffer_duration
        
        # Clean visual buffer
        while self.visual_buffer and self.visual_buffer[0]['collection_timestamp'] < cutoff_time:
            self.visual_buffer.popleft()
            
        # Clean audio buffer  
        while self.audio_buffer and self.audio_buffer[0]['collection_timestamp'] < cutoff_time:
            self.audio_buffer.popleft()

    def get_unified_state(self):
        """
        Create unified state by rolling up non-default values from most recent to oldest frames.
        Returns the combined state from both visual and audio data in the buffer.
        """
        unified_state = self._create_empty_unified_state()
        
        # Convert buffers to lists and sort by timestamp (most recent first)
        visual_frames = sorted(list(self.visual_buffer), 
                             key=lambda x: x.get('capture_timestamp', 0), reverse=True)
        audio_frames = sorted(list(self.audio_buffer),
                            key=lambda x: x.get('capture_timestamp', 0), reverse=True)
        
        # Update frame counts
        unified_state['visual_frames_in_update'] = len(visual_frames)  
        unified_state['audio_frames_in_update'] = len(audio_frames)
        unified_state['buffer_visual_frames'] = len(self.visual_buffer)
        unified_state['buffer_audio_frames'] = len(self.audio_buffer)
        
        # Set timestamp to most recent frame timestamp
        most_recent_timestamp = 0
        if visual_frames:
            most_recent_timestamp = max(most_recent_timestamp, visual_frames[0].get('capture_timestamp', 0))
        if audio_frames:
            most_recent_timestamp = max(most_recent_timestamp, audio_frames[0].get('capture_timestamp', 0))
        
        if most_recent_timestamp > 0:
            unified_state['timestamp'] = most_recent_timestamp
            unified_state['formatted_time'] = datetime.fromtimestamp(most_recent_timestamp).strftime('%H:%M:%S')

        # Process visual frames (most recent first)
        for frame in visual_frames:
            # Always update camera index from most recent frame
            if unified_state['visual_camera_index'] is None:
                unified_state['visual_camera_index'] = frame.get('camera_index')
            
            # Roll up non-default visual values
            for key in ['person_detected', 'person_match', 'person_match_probability', 'caption', 'vlm_timestamp']:
                if key in frame and self._is_default_value(key, unified_state[key]):
                    if not self._is_default_value(key, frame[key]):
                        unified_state[key] = frame[key]
                        logging.debug(f"Visual rollup: {key} = {frame[key]}")

        # Process audio frames (most recent first)  
        for frame in audio_frames:
            # Always update device index from most recent frame
            if unified_state['audio_device_index'] is None:
                unified_state['audio_device_index'] = frame.get('device_index')
            
            # Roll up non-default audio values
            for key in ['speech_detected', 'transcription', 'final_transcript', 
                       'voice_match', 'voice_probability', 'transcription_timestamp']:
                if key in frame and self._is_default_value(key, unified_state[key]):
                    if not self._is_default_value(key, frame[key]):
                        unified_state[key] = frame[key]
                        logging.debug(f"Audio rollup: {key} = {frame[key]}")

        self.stats['unified_updates_created'] += 1
        self.stats['last_update_time'] = time.time()
        self.last_unified_state = unified_state
        
        
        return unified_state
    
    def start_collection(self):
        """Start background frame collection"""
        if self.running:
            logging.warning("TemporalLobe collection already running")
            return
            
        self.running = True
        self.collection_thread = threading.Thread(target=self._collect_frames, daemon=True)
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
        
        # Clear buffers
        self.visual_buffer.clear()
        self.audio_buffer.clear()
        
        if had_failure:
            logging.error("Temporal Lobe shutdown completed with some failures")
        else:
            logging.info("Temporal Lobe shutdown completed successfully")
            
        return not had_failure

    def get_stats(self):
        """Get comprehensive processing statistics including all FPS data"""
        return {
            **self.stats,
            'buffer_visual_size': len(self.visual_buffer),
            'buffer_audio_size': len(self.audio_buffer),
            'collection_running': self.running
        }
    
    def get_last_unified_state(self):
        """Get the last created unified state without creating a new one"""
        return self.last_unified_state


"""
HTML Interface 
"""

@app.route('/')
def index():
    """Steampunk-styled HTML page with audio controls and display"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TemporalLobe Control Interface</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
        
        body {
            background: linear-gradient(45deg, #1a0f0a, #2d1b13, #1a0f0a);
            color: #d4af37;
            font-family: 'Orbitron', monospace;
            margin: 0;
            padding: 20px;
            min-height: 100vh;
            background-attachment: fixed;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            font-weight: 900;
            text-shadow: 0 0 20px #d4af37, 0 0 40px #d4af37;
            margin: 0;
            letter-spacing: 3px;
        }
        
        /* Stats Panel */
        .stats-panel {
            background: linear-gradient(135deg, #1a1a1a 0%, #2a1810 50%, #1a1a1a 100%);
            border: 3px solid #8b4513;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 
                inset 0 0 20px rgba(212, 175, 55, 0.1),
                0 0 30px rgba(139, 69, 19, 0.3);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .stat-item {
            background: rgba(42, 24, 16, 0.7);
            border: 1px solid #654321;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        
        .stat-item.error {
            background: rgba(139, 0, 0, 0.3);
            border-color: #ff6b6b;
        }
        
        .stat-label {
            font-size: 0.9em;
            color: #b8860b;
            margin-bottom: 5px;
        }
        
        .stat-value {
            font-size: 1.4em;
            font-weight: 700;
            color: #ffd700;
        }
        
        /* Audio Section */
        .audio-section {
            background: linear-gradient(135deg, #1a1a1a 0%, #2a1810 50%, #1a1a1a 100%);
            border: 3px solid #8b4513;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 
                inset 0 0 20px rgba(212, 175, 55, 0.1),
                0 0 30px rgba(139, 69, 19, 0.3);
        }
        
        .section-title {
            font-size: 1.8em;
            font-weight: 700;
            margin-bottom: 20px;
            text-align: center;
            color: #d4af37;
            text-shadow: 0 0 10px #d4af37;
        }
        
        .mic-controls {
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-bottom: 25px;
        }
        
        .mic-unit {
            text-align: center;
            background: linear-gradient(145deg, #b87333 0%, #cd853f 20%, #daa520 40%, #cd853f 70%, #8b4513 100%);
            border: 3px solid #654321;
            border-radius: 12px;
            padding: 20px;
            min-width: 200px;
            box-shadow: 
                inset 0 2px 4px rgba(255, 255, 255, 0.2),
                inset 0 -2px 4px rgba(0, 0, 0, 0.3),
                0 4px 8px rgba(0, 0, 0, 0.4);
            position: relative;
        }
        
        .mic-unit::before {
            content: '';
            position: absolute;
            top: 5px;
            left: 5px;
            right: 5px;
            height: 30%;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.3) 0%, rgba(255, 255, 255, 0.1) 50%, transparent 100%);
            border-radius: 8px 8px 0 0;
            pointer-events: none;
        }
        
        .mic-label {
            font-size: 1.2em;
            font-weight: 700;
            margin-bottom: 15px;
            color: #2f1b14;
            text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.3);
        }
        
        .control-button {
            background: linear-gradient(135deg, #8b4513, #a0522d, #8b4513);
            border: 2px solid #d4af37;
            border-radius: 8px;
            color: #fff;
            padding: 12px 24px;
            font-family: 'Orbitron', monospace;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 15px;
        }
        
        .control-button:hover {
            background: linear-gradient(135deg, #a0522d, #cd853f, #a0522d);
            box-shadow: 0 0 15px rgba(212, 175, 55, 0.5);
        }
        
        .control-button.active {
            background: linear-gradient(135deg, #228b22, #32cd32, #228b22);
            box-shadow: 0 0 20px rgba(50, 205, 50, 0.6);
        }
        
        .status-lights {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 10px;
        }
        
        .status-light {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            border: 2px solid #333;
            transition: all 0.3s ease;
        }
        
        .status-light.active-yellow {
            background: #ffd700;
            box-shadow: 0 0 15px #ffd700;
        }
        
        .status-light.speech-orange {
            background: #ff8c00;
            box-shadow: 0 0 15px #ff8c00;
        }
        
        .status-light.inactive {
            background: #444;
        }
        
        .transcripts {
            background: rgba(26, 15, 10, 0.9);
            border: 1px solid #654321;
            border-radius: 8px;
            padding: 15px;
            max-height: 150px;
            overflow-y: auto;
            font-family: 'Arial', sans-serif;
        }
        
        .transcript-item {
            background: rgba(42, 24, 16, 0.6);
            border-left: 3px solid #d4af37;
            padding: 8px 12px;
            margin: 5px 0;
            font-size: 1.1em;
            border-radius: 4px;
            font-family: 'Arial', sans-serif;
            color: #f0f0f0;
            line-height: 1.5;
        }
        
        /* Video Section */
        .video-section {
            background: linear-gradient(135deg, #1a1a1a 0%, #2a1810 50%, #1a1a1a 100%);
            border: 3px solid #8b4513;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 
                inset 0 0 20px rgba(212, 175, 55, 0.1),
                0 0 30px rgba(139, 69, 19, 0.3);
        }
        
        .camera-controls {
            display: flex;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
            margin-bottom: 25px;
        }
        
        .camera-unit {
            text-align: center;
            background: linear-gradient(145deg, #b87333 0%, #cd853f 20%, #daa520 40%, #cd853f 70%, #8b4513 100%);
            border: 3px solid #654321;
            border-radius: 12px;
            padding: 20px;
            min-width: 300px;
            box-shadow: 
                inset 0 2px 4px rgba(255, 255, 255, 0.2),
                inset 0 -2px 4px rgba(0, 0, 0, 0.3),
                0 4px 8px rgba(0, 0, 0, 0.4);
            position: relative;
        }
        
        .camera-unit::before {
            content: '';
            position: absolute;
            top: 5px;
            left: 5px;
            right: 5px;
            height: 30%;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.3) 0%, rgba(255, 255, 255, 0.1) 50%, transparent 100%);
            border-radius: 8px 8px 0 0;
            pointer-events: none;
        }
        
        .camera-label {
            font-size: 1.2em;
            font-weight: 700;
            margin-bottom: 15px;
            color: #2f1b14;
            text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.3);
        }
        
        .video-feed {
            width: 280px;
            height: 210px;
            border: 2px solid #654321;
            border-radius: 8px;
            background: #000;
            margin: 15px auto;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ff4444;
            font-family: 'Orbitron', monospace;
            font-size: 1.2em;
            font-weight: 700;
            text-shadow: 0 0 10px #ff4444;
            position: relative;
            overflow: hidden;
        }

        .video-feed.has-video {
            color: transparent;
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }
        
        /* Full-width captions section */
        .captions-section {
            margin-top: 25px;
            background: rgba(26, 15, 10, 0.9);
            border: 1px solid #654321;
            border-radius: 8px;
            padding: 20px;
        }
        
        .captions-title {
            font-size: 1.3em;
            font-weight: 700;
            margin-bottom: 15px;
            color: #d4af37;
            text-align: center;
        }
        
        .captions-container {
            max-height: 200px;
            overflow-y: auto;
            font-family: 'Arial', sans-serif;
        }
        
        .caption-item {
            background: rgba(42, 24, 16, 0.6);
            border-left: 3px solid #d4af37;
            padding: 12px 15px;
            margin: 8px 0;
            font-size: 1.1em;
            border-radius: 4px;
            line-height: 1.6;
            word-wrap: break-word;
            font-family: 'Arial', sans-serif;
        }
        
        .caption-item .camera-label {
            font-size: 0.9em;
            color: #b8860b;
            margin-bottom: 5px;
            font-weight: 700;
            font-family: 'Orbitron', monospace;
        }
        
        .caption-text {
            color: #f0f0f0;
            font-weight: 500;
        }
        
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        
        .pulsing {
            animation: pulse 2s infinite;
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(26, 15, 10, 0.3);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #654321;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #8b4513;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔅 TEMPORAL LOBE SYSTEM 🔅</h1>
        </div>
        
        <!-- Stats Panel -->
        <div class="stats-panel">
            <div class="section-title">System Performance</div>
            <div class="stats-grid" id="stats-grid">
                <!-- Stats will be populated by JavaScript -->
            </div>
        </div>
        
        <!-- Audio Section -->
        <div class="audio-section">
            <div class="section-title">Auditory Cortex</div>
            <div class="mic-controls">
                <div class="mic-unit">
                    <div class="mic-label">Microphone 0</div>
                    <button class="control-button" id="mic-0-btn" onclick="toggleMic(0)">ACTIVATE</button>
                    <div class="status-lights">
                        <div class="status-light inactive" id="mic-0-active" title="Device Active"></div>
                        <div class="status-light inactive" id="mic-0-speech" title="Speech Detected"></div>
                    </div>
                </div>
                <div class="mic-unit">
                    <div class="mic-label">Microphone 1</div>
                    <button class="control-button" id="mic-1-btn" onclick="toggleMic(1)">ACTIVATE</button>
                    <div class="status-lights">
                        <div class="status-light inactive" id="mic-1-active" title="Device Active"></div>
                        <div class="status-light inactive" id="mic-1-speech" title="Speech Detected"></div>
                    </div>
                </div>
            </div>
            <div class="transcripts" id="transcripts">
                <div style="text-align: center; color: #888;">Recent Transcriptions</div>
            </div>
        </div>
        
        <!-- Video Section -->
        <div class="video-section">
            <div class="section-title">Visual Cortex</div>
            <div class="camera-controls">
                <div class="camera-unit">
                    <div class="camera-label">Camera 0</div>
                    <div class="video-feed" id="video-0">No Signal</div>
                    <button class="control-button" id="cam-0-btn" onclick="toggleCamera(0)">ACTIVATE</button>
                    <div class="status-lights">
                        <div class="status-light inactive" id="cam-0-active" title="Camera Active"></div>
                        <div class="status-light inactive" id="cam-0-person" title="Person Detected"></div>
                    </div>
                </div>
                <div class="camera-unit">
                    <div class="camera-label">Camera 1</div>
                    <div class="video-feed" id="video-1">No Signal</div>
                    <button class="control-button" id="cam-1-btn" onclick="toggleCamera(1)">ACTIVATE</button>
                    <div class="status-lights">
                        <div class="status-light inactive" id="cam-1-active" title="Camera Active"></div>
                        <div class="status-light inactive" id="cam-1-person" title="Person Detected"></div>
                    </div>
                </div>
                <div class="camera-unit">
                    <div class="camera-label">Camera 2</div>
                    <div class="video-feed" id="video-2">No Signal</div>
                    <button class="control-button" id="cam-2-btn" onclick="toggleCamera(2)">ACTIVATE</button>
                    <div class="status-lights">
                        <div class="status-light inactive" id="cam-2-active" title="Camera Active"></div>
                        <div class="status-light inactive" id="cam-2-person" title="Person Detected"></div>
                    </div>
                </div>
            </div>
            
            <!-- Full-width captions section -->
            <div class="captions-section">
                <div class="captions-title">Scene Analysis & Captions</div>
                <div class="captions-container" id="all-captions">
                    <div style="text-align: center; color: #888;">Recent Scene Descriptions</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // State tracking
        let micStates = {0: false, 1: false};
        let camStates = {0: false, 1: false, 2: false};
        let recentTranscripts = [];
        
        // Store captions persistently for all cameras
        let persistentCaptions = [];
        
        // Microphone Control
        async function toggleMic(index) {
            const btn = document.getElementById(`mic-${index}-btn`);
            const isActive = micStates[index];
            
            try {
                const endpoint = isActive ? `/stop_audio/${index}` : `/start_audio/${index}`;
                const response = await fetch(endpoint);
                const result = await response.json();
                
                if (response.ok) {
                    micStates[index] = !isActive;
                    updateMicUI(index);
                }
            } catch (error) {
                console.error('Microphone toggle error:', error);
            }
        }
        
        function updateMicUI(index) {
            const btn = document.getElementById(`mic-${index}-btn`);
            const activeLight = document.getElementById(`mic-${index}-active`);
            
            if (micStates[index]) {
                btn.textContent = 'DEACTIVATE';
                btn.classList.add('active');
                activeLight.classList.remove('inactive');
                activeLight.classList.add('active-yellow');
            } else {
                btn.textContent = 'ACTIVATE';
                btn.classList.remove('active');
                activeLight.classList.remove('active-yellow');
                activeLight.classList.add('inactive');
                // Also clear speech indicator
                const speechLight = document.getElementById(`mic-${index}-speech`);
                speechLight.classList.remove('speech-orange');
                speechLight.classList.add('inactive');
            }
        }
        
        // Camera Control
        async function toggleCamera(index) {
            const btn = document.getElementById(`cam-${index}-btn`);
            const video = document.getElementById(`video-${index}`);
            const isActive = camStates[index];
            
            try {
                const endpoint = isActive ? `/stop/${index}` : `/start/${index}`;
                const response = await fetch(endpoint);
                const result = await response.json();
                
                if (response.ok) {
                    camStates[index] = !isActive;
                    updateCameraUI(index);
                }
            } catch (error) {
                console.error('Camera toggle error:', error);
            }
        }
        
        function updateCameraUI(index) {
            const btn = document.getElementById(`cam-${index}-btn`);
            const video = document.getElementById(`video-${index}`);
            const activeLight = document.getElementById(`cam-${index}-active`);
            
            if (camStates[index]) {
                btn.textContent = 'DEACTIVATE';
                btn.classList.add('active');
                activeLight.classList.remove('inactive');
                activeLight.classList.add('active-yellow');
                
                // Set video feed as background image and hide "No Signal" text
                video.style.backgroundImage = `url(/video_feed/${index})`;
                video.classList.add('has-video');
                video.textContent = '';
            } else {
                btn.textContent = 'ACTIVATE';
                btn.classList.remove('active');
                activeLight.classList.remove('active-yellow');
                activeLight.classList.add('inactive');
                
                // Remove video feed and show "No Signal" text
                video.style.backgroundImage = '';
                video.classList.remove('has-video');
                video.textContent = 'No Signal';
                
                // Also clear person detection
                const personLight = document.getElementById(`cam-${index}-person`);
                personLight.classList.remove('speech-orange');
                personLight.classList.add('inactive');
            }
        }
        
        // Update stats display dynamically from all stats keys
        function updateStats(stats) {
            const statsGrid = document.getElementById('stats-grid');
            
            // Helper function to format display names from keys
            const formatDisplayName = (key) => {
                return key
                    .replace(/_/g, ' ')  // Replace underscores with spaces
                    .split(' ')
                    .map(word => word.charAt(0).toUpperCase() + word.slice(1))  // Capitalize each word
                    .join(' ');
            };
            
            // Helper function to format values
            const formatValue = (key, value) => {
                if (value === undefined || value === null) return 'N/A';
                
                // Format FPS values
                if (key.includes('fps')) {
                    return typeof value === 'number' ? value.toFixed(1) : value.toString();
                }
                
                // Format boolean values
                if (typeof value === 'boolean') {
                    if (key.includes('running')) return value ? 'ACTIVE' : 'STOPPED';
                    if (key.includes('availible') || key.includes('available')) return value ? 'YES' : 'BUSY';
                    return value ? 'TRUE' : 'FALSE';
                }
                
                // Format timestamps (values that look like Unix timestamps)
                if (key.includes('last_') && typeof value === 'number' && value > 1000000000) {
                    const ageSeconds = Date.now() / 1000 - value;
                    return `${ageSeconds.toFixed(1)}s ago`;
                }
                
                // Default formatting
                return value.toString();
            };
            
            // Sort keys for better organization
            const sortedKeys = Object.keys(stats).sort((a, b) => {
                // Priority order for better visual organization
                const priority = {
                    'visual_frames_processed': 1,
                    'audio_frames_processed': 2,
                    'unified_updates_created': 3,
                    'buffer_visual_size': 4,
                    'buffer_audio_size': 5,
                    'collection_running': 6,
                    'visual_cortex_fps': 7,
                    'auditory_cortex_fps': 8,
                    'VLM_availible': 9
                };
                
                const priorityA = priority[a] || 100;
                const priorityB = priority[b] || 100;
                
                if (priorityA !== priorityB) {
                    return priorityA - priorityB;
                }
                
                // Secondary sort: FPS stats together, then alphabetical
                if (a.includes('fps') && !b.includes('fps')) return -1;
                if (!a.includes('fps') && b.includes('fps')) return 1;
                
                return a.localeCompare(b);
            });
            
            // Build HTML for all stats
            let html = '';
            sortedKeys.forEach(key => {
                const displayName = formatDisplayName(key);
                const displayValue = formatValue(key, stats[key]);
                
                html += `
                <div class="stat-item">
                    <div class="stat-label">${displayName}</div>
                    <div class="stat-value">${displayValue}</div>
                </div>
                `;
            });
            
            statsGrid.innerHTML = html;
        }

        // Store transcripts persistently
        let persistentTranscripts = []; 

        // Update transcripts
        function updateTranscripts(newTranscripts) {
            const container = document.getElementById('transcripts');
            
            // Add new transcripts to persistent storage (avoid duplicates)
            newTranscripts.forEach(transcript => {
                if (!persistentTranscripts.includes(transcript)) {
                    persistentTranscripts.unshift(transcript); // Add to beginning
                }
            });
            
            // Keep only last 5 transcripts
            persistentTranscripts = persistentTranscripts.slice(0, 5);
            
            if (persistentTranscripts.length === 0) {
                container.innerHTML = '<div style="text-align: center; color: #888;">Recent Transcriptions</div>';
                return;
            }
            
            const html = persistentTranscripts.map(transcript => 
                `<div class="transcript-item">${transcript}</div>`
            ).join('');
            container.innerHTML = html;
        }
        
        // Update all captions in unified section
        function updateAllCaptions() {
            const container = document.getElementById('all-captions');
            
            if (persistentCaptions.length === 0) {
                container.innerHTML = '<div style="text-align: center; color: #888;">Recent Scene Descriptions</div>';
                return;
            }
            
            const html = persistentCaptions.map(captionData => 
                `<div class="caption-item">
                    <div class="camera-label">Camera ${captionData.camera}</div>
                    <div class="caption-text">${captionData.caption}</div>
                </div>`
            ).join('');
            container.innerHTML = html;
        }
        
        // Update status indicators
        function updateStatusFromUnified(unified) {
            // Update speech detection indicators
            for (let i = 0; i < 2; i++) {
                const speechLight = document.getElementById(`mic-${i}-speech`);
                if (unified.speech_detected && unified.audio_device_index === i) {
                    speechLight.classList.remove('inactive');
                    speechLight.classList.add('speech-orange');
                } else {
                    speechLight.classList.remove('speech-orange');
                    speechLight.classList.add('inactive');
                }
            }
            
            // Update person detection indicators
            for (let i = 0; i < 3; i++) {
                const personLight = document.getElementById(`cam-${i}-person`);
                if (unified.person_detected && unified.visual_camera_index === i) {
                    personLight.classList.remove('inactive');
                    personLight.classList.add('speech-orange');
                } else {
                    personLight.classList.remove('speech-orange');
                    personLight.classList.add('inactive');
                }
            }
        }
        
        // Fetch and update all data
        async function updateData() {
            try {
                // Get stats
                const statsResponse = await fetch('/stats');
                if (statsResponse.ok) {
                    const stats = await statsResponse.json();
                    updateStats(stats);
                } else {
                    console.error('Failed to fetch stats:', statsResponse.status);
                }
                
                // Get unified state
                const unifiedResponse = await fetch('/unified_state');
                if (unifiedResponse.ok) {
                    const unified = await unifiedResponse.json();
                    updateStatusFromUnified(unified);
                } else {
                    console.error('Failed to fetch unified state:', unifiedResponse.status);
                }
                
                // Get recent transcripts
                const transcriptsResponse = await fetch('/recent_transcripts');
                if (transcriptsResponse.ok) {
                    const transcripts = await transcriptsResponse.json();
                    updateTranscripts(transcripts.transcripts || []);
                } else {
                    console.error('Failed to fetch transcripts:', transcriptsResponse.status);
                }
                
                // Get recent captions for each camera and store persistently
                for (let i = 0; i < 3; i++) {
                    const captionsResponse = await fetch(`/recent_captions/${i}`);
                    if (captionsResponse.ok) {
                        const captions = await captionsResponse.json();
                        
                        // Add new captions to persistent storage
                        (captions.captions || []).forEach(caption => {
                            if (caption.trim()) {
                                const captionData = {
                                    camera: i,
                                    caption: caption,
                                    timestamp: Date.now()
                                };
                                
                                // Check if this exact caption already exists
                                const exists = persistentCaptions.some(existing => 
                                    existing.camera === i && existing.caption === caption
                                );
                                
                                if (!exists) {
                                    persistentCaptions.unshift(captionData); // Add to beginning
                                }
                            }
                        });
                    } else {
                        console.error(`Failed to fetch captions for camera ${i}:`, captionsResponse.status);
                    }
                }
                
                // Keep only last 10 captions total and update display
                persistentCaptions = persistentCaptions.slice(0, 10);
                updateAllCaptions();
                
            } catch (error) {
                console.error('Data update error:', error);
                
                // Show error in stats grid if completely failed
                const statsGrid = document.getElementById('stats-grid');
                if (statsGrid) {
                    statsGrid.innerHTML = `
                        <div class="stat-item error">
                            <div class="stat-label">Error</div>
                            <div class="stat-value">Failed to load stats</div>
                        </div>
                    `;
                }
            }
        }
        
        // Initialize and start updates
        document.addEventListener('DOMContentLoaded', function() {
            updateData();
            setInterval(updateData, 1000); // Update every second
        });
    </script>
</body>
</html>'''

@app.route('/stats')
def stats():
    """API endpoint for performance statistics"""
    try:
        current_stats = temporal_lobe.get_stats()
        return jsonify(current_stats)
    except Exception as e:
        logging.error(f"Error getting stats: {e}")
        return jsonify({'error': 'Failed to get stats'}), 500

@app.route('/unified_state')
def unified_state():
    """API endpoint for current unified state"""
    try:
        unified = temporal_lobe.get_unified_state()
        return jsonify(unified)
    except Exception as e:
        logging.error(f"Error getting unified state: {e}")
        return jsonify({'error': 'Failed to get unified state'}), 500

@app.route('/recent_transcripts')
def recent_transcripts():
    """API endpoint for recent transcripts with persistence"""    
    try:
        # Get recent audio data and extract non-empty transcripts
        current_transcripts = []
        for frame in list(temporal_lobe.audio_buffer):
            if ('transcription' in frame and 
                frame['transcription'].strip() and 
                frame['transcription'] not in ['', ' ']):
                current_transcripts.append(frame['transcription'].strip())
        
        # Add new transcripts to persistent storage (avoid duplicates)
        for transcript in current_transcripts:
            if transcript not in temporal_lobe.persistent_transcripts:
                temporal_lobe.persistent_transcripts.insert(0, transcript)  # Add to beginning
        
        # Keep only last 10 transcripts in memory (return 5)
        temporal_lobe.persistent_transcripts = temporal_lobe.persistent_transcripts[:10]
        
        # Return last 5 unique transcripts
        return jsonify({'transcripts': temporal_lobe.persistent_transcripts[:5]})
        
    except Exception as e:
        logging.error(f"Error getting recent transcripts: {e}")
        return jsonify({'transcripts': temporal_lobe.persistent_transcripts[:5] if temporal_lobe.persistent_transcripts else []})
    

@app.route('/recent_captions/<int:camera_index>')
def recent_captions(camera_index):
    """API endpoint for recent captions from specific camera"""
    try:
        captions = []
        for frame in list(temporal_lobe.visual_buffer):
            if (frame.get('camera_index') == camera_index and 
                'caption' in frame and frame['caption'].strip()):
                captions.append(frame['caption'])
        
        # Return last 5 unique captions
        unique_captions = []
        for caption in reversed(captions):
            if caption not in unique_captions:
                unique_captions.append(caption)
                if len(unique_captions) >= 5:
                    break
        
        return jsonify({'captions': unique_captions})
    except Exception as e:
        logging.error(f"Error getting recent captions for camera {camera_index}: {e}")
        return jsonify({'captions': []})

# Visual System Routes
@app.route('/start/<int:camera_index>')
def start_camera(camera_index):
    """Start optic nerve and visual cortex processing for specific camera"""
    try:
        temporal_lobe.start_visual_nerve(camera_index)
        return jsonify({
            'status': 'started', 
            'message': f'Optic nerve {camera_index} started'
        })
    except Exception as e:
        logging.error(f"Error starting camera {camera_index}: {e}")
        return jsonify({'error': f'Failed to start camera {camera_index}'}), 500

@app.route('/stop/<int:camera_index>')
def stop_camera(camera_index):
    """Stop optic nerve for specific camera"""
    try:
        success = temporal_lobe.stop_visual_nerve(camera_index)
        return jsonify({
            'status': 'stopped' if success else 'error', 
            'message': f'Optic nerve {camera_index} {"stopped" if success else "failed to stop"}'
        })
    except Exception as e:
        logging.error(f"Error stopping camera {camera_index}: {e}")
        return jsonify({'error': f'Failed to stop camera {camera_index}'}), 500

# Audio System Routes  
@app.route('/start_audio/<int:device_index>')
def start_audio_device(device_index):
    """Start auditory nerve and auditory cortex processing for specific audio device"""
    try:
        temporal_lobe.start_audio_nerve(device_index)
        return jsonify({
            'status': 'started', 
            'message': f'Auditory nerve {device_index} started'
        })
    except Exception as e:
        logging.error(f"Error starting audio device {device_index}: {e}")
        return jsonify({'error': f'Failed to start audio device {device_index}'}), 500

@app.route('/stop_audio/<int:device_index>')
def stop_audio_device(device_index):
    """Stop auditory nerve for specific audio device"""
    try:
        success = temporal_lobe.stop_audio_nerve(device_index)
        return jsonify({
            'status': 'stopped' if success else 'error', 
            'message': f'Auditory nerve {device_index} {"stopped" if success else "failed to stop"}'
        })
    except Exception as e:
        logging.error(f"Error stopping audio device {device_index}: {e}")
        return jsonify({'error': f'Failed to stop audio device {device_index}'}), 500

# System Control Routes
@app.route('/stop_all')
def stop_all():
    """Stop all optic nerves, visual cortex, auditory nerves, and auditory cortex processing"""
    try:
        success = temporal_lobe.shutdown_all()
        return jsonify({
            'status': 'stopped' if success else 'partial_failure', 
            'message': 'All systems ' + ('stopped successfully' if success else 'stopped with some failures')
        })
    except Exception as e:
        logging.error(f"Error stopping all systems: {e}")
        return jsonify({'error': 'Failed to stop all systems'}), 500

# Video Feed Route
@app.route('/video_feed/<int:camera_index>')
def video_feed(camera_index):
    """Video streaming route for specific camera"""
    if temporal_lobe.visual_cortex:
        return Response(temporal_lobe.visual_cortex.generate_img_frames(camera_index),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Visual cortex not available", 404

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print("\nShutting down...")
    success = temporal_lobe.shutdown_all()
    print(f"Shutdown {'completed successfully' if success else 'completed with failures'}")
    sys.exit(0)

if __name__ == '__main__':
    import signal
    import socket

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Global multiprocessing queues and managers MUST go here because of issues with Windows
    manager = mp.Manager()
    
    # Visual Cortex - HARD CODE WARNING - GPU 1
    gpu_to_use=1
    logging.debug(f"Setting up Visual Cortex VLM on GPU {gpu_to_use}")
    vc = VisualCortex(mpm=manager,gpu_to_use=gpu_to_use)
    
    #Audio Cortex
    ac = AuditoryCortex(mpm=manager)

    #Temporal Lobe
    temporal_lobe = TemporalLobe(visual_cortex=vc, auditory_cortex=ac, mpm=manager)

    # Start temporal lobe collection
    temporal_lobe.start_collection()
    """
    # Initialize with existing cortex instances
    temporal_lobe = TemporalLobe(visual_cortex, auditory_cortex)

    # Start collection
    temporal_lobe.start_collection()

    # Get unified state (call this every second or as needed)
    unified_state = temporal_lobe.get_unified_state()

    unified_state = print({'timestamp': 1758855474.7669709, 
                        'formatted_time': '22:57:54', 
                        'person_detected': True, 
                        'person_match': (False,), 
                        'person_match_probability': 0.0, 
                        'caption': ' The image shows a man with a beard and short hair, wearing a black hoodie over a white t-shirt. He is looking upwards and slightly to his left. The background features a cream-colored wall with vertical lines and a dark brown or black ceiling or trim.', 
                        'vlm_timestamp': 1758855473.9019487, 
                        'visual_camera_index': 0, 
                        'speech_detected': True, 
                        'transcription': "Is that the unified state format? It's hard for me to tell. If I keep  talking maybe I'll be able to see a transcript I guess I will I'm not", 
                        'final_transcript': False, 
                        'voice_match': False, 
                        'voice_probability': 0.0, 
                        'audio_device_index': 0, 
                        'transcription_timestamp': 1758855474.7669709, 
                        'visual_frames_in_update': 1, 
                        'audio_frames_in_update': 31, 
                        'buffer_visual_frames': 1, 
                        'buffer_audio_frames': 31})
    """


    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    # Handle shutdown signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("\n" + "="*80)
        print(" TEMPORAL LOBE CONTROL SYSTEM STARTING ")
        print("="*80)
        print("Flask server with multi-camera OpenCV streaming and audio processing...")
        print("Visit http://localhost:5000 to view the interface")
        print("\n API Endpoints:")
        print("   Visual System:")
        print("    GET /start/<camera_index>       - Start specific optic nerve")
        print("    GET /stop/<camera_index>        - Stop specific optic nerve")
        print("    GET /video_feed/<camera_index>  - Stream from specific camera")
        print("    GET /recent_captions/<camera_index> - Get recent captions")
        print("Audio System:")
        print("    GET /start_audio/<device_index> - Start specific auditory nerve")
        print("    GET /stop_audio/<device_index>  - Stop specific auditory nerve")
        print("    GET /recent_transcripts         - Get recent transcriptions")
        print("   System Status:")
        print("    GET /stats                      - Get performance statistics")
        print("    GET /unified_state              - Get current unified state")
        print("    GET /stop_all                   - Stop all systems")
        print("="*80)
        
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        print(f"Flask server running on IP: {ip_address}")

        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\n🛑Keyboard interrupt received...")
    except Exception as e:
        print(f"\n❌Server error: {e}")
    finally:
        print("\nPerforming cleanup...")
        success = temporal_lobe.shutdown_all()
        print(f"Cleanup {'completed successfully' if success else 'completed with some failures'}")
        print("Temporal Lobe Control System shutdown complete")