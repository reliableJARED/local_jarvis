
import cv2
import multiprocessing as mp
import queue
import time
import numpy as np
from datetime import datetime
from PIL import Image
import torch
import gc
import logging


logging.basicConfig(level=logging.INFO) #ignore everything use (level=logging.CRITICAL + 1)

# Set the Werkzeug logger level to ERROR to ignore INFO messages
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


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
            """if frame_count % 30 == 0:  # Update every 30 frames
                current_time = time.time()
                fps = frame_count / (current_time - start_time)
                stats_data = {f'optic_nerve_fps_{camera_index}':fps,
                            f'last_optic_nerve_{camera_index}': current_time}
                try:
                    external_stats_queue.put_nowait(stats_data)# Will drop and lose this data if no put
                except:
                    logging.warning(f"Optic nerve {camera_index} unable to send stats data to external_stats_queue because it's full")"""
                
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
                    """frame_count += 1
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
                                pass"""
                    
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
        #these are just placeholders
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
        
        """try:
            external_stats_queue.put_nowait({'VLM_availible': status_boolean})
        except queue.Full:
            try:
                _ = external_stats_queue.get_nowait()
                external_stats_queue.put_nowait({'VLM_availible': status_boolean})
            except queue.Empty:
                pass"""
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
                visual_scene_data['person_match'] = person_match_result
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
    def __init__(self,cortex=visual_cortex_core,vlm=visual_cortex_worker_vlm,nerve=optic_nerve_connection,internal_nerve_queue=None,mpm=False,gpu_to_use=0):
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
        if internal_nerve_queue is None:
            self.internal_nerve_queue = mpm.Queue(maxsize=1)
        else:
            if isinstance(internal_nerve_queue, mp.Queue):
                logging.warning("a custom internal_nerve_queue passed to VisualCortex. Make sure it feeds at the correct rates and data struct or there will be errors. use optic_nerve_connection() as a guide")
                self.nerve_from_input_to_cortex = internal_nerve_queue
            else:
                logging.error("internal_nerve_queue passed to VisualCortex must be a multiprocessing queue")

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
                    cv2.putText(placeholder, f"No Signal Yet - {cam_text}", (150, 240), 
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

        

if __name__ == "__main__":
    # Ensure safe multiprocessing on different OSs
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    print("Initializing System...")
    
    # 1. Create the Multiprocessing Manager
    manager = mp.Manager()

    # 2. Initialize the Visual Cortex (starts the Core and VLM processes automatically)
    # Note: Ensure your 'yolo_.py' and 'moondream_.py' files are accessible
    vc = VisualCortex(mpm=manager)

    # 3. Start the Nerve (Camera)
    # Defaulting to camera index 0
    print("Starting Optic Nerve...")
    vc.start_nerve(device_index=0)

    print("\nSystem Running. Press Ctrl+C to stop.\n")

    try:
        while True:
            # --- HANDLE IMAGE FEED ---
            try:
                # Get the latest frame from the queue (non-blocking)
                img_data = vc.external_img_queue.get_nowait()
                
                frame = img_data.get('frame')
                if frame is not None:
                    cv2.imshow("Visual Cortex Feed", frame)

            except queue.Empty:
                pass

            # --- HANDLE CORTEX TEXT OUTPUT ---
            try:
                # Get the latest analysis data from the queue
                cortex_data = vc.external_cortex_queue.get_nowait()
                
                # Print the data to terminal
                # Formatting specifically to show the most relevant parts cleanly
                timestamp = cortex_data.get('formatted_time', 'N/A')
                detected = cortex_data.get('person_detected', False)
                caption = cortex_data.get('caption', '')
                
                #person recognition is real time, only show when we have a new caption to limit output spam
                if caption:
                    output_str = f"[{timestamp}] Person: {detected}"
                    output_str += f" | Caption: {caption}"

                    print("*"*50)
                    print(output_str)
                    print("="*50)
                    print(cortex_data)  # Full data for debugging
                    print("="*50)

            except queue.Empty:
                pass

            # --- OPENCV GUI  ---
            # Required for cv2.imshow to draw. waitKey(1) waits 1ms.
            # checks if 'q' is pressed as an alternative exit method
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Initiating shutdown sequence...")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    finally:
        # Cleanup
        vc.shutdown()
        cv2.destroyAllWindows()
        print("System shutdown complete.")