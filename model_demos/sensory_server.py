from flask import Flask, Response, jsonify, request
import cv2
import multiprocessing as mp
import queue
import time
import numpy as np
from datetime import datetime
import json
import signal
import sys
from PIL import Image
import torch
import gc
import threading

app = Flask(__name__)

# Configuration
QUEUE_SIZE = 5   # Maximum number of frames in queue
FPS_TARGET = 30   # Target FPS for camera capture, Moondream alone will make 1fps a stretch

# Global process tracking
optic_nerve_processes = {}
visual_cortex_processes = {}

# Global VLM (moondream) analysis state
VLM_availible = False
VLM_thread = None

def optic_nerve_worker(camera_index, raw_img_queue, stats_dict, fps_target):
    """Worker process for capturing frames from a specific camera (optic nerve function).
    Will drop frames if the queue is full to maintain real-time performance.
    will also update stats_dict with fps and last frame time.
    will stream to raw_img_queue."""
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
                raw_img_queue.put_nowait(frame_data)
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

def visual_cortex_worker(raw_img_queue, processed_queue, stats_dict, visual_scene_queue):
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
    from moondream_ import MoondreamWrapper

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
                        frame_data = raw_img_queue.get_nowait()
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

                        # Import and initialize VLM Moondream (done in thread to avoid blocking)
                        IMG_PROCESSING_VLM = MoondreamWrapper(local_files_only=True)

                        # Start VLM Moondream analysis in separate thread
                        VLM_thread = threading.Thread(
                            target=visual_cortex_process_vlm_analysis_thread,
                            args=(frame_data['frame'], frame_data['camera_index'], 
                                frame_data['capture_timestamp'], visual_scene_queue, stats_dict,IMG_PROCESSING_VLM)
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
                        processed_queue.put_nowait(processed_data)
                    except:
                        # Queue is full, try to remove oldest and add new
                        try:
                            processed_queue.get_nowait()
                            processed_queue.put_nowait(processed_data)
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
    #gc.collect()
    #torch.cuda.empty_cache()  # Force GPU memory cleanup
    
    if torch.cuda.is_available():
         # Monitor GPU memory before loading
        print(f"GPU memory after cleanup: {torch.cuda.memory_allocated()/1024**2:.1f}MB")

    return frame, person_detected

def visual_cortex_process_vlm_analysis_thread(frame, camera_index, timestamp, visual_scene_queue, stats_dict,IMG_PROCESSING_VLM):
    """Thread function to run VLM (Moondream) analysis without blocking main processing"""
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
        except:
            pass

        gc.collect()
        torch.cuda.empty_cache()
        stats_dict['VLM_availible'] = False
        print(f"VLM analysis thread finished for camera {camera_index}")

def generate_img_frames(camera_index=None):
    """Generator function for video streaming from specific camera or all cameras"""
    frame_count = 0
    start_time = time.time()
    
    while True:
        try:
            # Get processed frame (blocking with timeout)
            frame_data = processed_frame_queue.get(timeout=1.0)
            
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
HTML Interface - UPDATED with visual scene display functionality
"""
@app.route('/')
def index():
    """Basic HTML page to view streams from all cameras with VLM output"""
    return '''
    <html>
    <head>
        <title>Multi-Camera OpenCV Stream</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
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
            <h1>Multi-Camera OpenCV Stream with VLM Analysis</h1>
            
            <div class="controls">
                <button class="start-btn" onclick="startCamera(0)">Start Optic Nerve 0</button>
                <button class="start-btn" onclick="startCamera(1)">Start Optic Nerve 1</button>
                <button class="start-btn" onclick="startCamera(2)">Start Optic Nerve 2</button>
                <button class="stop-btn" onclick="stopCamera(0)">Stop Optic Nerve 0</button>
                <button class="stop-btn" onclick="stopCamera(1)">Stop Optic Nerve 1</button>
                <button class="stop-btn" onclick="stopCamera(2)">Stop Optic Nerve 2</button>
                <button class="stop-btn" onclick="stopAll()">Stop All</button>
            </div>
            
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
            
            <div class="stats" id="stats-container">
                <!-- Stats will be populated by JavaScript -->
            </div>
            
            <div class="visual-scene">
                <h3>Visual Scene Analysis - VLM Output</h3>
                <div id="vlm-status" class="vlm-status vlm-idle">VLM Status: Idle</div>
                <div id="scene-analysis">
                    <p>No scene analysis data available. Start a camera and detect a person to begin analysis.</p>
                </div>
            </div>

        </div>
        
        <script>
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
            setInterval(updateVisualScenes, 2000);  // Update scenes every 2 seconds
            
            // Initial updates
            updateStats();
            updateVisualScenes();
        </script>
    </body>
    </html>
    '''


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

@app.route('/stats')
def stats():
    """API endpoint for performance statistics"""
    current_stats = dict(stats_dict)
    current_stats['timestamp'] = datetime.now().isoformat()
    current_stats['raw_img_queue_size'] = raw_img_queue.qsize()
    current_stats['processed_queue_size'] = processed_frame_queue.qsize()
    current_stats['visual_scene_queue_size'] = visual_scene_queue.qsize()
    current_stats['active_optic_nerves'] = list(optic_nerve_processes.keys())
    return jsonify(current_stats)

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
        args=(camera_index, raw_img_queue, stats_dict, FPS_TARGET)
    )
    optic_nerve_process.start()
    optic_nerve_processes[camera_index] = optic_nerve_process
    
    # Start visual cortex if not already running
    if not visual_cortex_processes:
        visual_cortex_process = mp.Process(
            target=visual_cortex_worker,
            args=(raw_img_queue, processed_frame_queue, stats_dict, visual_scene_queue)
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

@app.route('/stop_all')
def stop_all():
    """Stop all optic nerves and visual cortex processing"""
    global optic_nerve_processes, visual_cortex_processes
    
    # Stop all optic nerve processes
    for camera_index in list(optic_nerve_processes.keys()):
        stop_camera(camera_index)
    
    # Stop visual cortex processes
    for process in visual_cortex_processes.values():
        process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
    
    visual_cortex_processes.clear()
    stats_dict.clear()
    
    return jsonify({
        'status': 'stopped', 
        'message': 'All optic nerves and visual cortex stopped'
    })

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print("\nShutting down...")
    stop_all()
    sys.exit(0)

if __name__ == '__main__':
    # Global multiprocessing queues and managers MUST go here because of issues with Windows
    manager = mp.Manager()
    raw_img_queue = manager.Queue(maxsize=QUEUE_SIZE * 4)  # Larger queue for multiple cameras
    processed_frame_queue = manager.Queue(maxsize=QUEUE_SIZE * 4)
    visual_scene_queue = manager.Queue(maxsize=10)  # Queue for Moondream analysis results
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