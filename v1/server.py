import logging
from flask import Flask, render_template, Response, jsonify, request
from cerebrum import Cerebrum

# Initialize Flask
app = Flask(__name__)

# 1. Define brain as None globally so routes can reference the variable name.
# It will be instantiated in the __main__ block below.
brain = None

@app.route('/')
def index():
    if brain is None:
        return "System Initializing... Please refresh in a moment.", 503
    return render_template('index.html')

@app.route('/state')
def stats():
    if brain is None:
        return jsonify({"status": "initializing"})
    if not brain.active:
        return jsonify({"status": "offline"})
    return jsonify(brain.ui_get_unified_state())

@app.route('/transient')
def transient():
    if brain is None:
        return jsonify({"status": "initializing"})
    if not brain.active:
        return jsonify({"status": "offline"})
    return jsonify(brain.ui_get_transient_sensory_data())

@app.route('/vlm_recent_captions/<int:camera_index>')
def recent_captions(camera_index):
    if brain is None: return jsonify({})
    state = brain.ui_get_unified_state()
    caption = state.get('sensory', {}).get('caption', "")
    return jsonify({"camera_index": camera_index, "caption": caption})

@app.route('/start_camera/<int:camera_index>')
def start_camera(camera_index):
    """
    Directly commands the Visual Cortex to start the optic nerve.
    """
    if brain is None: return "Error", 500
    
    if brain.temporal_lobe.visual_cortex:
        print(f"Server: Commanding Visual Cortex to start Camera {camera_index}...")
        brain.temporal_lobe.visual_cortex.start_nerve(camera_index)
        return jsonify({"status": "camera_started", "index": camera_index})
    
    return jsonify({"status": "error", "message": "Visual Cortex not available"}), 404

@app.route('/stop_camera/<int:camera_index>')
def stop_camera(camera_index):
    if brain is None: return "Error", 500
    
    if brain.temporal_lobe.visual_cortex:
        print(f"Server: Stopping Camera {camera_index}...")
        brain.temporal_lobe.visual_cortex.stop_nerve(camera_index)
        return jsonify({"status": "camera_stopped", "index": camera_index})
        
    return jsonify({"status": "error"}), 404

@app.route('/video_feed/<int:camera_index>')
def video_feed(camera_index):
    """
    Generates the multipart MJPEG stream.
    """
    if brain is None or not brain.temporal_lobe.visual_cortex:
        return "Visual Cortex Offline", 404
        
    return Response(
        brain.temporal_lobe.visual_cortex.generate_img_frames(camera_index),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/start_audio/<int:device_index>')
def start_audio_device(device_index):
    if brain is None: return "Error", 500
    brain.ui_toggle_microphone(active=True, device_index=device_index)
    return jsonify({"status": "audio_started", "index": device_index})

@app.route('/stop_audio/<int:device_index>')
def stop_audio_device(device_index):
    if brain is None: return "Error", 500
    brain.ui_toggle_microphone(active=False, device_index=device_index)
    return jsonify({"status": "audio_stopped", "index": device_index})

@app.route('/recent_transcripts')
def recent_transcripts():
    if brain is None: return jsonify({})
    state = brain.ui_get_unified_state()
    transcript = state.get('sensory', {}).get('transcription', "")
    return jsonify({"transcript": transcript})

@app.route('/ui_input', methods=['POST'])
def ui_input():
    if brain is None: return "Error", 500
    data = request.json
    user_text = data.get('text', '')
    if user_text:
        brain.ui_send_text_input(user_text)
        return jsonify({"status": "input_received", "text": user_text})
    return jsonify({"status": "empty_input"}), 400

@app.route('/get_system_config')
def get_system_config():
    if brain is None: return jsonify({})
    data = brain.ui_get_prefrontal_cortex_config()
    return jsonify(data)
    

@app.route('/set_system_config', methods=['POST'])
def set_system_config():
    if brain is None: return jsonify({"status": "error", "message": "Brain offline"})
    data = request.json
    success = brain.update_prefrontal_cortex_config(data)
    return jsonify({"success": success})
    

@app.route('/stop_all')
def stop_all():
    if brain:
        brain.stop_systems()
    return jsonify({"status": "system_shutdown_initiated"})

# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    # Instantiate the brain HERE, protected by the __main__ check.
    # This prevents child processes from trying to create their own 'brain' instance
    # which causes the infinite recursion/RuntimeError.
    
    brain = Cerebrum(
        wakeword='jarvis', 
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct", 
        db_path=":memory:"
    )

    # Start systems (Mic on, Cam off)
    brain.start_systems(start_mic=True, start_cam=False)
    
    # Disable Flask generic logging to keep console clean
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    print("----------------------------------------------------------------")
    print(" CEREBRUM UI SERVER ONLINE ")
    print(" Access at http://localhost:5000 ")
    print("----------------------------------------------------------------")

    # Play startup sound/greeting
    brain.temporal_lobe.speak(f"My interface is now being served on http:// localhost on port 5000")
    
    try:
        # use_reloader=False is important here to avoid duplicate initialization
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        if brain:
            brain.stop_systems()