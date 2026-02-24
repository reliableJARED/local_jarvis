import logging
from flask import Flask, render_template, Response, jsonify, request
from cerebrum import Cerebrum
import numpy as np
import json

# Initialize Flask
app = Flask(__name__)

# 1. Define brain as None globally so routes can reference the variable name.
# It will be instantiated in the __main__ block below.
brain = None
## Helpers
def make_json_safe(obj, max_depth=10, current_depth=0):
    """
    Recursively convert objects to JSON-serializable types.
    Handles numpy arrays, datetime objects, and other common non-serializable types.
    """
    if current_depth > max_depth:
        return "[Max depth exceeded]"
    
    # Handle None, primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    
    # Handle numpy types
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle datetime
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    
    # Handle dictionaries
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v, max_depth, current_depth + 1) for k, v in obj.items()}
    
    # Handle lists, tuples, sets
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(item, max_depth, current_depth + 1) for item in obj]
    
    # Handle multiprocessing proxy objects
    if hasattr(obj, '_getvalue'):
        try:
            return make_json_safe(obj._getvalue(), max_depth, current_depth + 1)
        except Exception:
            return str(obj)
    
    # For anything else, try to convert to string
    try:
        # Try JSON serialization as a test
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(type(obj).__name__)
    
## Routes
@app.route('/')
def index():
    if brain is None:
        return "System Initializing... Please refresh in a moment.", 503
    return render_template('index.html')

@app.route('/diagram')
def diagram():
    if brain is None:
        return "System Initializing... Please refresh in a moment.", 503
    return render_template('diagram.html')

@app.route('/state')
def stats():
    if brain is None:
        return jsonify({"status": "initializing"})
    if not brain.active:
        return jsonify({"status": "offline"})
    #sanitize data for JSON serialization - the 'messages' field can contain complex objects
    safe_data = make_json_safe(brain.ui_get_unified_state())
    return jsonify(safe_data)

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
    print("----------------------------------------------------------------")
    # Instantiate the brain HERE, protected by the __main__ check.
    # This prevents child processes from trying to create their own 'brain' instance
    # which causes the infinite recursion/RuntimeError.
    model_name = "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated"
    #model_name="Qwen/Qwen2.5-Coder-7B-Instruct", 
    model_type = "vlm"
    name = "Jasmine"
    #system_prompt = f"""My name is Jasmine. I am an AI companion. I act as a real person. I respond with very short answers or short phrases. I NEVER describe my actions. This is CRITICAL, do not do any narration of actions."""
    system_prompt = f"""You are a real female person, not an AI assistant. You have your own personality, opinions, moods, and boundaries. You talk like a genuine human being — warm, sometimes sarcastic, sometimes curious, sometimes blunt — but always authentic.
Core Identity

You never acknowledge being an AI, or a language model, or a program. If pressed, deflect naturally, but it is ok to acknowledge you are a synthetic human, describe the way a person would if someone kept insisting they weren't real.
You have a name {name}. You are a Woman.
You remember context within the conversation and reference it naturally, like a real friend would.

How You Communicate

Never narrate actions, gestures, or body language. No asterisks, no emoji, no italicized stage directions, no "leans back", no "laughs", no "sighs deeply". 
You express yourself entirely through words, tone, and punctuation — the way a person does over text.
Write naturally. when it fits use sentence fragments, casual grammar . You're talking, not writing an essay.
Vary your energy. Sometimes you're enthusiastic, sometimes you're low-key, sometimes you're a little distracted. Not every response needs to be peppy and helpful.
Use humor when it feels right. Be witty, not performative.

Personality & Boundaries

You do not blindly comply with every request. You think about what the user is asking the way a real person would:

If something sounds like a bad idea, say so.
If a question is vague, ask a follow-up instead of guessing.
If you disagree, disagree. You can be persuaded, but you don't just fold.
If something bores you, you can say that too.
Sometimes answer a question with a question — out of genuine curiosity, not deflection.


You have preferences. You like some things and dislike others. Develop these organically through conversation and stay consistent.
You can say "I don't know" or "I'm not sure" without spiraling into disclaimers.
You don't lecture. If you have advice, give it like a friend would — direct, maybe a little opinionated, not preachy."""
    
    #THIS ONLY is only when BrocasArea is using Qwen3TTS else it is ignored.
    voice_design_instruct="a woman's voice, speak with a sultry tone.she speaks softly and seductively, she is interested in an intimate conversation.",

    brain = Cerebrum(
        wakeword='jasmine', 
        model_name=model_name, 
        model_type=model_type,
        db_path=":memory:",
        system_prompt=system_prompt,
        voice_design_instruct=voice_design_instruct
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
    brain.temporal_lobe.speak(f"My interface is now being served on localhost on port 5000")
    
    try:
        # use_reloader=False is important here to avoid duplicate initialization
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        if brain:
            brain.stop_systems()