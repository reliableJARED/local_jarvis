from flask import Flask, render_template, Response, jsonify, request
import time
import logging
from collections import deque
from cerebrum import Cerebrum

# Initialize Flask
app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/state')
def stats():
    """
    The heartbeat of the UI. Returns the unified state JSON.
    Also updates local history buffers if new data is detected.
    """
    global last_transcript_id, last_caption_ts

    state = brain.ui_get_unified_state()
    sensory = state.get('sensory', {})
    
    # Update Transcript History
    # We check if the current transcription is 'final' and different from the last processed one
    curr_trans = sensory.get('transcription', '')
    is_final = sensory.get('final_transcript', False)
    ts = sensory.get('timestamp', 0)
    
    # Simple logic to prevent duplicates in history
    if is_final and curr_trans and curr_trans != last_transcript_id:
        transcript_history.appendleft({
            'time': sensory.get('formatted_time'),
            'text': curr_trans,
            'speaker': sensory.get('voice_id', 'Unknown')
        })
        last_transcript_id = curr_trans

    # Update Caption History
    curr_caption = sensory.get('caption', '')
    cap_ts = sensory.get('vlm_timestamp', 0)
    
    if curr_caption and cap_ts != last_caption_ts:
        caption_history.appendleft({
            'time': sensory.get('formatted_time'),
            'text': curr_caption
        })
        last_caption_ts = cap_ts

    # Inject history into the response
    state['history'] = {
        'transcripts': list(transcript_history),
        'captions': list(caption_history)
    }

    return jsonify(state)

### Visual System Routes ###

@app.route('/vlm_recent_captions/<int:camera_index>')
def recent_captions(camera_index):
    # Returns just the caption history, distinct from the full state if needed separate
    return jsonify(list(caption_history))

@app.route('/start_camera/<int:camera_index>')
def start_camera(camera_index):
    brain.ui_toggle_camera(active=True, device_index=camera_index)
    return jsonify({"status": "Camera Started", "index": camera_index})

@app.route('/stop_camera/<int:camera_index>')
def stop_camera(camera_index):
    brain.ui_toggle_camera(active=False, device_index=camera_index)
    return jsonify({"status": "Camera Stopped", "index": camera_index})

@app.route('/video_feed/<int:camera_index>')
def video_feed(camera_index):
    """
    Video streaming route. Put this in the src attribute of an img tag.
    """
    if not brain.visual_cortex:
        return "Visual Cortex Not Available"
        
    return Response(
        brain.visual_cortex.generate_img_frames(camera_index),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

### Audio System Routes ###

@app.route('/start_audio/<int:device_index>')
def start_audio_device(device_index):
    brain.ui_toggle_microphone(active=True, device_index=device_index)
    return jsonify({"status": "Audio Started", "index": device_index})

@app.route('/stop_audio/<int:device_index>')
def stop_audio_device(device_index):
    brain.ui_toggle_microphone(active=False, device_index=device_index)
    return jsonify({"status": "Audio Stopped", "index": device_index})

@app.route('/recent_transcripts')
def recent_transcripts():
    return jsonify(list(transcript_history))

### Interaction Routes ###

@app.route('/ui_input', methods=['POST'])
def ui_input():
    # Handles text input from the web form
    data = request.json
    text = data.get('input', '')
    if text:
        brain.ui_send_text_input(text)
        return jsonify({"status": "sent", "content": text})
    return jsonify({"status": "empty"})

@app.route('/stop_all')
def stop_all():
    brain.stop_systems()
    return jsonify({"status": "System Shutdown Initiated"})

if __name__ == '__main__':
    # Initialize the AI Brain
    # We start it immediately so it's ready when the UI loads
    brain = Cerebrum(wakeword='jarvis')
    brain.start_systems(start_mic=False, start_cam=False)

    # History Buffers (To track history on the UI side)
    transcript_history = deque(maxlen=5)
    caption_history = deque(maxlen=3)
    last_transcript_id = None
    last_caption_ts = None

    # Run threaded to allow the Brain loops and Flask loops to coexist
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)