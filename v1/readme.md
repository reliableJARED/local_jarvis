# Local AI Cerebrum Architecture

A fully local, offline, multimodal AI system designed with a biomimetic architecture that mimics the human brain's structure by separating functionalities into specific "cortices" (processes) that communicate via a central nervous system.

## Overview

This project implements a brain-inspired AI system that bypasses Python's Global Interpreter Lock (GIL) through heavy use of multiprocessing, allowing multiple heavy AI models (Vision, LLM, TTS, STT) to run simultaneously without freezing the UI.

## High-Level Architecture

The system is orchestrated by `cerebrum.py`, which distributes heavy computational tasks across independent processes. This ensures that resource-intensive operations like visual processing (YOLO/Moondream) or LLM generation (Qwen) do not block real-time responsiveness of audio wake-word detection or the web UI.

## Component Map

| Component | File | Biological Analogy | Function |
|-----------|------|-------------------|----------|
| Cerebrum | `cerebrum.py` | Central Nervous System | Orchestrates processes, manages shared memory (mp.Manager), handles startup/shutdown |
| Optic Nerve | `visualcortex.py` | Eyes | Captures raw video frames via OpenCV |
| Visual Cortex | `visualcortex.py` | Occipital Lobe | Processes images using YOLO (fast detection) and Moondream (deep understanding/VLM) |
| Auditory Nerve | `audiocortex.py` | Ears (Cochlea) | Captures raw audio streams via SoundDevice |
| Auditory Cortex | `audiocortex.py` | Temporal Lobe (Audio) | Voice Activity Detection (Silero), Speech-to-Text, Speaker Recognition |
| Broca's Area | `brocasArea.py` | Speech Center | Text-to-Speech synthesis and audio playback (Kokoro) |
| Temporal Lobe | `temporallobe.py` | Sensory Integration | Aggregates data from Audio and Visual cortices into unified state |
| Prefrontal Cortex | `prefrontalcortex.py` | Frontal Lobe (Logic) | LLM (Qwen) for decisions, tool use (Internet), conversational logic |
| Server | `server.py` | Conscious Interface | Flask web server for visualizing what the system sees, hears, and thinks |

## Data Flow & Multiprocessing Strategy

The system relies on `multiprocessing.Queue` for data transport and `multiprocessing.Dict` for shared state between isolated processes.

### 1. Visual Pathway

- **Capture**: `optic_nerve_connection` captures video frames
- **Queue**: Pushes to `internal_nerve_queue` (LIFO - drops old frames to reduce latency)
- **Core Processing**: `visual_cortex_core` consumes frames
  - Runs YOLOv8 for person detection
  - If enabled, copies frame to `internal_to_vlm_queue` for VLM worker
- **Output**: Processed data (bounding boxes, captions) pushed to `external_cortex_queue`

### 2. Auditory Pathway

- **Capture**: `auditory_nerve_connection` captures raw audio chunks
- **VAD**: `auditory_cortex_core` analyzes chunks, accumulates buffer when speech detected
- **Transcription**: When silence detected, buffer moves to `nerve_from_cortex_to_stt`
- **Recognition**: Audio simultaneously sent to `nerve_from_stt_to_vr` to identify speaker
- **Output**: Final transcript and Speaker ID pushed to `external_cortex_queue`

### 3. Integration (Temporal Lobe)

The TemporalLobe runs a background thread `_temporalLobe_State_Loop` that:

- Listens to both Visual and Auditory output queues
- Creates Unified State Packets:

```json
{
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
            'voice_id': False,
            'voice_probability': 0.0,
            'audio_device_index': None,
            'transcription_timestamp': None,
            'audio_capture_timestamp': None,
            
            # Active speaker tracking
            'locked_speaker_id': None,
            'locked_speaker_timestamp': None,
            'is_locked_speaker': False,

            #Speech Output
            'actively_speaking':False,

            #Interruption Attempt
            'is_interrupt_attempt':False,
            
            # UI Inputs
            'user_text_input': ""
            }
```

- Pushes packets to `external_temporallobe_to_prefrontalcortex`

### 4. Thought Process (Prefrontal Cortex)

The PrefrontalCortex:

- Waits for data on `external_temporallobe_to_prefrontalcortex`
- Constructs prompts using the Unified State
- Runs the LLM (Qwen) to generate tokens
- **Tool Use**: Parses output for XML/JSON tags, pauses to run Python functions (like `internet_search`), injects results, continues generating
- **Streaming Speech**: Sends completed sentences to BrocasArea for immediate synthesis

## Key Features

### Interrupt Handling

Real-time conversation requires the ability to be interrupted:

- **Detection**: `audiocortex.py` checks if `system_actively_speaking` is True AND `speech_active` (VAD) is True
- **Signal**: Updates `brocas_area_interrupt_dict` (shared memory)
- **Reaction**:
  - Broca's Area immediately calls `sd.stop()` to kill audio playback
  - Prefrontal Cortex checks flag during token generation and aborts to listen to new input

### Queue Types

- `nerve_from_input_to_cortex`: Raw high-speed data (Audio/Video) from hardware to processing
- `external_cortex_queue`: Finished processing data (Text/Captions) leaving a cortex
- `external_temporallobe_to_prefrontalcortex`: Final triggers causing the LLM to think

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (recommended for Qwen/Moondream/Kokoro)
- ffmpeg installed on system

### Dependencies

```bash
pip install torch transformers flask opencv-python sounddevice numpy scipy
```

Install specific model requirements for Kokoro, Moondream, and Silero.

### Running

```bash
python server.py
```

This initializes Cerebrum, which spins up all child processes. Access the UI at `http://localhost:5000`.

## Local-Only Operation

The system is designed to run completely offline:

- **Models**: Ensure Qwen, Moondream, and Kokoro weights are downloaded to your HuggingFace cache or specified local directories
- **Flags**: The code checks `check_internet()`. If offline, it forces `local_files_only=True` in transformers

## Architecture Benefits

- **True Parallelism**: Bypasses GIL through multiprocessing
- **Real-time Responsiveness**: Each cortex operates independently at its own speed
- **Modular Design**: Easy to upgrade individual components without affecting others
- **Scalable**: Add new cortices or modify existing ones without restructuring the entire system
- **Biomimetic**: Natural separation of concerns mirrors biological information processing