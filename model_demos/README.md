# Project Setup

Set up the virtual environment for Orenda.

## Prerequisites

- Python 3.12.x or 3.11.9 on your system DO NOT USE other python versions
- Git (for cloning the repository)
- Make sure you're in the model_demos directory
```cmd
   cd model_demos
```
## Check Python Version
### Windows
   ```cmd
   py --version
   ```
### OS/Linux
   ```cmd
   python --version
   ```

## Virtual Environment Setup

### Windows

1. **Create the virtual environment:**
get rid of the -3.12 arg if your system version was already 3.12
   ```cmd
   py -3.12 -m venv orenda
   ```

2. **Activate the virtual environment:**
   ```cmd
   orenda\Scripts\activate
   ```

3. **Verify activation:**
   You should see `(orenda)` at the beginning of your command prompt.

### Mac/Linux

1. **Create the virtual environment:**
get rid of the -3.12 arg if your system version was already 3.12
   ```bash
   python3.12 -m venv orenda
   ```

2. **Activate the virtual environment:**
   ```bash
   source orenda/bin/activate
   ```

3. **Verify activation:**
   You should see `(orenda)` at the beginning of your terminal prompt.

## Installing Dependencies
Once your virtual environment is activated, install the required packages:

Torch Libs for use with Nvidia 5000 series needs to be from the nightly build at the moment (Sept 2025). For those machines/GPU cards use

```bash
    pip install -r pytorch-cu128-requirements.txt
```
Else non-5000 series Nvida card OR mac/linux use:

```bash
    pip install -r pytorch-requirements.txt
```

then get the rest of the requirements.

```bash
    pip install -r requirements.txt
```
## Windows only Last Step libvips 
### Moondream2 Issue without libvips
We need to add libvips library to the system path on Windows to use moondream2.  The bin zip is already in this repo in the model_demos/dependencies/libvips/vips-dev-w64-web-8.17.2.zip you can unzip in place and add to system PATH, or run the helper function to automatically do that
```bash
python libvips_win_helper.py
```
## Pre-Load Models
All dependencies should be in. Restart terminal just to make sure (more of a windows issue), then download all the models
```bash
python model_loader.py
```
Done!

## Deactivating the Virtual Environment

When you're done working on the project, you can deactivate the virtual environment:

```bash
deactivate
```

## Troubleshooting

### Python Command Not Found (Windows)
If `python` is not recognized, try using `py` instead (typical for Windows):
```cmd
py -m venv orenda
```

### Permission Issues (Mac/Linux)
If you encounter permission issues, you might need to use `python3` instead of `python`:
```bash
python3 -m venv orenda
```

### Virtual Environment Not Activating
Make sure you're in the correct directory where you created the virtual environment, and double-check the activation command for your operating system.

---

**Note:** Always make sure your virtual environment is activated (you see `(orenda)` in your prompt) before installing packages or running the project.

**Dev:** Keep requirements.txt udpated
```bash 
pip list --format=freeze > requirements.txt
```



<div class="message-content "><div><h1>Sensory Fusion Pipeline</h1>
<p>This system aims to establish a robust, real-time, and multimodal sensory fusion pipeline that concurrently manages the acquisition and advanced processing of streaming video and audio inputs. It translates raw sensory data into high-level contextual awareness.</p>
<h2>Goals</h2>
<ol>
<li><strong>Low Latency:</strong> Using multiprocessing and non-blocking queues to separate real-time capture from computationally heavy processing.</li>
<li><strong>Modularity:</strong> Isolating Visual and Auditory processing into independent "Cortexes" and "Nerve" components.</li>
<li><strong>Contextual Synthesis:</strong> Fusing asynchronous data streams into a single, time-coherent state.</li>
<li><strong>Live Streaming Interface:</strong> Providing a live streaming interface via Flask for real-time monitoring of the visual processing chain.</li>
</ol>
<h2>Conceptual Breakdown</h2>
<p>The system is structured around a biological analogy, using multiprocessing and queues to handle concurrency and data flow.</p>
<table>
<thead>
<tr>
<th>Concept</th>
<th>Component</th>
<th>Function</th>
</tr>
</thead>
<tbody><tr>
<td><strong>Data Acquisition</strong></td>
<td><strong>Nerves</strong> (<code>optic_nerve_connection</code>, <code>auditory_nerve_connection</code>)</td>
<td>Real-time, low-latency capture of raw data. Prioritizes speed and drops older data if queues are full (LIFO).</td>
</tr>
<tr>
<td><strong>Primary Processing</strong></td>
<td><strong>Cortex Cores</strong> (<code>visual_cortex_core</code>, <code>auditory_cortex_core</code>)</td>
<td>Main loops for each modality. Apply initial filtering and manage data flow to specialized AI workers.</td>
</tr>
<tr>
<td><strong>Specialized AI Workers</strong></td>
<td><strong>VLM, STT, VR</strong></td>
<td>Dedicated processes for high-computation tasks. Isolated to prevent blocking the Cortex Cores.</td>
</tr>
<tr>
<td><strong>Multimodal Fusion</strong></td>
<td><strong>Temporal Lobe</strong></td>
<td>Central synthesis component. Collects, buffers, and synchronizes asynchronous outputs to create a unified state of awareness.</td>
</tr>
<tr>
<td><strong>Inter-Process Communication</strong></td>
<td><strong>Multiprocessing Queues</strong></td>
<td>Primary mechanism for data transfer. Queues are small and non-blocking, ensuring data is processed quickly or dropped/overwritten.</td>
</tr>
<tr>
<td><strong>Memory/Identity</strong></td>
<td><strong>Database/VR Worker</strong></td>
<td>Handles persistent storage of contextual information, specifically managing known speaker profiles and identities.</td>
</tr>
</tbody></table>
<h2>Class Definitions</h2>
<h3><code>VisualCortex</code></h3>
<ul>
<li><strong>Responsibility:</strong> Manages visual data capture and processing pipelines, setting up and managing multiprocessing queues, starting core vision processes, and providing an interface for external video streaming.</li>
<li><strong>Key Feature:</strong> Allows runtime instruction updates to enable or disable computationally intensive tasks.</li>
</ul>
<h3><code>AuditoryCortex</code></h3>
<ul>
<li><strong>Responsibility:</strong> Manages the complete auditory processing chain, including raw capture, real-time speech detection, transcription, and speaker identification.</li>
<li><strong>Key Feature:</strong> Uses multiple internal queues to segregate audio samples needing processing from final transcription results, ensuring specialized workers receive necessary data.</li>
</ul>
<h3><code>TemporalLobe</code></h3>
<ul>
<li><strong>Responsibility:</strong> Central integration point, responsible for high-level awareness and fusion. Bridges data output from the <code>VisualCortex</code> and <code>AuditoryCortex</code>.</li>
</ul>
<h2>Function Descriptions</h2>
<h3>Visual Functions</h3>
<table>
<thead>
<tr>
<th>Function</th>
<th>Role in Program</th>
</tr>
</thead>
<tbody><tr>
<td><code>optic_nerve_connection</code></td>
<td>Captures raw video frames from a camera index, using <code>put_nowait</code> to implement a strict LIFO queue strategy, ensuring low latency and prioritizing the freshest frame.</td>
</tr>
<tr>
<td><code>visual_cortex_core</code></td>
<td>Reads frames from the <code>internal_nerve_queue</code>, applies fast processing via <code>visual_cortex_process_img</code>, and calculates visual cortex FPS.</td>
</tr>
<tr>
<td><code>visual_cortex_process_img</code></td>
<td>Executes object detection using <code>YOLOhf</code>, annotates the frame, and returns the processed image and a boolean indicating if a person was detected. Handles memory cleanup.</td>
</tr>
<tr>
<td><code>visual_cortex_worker_vlm</code></td>
<td>A persistent multiprocessing worker responsible for time-consuming Visual Language Model tasks. Waits for tasks on <code>internal_to_vlm_queue</code>, processes the image, and returns results via <code>internal_from_vlm_queue</code>.</td>
</tr>
<tr>
<td><code>VisualCortex.generate_img_frames</code></td>
<td>A Flask generator function that pulls processed frames from <code>external_img_queue</code> and yields them as JPEG bytes for an MJPEG stream, enabling live visual monitoring.</td>
</tr>
<tr>
<td><code>VisualCortex.send_cortex_instructions</code></td>
<td>Sends configuration changes to the running <code>visual_cortex_core</code> process via a control queue.</td>
</tr>
</tbody></table>
<h3>Auditory Functions</h3>
<table>
<thead>
<tr>
<th>Function</th>
<th>Role in Program</th>
</tr>
</thead>
<tbody><tr>
<td><code>auditory_nerve_connection</code></td>
<td>Captures raw audio chunks using <code>sounddevice</code> and places them directly into <code>internal_nerve_queue</code>.</td>
</tr>
<tr>
<td><code>auditory_cortex_core</code></td>
<td>Performs real-time Voice Activity Detection, maintains a pre-speech buffer, and applies the Voice Lock logic.</td>
</tr>
<tr>
<td><code>auditory_cortex_worker_speechToText</code></td>
<td>A CPU-intensive worker that pulls speech segments from the Auditory Cortex and implements an incremental transcription strategy.</td>
</tr>
<tr>
<td><code>auditory_cortex_worker_voiceRecognition</code></td>
<td>A GPU-accelerated worker that processes audio samples received from the STT worker, identifies known speakers, and updates the identity database.</td>
</tr>
<tr>
<td><code>find_overlap_position</code></td>
<td>Used by the STT worker to splice new interim transcription results onto the existing working transcript, correcting past errors and removing redundant words.</td>
</tr>
</tbody></table>
<h3>Temporal Lobe Functions</h3>
<table>
<thead>
<tr>
<th>Function</th>
<th>Role in Program</th>
</tr>
</thead>
<tbody><tr>
<td><code>TemporalLobe._collect_frames</code></td>
<td>Asynchronously polls all Cortex output queues and adds data to time-based internal buffers.</td>
</tr>
<tr>
<td><code>TemporalLobe._clean_old_frames</code></td>
<td>Enforces the <code>buffer_duration</code> by removing old frames from the internal visual and audio buffers.</td>
</tr>
<tr>
<td><code>TemporalLobe.get_unified_state</code></td>
<td>Synthesizes the final contextual snapshot, merging non-default values into a single state object. Implements filtering based on <code>locked_speaker_id</code>.</td>
</tr>
<tr>
<td><code>TemporalLobe.unlock_speaker</code></td>
<td>Resets the <code>locked_speaker_id</code>, allowing the auditory system to listen for a new wake word or process any speaker's transcription.</td>
</tr>
</tbody></table>
<h3>Memory Function</h3>
<table>
<thead>
<tr>
<th>Function</th>
<th>Role in Program</th>
</tr>
</thead>
<tbody><tr>
<td><code>create_known_human_database</code></td>
<td>Sets up the SQLite database schema for persistent information about recognized humans.</td>
</tr>
</tbody></table>
</div></div>