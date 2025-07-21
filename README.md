# Jarvis Voice Assistant

A sophisticated voice assistant system with persistent memory, multi-speaker support, and natural conversation capabilities.

## Overview

Jarvis is a comprehensive voice assistant that combines state-of-the-art speech recognition, language modeling, and text-to-speech synthesis with an intelligent memory system. Unlike traditional voice assistants, Jarvis remembers your conversations and provides increasingly personalized responses over time.

## Key Features

- 🎤 **Real-time Speech Recognition** with word boundary detection
- 🧠 **Persistent Memory System** using semantic embeddings
- 👥 **Multi-Speaker Support** with voice identification and clustering
- 🗣️ **Natural Text-to-Speech** with multiple voice options
- 💬 **Continuous Conversations** without repeated wake words
- 🔄 **Context-Aware Responses** based on conversation history
- ⚙️ **Highly Configurable** with extensive customization options

## Tech Stack

### Core AI Models
| Component | Model | Purpose |
|-----------|-------|---------|
| Speech Recognition | OpenAI Whisper (small) | Speech-to-text conversion |
| Language Model | Qwen2.5-7B-Instruct | Intelligent response generation |
| Text-to-Speech | Kokoro-82M | Natural speech synthesis |
| Voice Activity Detection | Silero VAD (ONNX) | Speech detection |
| Speaker Identification | SpeechBrain ECAPA-VOXCELEB | Speaker recognition |
| Text Embeddings | MixedBread mxbai-embed-large-v1 | Semantic memory search |

### Supporting Technologies
- **Audio**: SoundDevice for real-time capture/playback
- **ML Framework**: PyTorch, Transformers, scikit-learn
- **Memory**: Pickle files with vector similarity search
- **Clustering**: DBSCAN and K-means algorithms

## Installation
⚠️ System Requirements Warning: This system loads multiple large AI models simultaneously and requires significant computational resources:

Memory: 32GB+ RAM recommended (peak usage ~32GB)
Storage: 30GB free disk space for model files
Processing: Modern multi-core CPU or Apple Silicon recommended
Tested Configuration: MacBook Pro M4 with 48GB RAM runs smoothly

Systems with less than 16GB RAM may experience performance issues or crashes. 

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Microphone and speakers
- Internet connection (for initial model downloads)

### System Dependencies
System dependencies (espeak-ng, portaudio) are automatically handled by the Kokoro TTS dependency manager on first run. The system will:

Linux: Auto-install via apt-get or yum
macOS: Auto-install via Homebrew
Windows: sorry nothing at the moment

Simply install Python dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers sounddevice soundfile numpy scipy scikit-learn
pip install speechbrain silero-vad kokoro>=0.9.4
pip install onnxruntime requests pickle5
```
requirement.txt and an easy setup .sh file on the todo list

## Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd jarvis-voice-assistant
```

### 2. Download Models (First Run)
On first run, models will download automatically with internet connection. For offline setup:
```bash
# Only Qwen has manual download option
python qwen_.py download  

# Kokoro, Whisper, VAD, and other models download automatically on first use
# They will fallback to cached/offline versions if no internet connection
```

### Basic Usage
```python
from local_llm import Jarvis, JarvisConfig

# Initialize with default settings
jarvis = Jarvis()

# Start the assistant
jarvis.start()
```
### OR

### Run the Assistant With The Existing Launch Script
```bash
python local_llm.py
```

## Configuration

### Basic Configuration
```python
config = JarvisConfig(
    wake_word="jarvis",                    # Wake word to activate
    interrupt_phrase="enough jarvis",      # Phrase to stop responses
    wake_word_timeout=30.0,               # Seconds to wait for prompt
    conversation_timeout=45.0,            # Conversation idle timeout
    continuous_conversation=True,          # Allow conversation without wake word
    debug_mode=True,                      # Enable debug output
    
    # Memory System
    memory_similarity_threshold=0.7,       # Memory relevance threshold
    max_similar_memories=2,               # Max memories to include in context
    embeddings_file="jarvis_memory.pkl"   # Memory storage file
)

jarvis = Jarvis(config)
```

### Voice Configuration
```python
# Available voices in Kokoro TTS
FEMALE_VOICES = ["af_heart", "af_alloy", "af_bella", "af_nova", "af_sky"]
MALE_VOICES = ["am_adam", "am_echo", "am_eric", "am_liam", "am_onyx"]

# Set voice during initialization
jarvis.tts.voice = "af_sky"  # Female American voice
```

## Usage Examples

### Basic Interaction
```
User: "Jarvis, what's the weather like?"
Jarvis: "I don't have access to current weather data, but I can help you find weather information if you'd like to check a specific weather service or app."

User: "Tell me a joke"
Jarvis: "Why don't scientists trust atoms? Because they make up everything!"
```

### Memory-Aware Conversations
```
# First conversation
User: "Jarvis, I love reading science fiction novels"
Jarvis: "That's wonderful! Science fiction is such a rich genre..."

# Later conversation (Jarvis remembers)
User: "Jarvis, recommend something new"
Jarvis: "Since you mentioned loving science fiction novels, I'd recommend..."
```

## System Architecture

### Conversation Flow
```mermaid
graph LR
    A[Audio Input] --> B[VAD Detection]
    B --> C[Speech Recognition]
    C --> D[Speaker ID]
    D --> E[Wake Word Check]
    E --> F[Memory Retrieval]
    F --> G[LLM Processing]
    G --> H[Memory Storage]
    H --> I[TTS Generation]
    I --> J[Audio Output]
```

### Memory System
1. **Input Processing**: User speech transcribed and speaker identified
2. **Memory Search**: Semantic similarity search of past conversations
3. **Context Building**: Relevant memories added to system prompt
4. **Response Generation**: LLM generates contextually aware response
5. **Memory Storage**: New conversation stored with embeddings

### State Management
- **IDLE**: Waiting for wake word
- **LISTENING**: Capturing user prompt after wake word
- **THINKING**: Processing prompt with LLM
- **RESPONDING**: Playing TTS response

## Advanced Features

### View All Memories
```bash
python memory_viewer.py
```

### Memory Management
```python
# Get memory statistics
stats = jarvis.get_memory_stats()
print(f"Total memories: {stats['total_memories']}")

# Search memories
similar_memories = jarvis.search_memories("cooking recipes", n=5)

# Export conversation history
jarvis.export_memories("my_conversations.txt")

# Clear all memories (requires confirmation)
jarvis.clear_memory(confirm=True)
```

### Speaker Analysis
```python
# Get speaker statistics
speaker_stats = jarvis.speaker_id.get_speaker_stats()

# Get clustering information
clustering_stats = jarvis.speaker_id.get_clustering_stats()

# Merge speaker profiles (if misidentified)
jarvis.speaker_id.merge_speakers("USER_01", "USER_02", keep_id="USER_01")
```

### Real-time Monitoring
```python
# Get current system status
status = jarvis.get_status()
print(f"State: {status['state']}")
print(f"Active speakers: {status['active_speakers']}")
print(f"Memory count: {status['memory_stats']['total_memories']}")
```

## Configuration Files

### Memory Storage
- `jarvis_memory.pkl`: Semantic embeddings and conversation history
- `clustered_speaker_profiles.pkl`: Speaker identification data

### Model Cache Locations
- **Linux/macOS**: `~/.cache/huggingface/hub/`
- **Windows**: `%USERPROFILE%\.cache\huggingface\hub\`

## Troubleshooting

### Common Issues

#### Audio Problems
```bash
# List audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"

# Test microphone
python -c "import sounddevice; sounddevice.rec(16000)"
```

#### Model Loading Issues
```bash
# Force offline mode
python local_llm.py --offline

# Download models manually
python qwen_.py download
```

#### Memory Issues
```bash
# Check memory file
ls -la jarvis_memory.pkl

# Reset memory (caution: deletes all conversations)
rm jarvis_memory.pkl clustered_speaker_profiles.pkl
```

### Performance Optimization

#### GPU Usage
```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

#### Memory Management
```python
# Limit memory usage
config = JarvisConfig(
    max_similar_memories=1,        # Reduce memory context
    memory_similarity_threshold=0.8  # Higher threshold = fewer memories
)
```

## Development

### Adding Custom Tools
```python
def get_weather(location: str) -> str:
    """Get weather for a location"""
    # Your weather API implementation
    return f"Weather in {location}: Sunny, 72°F"

# Register tool with Jarvis
jarvis.llm.register_tool(
    get_weather,
    description="Get current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"]
    }
)
```

### Custom Callbacks
```python
class CustomCallback(VoiceAssistantCallback):
    def on_transcript_final(self, segment):
        print(f"Custom processing: {segment.text}")
        # Your custom logic here

# Use custom callback
callback = CustomCallback()
processor = VoiceAssistantSpeechProcessor(callback)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for Whisper speech recognition
- Alibaba for Qwen language models
- Kokoro team for TTS synthesis
- SpeechBrain community for speaker recognition
- MixedBread AI for embedding models

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review existing GitHub issues
3. Create a new issue with detailed information

---

**Note**: This is an AI assistant for educational and research purposes. Ensure compliance with applicable laws and regulations when using voice recording capabilities.