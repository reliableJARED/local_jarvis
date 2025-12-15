# Wan2.2 5B Image-to-Video Generator

A modular Python wrapper for the Wan2.2 TI2V-5B model, optimized for 16GB GPUs (RTX 5060 Ti).

## 🚀 Quick Start

### 1. Setup (One-time)

Run the setup script to install dependencies and download the model:

```bash
python setup_wan22.py --all
```

This will:
- Install all required dependencies
- Clone the Wan2.2 repository
- Download the TI2V-5B model (~5GB)
- Verify the setup

### 2. Basic Usage

```python
from wan22_video_i2v import Wan22ImageToVideo

# Initialize generator (optimized for 16GB GPU)
generator = Wan22ImageToVideo()

# Generate video from image
output_path = generator.generate_video(
    image_path="your_image.jpg",
    prompt="Gentle wind blowing through the scene",
    output_path="output_video.mp4"
)

print(f"Video generated: {output_path}")
```

### 3. Quick Function

```python
from wan22_video_i2v import generate_video_from_image

# One-line video generation
output = generate_video_from_image(
    image_path="input.jpg",
    prompt="Beautiful cinematic movement"
)
```

## 📁 Files Overview

- **`wan22_video_i2v.py`** - Main wrapper class
- **`setup_wan22.py`** - Setup and installation script
- **`example_i2v_usage.py`** - Usage examples and tutorials
- **`wan22_requirements.txt`** - Python dependencies

## 🔧 Installation Details

### System Requirements

- **GPU**: 16GB VRAM minimum (RTX 5060 Ti or equivalent)
- **RAM**: 32GB+ recommended
- **Storage**: ~10GB free space
- **OS**: Windows 10/11
- **Python**: 3.8+

### Manual Installation

If the automatic setup fails, install manually:

```bash
# Install dependencies
pip install -r wan22_requirements.txt

# Install optional flash attention (may fail - that's OK)
pip install flash_attn

# Clone Wan2.2 repository
git clone https://github.com/Wan-Video/Wan2.2.git

# Download model via Hugging Face
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B

# OR download via ModelScope (alternative)
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Wan-AI/Wan2.2-TI2V-5B', cache_dir='.', local_dir='./Wan2.2-TI2V-5B')"
```

## 🎨 Usage Examples

### Basic Image-to-Video

```python
from wan22_video_i2v import Wan22ImageToVideo

generator = Wan22ImageToVideo()

# Simple generation
output = generator.generate_video(
    image_path="landscape.jpg",
    prompt="Clouds moving across the sky",
    size="1280*704"  # Optimized resolution for 5B model
)
```

### Advanced Parameters

```python
output = generator.generate_video(
    image_path="portrait.jpg",
    prompt="Cinematic lighting changes with gentle movement",
    output_path="custom_output.mp4",
    size="1280*704",
    seed=42,                    # Reproducible results
    guidance_scale=7.5,         # Control adherence to prompt
    num_inference_steps=50      # Quality vs speed trade-off
)
```

### Batch Processing

```python
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
prompts = ["Wind through trees", "Ocean waves", "Flying birds"]

for i, (img, prompt) in enumerate(zip(images, prompts)):
    output = generator.generate_video(
        image_path=img,
        prompt=prompt,
        output_path=f"batch_{i}.mp4"
    )
    generator.cleanup()  # Clear GPU memory
```

## ⚙️ Configuration

The wrapper automatically applies memory optimizations for 16GB GPUs:

- `offload_model=True` - Offload model parts to CPU
- `convert_model_dtype=True` - Use efficient data types
- `t5_cpu=True` - Run text encoder on CPU

### Custom Configuration

```python
generator = Wan22ImageToVideo(
    model_path="custom/path/to/model",
    device="cuda",
    optimize_for_low_memory=True
)
```

## 📊 Performance Tips

### For 16GB GPU (RTX 5060 Ti)

1. **Use 720p resolution**: `1280*704` or `704*1280`
2. **Enable all memory optimizations** (default)
3. **Process one video at a time**
4. **Clear GPU cache** between generations
5. **Close other GPU-intensive applications**

### Expected Performance

- **Resolution**: 720p (1280x704)
- **Duration**: 5 seconds @ 24fps
- **Generation Time**: ~8-15 minutes
- **Memory Usage**: ~14-16GB VRAM

## 🛠️ Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Enable all memory optimizations
generator = Wan22ImageToVideo(optimize_for_low_memory=True)
# Clear cache between generations
generator.cleanup()
```

**2. Model Not Found**
```bash
python setup_wan22.py --download-model --clone-repo
```

**3. Dependencies Missing**
```bash
python setup_wan22.py --install-deps
```

**4. flash_attn Installation Fails**
This is normal and optional. The model will work without it.

### Check Setup Status

```python
from wan22_video_i2v import Wan22ImageToVideo

generator = Wan22ImageToVideo()

# Print system info
print(generator.get_system_info())

# Check dependencies
deps = generator.check_dependencies()
for dep, status in deps.items():
    print(f"{dep}: {'OK' if status else 'MISSING'}")
```

## 📝 API Reference

### Wan22ImageToVideo Class

```python
class Wan22ImageToVideo:
    def __init__(self, model_path=None, device=None, optimize_for_low_memory=True)
    def download_model(self, use_modelscope=False) -> bool
    def generate_video(self, image_path, prompt="", **kwargs) -> str
    def check_dependencies(self) -> dict
    def get_system_info(self) -> dict
    def cleanup(self)
```

### Generate Video Parameters

- `image_path` (str): Path to input image
- `prompt` (str): Text description of desired animation
- `output_path` (str, optional): Output video path
- `size` (str): Resolution "1280*704" (default)
- `seed` (int, optional): Random seed for reproducibility
- `guidance_scale` (float): Prompt adherence (default: 7.5)
- `num_inference_steps` (int): Quality steps (default: 50)

## 🔄 Integration with Other Scripts

### Import and Use

```python
# In your other Python scripts
from wan22_video_i2v import Wan22ImageToVideo, generate_video_from_image

# Method 1: Class-based (recommended for multiple generations)
generator = Wan22ImageToVideo()
video1 = generator.generate_video("img1.jpg", "prompt1")
video2 = generator.generate_video("img2.jpg", "prompt2")

# Method 2: Function-based (for single generations)
video = generate_video_from_image("image.jpg", "animation prompt")
```

### As a Module

```python
import sys
sys.path.append("path/to/local_jarvis")
from wan22_video_i2v import Wan22ImageToVideo
```

## 📄 Model Information

- **Model**: Wan2.2-TI2V-5B
- **Size**: ~5GB download
- **Resolution**: Up to 720P (1280x704)
- **Frame Rate**: 24 FPS
- **Duration**: 5 seconds default
- **Tasks**: Both Text-to-Video and Image-to-Video

## 🤝 Support

For issues and questions:

1. Check the troubleshooting section
2. Run `python example_i2v_usage.py` to test setup
3. Check GPU memory usage with Task Manager
4. Refer to the original [Wan2.2 repository](https://github.com/Wan-Video/Wan2.2)

## 📄 License

This wrapper follows the same license as the original Wan2.2 project (Apache 2.0).

---

*Optimized for RTX 5060 Ti (16GB) - August 2025*
