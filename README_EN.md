# Qwen3 Multimodal Video Processing Project

English | [中文](README.md)

An intelligent video/audio/image processing system based on Qwen3-VL multimodal large language model, supporting automatic speech recognition, video segmentation, media content understanding, and intelligent editing project generation.

## Features

### 🎬 Media Processing
- **Video Processing**: Automatic video segmentation with intelligent splitting based on speech recognition results
- **Audio Processing**: Audio-to-subtitle conversion with silence detection and automatic segmentation
- **Image Processing**: Image content description and understanding
- **Multi-format Support**: Supports MP4, MOV, WAV, MP3, JPG, PNG, and many other formats

### 🧠 Multimodal Understanding
- **Video Understanding**: Uses Qwen3-VL model to describe and summarize video segments
- **Image Understanding**: Automatically generates scene, character, and overall descriptions for images
- **Intelligent Analysis**: Combines speech recognition and visual understanding to generate complete media metadata

### ✂️ Intelligent Editing
- **Project Generation**: Automatically generates video editing project configurations based on media resources and user requirements
- **Multi-track Support**: Supports multi-track editing with text, images, audio, and video
- **Timeline Alignment**: Automatically handles timestamps to ensure proper alignment of track elements

### 🎤 Speech Recognition
- **Multi-engine Support**: Supports both Faster-Whisper and WhisperX recognition engines
- **Word-level Timestamps**: Provides precise word-level time alignment
- **Silence Detection**: Automatically detects and processes silent segments

## Project Structure

```
qwen3/
├── qwen3/                    # Core processing modules
│   ├── llm_processor.py      # LLM processor (for video segmentation and project generation)
│   ├── media2track.py        # Media to track configuration generation
│   └── config.py             # Configuration file
├── qwen3_vl/                 # Qwen3-VL multimodal processing
│   ├── video_inference.py    # Video inference
│   └── batch_inference.py    # Batch inference
├── speech_recognizers/       # Speech recognition module
│   ├── speech_recognizer.py          # Speech recognizer base class
│   ├── faster_whisper_speech_recognizer.py  # Faster-Whisper implementation
│   ├── whisperx_speech_recognizer.py        # WhisperX implementation
│   └── speech_recognizer_factory.py         # Factory class
├── utils/                    # Utility functions
│   ├── data_process.py       # Main data processing pipeline
│   ├── video_seg.py          # Video segmentation
│   ├── audio2subtitle.py     # Audio to subtitle conversion
│   ├── media2simple.py       # Media simplification processing
│   └── pack_video.py         # Video packaging
└── data/                     # Data directory
    ├── instruct_input/       # Input media files
    ├── instruct_output/      # Processed output
    └── project_sample.json   # Project configuration example
```

## Installation

### System Requirements
- Python 3.8+
- CUDA (optional, for GPU acceleration)
- FFmpeg (for video/audio processing)

### Python Dependencies

```bash
pip install torch transformers
pip install opencv-python
pip install faster-whisper  # or whisperx
pip install librosa soundfile  # for audio processing
pip install openai  # for LLM API calls
pip install sympy
```

### FFmpeg Installation

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [FFmpeg official website](https://ffmpeg.org/download.html) and add to PATH

## Configuration

### Model Path Configuration

Configure in `qwen3/config.py`:

```python
# Qwen3-VL model path
QWEN3_VL_MODEL_PATH = "/path/to/Qwen3-VL-4B-Instruct"

# Qwen3 text model path
QWEN3_MODEL_PATH = "/path/to/Qwen3-4B"
```

### LLM API Configuration

Configure LLM API in `qwen3/config.py`:

```python
LLM_MODEL_OPTIONS = [
    {
        "model": "deepseek-v3-0324",
        "base_url": "https://api.lkeap.cloud.tencent.com/v1",
        "api_key_env_name": "DEEPSEEK_V3_API_KEY",
        "label": "DeepSeek-V3-0324",
        "max_tokens": 4096
    }
]
```

### Speech Recognition Configuration

Configure in `qwen3/config.py`:

```python
# Speech recognition engine type
SPEECH_RECOGNIZER_TYPE = 'faster-whisper'  # or 'whisperx'

# Whisper model configuration
WHISPER_MODEL_SIZE = 'large-v2'  # tiny, base, small, medium, large, large-v2, large-v3
WHISPER_DEVICE = 'cuda'  # or 'cpu'
WHISPER_COMPUTE_TYPE = 'float16'  # float16, float32, int8
```

## Usage

### 1. Process Media Files
> python utils/data_process.py

Process all media files (videos, audio, images) in the input directory:

```python
from utils.data_process import process_data, instruct_infer

# Set input and output directories
in_dir = '/path/to/input/media'
out_dir = '/path/to/instruct_output'

# Process media files (segmentation, recognition, etc.)
process_data(in_dir, out_dir)

# Use Qwen3-VL for multimodal understanding
instruct_infer(out_dir)
```

### 2. Generate Editing Project JSON
> python utils/media2simple.py

Generate editing project configuration based on processed media resources:

```python
from utils.media2simple import generate_project_from_media

media_out_dir = '/path/to/instruct_output'
user_request = "Generate editing result highlighting that manual video editing is very troublesome"
sample_format_path = '/path/to/project_sample.json'
output_path = '/path/to/project_out.json'

generate_project_from_media(
    media_out_dir=media_out_dir,
    user_request=user_request,
    sample_format_path=sample_format_path,
    output_path=output_path
)
```

### 3. Generate Video from Editing Project JSON
> python utils/pack_video.py

## License

This project is based on the Qwen3 series models. Please follow the corresponding open source license.

## Contributing

Issues and Pull Requests are welcome!

## Changelog

- Support for video/audio/image multi-format processing
- Integrated Qwen3-VL multimodal understanding
- Support for intelligent editing project generation
- Optimized memory management and processing performance

