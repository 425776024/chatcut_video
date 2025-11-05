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

Process all media files (videos, audio, images) in the input directory:

```python
from utils.data_process import process_data, instruct_infer

# Set input and output directories
in_dir = '/path/to/input/media'
out_dir = '/path/to/output'

# Process media files (segmentation, recognition, etc.)
process_data(in_dir, out_dir)

# Use Qwen3-VL for multimodal understanding
instruct_infer(out_dir)
```

### 2. Generate Editing Project

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

### 3. Use Speech Recognition Independently

```python
from utils.audio2subtitle import audio2subtitle

# Recognize audio/video file
segments = audio2subtitle('/path/to/audio.wav')
for segment in segments:
    print(f"{segment['start']:.2f}s - {segment['end']:.2f}s: {segment['text']}")
```

### 4. Use Video Segmentation Independently

```python
from utils.video_seg import video_seg

video_path = '/path/to/video.mp4'
output_dir = '/path/to/output'

media_data = video_seg(video_path, output_dir)
print(media_data)
```

## Output Format

### media.json Format

After processing, a `media.json` file will be generated containing metadata for all media:

```json
{
  "image": {
    "image1.jpg": {
      "des": "Image description"
    }
  },
  "audio": {
    "audio1.json/0001.wav": {
      "start": 0.0,
      "end": 3.5,
      "text": "Recognized text content"
    }
  },
  "video": {
    "video1/0001.mp4": {
      "start": 0.0,
      "end": 5.2,
      "text": "Speech recognition text",
      "des": "Video segment description"
    }
  }
}
```

### Project Configuration Format

Generated editing project configuration format:

```json
{
  "tracks": [
    {
      "type": "text",
      "clips": [
        [0.0, 8.0, "This is text content"],
        [2.0, 4.0, "Text content"]
      ]
    },
    {
      "type": "image",
      "clips": [
        [2.5, 6.5, "cutme.png"],
        [4.8, 8.5, "lv.jpeg"]
      ]
    },
    {
      "type": "audio",
      "clips": [
        [0.0, 3.8, "s1.mp3"],
        [3.8, 6.0, "s1.wav"]
      ]
    },
    {
      "type": "video",
      "clips": [
        [0.0, 8.5, "1.mp4"]
      ]
    }
  ]
}
```

## Workflow

1. **Media Preprocessing**
   - Video: Extract audio → Speech recognition → Segment video based on recognition results
   - Audio: Speech recognition → Segment audio based on recognition results
   - Image: Directly copy to output directory

2. **Multimodal Understanding**
   - Use Qwen3-VL model to describe video segments and images
   - Generate complete metadata including timestamps, text, and descriptions

3. **Project Generation**
   - Based on media metadata and user requirements
   - Use LLM to generate editing project configuration in the required format

## Notes

1. **Model Paths**: Ensure Qwen3-VL and Qwen3 models are properly downloaded and paths are configured
2. **GPU Memory**: Pay attention to GPU memory usage when processing large video files; code includes memory cleanup mechanisms
3. **FFmpeg**: Ensure FFmpeg is properly installed and accessible from command line
4. **API Keys**: Configure correct API keys when using LLM APIs
5. **File Paths**: Code contains some hardcoded paths that need to be modified according to actual usage

## Performance Optimization

- **Batch Processing**: Supports batch processing of multiple media files
- **Memory Management**: Automatically cleans up GPU cache and temporary tensors
- **Resolution Scaling**: Automatically scales large resolution videos to save memory
- **Silence Detection**: Automatically skips silent segments to improve processing efficiency

## License

This project is based on the Qwen3 series models. Please follow the corresponding open source license.

## Contributing

Issues and Pull Requests are welcome!

## Changelog

- Support for video/audio/image multi-format processing
- Integrated Qwen3-VL multimodal understanding
- Support for intelligent editing project generation
- Optimized memory management and processing performance

