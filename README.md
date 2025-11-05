# Qwen3 多模态视频处理项目

[English](README_EN.md) | 中文

基于 Qwen3-VL 多模态大模型的智能视频/音频/图片处理系统，支持自动语音识别、视频分割、媒体内容理解和智能剪辑项目生成。

## 功能特性

### 🎬 媒体处理
- **视频处理**：自动分割视频片段，基于语音识别结果进行智能切分
- **音频处理**：支持音频转字幕，静音检测和自动分割
- **图片处理**：图片内容描述和理解
- **多格式支持**：支持 MP4、MOV、WAV、MP3、JPG、PNG 等多种格式

### 🧠 多模态理解
- **视频理解**：使用 Qwen3-VL 模型对视频片段进行描述和总结
- **图片理解**：自动生成图片的场景、角色和整体描述
- **智能分析**：结合语音识别和视觉理解，生成完整的媒体元数据

### ✂️ 智能剪辑
- **项目生成**：根据媒体资源和用户需求，自动生成视频剪辑项目配置
- **多轨道支持**：支持文本、图片、音频、视频多轨道编辑
- **时间轴对齐**：自动处理时间戳，确保各轨道元素正确对齐

### 🎤 语音识别
- **多引擎支持**：支持 Faster-Whisper 和 WhisperX 两种识别引擎
- **字词级时间戳**：提供精确的字词级时间对齐
- **静音检测**：自动检测并处理静音片段

## 项目结构

```
qwen3/
├── qwen3/                    # 核心处理模块
│   ├── llm_processor.py      # LLM 处理器（用于视频分段和项目生成）
│   ├── media2track.py        # 媒体转轨道配置生成
│   └── config.py             # 配置文件
├── qwen3_vl/                 # Qwen3-VL 多模态处理
│   ├── video_inference.py    # 视频推理
│   └── batch_inference.py    # 批量推理
├── speech_recognizers/       # 语音识别模块
│   ├── speech_recognizer.py          # 语音识别基类
│   ├── faster_whisper_speech_recognizer.py  # Faster-Whisper 实现
│   ├── whisperx_speech_recognizer.py        # WhisperX 实现
│   └── speech_recognizer_factory.py         # 工厂类
├── utils/                    # 工具函数
│   ├── data_process.py       # 数据处理主流程
│   ├── video_seg.py          # 视频分割
│   ├── audio2subtitle.py     # 音频转字幕
│   ├── media2simple.py       # 媒体简化处理
│   └── pack_video.py         # 视频打包
└── data/                     # 数据目录
    ├── instruct_input/       # 输入媒体文件
    ├── instruct_output/      # 处理结果输出
    └── project_sample.json   # 项目配置示例
```

## 安装依赖

### 系统要求
- Python 3.8+
- CUDA（可选，用于 GPU 加速）
- FFmpeg（用于视频/音频处理）

### Python 依赖

```bash
pip install torch transformers
pip install opencv-python
pip install faster-whisper  # 或 whisperx
pip install librosa soundfile  # 用于音频处理
pip install openai  # 用于 LLM API 调用
pip install sympy
```

### FFmpeg 安装

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
从 [FFmpeg官网](https://ffmpeg.org/download.html) 下载并添加到 PATH

## 配置说明

### 模型路径配置

在 `qwen3/config.py` 中配置：

```python
# Qwen3-VL 模型路径
QWEN3_VL_MODEL_PATH = "/path/to/Qwen3-VL-4B-Instruct"

# Qwen3 文本模型路径
QWEN3_MODEL_PATH = "/path/to/Qwen3-4B"
```

### LLM API 配置

在 `qwen3/config.py` 中配置 LLM API：

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

### 语音识别配置

在 `qwen3/config.py` 中配置：

```python
# 语音识别引擎类型
SPEECH_RECOGNIZER_TYPE = 'faster-whisper'  # 或 'whisperx'

# Whisper 模型配置
WHISPER_MODEL_SIZE = 'large-v2'  # tiny, base, small, medium, large, large-v2, large-v3
WHISPER_DEVICE = 'cuda'  # 或 'cpu'
WHISPER_COMPUTE_TYPE = 'float16'  # float16, float32, int8
```

## 使用方法

### 1. 处理媒体文件

处理输入目录中的所有媒体文件（视频、音频、图片）：

```python
from utils.data_process import process_data, instruct_infer

# 设置输入和输出目录
in_dir = '/path/to/input/media'
out_dir = '/path/to/output'

# 处理媒体文件（分割、识别等）
process_data(in_dir, out_dir)

# 使用 Qwen3-VL 进行多模态理解
instruct_infer(out_dir)
```

### 2. 生成剪辑项目

根据处理后的媒体资源生成剪辑项目配置：

```python
from utils.media2simple import generate_project_from_media

media_out_dir = '/path/to/instruct_output'
user_request = "以突出手动剪辑视频很麻烦为主题生成剪辑结果"
sample_format_path = '/path/to/project_sample.json'
output_path = '/path/to/project_out.json'

generate_project_from_media(
    media_out_dir=media_out_dir,
    user_request=user_request,
    sample_format_path=sample_format_path,
    output_path=output_path
)
```

### 3. 单独使用语音识别

```python
from utils.audio2subtitle import audio2subtitle

# 识别音频/视频文件
segments = audio2subtitle('/path/to/audio.wav')
for segment in segments:
    print(f"{segment['start']:.2f}s - {segment['end']:.2f}s: {segment['text']}")
```

### 4. 单独使用视频分割

```python
from utils.video_seg import video_seg

video_path = '/path/to/video.mp4'
output_dir = '/path/to/output'

media_data = video_seg(video_path, output_dir)
print(media_data)
```

## 输出格式

### media.json 格式

处理完成后会生成 `media.json` 文件，包含所有媒体的元数据：

```json
{
  "image": {
    "image1.jpg": {
      "des": "图片描述"
    }
  },
  "audio": {
    "audio1.json/0001.wav": {
      "start": 0.0,
      "end": 3.5,
      "text": "识别的文本内容"
    }
  },
  "video": {
    "video1/0001.mp4": {
      "start": 0.0,
      "end": 5.2,
      "text": "语音识别文本",
      "des": "视频片段描述"
    }
  }
}
```

### 项目配置格式

生成的剪辑项目配置格式：

```json
{
  "tracks": [
    {
      "type": "text",
      "clips": [
        [0.0, 8.0, "这是文本内容"],
        [2.0, 4.0, "文本内容"]
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

## 工作流程

1. **媒体预处理**
   - 视频：提取音频 → 语音识别 → 根据识别结果分割视频片段
   - 音频：语音识别 → 根据识别结果分割音频片段
   - 图片：直接复制到输出目录

2. **多模态理解**
   - 使用 Qwen3-VL 模型对视频片段和图片进行描述
   - 生成包含时间戳、文本和描述的完整元数据

3. **项目生成**
   - 根据媒体元数据和用户需求
   - 使用 LLM 生成符合格式要求的剪辑项目配置

## 注意事项

1. **模型路径**：确保 Qwen3-VL 和 Qwen3 模型已正确下载并配置路径
2. **GPU 内存**：处理大视频文件时注意 GPU 内存使用，代码已包含内存清理机制
3. **FFmpeg**：确保 FFmpeg 已正确安装并可在命令行访问
4. **API 密钥**：使用 LLM API 时需要配置正确的 API 密钥
5. **文件路径**：代码中包含一些硬编码路径，使用前需要根据实际情况修改

## 性能优化

- **批量处理**：支持批量处理多个媒体文件
- **内存管理**：自动清理 GPU 缓存和临时张量
- **分辨率缩放**：自动缩放大分辨率视频以节省内存
- **静音检测**：自动跳过静音片段，提高处理效率

## 许可证

本项目基于 Qwen3 系列模型，请遵循相应的开源许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

- 支持视频/音频/图片多格式处理
- 集成 Qwen3-VL 多模态理解
- 支持智能剪辑项目生成
- 优化内存管理和处理性能


