import os
import tempfile
import subprocess
import json
from pathlib import Path
import numpy as np
from speech_recognizers.speech_recognizer_factory import SpeechRecognizerFactory

SPEECH_RECOGNIZER_TYPE = 'faster-whisper'
model_size = 'small'
# 静音检测参数
SILENCE_THRESHOLD = 0.01  # RMS能量阈值，低于此值认为是静音
MIN_SILENCE_DURATION = 1.5  # 连续静音的最小时长（秒），超过此时长的静音会被切割
FRAME_LENGTH = 2048  # 音频帧长度
HOP_LENGTH = 512  # 帧移

# 视频文件扩展名列表
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.3gp', '.mpg', '.mpeg'}


def is_video_file(file_path):
    """判断文件是否为视频格式"""
    ext = Path(file_path).suffix.lower()
    return ext in VIDEO_EXTENSIONS


def extract_audio_from_video(video_path, audio_output_path):
    """从视频中提取音频"""
    try:
        # 使用 ffmpeg 提取音频
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # 不包含视频
            '-acodec', 'pcm_s16le',  # 使用 PCM 16位编码
            '-ar', '16000',  # 采样率 16kHz
            '-ac', '1',  # 单声道
            '-y',  # 覆盖输出文件
            audio_output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"提取音频失败: {e}")
        return False
    except FileNotFoundError:
        print("错误: 未找到 ffmpeg，请确保已安装 ffmpeg")
        return False


def is_audio_silent(audio_path):
    """检测音频是否为静音"""
    try:
        import librosa
        
        # 加载音频文件
        audio, sr = librosa.load(audio_path, sr=None)
        
        # 计算 RMS (Root Mean Square) 能量
        rms = librosa.feature.rms(y=audio)[0]
        mean_rms = np.mean(rms)
        
        return mean_rms < SILENCE_THRESHOLD
    except ImportError:
        print("警告: 未安装 librosa，无法检测静音，将跳过静音检测")
        return False
    except Exception as e:
        print(f"检测静音时出错: {e}")
        # 如果检测失败，假设有声音，继续处理
        return False


def detect_silence_segments(audio_path):
    """
    检测音频中的静音片段
    返回静音片段的时间范围列表 [(start1, end1), (start2, end2), ...]
    """
    try:
        import librosa
        
        # 加载音频文件
        audio, sr = librosa.load(audio_path, sr=None)
        
        # 计算 RMS 能量
        rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        
        # 计算每帧对应的时间
        frames = range(len(rms))
        times = librosa.frames_to_time(frames, sr=sr, hop_length=HOP_LENGTH)
        
        # 检测静音帧
        silence_mask = rms < SILENCE_THRESHOLD
        
        # 找到连续的静音片段
        silence_segments = []
        in_silence = False
        silence_start = 0
        
        for i, is_silent in enumerate(silence_mask):
            if is_silent and not in_silence:
                # 开始静音
                in_silence = True
                silence_start = times[i]
            elif not is_silent and in_silence:
                # 结束静音
                silence_end = times[i-1] if i > 0 else 0
                silence_duration = silence_end - silence_start
                if silence_duration >= MIN_SILENCE_DURATION:
                    silence_segments.append((silence_start, silence_end))
                in_silence = False
        
        # 处理音频结尾的静音
        if in_silence:
            silence_end = times[-1]
            silence_duration = silence_end - silence_start
            if silence_duration >= MIN_SILENCE_DURATION:
                silence_segments.append((silence_start, silence_end))
        
        return silence_segments
    except ImportError:
        print("警告: 未安装 librosa，无法检测静音片段")
        return []
    except Exception as e:
        print(f"检测静音片段时出错: {e}")
        return []


def split_audio_by_silence(audio_path, silence_segments, output_dir=None):
    """
    根据静音片段切割音频文件
    返回包含文件路径和时间偏移量的元组列表 [(file_path, time_offset), ...]
    """
    if not silence_segments:
        return [(audio_path, 0.0)]
    
    try:
        import librosa
        import soundfile as sf
        
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=None)
        duration = len(audio) / sr
        
        # 确定切割点（静音片段的中间位置）
        split_points = []
        for start, end in silence_segments:
            split_point = (start + end) / 2
            split_points.append(split_point)
        
        # 添加开始和结束点
        split_points = [0.0] + sorted(split_points) + [duration]
        
        # 移除重复和无效的点
        split_points = [p for p in split_points if 0 <= p <= duration]
        split_points = sorted(list(set(split_points)))
        
        # 生成切割后的音频文件
        split_files = []
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        base_name = Path(audio_path).stem
        for i in range(len(split_points) - 1):
            start_time = split_points[i]
            end_time = split_points[i + 1]
            
            # 只保留长度超过0.1秒的片段
            if end_time - start_time < 0.1:
                continue
            
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = audio[start_sample:end_sample]
            
            output_path = os.path.join(output_dir, f"{base_name}_seg_{i:04d}.wav")
            sf.write(output_path, segment_audio, sr)
            # 返回文件路径和该片段在原音频中的起始时间偏移量
            split_files.append((output_path, start_time))
        
        return split_files if split_files else [(audio_path, 0.0)]
    except ImportError as e:
        print(f"警告: 缺少必要的库 ({e})，无法切割音频")
        return [(audio_path, 0.0)]
    except Exception as e:
        print(f"切割音频时出错: {e}")
        return [(audio_path, 0.0)]


def merge_segments_with_silence_cuts(segments, silence_segments):
    """
    合并语音识别结果，并在静音片段处插入标记
    返回处理后的segments列表，其中静音片段被标记或移除
    """
    if not silence_segments:
        return segments
    
    result = []
    silence_index = 0
    
    for segment in segments:
        seg_start = segment['start']
        seg_end = segment['end']
        
        # 检查当前segment是否在静音片段内
        is_in_silence = False
        for silence_start, silence_end in silence_segments:
            # 如果segment与静音片段有重叠，标记为静音
            if not (seg_end < silence_start or seg_start > silence_end):
                is_in_silence = True
                break
        
        # 只保留非静音片段
        if not is_in_silence:
            result.append(segment)
        else:
            # 可选：在静音处插入标记segment
            # 这里我们选择直接跳过，不插入静音标记
            pass
    
    return result


def audio2subtitle(file_path, enable_silence_detection=True, enable_auto_split=True):
    """
    将音频/视频文件转换为字幕
    
    参数:
        file_path: 音频或视频文件路径
        enable_silence_detection: 是否启用静音检测（默认True）
        enable_auto_split: 是否启用自动切割静音片段（默认True）
    
    返回:
        包含字幕segments的列表，每个segment包含:
        - start: 开始时间（秒）
        - end: 结束时间（秒）
        - text: 识别的文本
        - words: 字词级时间戳列表（如果可用）
    """
    print(f"开始语音识别: {file_path}")

    # 如果是视频文件，需要提取音频
    temp_audio = None
    temp_dir = None
    split_audio_files = []  # 用于存储切割后的文件路径，以便清理
    
    try:
        if is_video_file(file_path):
            print(f"检测到视频文件: {file_path}")
            # 创建临时音频文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                temp_audio = tmp_file.name
            
            # 提取音频
            print(f"正在从视频提取音频...")
            if not extract_audio_from_video(file_path, temp_audio):
                print("提取音频失败，返回空结果")
                return []
            
            # 检测是否为静音
            print("检测音频是否有声音...")
            if is_audio_silent(temp_audio):
                print("视频为静音，返回空结果")
                return []
            
            print("音频提取成功")
            file_path = temp_audio
        
        # 静音检测和自动切割
        silence_segments = []
        if enable_silence_detection:
            print("正在检测静音片段...")
            silence_segments = detect_silence_segments(file_path)
            if silence_segments:
                print(f"检测到 {len(silence_segments)} 个静音片段（超过{MIN_SILENCE_DURATION}秒）")
                for i, (start, end) in enumerate(silence_segments):
                    print(f"  静音片段 {i+1}: {start:.2f}s - {end:.2f}s (时长: {end-start:.2f}s)")
            else:
                print("未检测到符合条件的静音片段")
        
        # 根据静音片段切割音频
        audio_files_to_process = [(file_path, 0.0)]  # [(file_path, time_offset), ...]
        if enable_auto_split and silence_segments:
            print("正在根据静音片段切割音频...")
            temp_dir = tempfile.mkdtemp()
            split_audio_files_info = split_audio_by_silence(file_path, silence_segments, temp_dir)
            if len(split_audio_files_info) > 1:
                print(f"音频已切割为 {len(split_audio_files_info)} 个片段")
                audio_files_to_process = split_audio_files_info
                split_audio_files = [info[0] for info in split_audio_files_info]  # 用于清理
            else:
                print("音频无需切割")
                split_audio_files = []
        
        # 进行语音识别
        print("开始语音识别...")
        recognizer = SpeechRecognizerFactory.get_speech_recognizer_by_type(SPEECH_RECOGNIZER_TYPE, model_size)
        
        all_segments = []
        
        for audio_file, time_offset in audio_files_to_process:
            print(f"识别音频片段: {audio_file} (时间偏移: {time_offset:.2f}s)")
            result = recognizer.transcribe(audio_file)
            
            # 调整时间戳（添加时间偏移量）
            segments = result.get('segments', [])
            for segment in segments:
                segment['start'] = round(segment['start'] + time_offset, 1)
                segment['end'] = round(segment['end'] + time_offset, 1)
                # 调整字词级时间戳
                if 'words' in segment:
                    for word in segment['words']:
                        word['start'] = round(word['start'] + time_offset, 1)
                        word['end'] = round(word['end'] + time_offset, 1)
            
            all_segments.extend(segments)
        
        # 如果启用了静音检测但没有自动切割，则过滤掉静音片段中的segments
        if enable_silence_detection and not enable_auto_split and silence_segments:
            print("过滤静音片段中的识别结果...")
            all_segments = merge_segments_with_silence_cuts(all_segments, silence_segments)
        
        # 去重：如果多个segments有相同的时间范围和文本，只保留一个
        seen_segment_keys = set()
        unique_segments = []
        
        for segment in all_segments:
            start = round(segment.get('start', 0), 1)
            end = round(segment.get('end', 0), 1)
            text = segment.get('text', '').strip()
            
            # 使用(start, end, text)作为唯一key
            segment_key = (start, end, text)
            
            if segment_key not in seen_segment_keys:
                seen_segment_keys.add(segment_key)
                unique_segments.append(segment)
            else:
                print(f"跳过重复的segment: {start:.2f}s-{end:.2f}s, text='{text[:30]}...'")
        
        # 按时间顺序排序
        unique_segments.sort(key=lambda s: (s.get('start', 0), s.get('end', 0)))
        
        print(f"识别完成，共生成 {len(unique_segments)} 个字幕片段（去重后，原始 {len(all_segments)} 个）")
        if unique_segments and 'words' in unique_segments[0]:
            word_count = sum(len(seg.get('words', [])) for seg in unique_segments)
            print(f"字词级时间戳已启用，共识别 {word_count} 个字词")
        
        return unique_segments
        
    except Exception as e:
        print(f"处理文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        # 清理临时文件
        if temp_audio and os.path.exists(temp_audio):
            try:
                os.unlink(temp_audio)
            except Exception as e:
                print(f"清理临时文件失败: {e}")
        
        # 清理切割后的临时文件
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"清理临时目录失败: {e}")
        
        # 清理split_audio_files中的临时文件
        for split_file in split_audio_files:
            if split_file != file_path and os.path.exists(split_file):
                try:
                    os.unlink(split_file)
                except Exception as e:
                    print(f"清理切割文件失败: {e}")


def get_audio_duration(audio_path):
    """获取音频总时长（秒），使用ffprobe"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            audio_path
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        duration = float(result.stdout.strip())
        return duration
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        # 如果ffprobe失败，尝试使用librosa
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=None)
            duration = len(audio) / sr
            return duration
        except ImportError:
            print(f"错误: 无法获取音频时长，需要安装ffprobe或librosa: {e}")
            return 0.0
        except Exception as e:
            print(f"获取音频时长失败: {e}")
            return 0.0


def format_time(seconds):
    """将秒数格式化为 HH:MM:SS.mmm 格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def split_audio(audio_path, start_time, end_time, output_path):
    """使用 ffmpeg 分割音频片段
    
    Args:
        audio_path: 输入音频路径
        start_time: 开始时间（秒）
        end_time: 结束时间（秒）
        output_path: 输出音频路径
    """
    start_str = format_time(start_time)
    duration = end_time - start_time
    
    cmd = [
        'ffmpeg',
        '-ss', start_str,  # -ss 在 -i 之前，seek到最近关键帧
        '-i', audio_path,
        '-t', str(duration),
        '-c', 'copy',  # 使用流复制，速度快
        '-avoid_negative_ts', 'make_zero',
        '-y',  # 覆盖输出文件
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"分割音频失败: {e}")
        print(f"命令: {' '.join(cmd)}")
        return False
    except FileNotFoundError:
        print("错误: 未找到 ffmpeg，请确保已安装 ffmpeg")
        return False


def audio_seg(audio_path, output_dir):
    """将音频文件按照语音识别的segments进行分割存储
    
    Args:
        audio_path: 输入音频文件路径
        output_dir: 输出目录，用于存储分割后的音频片段和media.json
    
    Returns:
        media_data: 包含音频信息和segments的字典
    """
    # 调用audio2subtitle获取语音识别结果
    res = audio2subtitle(audio_path)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取音频信息（时长）
    total_duration = get_audio_duration(audio_path)
    
    print(f"音频总时长: {total_duration:.2f}秒")
    
    # 生成所有分割点
    split_points = [0.0]  # 从0开始
    
    # 检查音频是否有声音（res为空表示没有声音）
    has_audio = res and len(res) > 0
    if not has_audio:
        print("音频没有声音，按3秒一段进行切分")
        # 按3秒一段生成分割点
        current_time = 0.0
        while current_time < total_duration:
            split_points.append(current_time)
            current_time += 3.0
        # 确保最后一个分割点是音频结束时间
        if split_points[-1] < total_duration:
            split_points.append(total_duration)
        # 去重并排序
        split_points = sorted(set(split_points))
    else:
        # 从 segments 中提取所有时间点
        for segment in res:
            if 'start' in segment:
                split_points.append(segment['start'])
            if 'end' in segment:
                split_points.append(segment['end'])
        
        # 添加音频结束点
        if split_points[-1] < total_duration:
            split_points.append(total_duration)
        
        # 去重并排序
        split_points = sorted(set(split_points))
    
    print(f"分割点: {split_points}")
    
    # 创建字典用于快速查找 segment 信息
    # 如果多个segments有相同的(start, end)和text，只保留一个（去重）
    segment_map = {}
    seen_segment_keys = set()  # 用于去重完全相同的segments
    
    if res:
        for segment in res:
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            text = segment.get('text', '').strip()
            
            # 创建唯一key：使用(start, end, text)来识别重复的segment
            segment_key = (round(start, 2), round(end, 2), text)
            
            # 如果这个segment已经存在，跳过（去重）
            if segment_key in seen_segment_keys:
                print(f"跳过重复的segment: {start:.2f}s-{end:.2f}s, text='{text[:20]}...'")
                continue
            
            seen_segment_keys.add(segment_key)
            # 存储完整的 segment 信息，包括 text 和 words
            segment_map[(start, end)] = segment
    
    # 获取原始音频文件的扩展名，用于保存分割后的音频
    audio_ext = Path(audio_path).suffix.lower()
    if not audio_ext:
        audio_ext = '.m4a'  # 默认使用m4a格式
    
    # 分割音频并生成
    media_info = {}
    segment_index = 0
    
    for i in range(len(split_points) - 1):
        start_time = split_points[i]
        end_time = split_points[i + 1]
        segment_duration = end_time - start_time
        
        # 跳过长度为0的片段
        if segment_duration <= 0:
            continue
        
        # 对于无声音音频，如果是最后一段（接近音频结尾），即使小于1.5秒也保留
        is_near_end = abs(end_time - total_duration) < 0.1  # 容差0.1秒
        
        # 跳过时长小于1.5秒的片段（无声音音频的最后一段除外）
        if segment_duration < 1.5:
            print(f"跳过短片段: {start_time:.2f}s - {end_time:.2f}s (时长: {segment_duration:.2f}s < 1.5s)")
            continue
        
        segment_index += 1
        segment_name = f"{segment_index:04d}{audio_ext}"
        output_path = os.path.join(output_dir, segment_name)
        
        # 查找与当前音频片段最匹配的 segment（正常情况下只有一个）
        matched_segment = None
        best_overlap = 0
        
        for (seg_start, seg_end), segment_info in segment_map.items():
            # 检查当前片段是否与标记的片段重叠
            if not (end_time <= seg_start or start_time >= seg_end):
                # 计算重叠度
                overlap = min(end_time, seg_end) - max(start_time, seg_start)
                # 选择重叠度最大的segment
                if overlap > best_overlap:
                    best_overlap = overlap
                    matched_segment = segment_info
        
        # 提取文本和 words
        text = ''
        words = []
        
        if matched_segment:
            text = matched_segment.get('text', '').strip()
            seg_words = matched_segment.get('words', [])
            
            # 过滤并裁剪 words 到当前音频片段时间范围内
            for word in seg_words:
                word_start = word.get('start', 0)
                word_end = word.get('end', 0)
                
                # 只保留与音频片段有重叠的 words
                if not (word_end < start_time or word_start > end_time):
                    word_copy = word.copy()
                    # 裁剪时间戳到音频片段范围内
                    word_copy['start'] = max(start_time, min(word_start, end_time))
                    word_copy['end'] = max(start_time, min(word_end, end_time))
                    words.append(word_copy)
            
            # 按时间戳排序
            words.sort(key=lambda w: (w.get('start', 0), w.get('end', 0)))
        
        # 分割音频
        print(f"正在分割片段 {segment_index}: {start_time:.2f}s - {end_time:.2f}s")
        if split_audio(audio_path, start_time, end_time, output_path):
            print(f"片段 {segment_name} 生成成功")
            media_info[segment_name] = {
                'start': start_time,
                'end': end_time,
                'text': text
            }
            # 如果有 words 字段，也写入 JSON
            if words:
                media_info[segment_name]['words'] = words
        else:
            print(f"片段 {segment_name} 生成失败")
    
    media_data = {
        'path': audio_path,
        'seg_asr': media_info,
    }

    return media_data


if __name__ == '__main__':
    # rest = audio2subtitle("/Users/jxinfa/PycharmProjects/qwen3/data/spk_1743562887.wav")
    res = audio2subtitle("/data/1.mp4")
    print(res)
