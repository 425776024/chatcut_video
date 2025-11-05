import cv2
import os
import subprocess
import json

from utils.audio2subtitle import audio2subtitle


def get_video_info(video_path):
    """获取视频信息（时长、宽度、高度、FPS）"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()
    return {
        'duration': duration,
        'width': width,
        'height': height,
        'fps': fps
    }


def get_video_duration(video_path):
    """获取视频总时长（秒）"""
    info = get_video_info(video_path)
    return info['duration']


def calculate_scale_resolution(width, height, max_dimension=480):
    """计算等比例缩放后的分辨率，最大边不超过max_dimension
    
    Args:
        width: 原始宽度
        height: 原始高度
        max_dimension: 最大尺寸（默认480）
    
    Returns:
        (new_width, new_height): 缩放后的宽度和高度（都是偶数）
    """
    max_side = max(width, height)

    # 如果最大边已经小于等于限制，不需要缩放，但需要确保是偶数
    if max_side <= max_dimension:
        # 确保是偶数（H.264编码要求），向下取整为偶数
        new_width = width - (width % 2)
        new_height = height - (height % 2)
        return new_width, new_height

    # 计算缩放比例
    scale = max_dimension / max_side

    # 计算新尺寸
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 确保是偶数（H.264编码要求），向下取整为偶数
    new_width = new_width - (new_width % 2)
    new_height = new_height - (new_height % 2)

    return new_width, new_height


def format_time(seconds):
    """将秒数格式化为 HH:MM:SS.mmm 格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def split_video(video_path, start_time, end_time, output_path, target_width=None, target_height=None,
                use_reencode=False):
    """使用 ffmpeg 分割视频片段
    
    Args:
        video_path: 输入视频路径
        start_time: 开始时间（秒）
        end_time: 结束时间（秒）
        output_path: 输出视频路径
        target_width: 目标宽度（如果为None则不缩放）
        target_height: 目标高度（如果为None则不缩放）
        use_reencode: 是否使用重新编码（True=避免黑屏但慢，False=速度快但可能有黑屏）
    """
    start_str = format_time(start_time)
    duration = end_time - start_time

    if use_reencode:
        # 使用重新编码模式：完全避免黑屏，优化质量
        cmd = [
            'ffmpeg',
            '-ss', start_str,  # -ss 在 -i 之前，seek到关键帧
            '-i', video_path,
            '-t', str(duration),
            '-c:v', 'libx264',  # 重新编码视频
            '-preset', 'slow',  # 使用slow预设以获得更好的质量（不考虑速度）
            '-crf', '23',  # 视频质量（23是较好的默认值）
        ]

        # 如果需要缩放，添加scale过滤器
        if target_width and target_height:
            cmd.extend([
                '-vf', f'scale={target_width}:{target_height}',
            ])

        cmd.extend([
            '-c:a', 'aac',  # 重新编码音频
            '-b:a', '128k',  # 音频比特率
            '-avoid_negative_ts', 'make_zero',
            '-y',  # 覆盖输出文件
            output_path
        ])
    else:
        # 使用流复制模式：速度快，将 -ss 放在 -i 之前可以避免大部分黑屏
        cmd = [
            'ffmpeg',
            '-ss', start_str,  # -ss 在 -i 之前，seek到最近关键帧避免黑屏
            '-i', video_path,
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
        print(f"分割视频失败: {e}")
        print(f"命令: {' '.join(cmd)}")
        return False
    except FileNotFoundError:
        print("错误: 未找到 ffmpeg，请确保已安装 ffmpeg")
        return False


def video_seg(video_path, output_dir):
    res = audio2subtitle(video_path)
    # 基于返回的start'、 'end'切分视频片段，没有被标记的片段也需要分割，
    # 比如：视频共10s，[{'start': 1.0, 'end': 3.5},{'start': 5.5, 'end': 10}],则需要分割为4段：0-1.0\1.0-3.5\3.5-5.5\5.5-10，片段以序号命名
    # 输出到output_dir目录，且包含一个media.json文件，以：片段名:{'start': xxx, 'end': xxx, 'text':''}存储，没有标记的片段text为空

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取视频信息（时长、分辨率等）
    video_info = get_video_info(video_path)
    total_duration = video_info['duration']
    original_width = video_info['width']
    original_height = video_info['height']

    print(f"视频总时长: {total_duration:.2f}秒")
    print(f"原始分辨率: {original_width}x{original_height}")

    # 计算缩放后的分辨率（最大边不超过480p）
    target_width, target_height = calculate_scale_resolution(original_width, original_height, max_dimension=1080)

    if target_width != original_width or target_height != original_height:
        print(f"缩放分辨率: {target_width}x{target_height}")
    else:
        print(f"分辨率已满足要求，无需缩放")

    # 生成所有分割点
    split_points = [0.0]  # 从0开始

    # 检查视频是否有声音（res为空表示没有声音）
    has_audio = res and len(res) > 0
    if not has_audio:
        print("视频没有声音，按3秒一段进行切分")
        # 按3秒一段生成分割点
        current_time = 0.0
        while current_time < total_duration:
            split_points.append(current_time)
            current_time += 3.0
        # 确保最后一个分割点是视频结束时间
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

        # 添加视频结束点
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

    # 分割视频并生成 asr.json 数据
    media_info = {}
    segment_index = 0

    for i in range(len(split_points) - 1):
        start_time = split_points[i]
        end_time = split_points[i + 1]
        segment_duration = end_time - start_time

        # 跳过长度为0的片段
        if segment_duration <= 0:
            continue

        # 对于无声音视频，如果是最后一段（接近视频结尾），即使小于1.5秒也保留
        is_near_end = abs(end_time - total_duration) < 0.1  # 容差0.1秒

        # 跳过时长小于1.5秒的片段（无声音视频的最后一段除外）
        if segment_duration < 1.5:
            print(f"跳过短片段: {start_time:.2f}s - {end_time:.2f}s (时长: {segment_duration:.2f}s < 1.5s)")
            continue

        segment_index += 1
        segment_name = f"{segment_index:04d}.mp4"
        output_path = os.path.join(output_dir, segment_name)

        # 查找与当前视频片段最匹配的 segment（正常情况下只有一个）
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

            # 过滤并裁剪 words 到当前视频片段时间范围内
            for word in seg_words:
                word_start = word.get('start', 0)
                word_end = word.get('end', 0)

                # 只保留与视频片段有重叠的 words
                if not (word_end < start_time or word_start > end_time):
                    word_copy = word.copy()
                    # 裁剪时间戳到视频片段范围内
                    word_copy['start'] = max(start_time, min(word_start, end_time))
                    word_copy['end'] = max(start_time, min(word_end, end_time))
                    words.append(word_copy)

            # 按时间戳排序
            words.sort(key=lambda w: (w.get('start', 0), w.get('end', 0)))

        # 分割视频（使用重新编码避免黑屏，并应用分辨率缩放）
        print(f"正在分割片段 {segment_index}: {start_time:.2f}s - {end_time:.2f}s")
        # 使用重新编码模式，完全避免黑屏问题，并应用分辨率缩放
        if split_video(video_path, start_time, end_time, output_path,
                       target_width=target_width, target_height=target_height,
                       use_reencode=True):
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
        'width': original_width,
        'height': original_height,
        'path': video_path,
        'seg_asr': media_info,
    }
    return media_data


if __name__ == '__main__':
    output_dir = '/data/instruct_output/video/1'
    video_path = '/Users/jxinfa/PycharmProjects/qwen3/data/instruct_input/1.MOV'

    # output_dir = '/data/to_instruct/video_seg/2'
    # video_path = '/data/to_instruct/2.MOV'
    res = video_seg(video_path, output_dir)
    print(res)
