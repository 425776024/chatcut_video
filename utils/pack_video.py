#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于ffmpeg将project_out.json打包为视频的脚本
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple


def format_time(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS.mmm 格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def get_image_path(image_name: str, instruct_output_dir: str) -> str:
    """根据图片名称获取实际文件路径"""
    image_json_path = os.path.join(instruct_output_dir, 'image', f'{image_name}.json')
    if os.path.exists(image_json_path):
        with open(image_json_path, 'r', encoding='utf-8') as f:
            image_data = json.load(f)
            return image_data.get('path', '')
    
    # 如果JSON不存在，尝试直接查找
    possible_paths = [
        os.path.join(instruct_output_dir, 'image', image_name),
        os.path.join(instruct_output_dir, '..', 'instruct_input', image_name),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return ''


def create_image_video(image_path: str, duration: float, output_path: str, 
                       width: int = 1920, height: int = 1080) -> bool:
    """将图片转换为指定时长的视频片段，并添加静音音频轨道"""
    try:
        cmd = [
            'ffmpeg',
            '-loop', '1',
            '-i', image_path,
            '-f', 'lavfi',
            '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100',  # 添加静音音频
            '-t', str(duration),
            '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-shortest',  # 确保输出时长与最短输入一致
            '-y',
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"创建图片视频失败: {e}")
        print(f"命令: {' '.join(cmd)}")
        if e.stderr:
            print(f"错误输出: {e.stderr}")
        return False
    except FileNotFoundError:
        print("错误: 未找到 ffmpeg，请确保已安装 ffmpeg")
        return False


def get_video_info(video_path: str) -> Tuple[int, int, float]:
    """获取视频的宽度、高度和时长"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,duration',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        if 'streams' in data and len(data['streams']) > 0:
            stream = data['streams'][0]
            width = int(stream.get('width', 1920))
            height = int(stream.get('height', 1080))
            duration = float(stream.get('duration', 0))
            return width, height, duration
        
        return 1920, 1080, 0
    except Exception as e:
        print(f"获取视频信息失败: {e}")
        return 1920, 1080, 0


def pack_video_from_json(json_path: str, output_video_path: str, 
                         instruct_output_dir: str = None) -> bool:
    """
    根据project_out.json打包视频
    
    Args:
        json_path: project_out.json的路径
        output_video_path: 输出视频路径
        instruct_output_dir: instruct_output目录路径，如果为None则从json_path推断
    """
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        project_data = json.load(f)
    
    # 确定instruct_output_dir
    if instruct_output_dir is None:
        instruct_output_dir = os.path.dirname(json_path)
    
    # 解析所有轨道
    video_clips = []  # [(start, end, path)]
    image_clips = []  # [(start, end, image_path)]
    text_clips = []   # [(start, end, text)]
    
    tracks = project_data.get('tracks', [])
    for track in tracks:
        track_type = track.get('type', '')
        clips = track.get('clips', [])
        
        for clip in clips:
            if len(clip) < 3:
                continue
            start = float(clip[0])
            end = float(clip[1])
            content = clip[2]
            
            if track_type == 'video':
                video_path = os.path.join(instruct_output_dir, 'video', content)
                if os.path.exists(video_path):
                    video_clips.append((start, end, video_path))
                else:
                    print(f"警告: 视频文件不存在: {video_path}")
            elif track_type == 'image':
                image_path = get_image_path(content, instruct_output_dir)
                if image_path and os.path.exists(image_path):
                    image_clips.append((start, end, image_path))
                else:
                    print(f"警告: 图片文件不存在: {image_path}")
            elif track_type == 'text':
                text_clips.append((start, end, content))
    
    if not video_clips and not image_clips:
        print("错误: 没有找到任何视频或图片片段")
        return False
    
    # 确定输出视频的分辨率（使用第一个视频片段或默认值）
    target_width, target_height = 1920, 1080
    if video_clips:
        w, h, _ = get_video_info(video_clips[0][2])
        target_width, target_height = w, h
    
    # 计算总时长
    all_clips = video_clips + image_clips
    if not all_clips:
        print("错误: 没有可用的视频或图片片段")
        return False
    
    total_duration = max([end for _, end, _ in all_clips])
    
    print(f"总时长: {total_duration:.2f}秒")
    print(f"目标分辨率: {target_width}x{target_height}")
    print(f"视频片段数: {len(video_clips)}")
    print(f"图片片段数: {len(image_clips)}")
    print(f"文本片段数: {len(text_clips)}")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix='pack_video_')
    print(f"临时目录: {temp_dir}")
    
    try:
        # 步骤1: 准备所有视频片段（包括图片转换的视频）
        # 将所有片段按开始时间排序
        all_segments = []
        for start, end, path in video_clips:
            all_segments.append(('video', start, end, path))
        for start, end, path in image_clips:
            all_segments.append(('image', start, end, path))
        
        # 按开始时间排序
        all_segments.sort(key=lambda x: x[1])
        
        # 处理每个片段，转换为统一格式的视频文件
        processed_segments = []  # [(start, end, video_path)]
        last_end_time = 0.0
        
        for idx, (seg_type, start, end, path) in enumerate(all_segments):
            segment_duration = end - start
            
            # 如果有时间间隔，用黑色填充（带静音音频）
            if start > last_end_time:
                gap_duration = start - last_end_time
                gap_output = os.path.join(temp_dir, f'gap_{last_end_time:.2f}.mp4')
                cmd = [
                    'ffmpeg',
                    '-f', 'lavfi',
                    '-i', f'color=c=black:s={target_width}x{target_height}:d={gap_duration}',
                    '-f', 'lavfi',
                    '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100',
                    '-c:v', 'libx264',
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-preset', 'medium',
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    '-shortest',
                    '-y',
                    gap_output
                ]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                processed_segments.append((last_end_time, start, gap_output))
            
            # 处理当前片段
            if seg_type == 'video':
                # 视频片段：转换为统一格式，保留音频（如果没有音频则添加静音）
                segment_output = os.path.join(temp_dir, f'seg_{idx}_{start:.2f}.mp4')
                # 先检查视频是否有音频流
                probe_cmd = [
                    'ffprobe',
                    '-v', 'error',
                    '-select_streams', 'a:0',
                    '-show_entries', 'stream=codec_type',
                    '-of', 'json',
                    path
                ]
                has_audio = False
                try:
                    probe_result = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
                    probe_data = json.loads(probe_result.stdout)
                    has_audio = 'streams' in probe_data and len(probe_data['streams']) > 0
                except:
                    has_audio = False
                
                if has_audio:
                    # 有音频，保留并重新编码
                    cmd = [
                        'ffmpeg',
                        '-i', path,
                        '-t', str(segment_duration),
                        '-vf', f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2',
                        '-c:v', 'libx264',
                        '-c:a', 'aac',
                        '-b:a', '128k',
                        '-preset', 'medium',
                        '-crf', '23',
                        '-pix_fmt', 'yuv420p',
                        '-y',
                        segment_output
                    ]
                else:
                    # 没有音频，添加静音音频轨道
                    cmd = [
                        'ffmpeg',
                        '-i', path,
                        '-f', 'lavfi',
                        '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100',
                        '-t', str(segment_duration),
                        '-vf', f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2',
                        '-c:v', 'libx264',
                        '-c:a', 'aac',
                        '-b:a', '128k',
                        '-preset', 'medium',
                        '-crf', '23',
                        '-pix_fmt', 'yuv420p',
                        '-shortest',
                        '-y',
                        segment_output
                    ]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                processed_segments.append((start, end, segment_output))
            else:  # image
                # 图片片段：转换为视频
                segment_output = os.path.join(temp_dir, f'img_{idx}_{start:.2f}.mp4')
                if create_image_video(path, segment_duration, segment_output, target_width, target_height):
                    processed_segments.append((start, end, segment_output))
            
            last_end_time = end
        
        # 如果最后还有空白，用最后一个片段或黑色填充（带静音音频）
        if last_end_time < total_duration:
            fill_duration = total_duration - last_end_time
            fill_output = os.path.join(temp_dir, f'fill_end_{last_end_time:.2f}.mp4')
            if processed_segments:
                # 使用最后一个片段的最后一帧
                last_seg_path = processed_segments[-1][2]
                cmd = [
                    'ffmpeg',
                    '-sseof', '-0.1',  # 从最后0.1秒开始
                    '-i', last_seg_path,
                    '-t', str(fill_duration),
                    '-vf', f'scale={target_width}:{target_height}',
                    '-c:v', 'libx264',
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-preset', 'medium',
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    '-y',
                    fill_output
                ]
            else:
                # 创建黑色视频（带静音音频）
                cmd = [
                    'ffmpeg',
                    '-f', 'lavfi',
                    '-i', f'color=c=black:s={target_width}x{target_height}:d={fill_duration}',
                    '-f', 'lavfi',
                    '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100',
                    '-c:v', 'libx264',
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-preset', 'medium',
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    '-shortest',
                    '-y',
                    fill_output
                ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            processed_segments.append((last_end_time, total_duration, fill_output))
        
        # 步骤2: 创建concat文件列表（按时间顺序）
        concat_file = os.path.join(temp_dir, 'concat.txt')
        with open(concat_file, 'w') as f:
            for start, end, path in sorted(processed_segments, key=lambda x: x[0]):
                if os.path.exists(path):
                    f.write(f"file '{os.path.abspath(path)}'\n")
        
        # 步骤3: 拼接所有视频片段（确保音频轨道一致）
        concat_video = os.path.join(temp_dir, 'concat.mp4')
        # 使用重新编码模式确保兼容性，统一音频格式
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-y',
            concat_video
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # 步骤4: 添加文本字幕
        if text_clips:
            # 创建字幕文件（SRT格式）
            srt_file = os.path.join(temp_dir, 'subtitles.srt')
            with open(srt_file, 'w', encoding='utf-8') as f:
                for idx, (start, end, text) in enumerate(text_clips, 1):
                    f.write(f"{idx}\n")
                    # SRT格式时间分隔符是逗号
                    start_time = format_time(start).replace('.', ',')
                    end_time = format_time(end).replace('.', ',')
                    f.write(f"{start_time} --> {end_time}\n")
                    # 转义特殊字符
                    text_escaped = text.replace('\n', ' ').replace('\r', '')
                    f.write(f"{text_escaped}\n\n")
            
            # 使用ffmpeg的subtitles过滤器添加字幕
            # 注意：subtitles过滤器需要libass支持，如果不可用，可以使用drawtext
            srt_abs_path = os.path.abspath(srt_file)
            # 转义路径中的特殊字符
            srt_abs_path_escaped = srt_abs_path.replace('\\', '\\\\').replace(':', '\\:')
            
            cmd = [
                'ffmpeg',
                '-i', concat_video,
                '-vf', f"subtitles='{srt_abs_path_escaped}':force_style='FontSize=8,PrimaryColour=&Hffffff,OutlineColour=&H000000,Outline=2'",
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-preset', 'medium',
                '-crf', '23',
                '-y',
                output_video_path
            ]
        else:
            # 没有字幕，重新编码以确保音频一致
            cmd = [
                'ffmpeg',
                '-i', concat_video,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-preset', 'medium',
                '-crf', '23',
                '-y',
                output_video_path
            ]
        
        print(f"正在生成最终视频: {output_video_path}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"视频生成成功: {output_video_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg处理失败: {e}")
        if e.stderr:
            print(f"错误输出: {e.stderr}")
        return False
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时文件
        import shutil
        try:
            shutil.rmtree(temp_dir)
            print(f"已清理临时目录: {temp_dir}")
        except Exception as e:
            print(f"清理临时目录失败: {e}")


if __name__ == '__main__':
    import sys
    
    # 默认路径
    default_json_path = '/Users/jxinfa/PycharmProjects/qwen3/data/instruct_output/project_out.json'
    default_output_path = '/Users/jxinfa/PycharmProjects/qwen3/data/instruct_output/output.mp4'
    default_instruct_dir = '/Users/jxinfa/PycharmProjects/qwen3/data/instruct_output'
    
    # 从命令行参数获取路径
    json_path = sys.argv[1] if len(sys.argv) > 1 else default_json_path
    output_path = sys.argv[2] if len(sys.argv) > 2 else default_output_path
    instruct_dir = sys.argv[3] if len(sys.argv) > 3 else default_instruct_dir
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"输入JSON: {json_path}")
    print(f"输出视频: {output_path}")
    print(f"数据目录: {instruct_dir}")
    print("-" * 50)
    
    success = pack_video_from_json(json_path, output_path, instruct_dir)
    
    if success:
        print("\n✓ 视频打包完成！")
        sys.exit(0)
    else:
        print("\n✗ 视频打包失败！")
        sys.exit(1)

