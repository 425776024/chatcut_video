import os
import json
import random
import re
import shutil
import gc
from pathlib import Path

import cv2

from utils.video_seg import video_seg, calculate_scale_resolution
from utils.audio2subtitle import audio_seg
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

random.seed(42)
torch.manual_seed(42)

# 定义文件类型
video_extensions = {'.mp4', '.mov', '.MOV', '.MP4'}
audio_extensions = {'.wav', '.mp3', '.WAV', '.MP3', '.m4a'}
image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP', '.WEBP'}


def instruct_infer(instruct_output_dir):
    model = AutoModelForImageTextToText.from_pretrained("/Users/jxinfa/RustroverProjects/qwen3/Qwen3-VL-4B-Instruct",
                                                        dtype=torch.bfloat16,
                                                        device_map="auto")

    processor = AutoProcessor.from_pretrained("/Users/jxinfa/RustroverProjects/qwen3/Qwen3-VL-4B-Instruct")

    def get_msg_out(item):
        inputs = None
        generated_ids = None
        try:
            inputs = processor.apply_chat_template(
                item,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            # 存储结果：key为文件路径，value为输出文本（如果是列表则取第一个元素，否则直接使用）
            res = output_text[0] if isinstance(output_text, list) and len(output_text) > 0 else output_text

            # 打印原始输出以便调试
            print(f"原始输出: {res}")

            # 尝试提取JSON内容
            try:
                # 去除可能的代码块标记
                if res.strip().startswith("```json"):
                    res = res[7:].strip()
                    if res.endswith("```"):
                        res = res[:-3].strip()
                elif res.strip().startswith("```"):
                    res = res[3:].strip()
                    if res.endswith("```"):
                        res = res[:-3].strip()

                # 尝试找到JSON数组的开始和结束位置
                # 查找第一个 [ 和最后一个 ]
                match = re.search(r'\[.*\]', res, re.DOTALL)
                if match:
                    res = match.group(0)

                json_out = json.loads(res)
                if len(json_out) > 0:
                    json_out = json_out[0]
                return json_out
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                print(f"尝试解析的内容: {res}")
                print(f"内容长度: {len(res)}, 前100个字符: {res[:100]}")
                # 返回一个默认的空结构
                return {"des": "", "role_des": "", "sence": ""}
        finally:
            # 清理张量以释放内存
            try:
                if inputs is not None:
                    del inputs
                if generated_ids is not None:
                    del generated_ids
            except Exception:
                pass
            # 清理GPU缓存（最重要）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 强制垃圾回收
            gc.collect()

    output_image_dir = os.path.join(instruct_output_dir, 'image')
    output_video_dir = os.path.join(instruct_output_dir, 'video')
    output_audio_dir = os.path.join(instruct_output_dir, 'audio')

    full_info = {
        'image': {},
        'audio': {},
        'video': {},
    }

    # 音频asr加载
    for audio_name in sorted(os.listdir(output_audio_dir)):
        audio_json = os.path.join(output_audio_dir, audio_name)
        if not os.path.isdir(audio_json):
            with open(audio_json, 'r') as f:
                audio_asr_content = json.load(f)
            audio_path = audio_asr_content['path']
            seg_asr = audio_asr_content['seg_asr']
            base_name = os.path.basename(audio_path)
            file_name = base_name.split('.')[0]
            audio_dir = os.path.join(output_audio_dir, file_name)
            seg_list = sorted(os.listdir(audio_dir))
            full_info['audio'] = {}
            for audio_seg_name in seg_list:
                if not audio_seg_name.startswith('.'):
                    start = round(seg_asr[audio_seg_name]['start'], 1)
                    end = round(seg_asr[audio_seg_name]['end'], 1)
                    text = str(seg_asr[audio_seg_name]['text'])
                    if len(text) > 0:
                        full_info['audio'][audio_name + '/' + audio_seg_name] = {'start': start, 'end': end,
                                                                                 'text': text}
                        # full_info['audio'][file_name + '/' + audio_seg_name] = {'text': text}
                    # if 'words' in seg_asr[audio_seg_name]:
                    #     for word in seg_asr[audio_seg_name]['words']:
                    #         start = word['start']
                    #         end = word['end']
                    #         text = word['word']
                    #         full_info['audio'][base_name].append({'start': start, 'end': end, 'text': text})
                    # else:
                    #     full_info['audio'][base_name].append({'start': start, 'end': end, 'text': text})

    for img_name in sorted(os.listdir(output_image_dir)):
        img_json_path = os.path.join(output_image_dir, img_name)
        img_src = None
        img_resized = None
        try:
            with open(img_json_path, 'r') as f:
                image_content = json.load(f)
            img_path = image_content['path']
            img_src = cv2.imread(img_path)
            if img_src is None:
                print(f"警告: 无法读取图像 {img_path}")
                continue
            base_name = os.path.basename(img_path)
            height, width = img_src.shape[:2]
            if max(height, width) >= 1080:
                target_width, target_height = calculate_scale_resolution(width, height, max_dimension=1080)
            elif max(height, width) >= 720:
                target_width, target_height = calculate_scale_resolution(width, height, max_dimension=720)
            else:
                target_width, target_height = calculate_scale_resolution(width, height, max_dimension=480)

            # RGB
            img_resized = cv2.resize(img_src, (target_width, target_height))
            cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB, img_resized)

            img_item = [{"role": "user", "content": [
                {"type": "image", "image": img_resized},
                {"type": "text",
                 "text": '''描述图像，输出格式为[{"role_des:动物、人物、物品的描述","sence":"场景、背景、环境描述","des":"整体描述"}]，不存在的可以为空'''}]}]
            json_out = get_msg_out(img_item)

            print(img_path, json_out)
            full_info['image'][base_name] = {
                "des": json_out['des'],
            }
        finally:
            # 释放图像内存
            del img_src
            del img_resized
            # 定期清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    for video_name in sorted(os.listdir(output_video_dir)):
        video_json = os.path.join(output_video_dir, video_name)
        # json文件
        if video_json.endswith('.json'):
            with open(video_json, 'r') as f:
                video_asr_content = json.load(f)
            video_path = video_asr_content['path']
            seg_asr = video_asr_content['seg_asr']
            base_name = os.path.basename(video_path)
            file_name = base_name.split('.')[0]
            video_dir = os.path.join(output_video_dir, file_name)

            height, width = video_asr_content['height'], video_asr_content['width']
            if max(height, width) >= 1080:
                target_width, target_height = calculate_scale_resolution(width, height, max_dimension=1080)
            elif max(height, width) >= 720:
                target_width, target_height = calculate_scale_resolution(width, height, max_dimension=720)
            else:
                target_width, target_height = calculate_scale_resolution(width, height, max_dimension=480)

            seg_list = sorted(os.listdir(video_dir))
            for video_seg_name in seg_list:
                # 片段的asr
                if video_seg_name.endswith('.mp4') or video_seg_name.endswith('.mpv'):
                    video_seg_path = os.path.join(video_dir, video_seg_name)

                    video_item = [{"role": "user", "content": [
                        {"type": "video", "video": video_seg_path,
                         "fps": 1.0,
                         "max_frames": 4,
                         "max_pixels": target_height * target_width,
                         "resized_height": target_height, "resized_width": target_width, },
                        {"type": "text",
                         "text": '''描述视频，输出格式为[{""des":"整体总结描述"}]，不存在的可以为空'''}]}]
                    json_out = get_msg_out(video_item)
                    # 没有text就是空words
                    # if seg_asr[video_seg_name]['text'] == '':
                    #     seg_asr[video_seg_name]['words'] = []
                    # if 'words' in seg_asr[video_seg_name]:
                    #     del seg_asr[video_seg_name]['words']

                    # del seg_asr[video_seg_name]['text']

                    start = round(seg_asr[video_seg_name]['start'], 1)
                    end = round(seg_asr[video_seg_name]['end'], 1)
                    text = seg_asr[video_seg_name]['text']
                    des = json_out['des']
                    full_info['video'][file_name + '/' + video_seg_name] = {'start': start, 'end': end, 'text': text,
                                                                            'des': des}
                    # full_info['video'][file_name + '/' + video_seg_name] = {'text': text, 'des': des}
                    print(video_seg_path, json_out)
                    
                    # 每处理一个视频片段后清理内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

    # 将结果保存到 JSON 文件
    media_json_path = os.path.join(instruct_output_dir, 'media.json')
    with open(media_json_path, 'w', encoding='utf-8') as f:
        json.dump(full_info, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到: {media_json_path}")


def process_data(in_dir, out_dir):
    # 遍历in_dir中文件:
    #   1.如果是视频abc.mp4/abc.mov：调用video_seg函数，输出到out_dir/video/abc目录
    #   2.如果是音频abc.wav/abc.mp3：调用audio2subtitle函数，输出识别结果json到out_dir/abc目录/audio.json，同时拷贝音频进目录
    #   3.如果是图片，则直接拷贝进out_dir目录

    # 确保输出目录存在
    os.makedirs(out_dir, exist_ok=True)

    # 遍历输入目录中的所有文件
    if not os.path.exists(in_dir):
        print(f"错误: 输入目录不存在: {in_dir}")
        return

    output_image_dir = os.path.join(out_dir, 'image')
    output_video_dir = os.path.join(out_dir, 'video')
    output_audio_dir = os.path.join(out_dir, 'audio')
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_audio_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)

    dir_list = os.listdir(in_dir)
    for filename in dir_list:
        file_path = os.path.join(in_dir, filename)

        # 获取文件扩展名和基础名称（不含扩展名）
        file_ext = Path(filename).suffix
        base_name = Path(filename).stem

        print(f"处理文件: {filename}")

        # 处理视频文件
        if file_ext in video_extensions:
            output_video_file_dir = os.path.join(output_video_dir, base_name)
            os.makedirs(output_video_file_dir, exist_ok=True)
            print(f"调用 video 处理视频: {file_path} -> {output_video_file_dir}")
            try:
                media_data = video_seg(file_path, output_video_file_dir)
                media_json_path = os.path.join(output_video_dir, f'{filename}.json')

                with open(media_json_path, 'w', encoding='utf-8') as f:
                    json.dump(media_data, f, ensure_ascii=False, indent=2)
                print(f"json 已保存到: {media_json_path}")

            except Exception as e:
                print(f"处理视频时出错: {e}")

        # 处理音频文件
        elif file_ext in audio_extensions:
            output_audio_file_dir = os.path.join(output_audio_dir, base_name)
            os.makedirs(output_audio_file_dir, exist_ok=True)
            print(f"调用 audio2subtitle 处理音频: {file_path}")
            try:
                # 调用音频转字幕函数
                media_data = audio_seg(file_path, output_audio_file_dir)
                audio_json_path = os.path.join(output_audio_dir, f'{filename}.json')
                with open(audio_json_path, 'w', encoding='utf-8') as f:
                    json.dump(media_data, f, ensure_ascii=False, indent=2)
                print(f"音频识别结果已保存: {audio_json_path}")
            except Exception as e:
                print(f"处理音频时出错: {e}")

        # 处理图片文件
        elif file_ext in image_extensions:
            try:
                media_info = {}
                media_info['path'] = file_path
                audio_json_path = os.path.join(output_image_dir, f'{filename}.json')
                with open(audio_json_path, 'w', encoding='utf-8') as f:
                    json.dump(media_info, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"拷贝图片时出错: {e}")

        else:
            print(f"跳过不支持的文件类型: {filename} (扩展名: {file_ext})")

    print(f"所有文件处理完成！")


if __name__ == '__main__':
    in_dir = '/Users/jxinfa/PycharmProjects/qwen3/data/instruct_input2'
    out_dir = '/Users/jxinfa/PycharmProjects/qwen3/data/instruct_output2'
    process_data(in_dir, out_dir)
    instruct_infer(out_dir)
