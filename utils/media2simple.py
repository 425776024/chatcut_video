import os
import json
import re
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen3.llm_processor import LLMProcessor

# model_name = "Qwen/Qwen3-4B"
model_name = "/Users/jxinfa/RustroverProjects/qwen3/Qwen3-4B"


def generate_project_from_media(media_out_dir: str,
                                user_request: str,
                                sample_format_path: str = None,
                                output_path: str = None,
                                use_batch_generation: bool = True,
                                batch_size: int = 1024,  # 每批生成的最大输出token数（不是输入批次！用于避免单次生成过长导致数值不稳定）
                                max_total_tokens: int = 32768,
                                use_greedy: bool = False):  # 是否使用greedy解码（更稳定但更确定性）
    media_path = os.path.join(media_out_dir, "media.json")
    # 读取media.json
    with open(media_path, 'r', encoding='utf-8') as f:
        media_data = json.load(f)

    sample_format_path = os.path.abspath(sample_format_path)

    with open(sample_format_path, 'r', encoding='utf-8') as f:
        sample_format = json.load(f)

    # 构建系统提示词，说明返回格式
    format_example = json.dumps(sample_format, ensure_ascii=False, indent=2)
    system_prompt = f'''你是一个专业的视频剪辑助手，需要根据提供的媒体资源信息和用户要求，生成一个完整的视频项目配置。
    
返回格式必须是JSON，格式如下：
{format_example}

说明：
- tracks是轨道数组，可以有多个轨道，通常视频在最底层，然后是图片、文本、音频
- clips是剪辑片段数组
  - type可以是：text|image|video|audio
  - text类型:[起始时间,结束时间,文本]
  - image|audio|video类型:[起始时间,结束时间,名称]'''

    # 构建用户提示词
    user_content = f'''媒体资源信息：
{json.dumps(media_data, ensure_ascii=False, indent=2)}

用户诉求：
{user_request}

请根据上述媒体资源和用户诉求，生成一个完整的视频项目配置JSON，资源必须是媒体资源中存在的。合理利用des(描述)和text(说话内容)和起始结束时间，没有text的视频片段可以参考des描述选用并配上旁白文字也可以不用，有text的视频片段必须同时包含对应文本元素。图片可以搭配合适的描述文本，文本、图片元素时间必须重叠到音视频元素上，音视频元素片段之间不能重叠。'''

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    llm = LLMProcessor('DeepSeek-V3-0324', 0.3)
    # 调用OpenAI API
    response = llm.client.chat.completions.create(
        model=llm.model,
        messages=messages,
        temperature=llm.temperature,
        max_tokens=llm.max_tokens
    )

    # 解析响应
    result = response.choices[0].message.content

    # 尝试提取JSON内容
    try:
        # 去除可能的代码块标记
        if result.startswith("```json"):
            result = result[7:-3].strip()
        elif result.startswith("```"):
            result = result[3:-3].strip()

        segments = json.loads(result)
        print("thinking content:", result)
        print("content:", segments)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)
            print(f"项目配置已保存到: {output_path}")
    except Exception as e:
        print(e)


if __name__ == '__main__':
    media_out_dir = '/Users/jxinfa/PycharmProjects/qwen3/data/instruct_output'
    user_request = "以突出手动剪辑视频很麻烦像个牛马一样，为主题生成剪辑结果，可以添加合适的文本、旁白和关键词。"
    output_path = '/Users/jxinfa/PycharmProjects/qwen3/data/instruct_output/project_out.json'
    sample_format_path = '/Users/jxinfa/PycharmProjects/qwen3/data/project_sample.json'

    project_data = generate_project_from_media(media_out_dir, user_request, sample_format_path, output_path)
