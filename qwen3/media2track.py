import json

from sympy import content
from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "Qwen/Qwen3-4B"
model_name = "/Users/jxinfa/RustroverProjects/qwen3/Qwen3-4B"


def generate_tracks():
    system_prompt = '你是一个专业的视频剪辑助手，需要根据提供的多轨道内容和用户要求返回新的轨道内容。返回格式必须是JSON，和多轨道内容格式一致，tracks是整个多轨道数组，内部可以多个轨道，每个轨道中可以有多个clips元素数组，元素类型是text、image、video、audio，text有文本内容，其它的有本地路径'
    system_content = str(json.load(open('../data/clip.json', 'r')))

    user_prompt = "请根据以下语音识别内容，给视频中插入多个文本元素，显示在最前面轨道"
    content = '''多轨道内容:\n''' + system_content + '''
    字幕内容：
    [{'start': 0.672, 'end': 1.722, 'text': '今天早上开会时'}, {'start': 2.394, 'end': 4.62, 'text': '经理说我们需要加强Time'}, {'start': 4.683, 'end': 6.447, 'text': 'Management时间管理技能'}, {'start': 7.035, 'end': 9.639, 'text': '这样才能更高效的完成任务'}]'''

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt + '\n' + content}
    ]

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)


if __name__ == '__main__':
    media_path = '/Users/jxinfa/PycharmProjects/qwen3/data/instruct_output/media.json'
