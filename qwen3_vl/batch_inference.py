import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained("/Users/jxinfa/RustroverProjects/qwen3/Qwen3-VL-2B-Instruct",
                                                    dtype=torch.bfloat16,
                                                    device_map="auto")

processor = AutoProcessor.from_pretrained("/Users/jxinfa/RustroverProjects/qwen3/Qwen3-VL-2B-Instruct")

# for batch generation, padding_side should be set to left!
processor.tokenizer.padding_side = 'left'

system_prompt = '你是一个专业的视频剪辑助手，需要根据用户要求返回内容，返回格式必须是JSON'



messages = [
    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/Users/jxinfa/PycharmProjects/qwen3/data/1.mp4",
            },
            {
                "type": "video",
                "video": "/Users/jxinfa/PycharmProjects/qwen3/data/2.mp4",
            },
            {"type": "text",
             "text": '''把视频生成多个剪辑片段,同时对每个片段进行总结描述,每个动作片段不小于3秒,输出格式为[{"start_time:xx","end_time:xx","des":"描述内容"}]'''},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    padding=True  # padding should be set for batch generation!
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
