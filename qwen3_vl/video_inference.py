import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained("/Users/jxinfa/RustroverProjects/qwen3/Qwen3-VL-2B-Instruct",
                                                    dtype=torch.bfloat16,
                                                    device_map="auto")

processor = AutoProcessor.from_pretrained("/Users/jxinfa/RustroverProjects/qwen3/Qwen3-VL-2B-Instruct")

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "video",
#                 "video": "/Users/jxinfa/PycharmProjects/qwen3/data/10月31日.mp4",
#                 "total_pixels": 24576 * 32 * 32
#             },
#             {"type": "text",
#              "text": '''把视频生成多个剪辑片段,同时对每个片段进行总结描述,每个动作片段不小于3秒,输出格式为[{"start_time:xx","end_time:xx","des":"描述内容"}]'''},
#         ],
#     }
# ]

messages = [{"role": "user", "content": [
            {"type": "image", "image": '/Users/jxinfa/PycharmProjects/qwen3/data/instruct_input/cutme.png'},
            {"type": "text",
             "text": '''描述图像，输出格式为[{"role_des:动物、人物、物品的描述","sence":"场景、背景、环境描述","des":"整体描述"}]，不存在的可以为空'''}]}]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
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
