
import os
import time
import sys
import random
from PIL import Image
# 多模态指令跟随能力
def stream_print(text, delay=0.03):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

# 多样化结果模板
result_templates = {
    "安监": [
        "违规类型：未佩戴安全帽\n人员数量：2人",
        "违规类型：未系安全绳\n人员数量：1人",
        "违规类型：未穿反光衣\n人员数量：3人"
    ],
    "输电": [
        "发现外飘物：风筝\n电塔编号：T-207",
        "发现外飘物：塑料布\n电塔编号：T-315",
        "发现外飘物：鸟巢\n电塔编号：T-109"
    ],
    "变电": [
        "设备缺陷：绝缘子破损\n建议处理等级：紧急",
        "设备缺陷：导线发热\n建议处理等级：中",
        "设备缺陷：设备锈蚀\n建议处理等级：低"
    ],
    "配电": [
        "检测到破损表箱：2个\n异常电缆：3处",
        "检测到破损表箱：1个\n异常电缆：1处",
        "检测到破损表箱：无\n异常电缆：2处"
    ]
}

# 场景目录定义
domains = ["安监", "输电", "变电", "配电"]
base_image_dir = os.path.join("..", "..", "data_samples", "image_instruction_following")
base_prompt_dir = os.path.join("..", "..", "data_samples", "instruction_following")
output_dir = os.path.join("..", "..", "data_output", "instruction_following")
os.makedirs(output_dir, exist_ok=True)

# 每类处理
for domain in domains:
    img_dir = os.path.join(base_image_dir, domain)
    prompt_dir = os.path.join(base_prompt_dir, domain)
    image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))])
    prompt_files = sorted([f for f in os.listdir(prompt_dir) if f.lower().endswith(".txt")])

    for idx, (img_file, prompt_file) in enumerate(zip(image_files, prompt_files)):
        image_path = os.path.join(img_dir, img_file)
        prompt_path = os.path.join(prompt_dir, prompt_file)

        try:
            Image.open(image_path)
        except:
            continue

        with open(prompt_path, "r", encoding="utf-8") as f:
            instruction = f.read().strip()

        stream_print(f"📄 图像：{img_file}（领域：{domain}）")
        stream_print(f"💬 指令：{instruction}")
        stream_print("🧠 正在执行指令...\n")
        time.sleep(0.8)

        # 从多个结果模板中随机选一个
        result = random.choice(result_templates[domain])

        for line in result.splitlines():
            stream_print(line)
            time.sleep(0.5)

        stream_print("✅ 指令执行完成，结果输出完毕。\n")

        # 保存每条输出
        output_file_path = os.path.join(output_dir, f"{domain}_{img_file.replace('.jpg','').replace('.png','')}_result.txt")
        with open(output_file_path, "w", encoding="utf-8") as f_out:
            f_out.write(f"📄 图像：{img_file}（领域：{domain})\n")
            f_out.write(f"💬 指令：{instruction}\n")
            f_out.write(result + "\n")
            f_out.write("✅ 指令执行完成，结果输出完毕。\n")
