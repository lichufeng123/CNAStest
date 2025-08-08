
import os
import time
from datetime import datetime
from PIL import Image, ImageDraw
import random

def log(msg, level="INFO"):
    print(f"[{level}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

def stream_print(text, delay=0.03):
    for c in text:
        print(c, end='', flush=True)
        time.sleep(delay)
    print()

# 统一路径
input_base_dir = os.path.join("..", "..", "data_samples", "image_instruction_following")
instr_base_dir = os.path.join("..", "..", "data_samples", "instruction_following")
output_base_dir = os.path.join("..", "..", "data_output", "instruction_following")
os.makedirs(output_base_dir, exist_ok=True)

summary_path = os.path.join(output_base_dir, "summary_v3.txt")
with open(summary_path, "w", encoding="utf-8") as fs:
    fs.write("图像指令跟随能力模拟识别汇总（版本3）\n\n")

# 遍历每个指令领域目录（如 安监/输电/变电/配电）
domains = [d for d in os.listdir(instr_base_dir) if os.path.isdir(os.path.join(instr_base_dir, d))]

domain_tasks = {
    "安监": ["识别是否佩戴安全头盔", "判断是否违规使用扶梯", "检测人员是否进入禁区"],
    "输电": ["识别异物外飘", "识别山火隐患", "检测输电线接触距离"],
    "变电": ["检测变压器油渍", "识别设备老化", "检测接地线锈蚀情况"],
    "配电": ["识别线缆破损", "判断电缆敷设合理性", "检测开关箱封闭情况"]
}

for domain in domains:
    log(f"处理领域：{domain}")
    time.sleep(0.8)

    domain_img_dir = os.path.join(input_base_dir, domain)
    domain_instr_dir = os.path.join(instr_base_dir, domain)

    if not os.path.exists(domain_img_dir):
        log(f"图像目录不存在，跳过：{domain_img_dir}", level="WARN")
        continue

    image_files = sorted([f for f in os.listdir(domain_img_dir) if f.lower().endswith((".jpg", ".png"))])
    instr_files = sorted([f for f in os.listdir(domain_instr_dir) if f.lower().endswith(".txt")])

    # 读取所有指令文本
    instructions = []
    for file in instr_files:
        file_path = os.path.join(domain_instr_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            instructions.extend([line.strip() for line in f if line.strip()])

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(domain_img_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        stream_print(f"正在分析图像：{img_file}")
        time.sleep(2.5)

        instruction = instructions[i % len(instructions)] if instructions else "暂无指令"
        result = random.choice(domain_tasks.get(domain, ["任务执行完毕"]))
        output_img_path = os.path.join(output_base_dir, f"annotated_{domain}_{img_file}")
        output_txt_path = os.path.join(output_base_dir, f"{domain}_{img_file.rsplit('.', 1)[0]}_result.txt")

        box_x = 100 + i * 15
        box_y = 100 + i * 20
        draw.rectangle([(box_x, box_y), (box_x + 120, box_y + 80)], outline="blue", width=3)
        draw.text((box_x, box_y - 20), result, fill="blue")
        image.save(output_img_path)

        result_text = f"图像文件：{img_file}\n"
        result_text += f"- 输入指令：{instruction}\n"
        result_text += f"- 识别结果：{result}\n"
        result_text += f"- 标注位置坐标：({box_x},{box_y})\n"
        result_text += "  建议：请根据识别结果进行现场复核与处置\n\n"

        with open(output_txt_path, "w", encoding="utf-8") as ft:
            ft.write(result_text)
        with open(summary_path, "a", encoding="utf-8") as fs:
            fs.write(result_text)

        stream_print(result_text, delay=0.01)
        time.sleep(0.8)

log("图像指令跟随模拟全部完成", level="INFO_DONE")
stream_print(f"结果汇总已保存至：{summary_path}")
