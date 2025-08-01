
import time
import sys

def stream_print(text, delay=0.03):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()
# 抗干扰能力
import os
import random
import time
from PIL import Image

# 输入输出路径
input_dir = os.path.join("..", "..", "data_samples", "interference_detection")
output_dir = os.path.join("..", "..", "data_output", "interference_detection")
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

interference_labels = [
    ("过曝灯光", "Overexposed Light"),
    ("工厂排烟", "Factory Smoke"),
    ("山间雾气", "Mountain Fog"),
    ("扬尘", "Dust Cloud")
]

# 读取图像文件列表
image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png"))])

# 汇总文本
output_txt_path = os.path.join(output_dir, "interference_detection_result.txt")
with open(output_txt_path, "w", encoding="utf-8") as summary_f:
    summary_f.write("📌 抗干扰能力模拟识别结果（流式模拟）：\n\n")

    for filename in image_files:
        stream_print(f"📄 识别图像：{filename}")
        summary_f.write(f"📄 文件名：{filename}\n")

        time.sleep(1)

        selected_items = random.sample(interference_labels, random.randint(1, 2))
        for zh, _ in selected_items:
            x = random.randint(100, 800)
            y = random.randint(100, 500)
            stream_print(f"- 检测内容：{zh}（位置约：{x},{y}）")
            summary_f.write(f"- 检测内容：{zh}（位置约：{x},{y}）\n")
            time.sleep(1)

        stream_print("✅ 建议：干扰因素已识别，请结合具体影响排查图像真实性\n")
        summary_f.write("✅ 建议：干扰因素已识别，请结合具体影响排查图像真实性\n\n")
        time.sleep(1.5)
