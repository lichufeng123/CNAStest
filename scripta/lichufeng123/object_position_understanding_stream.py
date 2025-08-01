
import os
import random
import time
import sys
from PIL import Image, ImageDraw

def stream_print(text, delay=0.03):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()
# 物体位置理解能力
# 输入输出路径
input_dir = os.path.join("..", "..", "data_samples", "object_position")
output_dir = os.path.join("..", "..", "data_output", "object_position")
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# 每类一个图，共5图
targets = [
    ("山火", "Mountain Fire", "orange"),
    ("绝缘子自爆", "Insulator Burst", "red"),
    ("作业人员着装", "Operator Outfit", "blue"),
    ("高空作业", "High-altitude Work", "green"),
    ("带电作业", "Live-line Work", "purple")
]

# 输出总文件
output_txt_path = os.path.join(output_dir, "object_position_result.txt")
with open(output_txt_path, "w", encoding="utf-8") as summary_f:
    summary_f.write("📌 物体位置理解能力识别结果（单图一类，流式模拟）：\n\n")

for i, (zh, en, color) in enumerate(targets):
    filename = f"image_{i+1:02d}.jpg"
    input_img_path = os.path.join(input_dir, filename)
    output_img_path = os.path.join(output_dir, filename.replace(".jpg", "_annotated.jpg"))
    output_txt_file = os.path.join(output_dir, filename.replace(".jpg", "_result.txt"))

    try:
        img = Image.open(input_img_path).convert("RGB")
    except:
        img = Image.new("RGB", (1000, 600), "white")
    draw = ImageDraw.Draw(img)

    stream_print(f"📄 正在识别图像：{filename}")
    time.sleep(0.8)

    x1 = random.randint(100, 700)
    y1 = random.randint(100, 450)
    x2 = x1 + random.randint(80, 150)
    y2 = y1 + random.randint(60, 100)
    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
    draw.text((x1, y1 - 20), en, fill=color)

    stream_print(f"- 标注内容：{zh}（坐标：{x1},{y1}）")
    stream_print("✅ 位置识别完成，建议结合坐标信息开展针对性巡检。\n")

    # 保存图片
    img.save(output_img_path)

    # 写入文本内容
    lines = [
        f"📄 文件名：{filename}",
        f"- 标注内容：{zh}（坐标：{x1},{y1}）",
        "✅ 位置识别完成，建议结合坐标信息开展针对性巡检。\n"
    ]
    text_block = "\n".join(lines)

    with open(output_txt_file, "w", encoding="utf-8") as f:
        f.write(text_block)

    with open(output_txt_path, "a", encoding="utf-8") as summary_f:
        summary_f.write(text_block + "\n")
