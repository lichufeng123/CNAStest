
import os
import random
import time
from datetime import datetime
from PIL import Image, ImageDraw
from tqdm import tqdm

def log(text, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{level}] {timestamp} - {text}")

def stream_print(text, delay=0.02):
    for c in text:
        print(c, end='', flush=True)
        time.sleep(delay)
    print()

# 模拟的类别与颜色配置
classes = {
    "远距离山火": ("Suspected Fire", "orange"),
    "绝缘子自爆部分": ("Insulator Burst", "purple"),
    "高空俯拍绿膜": ("Green Film", "green"),
    "绝缘子污闪": ("Pollution Flash", "red"),
    "工器具标注": ("Tools", "blue")
}

# 输入输出目录
input_dir = os.path.join("..", "..", "data_samples", "small_object_detection")
output_dir = os.path.join("..", "..", "data_output", "small_object_detection_outputs")
os.makedirs(output_dir, exist_ok=True)

# 递归获取所有 images 子目录下的图片路径
image_files = []
for root, dirs, files in os.walk(input_dir):
    if os.path.basename(root) == "images":
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_files.append(os.path.join(root, file))
image_files.sort()

if not image_files:
    log("未找到任何图像文件", level="ERROR")
    exit()

stream_print(f"共读取图像数量：{len(image_files)}\n")

for idx, img_path in enumerate(tqdm(image_files, desc="小目标检测模拟")):
    filename = os.path.basename(img_path)
    category = os.path.basename(os.path.dirname(os.path.dirname(img_path)))  # 获取上两级作为类别
    label_en, color = classes.get(category, ("Unknown", "gray"))

    # 打开图像并绘制标注
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # 模拟标注区域（不同图略微偏移）
    box_x = 100 + (idx % 3) * 30
    box_y = 100 + (idx % 4) * 25
    box = [(box_x, box_y), (box_x + 100, box_y + 80)]
    draw.rectangle(box, outline=color, width=3)
    draw.text((box_x, box_y - 25), label_en, fill=color)

    # 保存图像与文本
    output_img_path = os.path.join(output_dir, f"annotated_{filename}")
    output_txt_path = os.path.join(output_dir, f"{filename.rsplit('.', 1)[0]}_result.txt")
    image.save(output_img_path)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(f"图像文件名：{filename}\n")
        f.write(f"- 检测内容：{label_en}（位置约：{box_x},{box_y}）\n")
        f.write("✅ 建议：已识别小目标，请根据类型安排专项巡检\n")

    with open(output_txt_path, "r", encoding="utf-8") as f:
        stream_print(f.read(), delay=0.01)

    stream_print(f"✅ 模拟完成：{filename}")
    stream_print("=" * 60)
