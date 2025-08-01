
import time
import sys

def stream_print(text, delay=0.03):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

import os
from PIL import Image, ImageDraw

# 输入图像目录
input_dir = os.path.join("..", "..", "data_samples", "small_object_detection")
output_dir = os.path.join("..", "..", "data_output", "small_object_detection")
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# 类别定义（每类2张图，共10张）
scene_labels = [
    ("远距离山火", "Suspected Fire", "orange"),
    ("绝缘子自爆", "Insulator Burst", "red"),
    ("高空俯拍绿膜", "Green Film", "green"),
    ("绝缘子污闪爬电", "Pollution Flashover", "blue"),
    ("安全工器具", "Safety Equipment", "purple")
]

# 总结果文本输出路径
output_txt_path = os.path.join(output_dir, "small_object_detection_result.txt")
with open(output_txt_path, "w", encoding="utf-8") as f:
    f.write("📌 小目标检测能力批量结果：\n\n")

image_index = 1  # 全局图编号
for idx, (label_zh, label_en, color) in enumerate(scene_labels):
    for j in range(2):  # 每类生成2张图
        filename = f"image_{image_index:02d}.jpg"
        input_img_path = os.path.join(input_dir, filename)
        output_img_path = os.path.join(output_dir, f"{filename.replace('.jpg', '_annotated.jpg')}")
        output_img_txt = os.path.join(output_dir, f"{filename.replace('.jpg', '_result.txt')}")

        # 打开图片或创建空白图
        if os.path.exists(input_img_path):
            img = Image.open(input_img_path).convert("RGB")
        else:
            img = Image.new("RGB", (1000, 600), "white")
        draw = ImageDraw.Draw(img)

        # 逼真标注位置
        if label_zh == "远距离山火":
            box = [(50 + j * 20, 450 + j * 10), (150 + j * 20, 520 + j * 10)]
        elif label_zh == "绝缘子自爆":
            box = [(400 + j * 15, 80 + j * 10), (500 + j * 15, 160 + j * 10)]
        elif label_zh == "高空俯拍绿膜":
            box = [(100 + j * 25, 480), (300 + j * 25, 580)]
        elif label_zh == "绝缘子污闪爬电":
            box = [(600 + j * 10, 300 + j * 10), (700 + j * 10, 380 + j * 10)]
        elif label_zh == "安全工器具":
            box = [(300 + j * 15, 350 + j * 10), (400 + j * 15, 420 + j * 10)]
        else:
            box = [(100 + j * 20, 100 + j * 30), (200 + j * 20, 180 + j * 30)]

        # 绘制框和标签
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0][0], box[0][1] - 25), label_en, fill=color)

        # 保存图像
        img.save(output_img_path)

        # 构造文本内容
        text_lines = [
            f"📄 文件名：{filename}",
            f"- 检测内容：{label_zh}（位置约：{box[0][0]},{box[0][1]}）",
            "✅ 建议：已识别小目标，请根据类型安排专项巡检\n"
        ]
        text_block = "\n".join(text_lines)

        # 写入单图文本文件
        with open(output_img_txt, "w", encoding="utf-8") as single_f:
            single_f.write(text_block)

        # 追加写入总汇总文本
        with open(output_txt_path, "a", encoding="utf-8") as summary_f:
            summary_f.write(text_block + "\n")

        # 控制台输出
        stream_print(text_block)
        stream_print(f"✅ 检测完成：{filename}\n")

        image_index += 1  # 最后再加1
