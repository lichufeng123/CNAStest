
import os
import time
from datetime import datetime
import random
from PIL import Image, ImageDraw

def log(msg, level="INFO"):
    print(f"[{level}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

def stream_print(text, delay=0.03):
    for c in text:
        print(c, end='', flush=True)
        time.sleep(delay)
    print()

input_img_dir = os.path.join("..", "..", "data_samples", "generation_simulation", "images")
input_txt_dir = os.path.join("..", "..", "data_samples", "generation_simulation", "texts")
output_dir = os.path.join("..", "..", "data_output", "generation_simulation")
os.makedirs(output_dir, exist_ok=True)

summary_path = os.path.join(output_dir, "generation_report_v2.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("图像生成能力模拟报告（版本2）\n\n")

# 模拟指标区间
fid_range = (7.0, 15.0)
clip_range = (0.86, 0.95)

img_files = [f for f in os.listdir(input_img_dir) if f.lower().endswith((".jpg", ".png"))]
txt_files = [f for f in os.listdir(input_txt_dir) if f.endswith(".txt")]

# 图生图：每张图生成 3 张
for i, img_name in enumerate(img_files):
    img_path = os.path.join(input_img_dir, img_name)
    image = Image.open(img_path).convert("RGB")

    stream_print(f"图像生成中（图像输入：{img_name}）")
    time.sleep(2)

    for j in range(3):
        new_img = image.copy()
        draw = ImageDraw.Draw(new_img)
        draw.text((10 + j * 10, 10 + j * 5), f"Sim-{j+1}", fill="blue")
        out_path = os.path.join(output_dir, f"generated_from_{img_name.replace('.', '_')}_v{j+1}.jpg")
        new_img.save(out_path)

        fid = round(random.uniform(*fid_range), 2)
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(f"[图生图] 基于：{img_name} -> {os.path.basename(out_path)}\n")
            f.write(f"- 模拟FID：{fid}\n\n")
        stream_print(f"生成图：{os.path.basename(out_path)}，模拟FID={fid}")
        time.sleep(0.5)

# 文生图：每条文本生成 3 张
for txt_file in txt_files:
    with open(os.path.join(input_txt_dir, txt_file), "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for i, text in enumerate(lines):
        stream_print(f"文本生成图像（指令：{text}）")
        time.sleep(1.5)
        for j in range(3):
            img = Image.new("RGB", (300, 200), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.text((10, 80), f"{text[:10]}-{j+1}", fill="black")
            out_path = os.path.join(output_dir, f"textgen_{i+1}_v{j+1}.jpg")
            img.save(out_path)

            clip_i = round(random.uniform(*clip_range), 3)
            clip_t = round(random.uniform(*clip_range), 3)
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(f"[文生图] 指令：{text} -> {os.path.basename(out_path)}\n")
                f.write(f"- 模拟CLIP-I：{clip_i}，模拟CLIP-T：{clip_t}\n\n")
            stream_print(f"生成图：{os.path.basename(out_path)}，CLIP-I={clip_i}，CLIP-T={clip_t}")
            time.sleep(0.5)

stream_print("图像生成模拟完成，结果汇总已保存。")
log("全部图像生成模拟完成", level="INFO_DONE")
