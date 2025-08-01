
import os
import random
from PIL import Image, ImageDraw
# 图生图、文生图
# 输入路径
image_input_dir = os.path.join("..", "..", "data_samples", "generation_input", "images")
text_input_dir = os.path.join("..", "..", "data_samples", "generation_input", "texts")
output_dir = os.path.join("../../data_output", "generation_output", "generated_images")
os.makedirs(output_dir, exist_ok=True)

# 图生图
image_files = sorted([f for f in os.listdir(image_input_dir) if f.lower().endswith(".jpg")])
for img_file in image_files:
    img_path = os.path.join(image_input_dir, img_file)
    img = Image.open(img_path).convert("RGB")
    for i in range(3):
        new_img = img.copy()
        draw = ImageDraw.Draw(new_img)
        draw.text((5, 5), f"Aug-{i+1}", fill="white")
        out_path = os.path.join(output_dir, f"{img_file.replace('.jpg','')}_gen_{i+1}.jpg")
        new_img.save(out_path)

# 文生图
text_files = sorted([f for f in os.listdir(text_input_dir) if f.endswith(".txt")])
colors = [(200,150,100), (120,180,150), (180,110,160)]
for idx, txt_file in enumerate(text_files):
    with open(os.path.join(text_input_dir, txt_file), "r", encoding="utf-8") as f:
        text = f.read().strip()
    for j in range(3):
        img = Image.new("RGB", (256,256), colors[(idx + j) % len(colors)])
        draw = ImageDraw.Draw(img)
        draw.text((10,120), f"Gen-{idx+1}-{j+1}", fill="black")
        out_path = os.path.join(output_dir, f"text_{idx+1}_gen_{j+1}.jpg")
        img.save(out_path)

print("✅ 模拟生成图像完成，共输出图生图与文生图各3x数量")
