
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

input_base = os.path.join("..", "..", "data_samples", "semantic_segmentation")
output_base = os.path.join("..", "..", "data_output", "semantic_segmentation")
os.makedirs(output_base, exist_ok=True)

summary_path = os.path.join(output_base, "segmentation_summary_v2.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("图像语义分割能力模拟结果（版本2）\n\n")

# 场景标签
label_sets = {
    "输电": ["输电铁塔", "导线", "异物", "山火"],
    "变电": ["变压器", "电容器", "绝缘子", "渗油区域"],
    "安监": ["作业人员", "安全带", "安全帽", "扶梯"],
    "配电": ["电缆线", "电表箱", "警示标识", "破损开关"]
}

for domain in os.listdir(input_base):
    domain_path = os.path.join(input_base, domain)
    if not os.path.isdir(domain_path):
        continue

    image_files = [f for f in os.listdir(domain_path) if f.lower().endswith((".jpg", ".png"))]
    if not image_files:
        continue

    log(f"处理场景：{domain}")
    time.sleep(1.2)

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(domain_path, img_file)
        image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        stream_print(f"分析图像：{img_file}")
        time.sleep(2)

        output_img_path = os.path.join(output_base, f"seg_{domain}_{img_file}")
        output_txt_path = os.path.join(output_base, f"{domain}_{img_file.rsplit('.', 1)[0]}_result.txt")

        labels = label_sets.get(domain, ["设备A", "设备B", "设备C"])
        segmented_labels = random.sample(labels, k=min(len(labels), 2 + i % len(labels)))

        result_lines = []
        for j, label in enumerate(segmented_labels):
            x1 = 50 + j * 60
            y1 = 50 + j * 40
            x2 = x1 + 120
            y2 = y1 + 80
            draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=3)
            draw.text((x1, y1 - 20), label, fill="green")
            result_lines.append(f"- 分割目标：{label}（区域坐标约：({x1},{y1})~({x2},{y2})）")

        image.save(output_img_path)

        with open(output_txt_path, "w", encoding="utf-8") as ft:
            ft.write(f"图像文件：{img_file}\n")
            ft.write(f"所属场景：{domain}\n")
            ft.write("\n".join(result_lines))
            ft.write("\n建议：根据分割结果，评估场景中目标是否完整、位置是否正确。\n")

        with open(summary_path, "a", encoding="utf-8") as fs:
            fs.write(f"[{domain}] {img_file}\n")
            fs.write("\n".join(result_lines))
            fs.write("\n\n")

        stream_print(f"{os.path.basename(output_img_path)} 图像已生成")
        time.sleep(0.6)

stream_print(f"语义分割模拟已完成，汇总保存在：{summary_path}")
log("模拟全部完成", level="INFO_DONE")
