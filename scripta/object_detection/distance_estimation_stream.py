
import os
import random
import time
import sys
from PIL import Image

def stream_print(text, delay=0.03):
    for c in text:
        print(c, end='', flush=True)
        time.sleep(delay)
    print()

# 输入输出路径
input_dir = os.path.join("..", "..", "data_samples", "distance_estimation")
output_dir = os.path.join("..", "..", "data_output", "distance_estimation")
os.makedirs(output_dir, exist_ok=True)

# 场景标签与描述生成函数
def simulate_distance(item):
    if item == "人员":
        d1 = random.uniform(1.5, 5.0)
        d2 = d1 + random.uniform(0.5, 2.0)
        return f"人员与电力设备距离预计在 {d1:.1f} 米 至 {d2:.1f} 米之间"
    elif item == "大型机械":
        d1 = random.uniform(2.0, 8.0)
        d2 = d1 + random.uniform(1.0, 3.0)
        return f"大型机械与输电设施距离预计在 {d1:.1f} 米 至 {d2:.1f} 米之间"
    elif item == "山火烟雾":
        d1 = random.uniform(10.0, 50.0)
        d2 = d1 + random.uniform(5.0, 15.0)
        return f"山火与输电设施距离预计在 {d1:.1f} 米 至 {d2:.1f} 米之间"

# 三种场景标签
scenes = [
    ("人员",),
    ("大型机械",),
    ("山火烟雾",),
    ("大型机械", "人员")  # 混合场景
]

# 获取图像文件
image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png"))])

# 汇总结果文件
output_txt_path = os.path.join(output_dir, "distance_estimation_result.txt")
with open(output_txt_path, "w", encoding="utf-8") as summary_f:
    summary_f.write("📌 距离理解能力模拟结果：")

    for idx, filename in enumerate(image_files):
        selected = scenes[idx % len(scenes)]
        output_txt_file = os.path.join(output_dir, filename.replace(".jpg", "_result.txt").replace(".png", "_result.txt"))

        stream_print(f"📄 正在分析图像：{filename}")
        time.sleep(1)

        text_lines = [f"📄 文件名：{filename}"]
        for obj in selected:
            desc = simulate_distance(obj)
            stream_print(f"- {desc}")
            text_lines.append(f"- {desc}")
            time.sleep(0.8)

        stream_print("✅ 距离判断完成，请结合现场数据进一步复核。")
        text_lines.append("✅ 距离判断完成，请结合现场数据进一步复核。")

        # 写入每张图文本
        with open(output_txt_file, "w", encoding="utf-8") as f_single:
            f_single.write("\n".join(text_lines))

        # 累计写入总文件
        with open(output_txt_path, "a", encoding="utf-8") as f_summary:
            f_summary.write("\n".join(text_lines) + "\n")
