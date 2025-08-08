
import time
import os
from PIL import Image, ImageDraw
from datetime import datetime

# 打印日志（带时间戳）
def log(text, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{level}] {timestamp} - {text}")

# 流式打印（逐字）
def stream_print(text, delay=0.03):
    for c in text:
        print(c, end='', flush=True)
        time.sleep(delay)
    print()

# 输入输出路径
input_dir = os.path.join("..", "..", "data_samples", "small_object_detection")
output_dir = os.path.join("..", "..", "data_output", "small_object_detection")
os.makedirs(output_dir, exist_ok=True)

summary_txt_path = os.path.join(output_dir, "summary_v2.txt")

# 标签与模拟内容
label_info = {
    "山火": ("red", "发现明火，远距离烟雾蔓延", 0.91, "高"),
    "绝缘子自爆": ("blue", "绝缘子表面炸裂", 0.89, "中"),
    "高空俯拍绿膜": ("green", "远距离地面覆盖绿膜", 0.93, "低"),
    "绝缘子污闪爬电": ("orange", "表面放电痕迹", 0.87, "中"),
    "安全工器具": ("purple", "工具未按规范放置", 0.85, "中")
}

log("开始处理小目标检测图像目录...")
time.sleep(1)
image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png"))])
image_index = 0

# 清空汇总文件
with open(summary_txt_path, "w", encoding="utf-8") as f:
    f.write("小目标检测汇总结果（版本2）：\n\n")

for filename in image_files:
    image_index += 1
    img_path = os.path.join(input_dir, filename)
    output_img_path = os.path.join(output_dir, f"annotated_{filename}")
    output_txt_path = os.path.join(output_dir, f"{filename.rsplit('.', 1)[0]}_result.txt")

    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # 模拟检测不同标签，每张图检测1个标签
    label = list(label_info.keys())[(image_index - 1) % len(label_info)]
    color, desc, prob, risk = label_info[label]

    # 模拟标注位置（偏移模拟）
    box_x = 100 + image_index * 10
    box_y = 120 + image_index * 5
    box = [(box_x, box_y), (box_x + 100, box_y + 80)]
    draw.rectangle(box, outline=color, width=3)
    draw.text((box_x, box_y - 20), label, fill=color)

    image.save(output_img_path)

    # 写入单文件结果
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(f"图像文件：{filename}\n")
        f.write(f"- 检测内容：{label}\n")
        f.write(f"  描述：{desc}\n")
        f.write(f"  置信度：{prob}\n")
        f.write(f"  风险等级：{risk}\n")
        f.write("  建议：请安排人员进行针对性巡查\n")

    # 写入汇总结果
    with open(summary_txt_path, "a", encoding="utf-8") as f:
        f.write(f"图像文件：{filename}\n")
        f.write(f"- 检测内容：{label}\n")
        f.write(f"  描述：{desc}\n")
        f.write(f"  置信度：{prob}\n")
        f.write(f"  风险等级：{risk}\n")
        f.write("  建议：请安排人员进行针对性巡查\n\n")

    # 打印当前处理结果
    stream_print(f"图像：{filename}")
    stream_print(f"检测目标：{label} | 置信度：{prob} | 风险：{risk}")
    time.sleep(0.8)

log("全部图像处理完成，检测结果已保存", level="INFO_DONE")
stream_print(f"结果汇总路径：{summary_txt_path}")
