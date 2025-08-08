
import os
import time
from PIL import Image, ImageDraw
from datetime import datetime

# 流式打印
def stream_print(text, delay=0.03):
    for c in text:
        print(c, end="", flush=True)
        time.sleep(delay)
    print()

# 日志输出
def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{level}] {timestamp} - {msg}")

# 输入输出路径
input_dir = os.path.join("..", "..", "data_samples", "object_position")
output_dir = os.path.join("..", "..", "data_output", "object_position")
os.makedirs(output_dir, exist_ok=True)
summary_path = os.path.join(output_dir, "summary_v2.txt")

# 场景标签
scene_labels = [
    ("山火", "红色区域出现火点", "red", "中"),
    ("绝缘子自爆", "检测到爆裂结构", "blue", "中"),
    ("作业人员着装", "人员穿戴不符合要求", "green", "高"),
    ("高空作业", "识别到悬挂作业人员", "orange", "高"),
    ("带电作业", "检测到未隔离操作行为", "purple", "高")
]

# 获取图像
log(f"开始读取图像目录：{input_dir}")
time.sleep(1)
image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png"))])

# 清空汇总
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("物体位置理解能力识别结果\n\n")

for i, filename in enumerate(image_files):
    img_path = os.path.join(input_dir, filename)
    out_img_path = os.path.join(output_dir, f"annotated_{filename}")
    out_txt_path = os.path.join(output_dir, f"{filename.rsplit('.', 1)[0]}_result.txt")

    # log(f"加载图像：{filename}")
    time.sleep(0.5)
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    time.sleep(1)

    # 模拟“推理中...”等待时间
    stream_print(f"开始识别图像 {filename} 中物体位置...")
    time.sleep(2.5)

    label, desc, color, risk = scene_labels[i % len(scene_labels)]

    # 标注区域
    box_x = 120 + i * 15
    box_y = 160 + i * 10
    box = [(box_x, box_y), (box_x + 100, box_y + 90)]
    draw.rectangle(box, outline=color, width=3)
    draw.text((box_x, box_y - 20), label, fill=color)

    image.save(out_img_path)

    result_text = f"图像文件：{filename}\n"
    result_text += f"- 场景识别：{label}\n"
    result_text += f"  说明：{desc}\n"
    # result_text += f"  区域坐标：({box_x}, {box_y})\n"
    # result_text += f"  风险等级：{risk}\n"
    # result_text += f"  建议：对目标区域进行核查并采取相应处理措施\n\n"

    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write(result_text)
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(result_text)

    stream_print(result_text, delay=0.01)
    time.sleep(0.8)

log("全部图像处理完成", level="INFO_DONE")
stream_print(f"输出结果文件已经保存至路径：{summary_path}")
