
import os
import time
from datetime import datetime
from PIL import Image, ImageDraw

# 日志函数
def log(text, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{level}] {timestamp} - {text}")

# 流式打印
def stream_print(text, delay=0.03):
    for c in text:
        print(c, end='', flush=True)
        time.sleep(delay)
    print()

# 干扰类型模拟信息
interference_info = {
    "过曝灯光": ("yellow", "画面中心区域光线过曝", 0.88, "中"),
    "工厂排烟": ("gray", "远处建筑物上方排出浓烟", 0.91, "高"),
    "山间雾气": ("lightblue", "图像整体被雾气笼罩", 0.86, "中"),
    "扬尘": ("brown", "地面扬起大量灰尘", 0.84, "低")
}

# 输入输出路径
input_dir = os.path.join("..", "..", "data_samples", "interference_detection")
output_dir = os.path.join("..", "..", "data_output", "interference_detection")
os.makedirs(output_dir, exist_ok=True)
summary_txt_path = os.path.join(output_dir, "summary_v2.txt")

# 图像列表
log(f"开始读取图像目录：{input_dir}")
time.sleep(1)
image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png"))])

# 清空汇总文本
with open(summary_txt_path, "w", encoding="utf-8") as f:
    f.write("抗干扰能力识别结果汇总\n\n")

for i, filename in enumerate(image_files):
    img_path = os.path.join(input_dir, filename)
    output_img_path = os.path.join(output_dir, f"annotated_{filename}")
    output_txt_path = os.path.join(output_dir, f"{filename.rsplit('.', 1)[0]}_result.txt")

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # 模拟每张图最多2种干扰类型
    current_types = list(interference_info.keys())
    selected = current_types[i % len(current_types):][:2]

    with open(output_txt_path, "w", encoding="utf-8") as ftxt:
        ftxt.write(f"图像文件：{filename}\n")
        for j, interference in enumerate(selected):
            color, desc, score, risk = interference_info[interference]
            box_x = 100 + j * 120
            box_y = 100 + j * 60
            box = [(box_x, box_y), (box_x + 120, box_y + 80)]
            draw.rectangle(box, outline=color, width=3)
            draw.text((box_x, box_y - 20), interference, fill=color)

            ftxt.write(f"- 干扰类型：{interference}\n")
            ftxt.write(f"  描述：{desc}\n")
            ftxt.write(f"  置信度：{score}\n")
            ftxt.write(f"  风险等级：{risk}\n")
            ftxt.write("  建议：请根据干扰等级判断图像是否可用，必要时重新采集图像\n\n")

            with open(summary_txt_path, "a", encoding="utf-8") as fs:
                fs.write(f"图像文件：{filename}\n")
                fs.write(f"- 干扰类型：{interference}\n")
                fs.write(f"  描述：{desc}\n")
                fs.write(f"  置信度：{score}\n")
                fs.write(f"  风险等级：{risk}\n")
                fs.write("  建议：请根据干扰等级判断图像是否可用，必要时重新采集图像\n\n")

    img.save(output_img_path)

    stream_print(f"图像：{filename}")
    time.sleep(2)
    stream_print(f"检测到干扰：{', '.join(selected)}")
    time.sleep(0.8)

log("干扰检测模拟完成，结果已保存", level="INFO_DONE")
stream_print(f"结果汇总路径：{summary_txt_path}")
