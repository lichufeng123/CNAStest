
import time
import os
import sys
import random
from PIL import Image, ImageDraw
from datetime import datetime


# 基础路径
base_input_url = '../../data_samples/'
base_output_url = '../../data_output/'

# 输入输出目录
img_dir = base_input_url + "scene_classification"
output_dir = base_output_url + "scene_classification_outputs"
os.makedirs(output_dir, exist_ok=True)

# 模拟逐字打印，并将输出信息也写入 info 文件
def stream_print(text, delay=0.03, info_path=os.path.join(output_dir, "info_v4_1.txt")):
    with open(info_path, "a", encoding="utf-8") as info_file:
        info_file.write(text + "\n")
    for c in text:
        print(c, end='', flush=True)
        time.sleep(delay)
    print()

# 打印日志（带时间戳）
def log(text, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{level}] {timestamp} - {text}")



# 获取所有图片
img_files = []
for root, _, files in os.walk(img_dir):
    for f in files:
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_files.append(os.path.join(root, f))

if not img_files:
    log(f"未在目录 {img_dir} 中发现任何图片文件，请检查路径和扩展名", level="ERROR")

# ✅ 读取图像目录一次性输出
log(f"开始读取图像目录：{img_dir}")

for img_path in img_files:
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    output_img_path = os.path.join(output_dir, f"{base_name}_annotated_v4_1.jpg")
    output_txt_path = os.path.join(output_dir, f"{base_name}_output_v4_1.txt")

    # 打开图像（但不输出“图像加载完成”）
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # 模拟绘制标注
    draw.rectangle([(500, 150), (580, 250)], outline="red", width=3)
    draw.text((500, 125), "Foreign Object", fill="red")
    image.save(output_img_path)

    # ✅ 场景输出内容（单项输出，无置信度等）
    scene_outputs = [
        {
            "scene": "输电通道类",
            "items": [
                ("异物外飘", "导线附近有悬挂物体"),
                ("疑似山火", "图像左下角有火光迹象")
            ]
        },
        {
            "scene": "变电站设备缺陷检测类",
            "items": [
                ("变压器渗油", "密封件老化引起渗油"),
                ("绝缘子破损", "检测到裂纹")
            ]
        },
        {
            "scene": "作业违章识别类",
            "items": [
                ("高空作业未戴头盔", "人员未佩戴防护装备"),
                ("带电作业无绝缘手套", "存在触电风险"),
                ("攀爬扶梯无人协助", "有跌落风险")
            ]
        },
        {
            "scene": "识别失败",
            "error": random.choice([
                "图像文件损坏：image file is truncated",
                "模型推理超时：vLLM request timeout",
                "图像格式不受支持：unsupported image mode",
                "模型内部错误：CUDA out of memory"
            ])
        }
    ]

    selected = random.choice(scene_outputs)

    if "error" in selected:
        result_text = f"图像文件：{base_name}.jpg\n"
        result_text += f"识别失败：{selected['error']}\n"
    else:
        label, desc = random.choice(selected["items"])
        result_text = f"图像文件：{base_name}.jpg\n"
        result_text += f"场景类型：{selected['scene']}\n"
        result_text += f"- 检测内容：{label}\n"
        result_text += f"  描述：{desc}\n"

    # ✅ 输出到控制台
    stream_print(result_text)

    # ✅ 保存到文件
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(result_text)

# ✅ 输出图片目录路径作为末尾说明
stream_print(f"所有识别图像输出至：{output_dir}")
