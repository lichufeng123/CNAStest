import time
import os
import sys
from PIL import Image, ImageDraw
from datetime import datetime
from utils.progress_helper import ProgressHelper
from utils.thinking_simulator import ThinkingSimulator
# 模拟逐字打印
def stream_print(text, delay=0.03):
    for c in text:
        print(c, end='', flush=True)
        time.sleep(delay)
    print()

# 打印日志（带时间戳）
def log(text, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{level}] {timestamp} - {text}")

base_input_url = '../../data_samples/'
base_output_url = '../../data_output/'
# 设置输入输出路径
img_dir = base_input_url + "scene_classification"
output_dir = base_output_url + "scene_classification_outputs"
os.makedirs(output_dir, exist_ok=True)

# ✅ 递归读取所有子目录下的图片
img_files = []
for root, _, files in os.walk(img_dir):
    for f in files:
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_files.append(os.path.join(root, f))

if not img_files:
    log(f"未在目录 {img_dir} 中发现任何图片文件，请检查路径和扩展名", level="ERROR")


for img_path in img_files:
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    output_img_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
    output_txt_path = os.path.join(output_dir, f"{base_name}_output.txt")

    # 加载图像
    log(f"开始读取图像：{img_path}")
    time.sleep(1)
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    log("图像加载完成", level="INFO_DONE")

    # 模型调用 & 场景识别（保持你原来的模拟流程）
    log("开始识别场景内容")

    draw.rectangle([(500, 150), (580, 250)], outline="red", width=3)
    draw.text((500, 125), "Foreign Object", fill="red")
    draw.ellipse([(200, 380), (260, 440)], outline="orange", width=3)
    draw.text((200, 350), "Suspected Fire", fill="orange")
    time.sleep(1.2)

    draw.rectangle([(100, 100), (180, 180)], outline="blue", width=3)
    draw.text((100, 80), "Oil Leakage", fill="blue")
    draw.rectangle([(320, 100), (390, 160)], outline="purple", width=3)
    draw.text((320, 80), "Broken Insulator", fill="purple")
    time.sleep(1.2)

    draw.rectangle([(600, 300), (680, 380)], outline="green", width=3)
    draw.text((600, 280), "No Helmet", fill="green")
    draw.rectangle([(400, 300), (480, 380)], outline="brown", width=3)
    draw.text((400, 280), "No Gloves", fill="brown")
    draw.rectangle([(700, 150), (780, 230)], outline="gray", width=3)
    draw.text((700, 130), "No Ladder Support", fill="gray")
    time.sleep(1)

    # 保存图像
    image.save(output_img_path)
    log(f"图像标注结果已保存至 {output_img_path}", level="INFO_DONE")

    # 模拟输出文本
    scene_outputs = [
        {
            "scene": "输电通道类",
            "items": [
                ("异物外飘", "导线附近有悬挂物体", 0.93, "中", "建议派员清除异物"),
                ("疑似山火", "图像左下角有火光迹象", 0.88, "高", "立即通知消防并调度巡检")
            ]
        },
        {
            "scene": "变电站设备缺陷检测类",
            "items": [
                ("变压器渗油", "密封件老化引起渗油", 0.91, "中", "安排技术人员检修"),
                ("绝缘子破损", "检测到裂纹", 0.86, "中", "建议更换设备")
            ]
        },
        {
            "scene": "作业违章识别类",
            "items": [
                ("高空作业未戴头盔", "人员未佩戴防护装备", 0.94, "高", "立即叫停并整改"),
                ("带电作业无绝缘手套", "存在触电风险", 0.89, "高", "加强现场监管"),
                ("攀爬扶梯无人协助", "有跌落风险", 0.87, "中", "补充现场监护人员")
            ]
        }
    ]

    with open(output_txt_path, "w", encoding="utf-8") as f:
        for block in scene_outputs:
            f.write(f"场景类型：{block['scene']}\n")
            for label, desc, prob, level, suggest in block["items"]:
                f.write(f"- 检测内容：{label}\n")
                f.write(f"  描述：{desc}\n")
            f.write("\n")

    # 终端输出内容
    with open(output_txt_path, "r", encoding="utf-8") as f:
        content = f.read()
        stream_print("\n文本输出如下：\n")

    stream_print(content)
    stream_print(f"✅ 已完成图像 [{base_name}] 的识别与输出\n" + "=" * 60)
