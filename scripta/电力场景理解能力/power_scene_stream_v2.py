
import time
import os
import random
from PIL import Image, ImageDraw
from datetime import datetime
from tqdm import tqdm

def stream_print(text, delay=0.03):
    for c in text:
        print(c, end='', flush=True)
        time.sleep(delay)
    print()

def log(text, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{level}] {timestamp} - {text}")

# 场景输出候选
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
    }
]

# 输入输出路径
base_input_dir = os.path.join("..", "..", "data_samples", "scene_classification")
base_output_dir = os.path.join("..", "..", "data_output", "scene_classification_outputs")
os.makedirs(base_output_dir, exist_ok=True)

# 获取所有图像文件（支持多级 images 子目录）
image_files = []
for root, dirs, files in os.walk(base_input_dir):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(os.path.join(root, f))
image_files.sort()

if not image_files:
    log(f"未找到任何图像文件：{base_input_dir}", level="ERROR")
    exit()

progress = tqdm(total=len(image_files), desc="电力场景理解能力模拟", dynamic_ncols=True)

for img_path in image_files:
    filename = os.path.basename(img_path)
    base_name = os.path.splitext(filename)[0]
    output_img_path = os.path.join(base_output_dir, f"{base_name}_annotated_v7.jpg")
    output_txt_path = os.path.join(base_output_dir, f"{base_name}_result_v7.txt")

    log(f"处理图像：{filename}")
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    time.sleep(0.5)

    # 模拟画图标注
    draw.rectangle([(500, 150), (580, 250)], outline="red", width=3)
    draw.text((500, 125), "Foreign Object", fill="red")
    draw.ellipse([(200, 380), (260, 440)], outline="orange", width=3)
    draw.text((200, 350), "Suspected Fire", fill="orange")
    draw.rectangle([(100, 100), (180, 180)], outline="blue", width=3)
    draw.text((100, 80), "Oil Leakage", fill="blue")
    draw.rectangle([(320, 100), (390, 160)], outline="purple", width=3)
    draw.text((320, 80), "Broken Insulator", fill="purple")
    draw.rectangle([(600, 300), (680, 380)], outline="green", width=3)
    draw.text((600, 280), "No Helmet", fill="green")
    draw.rectangle([(400, 300), (480, 380)], outline="brown", width=3)
    draw.text((400, 280), "No Gloves", fill="brown")
    draw.rectangle([(700, 150), (780, 230)], outline="gray", width=3)
    draw.text((700, 130), "No Ladder Support", fill="gray")

    image.save(output_img_path)
    log(f"标注图保存至：{output_img_path}")

    # 选择一个随机场景输出
    selected = random.choice(scene_outputs)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(f"场景类型：{selected['scene']}\n")
        for label, desc in selected["items"]:
            f.write(f"- 检测内容：{label}\n")
            f.write(f"  描述：{desc}\n")
        f.write("\n")

    with open(output_txt_path, "r", encoding="utf-8") as f:
        stream_print(f.read(), delay=0.01)

    stream_print(f"✅ 图像 [{base_name}] 场景识别完成\n" + "=" * 60, delay=0.01)
    progress.update(1)

progress.close()
