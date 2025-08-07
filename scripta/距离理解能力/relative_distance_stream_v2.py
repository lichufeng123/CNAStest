
import os
import time
from datetime import datetime
from PIL import Image, ImageDraw

# 打印日志
def log(text, level="INFO"):
    print(f"[{level}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {text}")

# 模拟逐字打印
def stream_print(text, delay=0.03):
    for c in text:
        print(c, end='', flush=True)
        time.sleep(delay)
    print()

# 输入输出路径
input_dir = os.path.join("..", "..", "data_samples", "distance_estimation")
output_dir = os.path.join("..", "..", "data_output", "distance_estimation")
os.makedirs(output_dir, exist_ok=True)
summary_path = os.path.join(output_dir, "summary_v2.txt")

# 场景模拟模板
scenes = [
    {
        "name": "人员与电力设备",
        "obj1": "工作人员",
        "obj2": "高压电塔",
        "distance": "约5米~8米",
        "color": "green"
    },
    {
        "name": "大型机械与电力设施",
        "obj1": "施工吊车",
        "obj2": "变压器",
        "distance": "约3米~6米",
        "color": "orange"
    },
    {
        "name": "山火与电力设施",
        "obj1": "明火区域",
        "obj2": "输电线路",
        "distance": "约10米~15米",
        "color": "red"
    }
]

# 获取图像文件
log("开始读取图像目录...")
time.sleep(1)
image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png"))])

# 写入汇总头部
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("电力图像中目标物相对距离模拟识别结果（版本2）\n\n")

for i, filename in enumerate(image_files):
    img_path = os.path.join(input_dir, filename)
    output_img_path = os.path.join(output_dir, f"annotated_{filename}")
    output_txt_path = os.path.join(output_dir, f"{filename.rsplit('.', 1)[0]}_result.txt")

    log(f"加载图像：{filename}")
    time.sleep(0.8)
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    time.sleep(1)

    stream_print(f"分析图像 {filename} 中目标物相对距离...")
    time.sleep(2.5)

    scene = scenes[i % len(scenes)]
    obj1_x, obj1_y = 100 + i * 10, 100 + i * 10
    obj2_x, obj2_y = obj1_x + 120, obj1_y + 80

    draw.rectangle([(obj1_x, obj1_y), (obj1_x + 80, obj1_y + 60)], outline=scene["color"], width=3)
    draw.text((obj1_x, obj1_y - 15), scene["obj1"], fill=scene["color"])

    draw.rectangle([(obj2_x, obj2_y), (obj2_x + 80, obj2_y + 60)], outline=scene["color"], width=3)
    draw.text((obj2_x, obj2_y - 15), scene["obj2"], fill=scene["color"])

    image.save(output_img_path)

    result_text = f"图像文件：{filename}\n"
    result_text += f"- 场景类型：{scene['name']}\n"
    result_text += f"  目标1：{scene['obj1']} 坐标：({obj1_x},{obj1_y})\n"
    result_text += f"  目标2：{scene['obj2']} 坐标：({obj2_x},{obj2_y})\n"
    result_text += f"  预估相对距离：{scene['distance']}\n"
    result_text += f"  建议：若低于5米，请立即排查风险区域\n\n"

    with open(output_txt_path, "w", encoding="utf-8") as ftxt:
        ftxt.write(result_text)
    with open(summary_path, "a", encoding="utf-8") as fsum:
        fsum.write(result_text)

    stream_print(result_text, delay=0.01)
    time.sleep(0.8)

log("目标物相对距离识别模拟完成", level="INFO_DONE")
stream_print(f"结果汇总已保存至：{summary_path}")
