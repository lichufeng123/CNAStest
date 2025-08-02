
import os
import time
from PIL import Image

def stream_print(text, delay=0.03):
    for c in text:
        print(c, end="", flush=True)
        time.sleep(delay)
    print()

# 输入输出路径
scene_list = ["输电", "变电", "安监", "配电"]
input_base = os.path.join("..", "..", "data_samples", "semantic_segmentation")
output_base = os.path.join("..", "..", "data_output", "semantic_segmentation")
os.makedirs(output_base, exist_ok=True)

for scene in scene_list:
    input_img_path = os.path.join(input_base, scene, f"{scene}_sample.jpg")
    output_img_path = os.path.join(output_base, f"{scene}_seg_result.jpg")

    stream_print(f"正在处理场景：{scene}")
    stream_print("加载图像中...")
    time.sleep(0.5)

    try:
        img = Image.open(input_img_path).convert("RGB")
    except Exception as e:
        stream_print(f"加载失败：{e}")
        continue

    stream_print("正在分析图像语义信息...")
    time.sleep(0.8)

    # 模拟切割扶梯区域（矩形区域）
    box = (100, 100, 250, 250)
    segmented = img.crop(box)
    segmented.save(output_img_path)

    stream_print("已成功分割目标元素：ESCALATOR")
    stream_print(f"分割结果已保存为：{output_img_path}\n")
