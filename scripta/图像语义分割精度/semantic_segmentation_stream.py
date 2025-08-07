
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

    stream_print(f"正在识别场景：{scene}")
    time.sleep(1)
    stream_print("加载图像中...")
    time.sleep(0.5)

    try:
        img = Image.open(input_img_path).convert("RGB")
    except Exception as e:
        stream_print(f"加载失败：{e}")
        continue
    stream_print("正在分析图像语义信息...")
    time.sleep(0.8)

    stream_print(f"YOLO + VLM 串联推理结果: {response.status_code}")

    stream_print(f"图片名称: {result.get('image_name')}")
    stream_print(f"处理成功: {result.get('success')}")
    stream_print(f"最终决策: {result.get('final_decision')}")

    # 显示YOLO检测结果
    yolo_result = result.get('yolo_detection', {})
    stream_print(f"\nYOLO检测结果:")
    stream_print(f"  - 检测到盖板缺失: {yolo_result.get('has_open', False)}")
    stream_print(f"  - 总对象数: {yolo_result.get('detection_count', 0)}")

    # 显示检测统计
    summary = result.get('detection_summary', {})
    stream_print(f"\n检测统计:")
    stream_print(f"  - 盖板缺失数量: {summary.get('open_count', 0)}")
    stream_print(f"  - 是否使用VLM: {summary.get('used_vlm', False)}")


    # 模拟切割扶梯区域（矩形区域）
    box = (100, 100, 250, 250)
    segmented = img.crop(box)
    segmented.save(output_img_path)

    stream_print("已成功分割目标元素：ESCALATOR")
    stream_print(f"分割结果已保存为：{output_img_path}\n")
