
import os
import json
from PIL import Image, ImageDraw, ImageFont

# ==== 用户自定义路径 ====
JSON_FILE = "../data_samples/small_object_detection/工器具标注/json/all_annotations.json"           # 标注文件路径
IMAGES_DIR = "../data_samples/small_object_detection/工器具标注/images"                    # 原始图片目录
OUTPUT_DIR = "../data_output/small_object_detection_outputs/工器具标注标注结果"         # 输出带框图片目录

def load_font():
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",  # 黑体（推荐）
        "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=18)
            except Exception:
                continue
    return None

def draw_annotations(image_path, annotation_results, save_path, font):
    with Image.open(image_path).convert("RGB") as img:
        draw = ImageDraw.Draw(img)
        width, height = img.size
        for result in annotation_results:
            val = result["value"]
            x = val["x"] / 100 * width
            y = val["y"] / 100 * height
            w = val["width"] / 100 * width
            h = val["height"] / 100 * height
            label = val["rectanglelabels"][0]
            draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
            if font:
                draw.text((x + 2, y + 2), label, fill="red", font=font)
        img.save(save_path)

def process_annotations(json_file, images_dir, output_dir):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    os.makedirs(output_dir, exist_ok=True)
    font = load_font()
    for item in data:
        image_name = item["data"]["image"]
        image_path = os.path.join(images_dir, image_name)
        if os.path.exists(image_path):
            for ann in item["annotations"]:
                results = ann["result"]
                save_path = os.path.join(output_dir, image_name)
                draw_annotations(image_path, results, save_path, font)

if __name__ == "__main__":
    process_annotations(JSON_FILE, IMAGES_DIR, OUTPUT_DIR)
