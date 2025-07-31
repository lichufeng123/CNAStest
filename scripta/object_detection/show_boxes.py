import requests
import base64
import json
import os
import glob
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()

def draw_bboxes(boxes_info: list,  raw_img: Image.Image):
    """在模型上画框 (PIL版本)"""
    if len(boxes_info) > 0:
        rw, rh = raw_img.size
        line_thickness = max(round(sum([rw, rh]) / 2 * 0.002), 3)
        draw = ImageDraw.Draw(raw_img)
        font = ImageFont.truetype("SimHei.ttf", size=int(line_thickness * 10))

        for i, det in enumerate(boxes_info):
            if len(det) == 5:
                x1, y1, x2, y2, cls = det
                conf = 1
            else:
                x1, y1, x2, y2, conf, cls = det
            color = (255,0,0)
            box = [x1, y1, x2, y2] 
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[2]), int(box[3]))
            
            # 画框
            draw.rectangle([p1, p2], outline=color, width=line_thickness)
            
            # 写文字
            label = f'{cls} {conf:.2f}'
            bbox = font.getbbox(label)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            out_line = p1[1] - text_h - 3 < 0
            
            # 文字背景框
            if not out_line:
                text_box_p1 = (p1[0], p1[1] - text_h - 3)
                text_box_p2 = (p1[0] + text_w, p1[1])
                draw.rectangle([text_box_p1, text_box_p2], fill=color)
            
                # 写文字
                text_pos = (p1[0], p1[1] - text_h - 2)
                draw.text(text_pos, label, fill=(255, 255, 255), font=font)
            else:
                text_box_p1 = p1
                text_box_p2 = (p1[0] + text_w, p1[1] + text_h + 3)
                draw.rectangle([text_box_p1, text_box_p2], fill=color)
                
                # 写文字
                text_pos = (p1[0], p1[1] + 2)
                draw.text(text_pos, label, fill=(255, 255, 255), font=font)

    return raw_img

if __name__ == "__main__":
    info_path = "../data_output/inference_result/complete_directory_test_20250707_034445.json"
    image_folder = "/mnt/nas_data2/chenjn_workspace/datasets/project/online_feedback_test/jiujiang/577_火焰烟雾_cropped"
    save_folder = "outputs"
    os.makedirs(save_folder, exist_ok=True)
    with open(info_path, "r") as f:
        info = json.load(f)["all_results"]
    for item in tqdm(info):
        image_path = os.path.join(image_folder, item["image_name"])
        image = Image.open(image_path)
        if item["final_decision"] == "盖板缺失":
            boxes = [[*box['bbox'],box['conf'],box['class_name']] for box in item["detection_boxes"]]
            draw_bboxes(boxes, image)
            image.save(os.path.join(save_folder, os.path.basename(image_path)))