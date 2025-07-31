import requests
import base64
import json
import os
import glob
from PIL import Image, ImageDraw, ImageFont

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

def draw_bboxes(boxes_info: list, class_names: list, raw_img: Image.Image):
    """在模型上画框 (PIL版本)"""
    if len(boxes_info) > 0:
        rw, rh = raw_img.size
        line_thickness = max(round(sum([rw, rh]) / 2 * 0.002), 3)
        draw = ImageDraw.Draw(raw_img)
        font = ImageFont.truetype("SimHei.ttf", size=int(line_thickness * 10))

        for i, det in enumerate(boxes_info):
            if len(det) == 5:
                x1, y1, x2, y2, cls = det
            else:
                x1, y1, x2, y2, _, cls = det
            color = (255,0,0)
            box = [x1, y1, x2, y2] 
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[2]), int(box[3]))
            
            # 画框
            draw.rectangle([p1, p2], outline=color, width=line_thickness)
            
            # 写文字
            label = f'{cls}' if isinstance(cls,str) else str(class_names[cls])
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

def image_to_base64(image_path):
    """将图片文件转换为base64编码"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def find_test_images():
    """自动查找测试图片"""
    # 常见的图片扩展名
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.gif']
    
    test_images = []
    
    # 在测试目录查找
    test_dirs = ['./test_images']
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for ext in image_extensions:
                test_images.extend(glob.glob(os.path.join(test_dir, ext)))
                test_images.extend(glob.glob(os.path.join(test_dir, ext.upper())))
    
    return test_images

def test_hybrid_inference(image_path, base_url="http://localhost:5000"):
    """测试YOLO + VLM串联推理"""
    try:
        # 转换图片为base64
        image_base64 = image_to_base64(image_path)
        
        # 准备请求数据
        data = {
            "image_base64": image_base64,
            "image_name": os.path.basename(image_path)
        }
        
        # 发送请求
        response = requests.post(f"{base_url}/hybrid_inference", json=data)
        
        print(f"YOLO + VLM 串联推理结果: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            print(f"图片名称: {result.get('image_name')}")
            print(f"处理成功: {result.get('success')}")
            print(f"最终决策: {result.get('final_decision')}")
            
            # 显示YOLO检测结果
            yolo_result = result.get('yolo_detection', {})
            print(f"\nYOLO检测结果:")
            print(f"  - 检测到盖板缺失: {yolo_result.get('has_open', False)}")
            print(f"  - 总对象数: {yolo_result.get('detection_count', 0)}")
            
            # 显示检测统计
            summary = result.get('detection_summary', {})
            print(f"\n检测统计:")
            print(f"  - 盖板缺失数量: {summary.get('open_count', 0)}")
            print(f"  - 是否使用VLM: {summary.get('used_vlm', False)}")
            print(f"  - 返回目标框数量: {summary.get('boxes_returned', 0)}")
            
            # 显示检测框信息
            detection_boxes = result.get('detection_boxes', [])
            if detection_boxes:
                print(f"\n检测框信息:")
                for i, box in enumerate(detection_boxes, 1):
                    print(f"  框{i}: {box.get('class_name', 'N/A')} - 置信度: {box.get('conf', 0):.2f}")
                    print(f"       边界框: {box.get('bbox', [])}")
            else:
                print(f"\n检测框信息: 无")
            
            # 显示处理步骤
            steps = result.get('processing_steps', [])
            print(f"\n处理步骤:")
            for i, step in enumerate(steps, 1):
                print(f"  {i}. {step}")
            
            # 如果有VLM分析结果，显示它
            vlm_result = result.get('vlm_analysis')
            if vlm_result:
                print(f"\nVLM分析结果:")
                print(f"  - 分析成功: {vlm_result.get('success', False)}")
                if vlm_result.get('success'):
                    print(f"  - VLM预测: {vlm_result.get('predict', 'N/A')}")
                    print(f"  - 处理时间: {vlm_result.get('processing_time', 0):.2f}秒")
                    print(f"  - VLM思考: {vlm_result.get('think', 'N/A')}")
                    print(f"  - VLM答案: {vlm_result.get('answer', 'N/A')}")
                else:
                    print(f"  - VLM错误: {vlm_result.get('error', 'N/A')}")
            
        else:
            print(f"请求失败: {result}")
        
        return result
        
    except Exception as e:
        print(f"串联推理测试失败: {e}")
        return None

def test_hybrid_inference_batch(image_paths, base_url="http://localhost:5000"):
    """测试YOLO + VLM串联批量推理"""
    try:
        # 准备批量数据
        images = []
        for image_path in image_paths:
            image_base64 = image_to_base64(image_path)
            images.append({
                "image_base64": image_base64,
                "image_name": os.path.basename(image_path)
            })
        
        data = {
            "images": images
        }
        
        # 发送请求
        response = requests.post(f"{base_url}/hybrid_inference/batch", json=data)
        
        print(f"YOLO + VLM 串联批量推理结果: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            # 显示批量处理摘要
            batch_summary = result.get('batch_summary', {})
            print(f"\n批量处理摘要:")
            print(f"  - 总数量: {batch_summary.get('total_count', 0)}")
            print(f"  - 成功数量: {batch_summary.get('success_count', 0)}")
            print(f"  - 盖板缺失数量: {batch_summary.get('open_alarm_count', 0)}")
            print(f"  - VLM使用次数: {batch_summary.get('vlm_used_count', 0)}")
            print(f"  - 返回目标框的图片数量: {batch_summary.get('boxes_returned_count', 0)}")
            
            # 显示每个图片的处理结果
            results = result.get('results', [])
            print(f"\n详细处理结果:")
            for i, res in enumerate(results, 1):
                print(f"\n图片 {i}: {res.get('image_name')}")
                print(f"  - 处理成功: {res.get('success', False)}")
                
                if res.get('success'):
                    print(f"  - 最终决策: {res.get('final_decision')}")
                    
                    # YOLO检测结果
                    yolo_result = res.get('yolo_detection', {})
                    print(f"  - YOLO检测到盖板缺失: {yolo_result.get('has_open', False)}")
                    print(f"  - 检测对象数: {yolo_result.get('detection_count', 0)}")
                    
                    # 检测统计
                    summary = res.get('detection_summary', {})
                    print(f"  - 盖板缺失数量: {summary.get('open_count', 0)}")
                    print(f"  - 使用VLM: {summary.get('used_vlm', False)}")
                    print(f"  - 返回目标框数量: {summary.get('boxes_returned', 0)}")
                    
                    # 检测框信息
                    detection_boxes = res.get('detection_boxes', [])
                    if detection_boxes:
                        print(f"  - 检测框:")
                        for j, box in enumerate(detection_boxes, 1):
                            print(f"    框{j}: {box} - 置信度: {box.get('conf', 0):.2f}")
                    else:
                        print(f"  - 检测框: 无")
                    
                    # 处理步骤
                    steps = res.get('processing_steps', [])
                    print(f"  - 处理步骤: {' -> '.join(steps)}")
                    
                    # VLM分析结果
                    vlm_result = res.get('vlm_analysis')
                    if vlm_result:
                        print(f"  - VLM分析成功: {vlm_result.get('success', False)}")
                        if vlm_result.get('success'):
                            print(f"  - VLM预测: {vlm_result.get('predict', 'N/A')}")
                            print(f"  - VLM处理时间: {vlm_result.get('processing_time', 0):.2f}秒")
                            print(f"  - VLM思考: {vlm_result.get('think', 'N/A')}")
                            print(f"  - VLM答案: {vlm_result.get('answer', 'N/A')}")
                else:
                    print(f"  - 错误: {res.get('error', 'N/A')}")
        else:
            print(f"批量请求失败: {result}")
        
        return result
        
    except Exception as e:
        print(f"串联批量推理测试失败: {e}")
        return None

def main():
    """主测试函数"""
    base_url = "http://localhost:5000"
    
    print("=== YOLO + VLM 串联推理服务测试 ===\n")
    
    # 1. 查找测试图片
    print("1. 查找测试图片...")
    test_images = find_test_images()
    
    if not test_images:
        print("未找到测试图片，请确保有以下格式的图片文件：")
        print("  - jpg, jpeg, png, bmp, tiff, gif")
        print("  - 可以放在当前目录或 ./test_images 目录下")
        return
    
    print(f"找到 {len(test_images)} 张测试图片:")
    for img in test_images:
        print(f"  - {img}")
    print()
    
    # 2. 测试单张图片YOLO + VLM串联推理
    print("2. 测试单张图片YOLO + VLM串联推理...")
    if test_images:
        test_hybrid_inference(test_images[0], base_url)
    print()
    
    # 3. 测试批量YOLO + VLM串联推理
    print("3. 测试批量YOLO + VLM串联推理...")
    if len(test_images) > 1:
        test_hybrid_inference_batch(test_images[:16], base_url)
    else:
        print("图片数量不足，使用单张图片进行批量测试")
        test_hybrid_inference_batch([test_images[0]], base_url)
    print()
    
    print("所有测试完成!")

if __name__ == "__main__":
    main() 