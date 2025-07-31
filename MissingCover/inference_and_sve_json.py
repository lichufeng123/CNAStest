import requests
import base64
import json
import os
import glob
from PIL import Image
from datetime import datetime
import time
# === 配置参数 ===
DEFAULT_BASE_URL = "http://localhost:5000"  # 默认服务器地址
DEFAULT_CONF_THRESHOLD = 0.25  # 默认置信度阈值
DEFAULT_BATCH_SIZE = 16  # 分批处理时的批次大小

def configure_test_parameters(base_url=None, conf_threshold=None, batch_size=None):
    """配置测试参数"""
    global DEFAULT_BASE_URL, DEFAULT_CONF_THRESHOLD, DEFAULT_BATCH_SIZE
    
    if base_url is not None:
        DEFAULT_BASE_URL = base_url
        print(f"已设置服务器地址为: {base_url}")
    
    if conf_threshold is not None:
        DEFAULT_CONF_THRESHOLD = conf_threshold
        print(f"已设置置信度阈值为: {conf_threshold}")
        
    if batch_size is not None:
        DEFAULT_BATCH_SIZE = batch_size
        print(f"已设置批次大小为: {batch_size}")

def image_to_base64(image_path):
    """将图片文件转换为base64编码"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def find_test_images(test_dirs=None):
    """自动查找测试图片"""
    # 常见的图片扩展名
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.gif']
    
    test_images = []
    
    # 如果没有指定测试目录，使用默认目录
    if test_dirs is None:
        test_dirs = ['./test_images', './']
    
    # 在指定目录查找
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for ext in image_extensions:
                test_images.extend(glob.glob(os.path.join(test_dir, ext)))
                test_images.extend(glob.glob(os.path.join(test_dir, ext.upper())))
    
    return test_images

def test_hybrid_inference(image_path, base_url="http://localhost:5000", conf_threshold=None):
    """测试YOLO + VLM串联推理"""
    try:
        # 使用配置的置信度阈值
        if conf_threshold is None:
            conf_threshold = DEFAULT_CONF_THRESHOLD
            
        # 转换图片为base64
        image_base64 = image_to_base64(image_path)
        
        # 准备请求数据
        data = {
            "image_base64": image_base64,
            "image_name": os.path.basename(image_path),
            "conf_threshold": conf_threshold
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
                else:
                    print(f"  - VLM错误: {vlm_result.get('error', 'N/A')}")
            
        else:
            print(f"请求失败: {result}")
        
        return result
        
    except Exception as e:
        print(f"串联推理测试失败: {e}")
        return None

def test_hybrid_inference_batch(image_paths, base_url="http://localhost:5000", conf_threshold=None):
    """测试YOLO + VLM串联批量推理"""
    try:
        # 使用配置的置信度阈值
        if conf_threshold is None:
            conf_threshold = DEFAULT_CONF_THRESHOLD
            
        # 准备批量数据
        images = []
        for image_path in image_paths:
            image_base64 = image_to_base64(image_path)
            images.append({
                "image_base64": image_base64,
                "image_name": os.path.basename(image_path),
                "conf_threshold": conf_threshold
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
            print(f"  - 盖板缺失数量: {batch_summary.get('open_count', 0)}")
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
                else:
                    print(f"  - 错误: {res.get('error', 'N/A')}")
        else:
            print(f"批量请求失败: {result}")
        
        return result
        
    except Exception as e:
        print(f"串联批量推理测试失败: {e}")
        return None

def test_complete_directory(image_paths, base_url="http://localhost:5000", conf_threshold=None, batch_size=None):
    """测试完整目录的所有图片（分批处理）"""
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
    if conf_threshold is None:
        conf_threshold = DEFAULT_CONF_THRESHOLD
    
    total_images = len(image_paths)
    total_batches = (total_images + batch_size - 1) // batch_size  # 向上取整
    
    print(f"开始完整目录测试:")
    print(f"  - 总图片数量: {total_images}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 总批次数: {total_batches}")
    print()
    
    all_results = []
    batch_summaries = []
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        batch_images = image_paths[start_idx:end_idx]
        
        print(f"正在处理批次 {batch_idx + 1}/{total_batches} (图片 {start_idx + 1}-{end_idx})...")
        
        # 调用批量推理
        batch_result = test_hybrid_inference_batch(batch_images, base_url, conf_threshold)
        
        if batch_result:
            all_results.extend(batch_result.get('results', []))
            batch_summary = batch_result.get('batch_summary', {})
            batch_summaries.append({
                'batch_index': batch_idx + 1,
                'image_range': f"{start_idx + 1}-{end_idx}",
                'summary': batch_summary
            })
            print(f"批次 {batch_idx + 1} 完成: 成功 {batch_summary.get('success_count', 0)}/{batch_summary.get('total_count', 0)}")
        else:
            print(f"批次 {batch_idx + 1} 失败!")
            batch_summaries.append({
                'batch_index': batch_idx + 1,
                'image_range': f"{start_idx + 1}-{end_idx}",
                'summary': {'error': '批次处理失败'}
            })
        
        print()
    
    # 汇总所有批次的结果
    total_success = len([r for r in all_results if r.get('success', False)])
    total_open_count = len([r for r in all_results if r.get('final_decision') == '盖板缺失'])
    total_vlm_used = len([r for r in all_results if r.get('detection_summary', {}).get('used_vlm', False)])
    total_boxes_returned = len([r for r in all_results if r.get('detection_summary', {}).get('boxes_returned', 0) > 0])
    
    complete_result = {
        'test_type': 'complete_directory',
        'total_images': total_images,
        'total_batches': total_batches,
        'batch_size': batch_size,
        'overall_summary': {
            'total_count': total_images,
            'success_count': total_success,
            'open_count': total_open_count,
            'vlm_used_count': total_vlm_used,
            'boxes_returned_count': total_boxes_returned
        },
        'batch_summaries': batch_summaries,
        'all_results': all_results
    }
    
    print("=== 完整目录测试汇总 ===")
    print(f"总图片数量: {total_images}")
    print(f"处理成功: {total_success}")
    print(f"盖板缺失: {total_open_count}")
    print(f"VLM使用次数: {total_vlm_used}")
    print(f"返回目标框的图片: {total_boxes_returned}")
    print()
    
    return complete_result

def save_result_to_json(result, filename=None):
    """将结果保存到JSON文件"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./inference_result/hybrid_inference_result_{timestamp}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到文件: {filename}")
        return filename
    except Exception as e:
        print(f"保存JSON文件失败: {e}")
        return None

def main():
    """主测试函数"""
    base_url = DEFAULT_BASE_URL
    conf_threshold = DEFAULT_CONF_THRESHOLD
    batch_size = DEFAULT_BATCH_SIZE
    
    print("=== YOLO + VLM 完整目录测试服务 ===\n")
    
    # 显示当前配置
    print("当前配置参数:")
    print(f"  - 服务器地址: {base_url}")
    print(f"  - 置信度阈值: {conf_threshold}")
    print(f"  - 批次大小: {batch_size}")
    print("  - 提示: 可以通过调用 configure_test_parameters() 函数来修改这些参数")
    print()
    
    # 1. 查找测试图片
    print("1. 查找测试图片...")
    # 可以指定多个测试目录
    test_dirs = ["/mnt/nas_data2/gjx_workspace/Datasets/MissingCoverPlate/20250715/不存在盖板缺失"]  # 可以根据需要修改这里的路径
    test_images = find_test_images(test_dirs)
    
    if not test_images:
        print("未找到测试图片，请确保有以下格式的图片文件：")
        print("  - jpg, jpeg, png, bmp, tiff, gif")
        print(f"  - 可以放在以下目录中: {', '.join(test_dirs)}")
        return
    
    print(f"找到 {len(test_images)} 张测试图片:")
    for img in test_images[:5]:  # 只显示前5张
        print(f"  - {img}")
    if len(test_images) > 5:
        print(f"  - ... 还有 {len(test_images) - 5} 张图片")
    print()
    
    # 2. 开始完整目录测试
    print("2. 开始完整目录测试...")
    time_start = time.time()
    result = test_complete_directory(test_images, base_url, conf_threshold, batch_size)
    time_end = time.time()
    print(f"完整目录测试完成，用时: {time_end - time_start:.2f}秒")
    
    if result:
        # 3. 保存结果到JSON文件
        print("3. 保存结果到JSON文件...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./inference_result/complete_directory_test_{timestamp}.json"
        os.makedirs("./inference_result", exist_ok=True)
        json_filename = save_result_to_json(result, filename)
        
        if json_filename:
            print(f"测试完成！结果已保存到: {json_filename}")
        else:
            print("测试完成！但结果保存失败")
    else:
        print("完整目录测试失败！")

if __name__ == "__main__":
    # 运行完整目录测试
    main()