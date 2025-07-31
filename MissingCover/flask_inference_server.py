from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
import traceback
from vlm_inference import inference_single_base64, VLLM_MODEL
from yolo_inference import detect_base64
from config_loader import config

app = Flask(__name__)

# 获取配置
flask_config = config.get_flask_config()
PROMPT = config.get_prompt()

@app.route('/hybrid_inference', methods=['POST'])
def hybrid_inference():
    """
    YOLO + VLM 串联推理接口
    先使用YOLO检测，如果检测到盖板缺失（类别1），则调用VLM进行进一步分析        
    """
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "请求数据为空"}), 400
        
        # 检查必需的字段
        if 'image_base64' not in data:
            return jsonify({"error": "缺少image_base64字段"}), 400
        
        image_base64 = data['image_base64']
        image_name = data.get('image_name', 'hybrid_image')
        conf_threshold = data.get('conf_threshold', None)  # 使用配置文件默认值
        
        # 验证base64格式
        if not image_base64:
            return jsonify({"error": "image_base64不能为空"}), 400
        
        # 步骤1：使用YOLO进行检测
        print(f"步骤1：对图像 {image_name} 进行YOLO检测...")
        yolo_result = detect_base64(image_base64, conf_threshold)
        
        # 初始化返回结果
        result = {
            "image_name": image_name,
            "yolo_detection": yolo_result,
            "vlm_analysis": None,
            "final_decision": "不需要预警",
            "processing_steps": ["YOLO检测完成"],
            "success": True,
            "detection_boxes": []  # 新增：检测框列表
        }
        
        # 步骤2：判断是否需要VLM分析
        if yolo_result.get('has_open', False):
            print(f"步骤2：检测到盖板缺失，调用VLM进行进一步分析...")
            result["processing_steps"].append("检测到盖板缺失，启动VLM分析")
            
            # 调用VLM进行分析
            vlm_result = inference_single_base64(image_base64, PROMPT, image_name)
            result["vlm_analysis"] = vlm_result
            
            # 根据VLM结果确定最终决策
            if vlm_result.get('success', False):
                result["processing_steps"].append("VLM分析完成")
                vlm_answer = vlm_result.get('answer', '')
                
                # 解析VLM预测结果
                if "盖板缺失" in vlm_answer:
                    result["final_decision"] = "盖板缺失"
                    result["processing_steps"].append("VLM确认盖板缺失")
                    # 当VLM确认盖板缺失时，返回YOLO检测到的盖板缺失目标框
                    result["detection_boxes"] = yolo_result.get('open_objects', [])
                else:
                    result["final_decision"] = "盖板存在"
                    result["processing_steps"].append("VLM判断盖板存在")
                    # 其他情况返回空列表
                    result["detection_boxes"] = []
            else:
                result["processing_steps"].append("VLM分析失败，基于YOLO结果判断")
                result["final_decision"] = "盖板存在"  # YOLO检测到盖板缺失，默认盖板存在
                # VLM分析失败但YOLO检测到盖板缺失，也返回目标框
                result["detection_boxes"] = yolo_result.get('open_objects', [])
                result["vlm_analysis"] = {
                    "success": False,
                    "error": vlm_result.get('error', '未知错误')
                }
        else:
            # 没有检测到盖板缺失，不需要VLM分析
            result["processing_steps"].append("未检测到盖板缺失，不需要VLM分析")
            result["final_decision"] = "盖板存在"
            # 未检测到盖板缺失，返回空列表
            result["detection_boxes"] = []
        
        # 添加检测统计信息
        result["detection_summary"] = {
            "open_count": len(yolo_result.get('open_objects', [])),
            "total_objects": yolo_result.get('detection_count', 0),
            "used_vlm": yolo_result.get('has_open', False),
            "boxes_returned": len(result["detection_boxes"])
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"混合推理过程中出现错误: {e}")
        print(traceback.format_exc())
        return jsonify({
            "error": f"服务器内部错误: {str(e)}",
            "success": False
        }), 500

@app.route('/hybrid_inference/batch', methods=['POST'])
def hybrid_inference_batch():
    """
    YOLO + VLM 串联批量推理接口
    对多张图片进行串联推理
    """
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data or 'images' not in data:
            return jsonify({"error": "请求数据格式错误，需要包含images字段"}), 400
        
        images = data['images']
        if not isinstance(images, list):
            return jsonify({"error": "images字段必须是数组"}), 400
        
        if len(images) == 0:
            return jsonify({"error": "images数组不能为空"}), 400
        
        # 检查批量大小限制
        max_batch_size = flask_config.get('max_batch_size', 16)
        if len(images) > max_batch_size:
            return jsonify({"error": f"批量大小超过限制，最大允许 {max_batch_size} 张图片"}), 400
        
        # 验证每个图片数据
        for i, image in enumerate(images):
            if 'image_base64' not in image:
                return jsonify({"error": f"第{i+1}个图片对象缺少image_base64字段"}), 400
            if not image['image_base64']:
                return jsonify({"error": f"第{i+1}个图片的image_base64不能为空"}), 400
        
        def process_single_image(image_data):
            """处理单张图片的串联推理"""
            try:
                image_base64 = image_data['image_base64']
                image_name = image_data.get('image_name', 'batch_image')
                conf_threshold = image_data.get('conf_threshold', None)  # 使用配置文件默认值
                
                # 步骤1：使用YOLO进行检测
                print(f"批量处理：对图像 {image_name} 进行YOLO检测...")
                yolo_result = detect_base64(image_base64, conf_threshold)
                
                # 初始化返回结果
                result = {
                    "image_name": image_name,
                    "yolo_detection": yolo_result,
                    "vlm_analysis": None,
                    "final_decision": "不需要预警",
                    "processing_steps": ["YOLO检测完成"],
                    "success": True,
                    "detection_boxes": []  # 新增：检测框列表
                }
                
                # 步骤2：判断是否需要VLM分析
                if yolo_result.get('has_open', False):  
                    print(f"批量处理：检测到盖板缺失，调用VLM进行进一步分析...")
                    result["processing_steps"].append("检测到盖板缺失，启动VLM分析")
                    
                    # 调用VLM进行分析
                    vlm_result = inference_single_base64(image_base64, PROMPT, image_name)
                    result["vlm_analysis"] = vlm_result
                    
                    # 根据VLM结果确定最终决策
                    if vlm_result.get('success', False):
                        result["processing_steps"].append("VLM分析完成")
                        vlm_answer = vlm_result.get('answer', '')
                        
                        # 解析VLM预测结果
                        if "盖板缺失" in vlm_answer:
                            result["final_decision"] = "盖板缺失"
                            result["processing_steps"].append("VLM确认盖板缺失")
                            # 当VLM确认盖板缺失时，返回YOLO检测到的盖板缺失目标框
                            result["detection_boxes"] = yolo_result.get('open_objects', [])
                        else:
                            result["final_decision"] = "盖板存在"
                            result["processing_steps"].append("VLM判断盖板存在")
                            # 其他情况返回空列表
                            result["detection_boxes"] = []
                    else:
                        result["processing_steps"].append("VLM分析失败，基于YOLO结果判断")
                        result["final_decision"] = "盖板存在"  # YOLO检测到盖板缺失，默认盖板存在
                        # VLM分析失败但YOLO检测到盖板缺失，也返回目标框
                        result["detection_boxes"] = yolo_result.get('open_objects', [])
                        result["vlm_analysis"] = {
                            "success": False,
                            "error": vlm_result.get('error', '未知错误')
                        }
                else:
                    # 没有检测到盖板缺失，不需要VLM分析
                    result["processing_steps"].append("未检测到盖板缺失，不需要VLM分析")
                    result["final_decision"] = "盖板存在"
                    # 未检测到盖板缺失，返回空列表
                    result["detection_boxes"] = []
                
                # 添加检测统计信息
                result["detection_summary"] = {
                    "open_count": len(yolo_result.get('open_objects', [])),
                    "total_objects": yolo_result.get('detection_count', 0),
                    "used_vlm": yolo_result.get('has_open', False),
                    "boxes_returned": len(result["detection_boxes"])
                }
                
                return result
                
            except Exception as e:
                print(f"处理图片 {image_data.get('image_name', 'unknown')} 时出错: {e}")
                return {
                    "image_name": image_data.get('image_name', 'unknown'),
                    "error": f"处理失败: {str(e)}",
                    "success": False,
                    "detection_boxes": []
                }
        
        # 使用线程池并行处理
        max_workers = min(len(images), flask_config.get('max_workers', 16))
        print(f"开始批量处理 {len(images)} 张图片，使用 {max_workers} 个线程...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single_image, images))
        
        # 统计结果
        success_count = len([r for r in results if r.get('success', False)])
        open_alarm_count = len([r for r in results if r.get('final_decision') == '盖板缺失'])
        vlm_used_count = len([r for r in results if r.get('detection_summary', {}).get('used_vlm', False)])
        boxes_returned_count = len([r for r in results if len(r.get('detection_boxes', [])) > 0])
        
        return jsonify({
            "results": results,
            "batch_summary": {
                "total_count": len(images),
                "success_count": success_count,
                "open_alarm_count": open_alarm_count,
                "vlm_used_count": vlm_used_count,
                "boxes_returned_count": boxes_returned_count,
                "processing_time": "批量处理完成"
            }
        })
        
    except Exception as e:
        print(f"批量串联推理过程中出现错误: {e}")
        print(traceback.format_exc())
        return jsonify({
            "error": f"服务器内部错误: {str(e)}",
            "success": False
        }), 500

if __name__ == '__main__':
    print("启动 YOLO + VLM 串联推理服务...")
    print(f"使用模型: {VLLM_MODEL}")
    print("可用接口:")
    print("  POST /hybrid_inference - YOLO + VLM 单张图片串联推理")
    print("  POST /hybrid_inference/batch - YOLO + VLM 批量图片串联推理")
    print(f"\n服务启动在 http://{flask_config.get('host', '0.0.0.0')}:{flask_config.get('port', 5000)}")
    
    app.run(
        host=flask_config.get('host', '0.0.0.0'), 
        port=flask_config.get('port', 5000), 
        debug=flask_config.get('debug', True)
    ) 