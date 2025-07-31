import os
import json
import re
from tqdm import tqdm
import base64
from PIL import Image
import traceback
import datetime
from openai import OpenAI
import requests
import re
import time
from config_loader import config

# 获取VLM配置
vlm_config = config.get_vlm_config()

# VLM API 配置
VLLM_API_BASE = vlm_config.get("api_base", "http://localhost:9999/v1")
VLLM_MODEL = vlm_config.get("model_name", "MissCover-Qwen2.5VL-7B")
TIMEOUT = vlm_config.get("timeout", 30)
MAX_RETRIES = vlm_config.get("max_retries", 3)

def encode_image_from_base64(base64_string):
    """直接处理base64字符串，转换为API需要的格式"""
    try:
        # 如果base64字符串不包含数据URL前缀，添加它
        if not base64_string.startswith('data:image'):
            base64_string = f"data:image;base64,{base64_string}"
        return base64_string
    except Exception as e:
        raise ValueError(f"无法处理base64图片: {str(e)}")
    
def parse_vlm_result(result):
    """解析出think和answer"""
    think_match = re.search(r"<think>(.*?)</think>", result, re.S)
    if think_match:
        think = think_match.group(1).strip()
    else:
        think = ""
    answer_match = re.search(r"<answer>(.*?)</answer>", result, re.S)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        answer = ""
    return think, answer


def inference_single_base64(image_base64, prompt, image_name="image"):
    """
    对单张base64编码的图像进行VLM推理
    
    Args:
        image_base64: base64编码的图像数据
        prompt: 推理提示词
        image_name: 图像名称，用于日志记录
    
    Returns:
        dict: 推理结果，包含：
            - 'success': 是否成功
            - 'predict': 预测结果
            - 'error': 错误信息（如果失败）
            - 'processing_time': 处理时间
    """
    start_time = time.time()
    
    try:
        # 准备请求数据
        if image_base64.startswith('data:image'):
            # 移除数据URL前缀
            image_base64 = image_base64.split(',')[1]
        
        # 构建请求数据
        request_data = {
            "model": VLLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.0
        }
        
        # 发送请求，支持重试
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    f"{VLLM_API_BASE}/chat/completions",
                    json=request_data,
                    timeout=TIMEOUT
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # 提取预测结果
                    if "choices" in result and len(result["choices"]) > 0:
                        predict = result["choices"][0]["message"]["content"]

                        think, answer = parse_vlm_result(predict)
                        
                        processing_time = time.time() - start_time
                        
                        return {
                            "success": True,
                            "think": think,
                            "answer": answer,
                            "predict": predict,
                            "processing_time": processing_time,
                            "model": VLLM_MODEL,
                            "image_name": image_name
                        }
                    else:
                        return {
                            "success": False,
                            "error": "VLM响应格式错误",
                            "processing_time": time.time() - start_time,
                            "image_name": image_name
                        }
                else:
                    error_msg = f"VLM API请求失败，状态码: {response.status_code}"
                    if attempt < MAX_RETRIES - 1:
                        print(f"VLM请求失败，正在重试... (尝试 {attempt + 1}/{MAX_RETRIES})")
                        time.sleep(1)  # 等待1秒后重试
                        continue
                    else:
                        return {
                            "success": False,
                            "error": error_msg,
                            "processing_time": time.time() - start_time,
                            "image_name": image_name
                        }
                        
            except requests.exceptions.Timeout:
                error_msg = f"VLM请求超时 (超时时间: {TIMEOUT}秒)"
                if attempt < MAX_RETRIES - 1:
                    print(f"VLM请求超时，正在重试... (尝试 {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(2)  # 等待2秒后重试
                    continue
                else:
                    return {
                        "success": False,
                        "error": error_msg,
                        "processing_time": time.time() - start_time,
                        "image_name": image_name
                    }
            except requests.exceptions.ConnectionError:
                error_msg = f"VLM连接错误，请检查服务是否运行在 {VLLM_API_BASE}"
                if attempt < MAX_RETRIES - 1:
                    print(f"VLM连接失败，正在重试... (尝试 {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(2)  # 等待2秒后重试
                    continue
                else:
                    return {
                        "success": False,
                        "error": error_msg,
                        "processing_time": time.time() - start_time,
                        "image_name": image_name
                    }
                    
    except Exception as e:
        return {
            "success": False,
            "error": f"VLM推理过程中出现未知错误: {str(e)}",
            "processing_time": time.time() - start_time,
            "image_name": image_name
        }

def inference_batch_base64(images_data, prompt):
    """
    对多张base64编码的图像进行批量VLM推理
    
    Args:
        images_data: 图像数据列表，每个元素包含：
            - 'image_base64': base64编码的图像数据
            - 'image_name': 图像名称
        prompt: 推理提示词
    
    Returns:
        list: 推理结果列表
    """
    results = []
    
    for image_data in images_data:
        image_base64 = image_data.get('image_base64', '')
        image_name = image_data.get('image_name', 'unknown')
        
        result = inference_single_base64(image_base64, prompt, image_name)
        results.append(result)
    
    return results

if __name__ == "__main__":
    # 测试VLM推理
    print("测试VLM推理...")
    print(f"使用模型: {VLLM_MODEL}")
    print(f"API地址: {VLLM_API_BASE}")
    
    # 这里可以添加测试代码
    print("VLM推理模块已就绪")
