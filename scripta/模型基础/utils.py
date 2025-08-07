# utils.py - 工具模块
"""
电力场景AI模型测试系统工具函数
"""

import sys
import time
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

def stream_print(text: str, delay: float = 0.03):
    """模拟流式输出"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def stream_print_progress(text: str, delay: float = 0.1):
    """模拟进度条式输出"""
    sys.stdout.write(text)
    sys.stdout.flush()
    for i in range(3):
        time.sleep(delay)
        sys.stdout.write('.')
        sys.stdout.flush()
    print(" 完成")
    time.sleep(0.2)

def create_progress_bar(current: int, total: int, length: int = 30) -> str:
    """创建进度条"""
    filled_length = int(length * current // total)  
    bar = '█' * filled_length + '░' * (length - filled_length)
    percentage = current / total * 100
    return f"[{bar}] {percentage:5.1f}%"

def generate_random_scores(min_val: float, max_val: float, count: int = 1) -> List[float]:
    """生成随机分数"""
    if count == 1:
        return random.uniform(min_val, max_val)
    return [random.uniform(min_val, max_val) for _ in range(count)]

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """计算统计信息"""
    return {
        "mean": np.mean(values),
        "max": np.max(values), 
        "min": np.min(values),
        "std": np.std(values)
    }

def format_timestamp(timestamp_format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """格式化时间戳"""
    return datetime.now().strftime(timestamp_format)

def validate_requirements(actual: float, required: float, 
                         operator: str = ">=") -> bool:
    """验证是否满足要求"""
    if operator == ">=":
        return actual >= required
    elif operator == "<=": 
        return actual <= required
    elif operator == ">":
        return actual > required
    elif operator == "<":
        return actual < required
    elif operator == "==":
        return actual == required
    else:
        raise ValueError(f"不支持的操作符: {operator}")

def create_test_result(name: str, actual: Any, required: Any, 
                      operator: str = ">=", unit: str = "") -> Dict:
    """创建测试结果字典"""
    if isinstance(actual, (int, float)) and isinstance(required, (int, float)):
        passed = validate_requirements(actual, required, operator)
    else:
        passed = actual == required
        
    return {
        "name": name,
        "actual": f"{actual}{unit}",
        "required": f"{operator}{required}{unit}" if operator != "==" else f"{required}{unit}",
        "passed": passed,
        "status": "达标" if passed else "不达标"
    }

class TestTimer:
    """测试计时器"""
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        self.end_time = time.time()
    
    def get_duration(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    def get_duration_str(self) -> str:
        duration = self.get_duration()
        return f"{duration:.2f}s"