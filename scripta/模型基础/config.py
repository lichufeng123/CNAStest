# config.py - 配置模块
"""
电力场景AI模型测试系统配置文件
"""

import os
from typing import Dict, List

class ModelConfig:
    """模型基础配置"""
    MODEL_NAME = "PowerVision-Reasoning-72B"
    MODEL_TYPE = "Reasoning Model"
    MIN_PARAMS_B = 70  # 最小参数量（亿）
    MAX_RESOLUTION = "1080p"
    REQUIRED_MODALITIES = 5  # 需要支持的模态数量

class PerformanceConfig:
    """性能测试配置"""
    MAX_INFERENCE_TIME = 3.0  # 最大推理时间（秒）
    MIN_SEGMENTATION_ACCURACY = 90.0  # 最小分割精度（%）
    MIN_IMPROVEMENT_RATIO = 50.0  # 最小性能提升比例（%）
    MAX_PERPLEXITY_INCREASE = 5.0  # 最大困惑度上升（%）
    MIN_SIZE_REDUCTION = 85.0  # 最小模型压缩比例（%）

class FunctionConfig:
    """功能测试配置"""
    
    # 电力场景配置
    POWER_SCENARIOS = {
        '输电通道类': ['异物外飘安全隐患', '山火检测'],
        '变电站设备缺陷检测类': ['设备缺陷检测'],
        '作业违章识别类': ['高空作业', '带电作业', '攀爬扶梯']
    }
    
    # 小目标检测配置
    SMALL_TARGETS = [
        '远距离山火', '绝缘子自爆', '高空俯拍绿膜', 
        '绝缘子污闪爬电', '安全工器具'
    ]
    
    # 推理能力配置
    POSITION_SCENARIOS = [
        '山火', '绝缘子自爆', '作业人员着装', '高空作业', '带电作业'
    ]
    
    INTERFERENCE_FACTORS = [
        '过曝灯光', '工厂排烟', '山间雾气', '扬尘'
    ]
    
    DISTANCE_SCENARIOS = [
        '人员与电力设备之间的相对距离',
        '大型车辆与输电设施之间的相对距离',  
        '明火和输电设施之间的相对距离'
    ]
    
    # 业务领域配置
    INSTRUCTION_DOMAINS = ['安监', '输电', '变电', '配电']
    KNOWLEDGE_DOMAINS = ['输电', '变电', '配电', '安监']
    
    # 生成模型配置
    MIN_GENERATION_RATIO = 300.0  # 最小生成比例（%）

class TestConfig:
    """测试配置"""
    OUTPUT_DIR = "model_test_output"
    ANNOTATION_DIR = "annotated_images"
    TEST_IMAGE_COUNT = 100
    STREAM_DELAY = 0.03
    PROGRESS_DELAY = 0.1
    
    # 文件名配置
    REPORT_PREFIX = "power_model_test_report"
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# 导出所有配置
__all__ = [
    'ModelConfig', 'PerformanceConfig', 
    'FunctionConfig', 'TestConfig'
]