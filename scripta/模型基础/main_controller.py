# main_controller.py - 主控制器模块
"""
电力场景AI模型测试系统主控制器
"""

import json
import time
from datetime import datetime
from typing import Dict
from model_basics import ModelBasicsTest
from model_performance import ModelPerformanceTest
from model_functions import ModelFunctionsTest
from utils import stream_print, TestTimer
from config import TestConfig, ModelConfig

class PowerAIModelTestController:
    """电力AI模型测试主控制器"""
    
    def __init__(self):
        self.model_name = ModelConfig.MODEL_NAME
        self.model_type = ModelConfig.MODEL_TYPE
        self.test_results = {}
        self.timer = TestTimer()
        
        # 初始化各测试模块
        self.basics_test = ModelBasicsTest()
        self.performance_test = ModelPerformanceTest()
        self.functions_test = ModelFunctionsTest()
    
    def print_header(self):
        """打印测试开始信息"""
        stream_print("=" * 70)
        stream_print("开始执行电力场景推理模型完整性能测试")
        stream_print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        stream_print(f"测试模型: {self.model_name} ({self.model_type})")
        stream_print("=" * 70)
    
    def run_comprehensive_test(self) -> Dict:
        """运行完整测试"""
        self.print_header()
        self.timer.start()
        
        # 执行各模块测试
        stream_print("开始执行模型基础测试...")
        basics_results = self.basics_test.run_all_tests()
        
        stream_print("\n开始执行模型性能测试...")
        performance_results = self.performance_test.run_all_tests()
        
        stream_print("\n开始执行模型功能测试...")
        functions_results = self.functions_test.run_all_tests()
        
        self.timer.stop()
        
        # 汇总测试结果
        self.test_results = {
            "测试信息": {
                "测试时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "模型名称": self.model_name,
                "模型类型": self.model_type,
                "测试版本": "v3.0",
                "测试耗时": self.timer.get_duration_str()
            },
            "模型基础": basics_results,
            "模型性能": performance_results,
            "模型功能": functions_results
        }
        
        return self.test_results
    
    def print_final_summary(self):
        """打印最终测试总结"""
        if not self.test_results:
            stream_print("请先运行性能测试")
            return
        
        stream_print("\n" + "=" * 70)
        stream_print("推理模型测试完成 - 最终总结报告")
        stream_print("=" * 70)
        
        # 提取各模块结果
        basics = self.test_results["模型基础"]
        performance = self.test_results["模型性能"]
        functions = self.test_results["模型功能"]
        
        # 打印基础指标
        stream_print("\n模型基础指标:")
        basics_stats = basics["测试统计"]
        stream_print(f"  通过率: {basics_stats['通过率']} ({basics_stats['通过测试']}/{basics_stats['总测试数']})")
        
        # 打印性能指标
        stream_print("\n模型性能指标:")
        perf_stats = performance["测试统计"]
        stream_print(f"  通过率: {perf_stats['通过率']} ({perf_stats['通过测试']}/{perf_stats['总测试数']})")
        
        # 打印功能指标
        stream_print("\n模型功能指标:")
        func_stats = functions["测试统计"]
        stream_print(f"  通过率: {func_stats['通过率']} ({func_stats['通过测试']}/{func_stats['总测试数']})")
        
        # 统计总体情况
        total_passed = (basics_stats['通过测试'] + 
                       perf_stats['通过测试'] + 
                       func_stats['通过测试'])
        total_tests = (basics_stats['总测试数'] + 
                      perf_stats['总测试数'] + 
                      func_stats['总测试数'])
        overall_rate = total_passed / total_tests * 100 if total_tests > 0 else 100
        
        # 功能覆盖统计
        scene_count = functions["场景理解测试"]["具体场景数"]
        target_count = functions["小目标检测测试"]["检测目标数量"]
        
        reasoning = functions["推理能力测试"]
        capability_count = (reasoning["位置理解能力"]["场景数量"] +
                          reasoning["抗干扰能力"]["干扰类型数"] +
                          reasoning["距离理解能力"]["场景数量"] +
                          functions["指令跟随测试"]["领域数量"] +
                          functions["专业知识测试"]["业务域数量"])
        
        stream_print(f"\n推理模型测试总结:")
        stream_print(f"  总体达标率: {overall_rate:.1f}% ({total_passed}/{total_tests})")
        stream_print(f"  功能覆盖度: {scene_count}种场景 + {target_count}种目标 + {capability_count}项能力")
        stream_print(f"  测试耗时: {self.test_results['测试信息']['测试耗时']}")
        
        # 根据通过率给出评价
        if overall_rate >= 95:
            stream_print(f"\n恭喜！{self.model_name} 所有性能指标均达到预期要求！")
            stream_print("模型功能覆盖全面，能力部署完整！")
        elif overall_rate >= 85:
            stream_print(f"\n{self.model_name} 整体表现优秀，绝大部分指标达标！")
            stream_print("模型功能覆盖度高，能力部署良好！")
        else:
            stream_print(f"\n{self.model_name} 部分指标需要进一步优化。")
    
    def save_results_to_file(self, filename: str = None):
        """保存测试结果到文件"""
        if not self.test_results:
            stream_print("请先运行性能测试")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime(TestConfig.TIMESTAMP_FORMAT)
            filename = f"{TestConfig.REPORT_PREFIX}_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, ensure_ascii=False, indent=2)
            stream_print(f"测试结果已成功保存到: {filename}")
        except Exception as e:
            stream_print(f"保存文件时出错: {str(e)}")