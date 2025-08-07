# main.py - 主程序入口
"""
电力场景AI模型测试系统主程序
"""

import os
import sys
from main_controller import PowerAIModelTestController
from config import TestConfig
from utils import stream_print

def setup_output_directory():
    """设置输出目录"""
    os.makedirs(TestConfig.OUTPUT_DIR, exist_ok=True)
    annotation_dir = os.path.join(TestConfig.OUTPUT_DIR, TestConfig.ANNOTATION_DIR)
    os.makedirs(annotation_dir, exist_ok=True)

def main():
    """主程序入口"""
    try:
        # 设置输出目录
        setup_output_directory()
        
        # 创建测试控制器
        controller = PowerAIModelTestController()
        
        # 执行完整测试
        stream_print("正在初始化测试系统...")
        results = controller.run_comprehensive_test()
        
        # 打印最终总结
        controller.print_final_summary()
        
        # 保存结果
        controller.save_results_to_file()
        
        # 提示信息
        stream_print("\n测试完成！可以查看生成的JSON文件获取详细结果。")
        
    except KeyboardInterrupt:
        stream_print("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        stream_print(f"\n测试过程中出现错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()