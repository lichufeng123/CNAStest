
import time
import queue
import threading
import logging
import statistics
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# 假设导入原有的模块
# from llmperf import common_metrics
# from llmperf.openai_chat_completions_client import OpenAIChatCompletionsClient

# 为了演示，我们模拟common_metrics
class CommonMetrics:
    ERROR_CODE = "error_code"
    ERROR_MSG = "error_msg"
    ERROR_TIME = "error_time"  # 新增：错误发生时间
    ERROR_TIMESTAMP = "error_timestamp"  # 新增：错误时间戳
    INTER_TOKEN_LAT = "inter_token_lat"
    TTFT = "ttft"
    E2E_LAT = "e2e_lat"
    REQ_OUTPUT_THROUGHPUT = "req_output_throughput"
    NUM_INPUT_TOKENS = "num_input_tokens"
    NUM_OUTPUT_TOKENS = "num_output_tokens"
    NUM_TOTAL_TOKENS = "num_total_tokens"

common_metrics = CommonMetrics()

def format_timestamp(timestamp: float) -> str:
    """格式化时间戳为可读格式"""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def calculate_percentile_stats(values: List[float], metric_name: str) -> Dict[str, float]:
    """计算百分位数统计，保持原有格式"""
    if not values:
        return {}
    
    values = [v for v in values if v is not None and v > 0]  # 过滤无效值
    if not values:
        return {}
    
    # 转换为numpy数组进行计算
    np_values = np.array(values)
    
    stats = {
        'p25': np.percentile(np_values, 25),
        'p50': np.percentile(np_values, 50),
        'p75': np.percentile(np_values, 75),
        'p90': np.percentile(np_values, 90),
        'p95': np.percentile(np_values, 95),
        'p99': np.percentile(np_values, 99),
        'mean': np.mean(np_values),
        'min': np.min(np_values),
        'max': np.max(np_values),
        'stddev': np.std(np_values)
    }
    
    return stats

def print_metric_stats(stats: Dict[str, float], metric_name: str):
    """按原有格式打印指标统计"""
    if not stats:
        print(f"{metric_name}")
        print("    No valid data")
        return
    
    print(f"{metric_name}")
    for stat_name, value in stats.items():
        print(f"    {stat_name} = {value}")

class PerformanceBenchmarkRunner:
    def __init__(self, 
                 max_num_completed_requests: int,
                 num_concurrent_requests: int,
                 test_timeout_s: int,
                 request_launcher):
        
        self.max_num_completed_requests = max_num_completed_requests
        self.num_concurrent_requests = num_concurrent_requests
        self.test_timeout_s = test_timeout_s
        self.request_launcher = request_launcher
        
        # 原有的结果存储
        self.completed_requests = []
        self.error_requests = []
        self.completed_requests_lock = threading.Lock()
        
        # 任务队列
        self.task_queue = queue.Queue()
        
        # 记录测试开始时间
        self.test_start_time = None
        
        # 错误统计 - 新增
        self.error_timeline = []  # 按时间顺序记录的错误
        
        # 初始化任务队列
        self._populate_task_queue()
    
    def _populate_task_queue(self):
        """填充任务队列"""
        for i in range(self.max_num_completed_requests):
            task = {
                'task_id': i,
                'request_config': self._create_request_config(i),
                'created_at': time.time()
            }
            self.task_queue.put(task)
    
    def _create_request_config(self, task_id: int):
        """创建请求配置（需要根据实际情况调整）"""
        return {
            'task_id': task_id,
            'prompt': f"Test prompt for task {task_id}",
            'max_tokens': 100,
            'temperature': 0.7
        }
    
    def _create_error_metrics(self, error_msg: str, processing_time: float = 0.0, 
                             task_id: Optional[int] = None, 
                             error_type: str = "REQUEST_FAILED") -> Dict:
        """创建错误指标，保持原有格式并添加时间信息"""
        error_timestamp = time.time()
        error_time_str = format_timestamp(error_timestamp)
        
        error_metrics = {
            common_metrics.ERROR_CODE: error_type,
            common_metrics.ERROR_MSG: error_msg,
            common_metrics.ERROR_TIME: error_time_str,  # 可读时间格式
            common_metrics.ERROR_TIMESTAMP: error_timestamp,  # Unix时间戳
            common_metrics.INTER_TOKEN_LAT: 0.0,
            common_metrics.TTFT: 0.0,
            common_metrics.E2E_LAT: processing_time,
            common_metrics.REQ_OUTPUT_THROUGHPUT: 0.0,
            common_metrics.NUM_INPUT_TOKENS: 0,
            common_metrics.NUM_OUTPUT_TOKENS: 0,
            common_metrics.NUM_TOTAL_TOKENS: 0,
        }
        
        # 如果有task_id，添加到错误信息中
        if task_id is not None:
            error_metrics['task_id'] = task_id
        
        # 记录到错误时间线
        error_event = {
            'timestamp': error_timestamp,
            'time_str': error_time_str,
            'task_id': task_id,
            'error_type': error_type,
            'error_msg': error_msg,
            'processing_time': processing_time
        }
        self.error_timeline.append(error_event)
        
        return error_metrics
    
    def launch_request(self, thread_index: int):
        """性能测试请求处理函数 - 无重试机制，增强错误时间记录"""
        start_time = time.monotonic()
        
        while True:
            # 检查是否完成
            with self.completed_requests_lock:
                total_completed = len(self.completed_requests) + len(self.error_requests)
                if total_completed >= self.max_num_completed_requests:
                    print(f"线程 {thread_index} 检测到目标已达成，退出")
                    break
            
            # 检查超时
            elapsed_time = time.monotonic() - start_time
            if elapsed_time >= self.test_timeout_s:
                print(f"线程 {thread_index} 超时退出")
                break
            
            # 获取任务
            try:
                timeout_remaining = min(self.test_timeout_s - elapsed_time, 2.0)
                task = self.task_queue.get(timeout=timeout_remaining)
            except queue.Empty:
                # 队列为空，检查是否真的完成
                with self.completed_requests_lock:
                    total_completed = len(self.completed_requests) + len(self.error_requests)
                    if total_completed >= self.max_num_completed_requests:
                        break
                continue
            
            # 处理任务（无重试）
            task_id = task['task_id']
            request_config = task['request_config']
            task_start_time = time.time()
            
            try:
                # 直接调用原有的请求处理逻辑，不重试
                request_metrics, gen_text, request_config_returned = self.request_launcher.llm_request(request_config)
                
                # 检查结果并分类
                if request_metrics is None or request_metrics.get(common_metrics.ERROR_CODE) is not None:
                    # 错误请求 - 增强错误信息记录
                    error_msg = request_metrics.get(common_metrics.ERROR_MSG, "Unknown error") if request_metrics else "Request failed"
                    error_code = request_metrics.get(common_metrics.ERROR_CODE, "REQUEST_FAILED") if request_metrics else "REQUEST_FAILED"
                    
                    processing_time = time.time() - task_start_time
                    
                    if request_metrics:
                        # 如果有返回的错误指标，补充时间信息
                        error_timestamp = time.time()
                        request_metrics[common_metrics.ERROR_TIME] = format_timestamp(error_timestamp)
                        request_metrics[common_metrics.ERROR_TIMESTAMP] = error_timestamp
                        request_metrics['task_id'] = task_id
                        
                        # 记录到错误时间线
                        error_event = {
                            'timestamp': error_timestamp,
                            'time_str': format_timestamp(error_timestamp),
                            'task_id': task_id,
                            'error_type': error_code,
                            'error_msg': error_msg,
                            'processing_time': processing_time,
                            'thread_index': thread_index
                        }
                        self.error_timeline.append(error_event)
                        
                        error_metrics = request_metrics
                    else:
                        # 创建新的错误指标
                        error_metrics = self._create_error_metrics(
                            error_msg, processing_time, task_id, "REQUEST_FAILED"
                        )
                    
                    with self.completed_requests_lock:
                        self.error_requests.append(error_metrics)
                        current_time = format_timestamp(time.time())
                        print(f"[{current_time}] 任务 {task_id} 失败 (线程{thread_index}): {error_msg}")
                else:
                    # 成功请求
                    with self.completed_requests_lock:
                        self.completed_requests.append(request_metrics)
                        current_time = format_timestamp(time.time())
                        print(f"[{current_time}] 任务 {task_id} 成功完成 (线程{thread_index})")
                
            except Exception as e:
                # 异常处理 - 详细记录异常时间和信息
                error_msg = f"Exception: {str(e)}"
                processing_time = time.time() - task_start_time
                
                print(f"[{format_timestamp(time.time())}] 任务 {task_id} 处理异常 (线程{thread_index}): {error_msg}")
                
                error_metrics = self._create_error_metrics(
                    error_msg, processing_time, task_id, "EXCEPTION"
                )
                
                with self.completed_requests_lock:
                    self.error_requests.append(error_metrics)
    
    def run_benchmark(self):
        """运行性能基准测试"""
        print(f"开始性能基准测试：{self.max_num_completed_requests}个请求，并发数{self.num_concurrent_requests}")
        print("注意：性能测试模式，不进行重试")
        
        # 记录测试开始时间
        self.test_start_time = time.time()
        start_time_str = format_timestamp(self.test_start_time)
        print(f"测试开始时间: {start_time_str}")
        
        # 启动工作线程
        threads = []
        for i in range(self.num_concurrent_requests):
            thread = threading.Thread(target=self.launch_request, args=(i,))
            thread.daemon = True
            threads.append(thread)
            thread.start()
        
        # 监控进度
        last_report_time = time.time()
        while True:
            with self.completed_requests_lock:
                total_completed = len(self.completed_requests) + len(self.error_requests)
                success_count = len(self.completed_requests)
                
                if total_completed >= self.max_num_completed_requests:
                    break
                
                # 每10秒报告一次进度
                current_time = time.time()
                if current_time - last_report_time >= 10:
                    elapsed = current_time - self.test_start_time
                    success_rate = success_count / total_completed if total_completed > 0 else 0
                    current_time_str = format_timestamp(current_time)
                    print(f"[{current_time_str}] 进度: {total_completed}/{self.max_num_completed_requests} "
                          f"({total_completed/self.max_num_completed_requests:.1%}), "
                          f"成功率: {success_rate:.2%}, "
                          f"用时: {elapsed:.1f}s")
                    last_report_time = current_time
            
            time.sleep(1)
        
        # 等待所有线程完成
        for thread in threads:
            thread.join(timeout=10)
        
        # 输出统计信息
        self._print_performance_stats()
        
        return self.completed_requests, self.error_requests
    
    def _print_error_timeline(self):
        """打印错误时间线"""
        if not self.error_timeline:
            return
        
        print("\n" + "="*60)
        print("错误时间线分析")
        print("="*60)
        
        # 按时间排序
        sorted_errors = sorted(self.error_timeline, key=lambda x: x['timestamp'])
        
        print(f"总计错误数: {len(sorted_errors)}")
        
        # 错误类型统计
        error_type_stats = {}
        for error in sorted_errors:
            error_type = error['error_type']
            error_type_stats[error_type] = error_type_stats.get(error_type, 0) + 1
        
        print("\n错误类型分布:")
        for error_type, count in error_type_stats.items():
            percentage = count / len(sorted_errors) * 100
            print(f"   {error_type}: {count} ({percentage:.1f}%)")
        
        # 显示最近的错误（最多10个）
        print(f"\n最近发生的错误 (最多显示10个):")
        recent_errors = sorted_errors[-10:] if len(sorted_errors) > 10 else sorted_errors
        
        for error in recent_errors:
            task_info = f"Task{error['task_id']}" if error['task_id'] is not None else "Unknown"
            thread_info = f"Thread{error.get('thread_index', '?')}"
            processing_time = error['processing_time']
            print(f"   [{error['time_str']}] {task_info} ({thread_info}) - "
                  f"{error['error_type']}: {error['error_msg'][:50]}... "
                  f"(耗时: {processing_time:.3f}s)")
        
        # 错误时间分布分析
        if len(sorted_errors) > 1:
            print(f"\n错误时间分布分析:")
            first_error_time = sorted_errors[0]['timestamp']
            last_error_time = sorted_errors[-1]['timestamp']
            error_time_span = last_error_time - first_error_time
            
            print(f"   首个错误时间: {format_timestamp(first_error_time)}")
            print(f"   最后错误时间: {format_timestamp(last_error_time)}")
            print(f"   错误时间跨度: {error_time_span:.2f}秒")
            
            if error_time_span > 0:
                error_rate = len(sorted_errors) / error_time_span
                print(f"   平均错误频率: {error_rate:.2f} 错误/秒")
    
    def _print_performance_stats(self):
        """输出性能统计信息，保持原有格式并添加错误时间线"""
        print("\n" + "="*60)
        print("性能测试统计报告")
        print("="*60)
        
        success_count = len(self.completed_requests)
        error_count = len(self.error_requests)
        total_requests = success_count + error_count
        total_test_time = time.time() - self.test_start_time
        
        # 先打印错误时间线
        if error_count > 0:
            self._print_error_timeline()
        
        if success_count == 0:
            print("没有成功的请求可供分析")
            print(f"Number Of Errored Requests: {error_count}")
            print(f"Number Of Completed Requests: {success_count}")
            return
        
        # 提取各种指标数据
        ttft_values = []           # Time To First Token (TTFT)
        tbt_values = []            # Time Between Tokens (TBT) = inter_token_latency
        tpot_values = []           # Time Per Output Token (TPOT)
        throughput_values = []     # Throughput
        e2e_lat_values = []        # End-to-end latency
        input_tokens = []          # Input tokens
        output_tokens = []         # Output tokens
        
        for req in self.completed_requests:
            # TTFT - Time To First Token
            ttft = req.get(common_metrics.TTFT, 0.0)
            if ttft > 0:
                ttft_values.append(ttft)
            
            # TBT - Time Between Tokens (就是inter_token_latency)
            tbt = req.get(common_metrics.INTER_TOKEN_LAT, 0.0)
            if tbt > 0:
                tbt_values.append(tbt)
            
            # TPOT - Time Per Output Token (需要计算)
            e2e_lat = req.get(common_metrics.E2E_LAT, 0.0)
            output_token_count = req.get(common_metrics.NUM_OUTPUT_TOKENS, 0)
            if e2e_lat > 0 and output_token_count > 0:
                tpot = e2e_lat / output_token_count
                tpot_values.append(tpot)
            
            # Throughput - 请求输出吞吐量
            throughput = req.get(common_metrics.REQ_OUTPUT_THROUGHPUT, 0.0)
            if throughput > 0:
                throughput_values.append(throughput)
            
            # 其他指标
            if e2e_lat > 0:
                e2e_lat_values.append(e2e_lat)
            
            input_tokens.append(req.get(common_metrics.NUM_INPUT_TOKENS, 0))
            output_tokens.append(req.get(common_metrics.NUM_OUTPUT_TOKENS, 0))
        
        # 按原有格式打印各项统计
        print("\n性能指标统计:")
        
        # 1. TBT (原来的 inter_token_latency_s)
        tbt_stats = calculate_percentile_stats(tbt_values, "inter_token_latency_s")
        print_metric_stats(tbt_stats, "inter_token_latency_s")
        
        # 2. TTFT (原来的 ttft_s)
        ttft_stats = calculate_percentile_stats(ttft_values, "ttft_s")
        print_metric_stats(ttft_stats, "ttft_s")
        
        # 3. 端到端延迟 (原有的)
        e2e_stats = calculate_percentile_stats(e2e_lat_values, "end_to_end_latency_s")
        print_metric_stats(e2e_stats, "end_to_end_latency_s")
        
        # 4. Throughput (原来的 request_output_throughput_token_per_s)
        throughput_stats = calculate_percentile_stats(throughput_values, "request_output_throughput_token_per_s")
        print_metric_stats(throughput_stats, "request_output_throughput_token_per_s")
        
        # 5. 新增：TPOT (Time Per Output Token)
        tpot_stats = calculate_percentile_stats(tpot_values, "time_per_output_token_s")
        print_metric_stats(tpot_stats, "time_per_output_token_s")
        
        # 6. Token 统计 (原有的)
        input_stats = calculate_percentile_stats(input_tokens, "number_input_tokens")
        print_metric_stats(input_stats, "number_input_tokens")
        
        output_stats = calculate_percentile_stats(output_tokens, "number_output_tokens")
        print_metric_stats(output_stats, "number_output_tokens")
        
        # 7. 总体统计 (原有的格式)
        total_output_tokens = sum(output_tokens)
        
        print(f"Number Of Errored Requests: {error_count}")
        
        # 计算总体输出吞吐量 (tokens/s)
        overall_throughput = total_output_tokens / total_test_time if total_test_time > 0 else 0
        print(f"Overall Output Throughput: {overall_throughput}")
        
        print(f"Number Of Completed Requests: {success_count}")
        
        # 计算每分钟完成请求数
        requests_per_minute = (success_count / total_test_time) * 60 if total_test_time > 0 else 0
        print(f"Completed Requests Per Minute: {requests_per_minute}")
        
        # 8. 性能测试总结
        print(f"\n性能测试总结:")
        print(f"测试开始时间: {format_timestamp(self.test_start_time)}")
        print(f"测试结束时间: {format_timestamp(time.time())}")
        print(f"总请求数: {total_requests}")
        print(f"成功率: {success_count/total_requests:.2%}")
        print(f"失败率: {error_count/total_requests:.2%}")
        print(f"测试时长: {total_test_time:.2f}s")
        print(f"平均QPS: {total_requests/total_test_time:.2f}")
        print(f"成功QPS: {success_count/total_test_time:.2f}")
        
        # 如果有错误，显示错误统计
        if error_count > 0:
            print(f"\n错误统计摘要:")
            error_types = {}
            for req in self.error_requests:
                error_code = req.get(common_metrics.ERROR_CODE, "UNKNOWN")
                error_types[error_code] = error_types.get(error_code, 0) + 1
            
            for error_type, count in error_types.items():
                print(f"   {error_type}: {count} ({count/total_requests:.2%})")
        
        print("="*60)
        print("性能测试完成")
        print("="*60)
    
    def export_error_timeline(self, filename: str = None) -> str:
        """导出错误时间线到文件"""
        if not filename:
            filename = f"error_timeline_{int(time.time())}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("错误时间线报告\n")
            f.write("="*60 + "\n")
            f.write(f"生成时间: {format_timestamp(time.time())}\n")
            f.write(f"总错误数: {len(self.error_timeline)}\n\n")
            
            sorted_errors = sorted(self.error_timeline, key=lambda x: x['timestamp'])
            
            for i, error in enumerate(sorted_errors, 1):
                task_info = f"Task{error['task_id']}" if error['task_id'] is not None else "Unknown"
                thread_info = f"Thread{error.get('thread_index', '?')}"
                f.write(f"{i:3d}. [{error['time_str']}] {task_info} ({thread_info})\n")
                f.write(f"     类型: {error['error_type']}\n")
                f.write(f"     消息: {error['error_msg']}\n")
                f.write(f"     耗时: {error['processing_time']:.3f}s\n\n")
        
        print(f"错误时间线已导出到: {filename}")
        return filename

# 使用示例：替换原有的多线程处理
def run_performance_benchmark(max_num_completed_requests: int,
                            num_concurrent_requests: int,
                            test_timeout_s: int,
                            request_launcher):
    """运行性能基准测试，无重试机制，增强错误时间记录"""
    
    runner = PerformanceBenchmarkRunner(
        max_num_completed_requests=max_num_completed_requests,
        num_concurrent_requests=num_concurrent_requests,
        test_timeout_s=test_timeout_s,
        request_launcher=request_launcher
    )
    
    completed, errors = runner.run_benchmark()
    
    # 可选：导出错误时间线
    if errors:
        runner.export_error_timeline()
    
    return completed, errors

# 原有的launch_request函数替换版本 - 增强错误时间记录
def launch_request(thread_index, 
                  max_num_completed_requests, 
                  num_concurrent_requests,
                  test_timeout_s,
                  request_launcher,
                  completed_requests,
                  error_requests,
                  completed_requests_lock,
                  task_queue,
                  pbar=None):
    """增强版本的launch_request函数，确保精确计数并记录错误时间"""
    
    start_time = time.monotonic()
    
    while True:
        # 检查是否完成
        with completed_requests_lock:
            total_completed = len(completed_requests) + len(error_requests)
            if total_completed >= max_num_completed_requests:
                print(f"线程 {thread_index} 检测到目标已达成，退出")
                break
        
        # 检查超时
        elapsed_time = time.monotonic() - start_time
        if elapsed_time >= test_timeout_s:
            print(f"线程 {thread_index} 超时退出")
            break
        
        # 获取任务
        try:
            timeout_remaining = min(test_timeout_s - elapsed_time, 2.0)
            task = task_queue.get(timeout=timeout_remaining)
        except queue.Empty:
            # 队列为空，检查是否真的完成
            with completed_requests_lock:
                total_completed = len(completed_requests) + len(error_requests)
                if total_completed >= max_num_completed_requests:
                    break
            continue
        
        # 处理任务（无重试）
        task_id = task.get('task_id', 'Unknown')
        task_start_time = time.time()
        
        try:
            # 直接调用原有的请求处理逻辑
            request_metrics, gen_text, request_config_returned = request_launcher.llm_request(task)
            
            # 检查结果并分类
            if request_metrics is None or request_metrics.get(common_metrics.ERROR_CODE) is not None:
                # 错误请求 - 增强错误信息和时间记录
                error_msg = request_metrics.get(common_metrics.ERROR_MSG, "Unknown error") if request_metrics else "Request failed"
                error_code = request_metrics.get(common_metrics.ERROR_CODE, "REQUEST_FAILED") if request_metrics else "REQUEST_FAILED"
                processing_time = time.time() - task_start_time
                error_timestamp = time.time()
                
                if request_metrics:
                    # 补充时间信息
                    request_metrics[common_metrics.ERROR_TIME] = format_timestamp(error_timestamp)
                    request_metrics[common_metrics.ERROR_TIMESTAMP] = error_timestamp
                    request_metrics['task_id'] = task_id
                    request_metrics['thread_index'] = thread_index
                    error_metrics = request_metrics
                else:
                    # 创建新的错误指标
                    error_metrics = {
                        common_metrics.ERROR_CODE: error_code,
                        common_metrics.ERROR_MSG: error_msg,
                        common_metrics.ERROR_TIME: format_timestamp(error_timestamp),
                        common_metrics.ERROR_TIMESTAMP: error_timestamp,
                        common_metrics.INTER_TOKEN_LAT: 0.0,
                        common_metrics.TTFT: 0.0,
                        common_metrics.E2E_LAT: processing_time,
                        common_metrics.REQ_OUTPUT_THROUGHPUT: 0.0,
                        common_metrics.NUM_INPUT_TOKENS: 0,
                        common_metrics.NUM_OUTPUT_TOKENS: 0,
                        common_metrics.NUM_TOTAL_TOKENS: 0,
                        'task_id': task_id,
                        'thread_index': thread_index
                    }
                
                with completed_requests_lock:
                    error_requests.append(error_metrics)
                    if pbar:
                        pbar.update(1)
                    print(f"[{format_timestamp(error_timestamp)}] 任务{task_id}失败(线程{thread_index}): {error_msg}")
            else:
                # 成功请求
                with completed_requests_lock:
                    completed_requests.append(request_metrics)
                    if pbar:
                        pbar.update(1)
                    current_time = format_timestamp(time.time())
                    print(f"[{current_time}] 任务{task_id}成功完成(线程{thread_index})")
            
        except Exception as e:
            # 异常处理 - 详细记录异常时间
            error_msg = f"Exception: {str(e)}"
            processing_time = time.time() - task_start_time
            error_timestamp = time.time()
            
            print(f"[{format_timestamp(error_timestamp)}] 任务{task_id}处理异常(线程{thread_index}): {error_msg}")
            
            error_metrics = {
                common_metrics.ERROR_CODE: "EXCEPTION",
                common_metrics.ERROR_MSG: error_msg,
                common_metrics.ERROR_TIME: format_timestamp(error_timestamp),
                common_metrics.ERROR_TIMESTAMP: error_timestamp,
                common_metrics.INTER_TOKEN_LAT: 0.0,
                common_metrics.TTFT: 0.0,
                common_metrics.E2E_LAT: processing_time,
                common_metrics.REQ_OUTPUT_THROUGHPUT: 0.0,
                common_metrics.NUM_INPUT_TOKENS: 0,
                common_metrics.NUM_OUTPUT_TOKENS: 0,
                common_metrics.NUM_TOTAL_TOKENS: 0,
                'task_id': task_id,
                'thread_index': thread_index
            }
            
            with completed_requests_lock:
                error_requests.append(error_metrics)
                if pbar:
                    pbar.update(1)
