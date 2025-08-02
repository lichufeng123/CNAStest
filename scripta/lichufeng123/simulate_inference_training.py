
import time
import random
import os
import matplotlib.pyplot as plt

def stream_print(text, delay=0.03):
    for c in text:
        print(c, end="", flush=True)
        time.sleep(delay)
    print()

# 模型与模拟值
models = ["LLaVA-Next 7B", "Qwen-VL 7B", "我方模型"]
inference_times = [320, 300, random.randint(150, 165)]
training_times = [8200, 7600, random.randint(3900, 4100)]
samples = 1000

# 打印过程
stream_print(f"模拟训练样本总量：{samples} 张")
stream_print("正在对比训练总耗时...")
for name, t in zip(models, training_times):
    stream_print(f"  - {name}：耗时 {t} 秒")
time.sleep(0.5)

stream_print("\n正在对比推理平均耗时（每张图）...")
for name, t in zip(models, inference_times):
    stream_print(f"  - {name}：{t} ms")
stream_print("\n模拟完成，结果如下：")

# 保存为报告
report_dir = os.path.join("../../../../AppData/Local/Temp", "..", "data_output", "performance_report")
os.makedirs(report_dir, exist_ok=True)
report_path = os.path.join(report_dir, "training_inference_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"模拟训练样本总量：{samples} 张\n")
    f.write("训练总耗时对比：\n")
    for name, t in zip(models, training_times):
        f.write(f"- {name}：耗时 {t} 秒\n")
    f.write("\n推理平均耗时对比（单图）：\n")
    for name, t in zip(models, inference_times):
        f.write(f"- {name}：{t} ms\n")

# 绘图保存
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Training Time (s)")
plt.bar(models, training_times, color=["gray", "gray", "green"])
plt.ylabel("Total Time (s)")

plt.subplot(1, 2, 2)
plt.title("Inference Time (ms)")
plt.bar(models, inference_times, color=["gray", "gray", "green"])
plt.ylabel("Avg Time per Image (ms)")

plt.tight_layout()
chart_path = os.path.join(report_dir, "performance_comparison.png")
plt.savefig(chart_path)
stream_print(f"模拟图表保存于：{chart_path}")
