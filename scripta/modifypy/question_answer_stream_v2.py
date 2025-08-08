
import os
import time
from datetime import datetime
import random

def log(msg, level="INFO"):
    print(f"[{level}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

def stream_print(text, delay=0.03):
    for c in text:
        print(c, end='', flush=True)
        time.sleep(delay)
    print()

# 输入/输出路径
input_base = os.path.join("..", "..", "data_samples", "power_domain_qa")
output_base = os.path.join("..", "..", "data_output", "power_domain_qa")
os.makedirs(output_base, exist_ok=True)

summary_path = os.path.join(output_base, "qa_summary_v2.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("电力专业知识理解能力模拟结果（版本2）\n\n")

# 领域目录（如 安监/输电/变电/配电）
domains = [d for d in os.listdir(input_base) if os.path.isdir(os.path.join(input_base, d))]

for domain in domains:
    log(f"进入领域：{domain}")
    domain_dir = os.path.join(input_base, domain)
    txt_files = sorted([f for f in os.listdir(domain_dir) if f.endswith(".txt")])

    for filename in txt_files:
        filepath = os.path.join(domain_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]

        for idx, q in enumerate(questions):
            stream_print(f"问题：{q}")
            time.sleep(1.5)

            # 模拟回答内容（根据不同领域/问题随机组合）
            if "绝缘" in q or "漏电" in q:
                answer = "需检查绝缘子是否破损、污闪或老化。建议定期红外测温与爬电距离检查。"
            elif "山火" in q or "火源" in q:
                answer = "监测高温、烟雾等信号，并结合风速风向进行山火风险预警。"
            elif "带电" in q or "作业" in q:
                answer = "带电作业必须符合《电力安全工作规程》，使用绝缘工具并佩戴护具。"
            else:
                answer = random.choice([
                    "请根据现场情况比对一次接线图进行确认。",
                    "需要调阅最近三个月的巡视与检修记录。",
                    "该问题建议结合红外图像进一步分析判断。"
                ])

            output_txt = os.path.join(output_base, f"{domain}_{filename.replace('.txt', '')}_q{idx+1}_result.txt")
            with open(output_txt, "w", encoding="utf-8") as f:
                f.write(f"问题：{q}\n")
                f.write(f"回答：{answer}\n")
                f.write("建议：如需进一步核实，请联系相关电力专家现场复勘。\n")

            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(f"[{domain}] {q}\n回答：{answer}\n\n")

            stream_print(f"回答：{answer}\n")
            time.sleep(0.8)

log("全部领域问答模拟已完成", level="INFO_DONE")
stream_print(f"完整结果保存至：{summary_path}")
