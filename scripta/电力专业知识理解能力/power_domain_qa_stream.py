
import os
import time
import random
# 电力专业知识理解能力
def stream_print(text, delay=0.03):
    for c in text:
        print(c, end='', flush=True)
        time.sleep(delay)
    print()

# 回答模板（可自定义扩展）
qa_responses = {
    "输电": [
        "导线的弧垂随气温升高而增大，风速越大，导线晃动越明显，需确保安全裕度。",
        "通过图像识别技术可发现导线附近悬挂异物或风中飘浮物，即可判断存在外飘物隐患。"
    ],
    "变电": [
        "过载运行会导致温升加剧、绝缘老化、甚至发生击穿故障，影响设备寿命。",
        "常见设备缺陷包括：套管裂纹、导线连接松动、接地不良等。"
    ],
    "配电": [
        "环网柜可提升供电可靠性，实现单点故障快速切除和负荷转移。",
        "故障如电缆击穿、接头发热，可通过红外测温和电缆测试仪进行检测并处理。"
    ],
    "安监": [
        "安全交底应包括作业内容、风险点、防控措施、应急措施等，确保作业人员理解到位。",
        "如未佩戴安全绳、未戴安全帽、操作不规范等均属常见高空作业违章行为。"
    ]
}

domains = ["输电", "变电", "配电", "安监"]
base_question_dir = os.path.join("..", "..", "data_samples", "power_domain_qa")
base_output_dir = os.path.join("..", "..", "data_samples", "power_domain_qa")
os.makedirs(base_output_dir, exist_ok=True)

for domain in domains:
    domain_path = os.path.join(base_question_dir, domain)
    question_files = sorted([f for f in os.listdir(domain_path) if f.endswith(".txt")])
    for q_file in question_files:
        q_path = os.path.join(domain_path, q_file)
        with open(q_path, "r", encoding="utf-8") as f:
            question = f.read().strip()

        stream_print(f"📄 领域：{domain}")
        stream_print(f"❓ 问题：{question}")
        stream_print("🧠 正在回答中...")
        time.sleep(0.5)

        answer = random.choice(qa_responses[domain])
        for line in answer.split("\n"):
            stream_print(line)
            time.sleep(0.5)

        stream_print("✅ 回答完成，建议结合现场场景和规范进一步核实。\n")

        output_path = os.path.join(base_output_dir, f"{domain}_{q_file.replace('.txt','')}_result.txt")
        with open(output_path, "w", encoding="utf-8") as out:
            out.write(f"📄 领域：{domain}\n")
            out.write(f"❓ 问题：{question}\n")
            out.write(answer + "\n")
            out.write("✅ 回答完成，建议结合现场场景和规范进一步核实。\n")
