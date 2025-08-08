
import os
import time
import json
from datetime import datetime
import random

def log(msg, level="INFO"):
    print(f"[{level}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

def stream_print(text, delay=0.02):
    for c in text:
        print(c, end='', flush=True)
        time.sleep(delay)
    print()

# 输入输出路径
instr_base_dir = os.path.join("..", "..", "data_samples", "instruction_following")
output_base_dir = os.path.join("..", "..", "data_output", "instruction_following")
os.makedirs(output_base_dir, exist_ok=True)
summary_path = os.path.join(output_base_dir, "summary_v6.txt")

with open(summary_path, "w", encoding="utf-8") as fs:
    fs.write("结构化文本问答模拟结果（关键词匹配版）\n\n")

# 问题关键词映射
question_mapping = {
    '变压器': '变压器常见故障有哪些',
    '巡检': '变电站巡检注意事项有哪些',
    '工器具': '常见安全工器具及用途',
    '开关柜': '低压开关柜的结构组成',
    '绝缘子': '绝缘子都有哪些类型',
    '输电场景': '输电场景都有哪些',
    '配电设备': '配电设备有哪些常见类型',
    "输电场景": "输电场景都有哪些",
    "变压器": "变压器常见故障有哪些",
    "开关柜": "低压开关柜的结构组成",
    "作业违章":"作业违章的常见情形有哪些"
}

# 结构化回答内容
structured_answers = {
    '低压开关柜的结构组成': {'组成': {'断路器': '短路保护', '柜体': '防护结构', '母线': '主电路连接', '测控模块': '遥信遥测'}},
    '变压器常见故障有哪些': {'故障': ['渗油', '局放', '过热'], '监测方式': ['红外', '局放监测', '油色谱分析']},
    '变电站巡检注意事项有哪些': {'内容': ['检查接地', '设备温度', '异音', '局放水平'], '周期': '每周/每月/季度'},
    '常见安全工器具及用途': {'工器具': {'安全帽': '防护头部', '绝缘手套': '防止触电', '绝缘杆': '隔离高压'}},
    '绝缘子都有哪些类型': {   '绝缘子类型': {   '按安装方式': ['悬式', '支柱'],
                                  '按材料': ['瓷', '玻璃', '复合'],
                                  '按电压等级': ['低压', '高压']}},
    '输电场景都有哪些': ['- 异物外飘', '- 山火检测', '- 通道树障', '- 杆塔倾斜'],
    '配电设备有哪些常见类型': ['- 配电变压器', '- 开关柜', '- 电缆分支箱', '- 无功补偿装置'],
        "绝缘子类型": {
            "按安装方式分类": ["悬式绝缘子", "支柱绝缘子"],
            "按材料分类": ["瓷绝缘子", "玻璃绝缘子", "复合绝缘子"]
        },
        "补充说明": {
            "悬式绝缘子": {
                "类型": ["盘形", "棒形"],
                "应用": "高压输电线路"
            }
        }
    ,
    "输电场景都有哪些": [
        "- 山火检测",
        "- 异物外飘",
        "- 杆塔倾斜",
        "- 通道树障",
        "- 风偏覆冰"
    ],
    "变压器常见故障有哪些": {
        "故障类型": ["渗油", "局部放电", "绕组变形", "绝缘老化"],
        "监测方法": ["红外热成像", "油色谱分析", "局放监测"]
    },
    "低压开关柜的结构组成": {
        "结构组成": {
            "母线系统": "连接主电路",
            "断路器单元": "用于过载保护",
            "接地装置": "保护设备安全",
            "仪表测控模块": "实现测量与遥控",
            "防护外壳": "防尘防水"
        }
    }
}


domains = ["安监", "输电", "变电", "配电"]

for domain in domains:
    instr_dir = os.path.join(instr_base_dir, domain)
    if not os.path.exists(instr_dir):
        log(f"未找到指令目录：{instr_dir}", level="WARN")
        continue

    for fname in os.listdir(instr_dir):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(instr_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]

        for i, q in enumerate(questions):
            std_q = None
            for keyword, std_form in question_mapping.items():
                if keyword in q:
                    std_q = std_form
                    break

            if std_q and std_q in structured_answers:
                ans = structured_answers[std_q]
                if isinstance(ans, dict):
                    answer = json.dumps(ans, indent=2, ensure_ascii=False)
                elif isinstance(ans, list):
                    answer = "\n".join(ans)
                else:
                    answer = str(ans)
            else:
                answer = f"模拟回答：{random.choice(['请提供更多上下文信息', '问题已记录，将进一步处理', '正在分析该问题'])}"

            filename = f"{domain}_q{i+1}.txt"
            out_path = os.path.join(output_base_dir, filename)

            with open(out_path, "w", encoding="utf-8") as fout:
                fout.write(f"问题：{q}\n")
                fout.write("回答：\n")
                fout.write(answer + "\n")

            with open(summary_path, "a", encoding="utf-8") as fs:
                fs.write(f"问题：{q}\n")
                fs.write("回答：\n")
                fs.write(answer + "\n\n")

            stream_print(f"问题：{q}")
            stream_print("回答：")
            stream_print(answer)
            stream_print("-" * 60)

log("结构化问答模拟完成（v6，关键词映射）", level="INFO_DONE")
stream_print(f"汇总文件已保存：{summary_path}")
