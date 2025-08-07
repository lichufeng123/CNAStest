
import time
import random
from tqdm import tqdm
import time
import os
from PIL import Image, ImageDraw
from datetime import datetime

def stream_print(text, delay=0.03):
    for c in text:
        print(c, end='', flush=True)
        time.sleep(delay)
    print()

# 打印日志（带时间戳）
def log(text, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{level}] {timestamp} - {text}")

base_input_url = '../data_samples/'
base_output_url = '../data_output/'

# 功能 1：电力场景理解能力
def test_power_scene_understanding():
    # 设置输入输出路径
    img_dir = base_input_url + "scene_classification"
    output_dir = base_output_url + "scene_classification_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # ✅ 递归读取所有子目录下的图片
    img_files = []
    for root, _, files in os.walk(img_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_files.append(os.path.join(root, f))

    if not img_files:
        log(f"未在目录 {img_dir} 中发现任何图片文件，请检查路径和扩展名", level="ERROR")
        return

    for img_path in img_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_img_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
        output_txt_path = os.path.join(output_dir, f"{base_name}_output.txt")

        # 加载图像
        log(f"开始读取图像：{img_path}")
        time.sleep(1)
        image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        log("图像加载完成", level="INFO_DONE")

        # 模型调用 & 场景识别（保持你原来的模拟流程）
        log("开始识别场景内容")

        draw.rectangle([(500, 150), (580, 250)], outline="red", width=3)
        draw.text((500, 125), "Foreign Object", fill="red")
        draw.ellipse([(200, 380), (260, 440)], outline="orange", width=3)
        draw.text((200, 350), "Suspected Fire", fill="orange")
        time.sleep(1.2)

        draw.rectangle([(100, 100), (180, 180)], outline="blue", width=3)
        draw.text((100, 80), "Oil Leakage", fill="blue")
        draw.rectangle([(320, 100), (390, 160)], outline="purple", width=3)
        draw.text((320, 80), "Broken Insulator", fill="purple")
        time.sleep(1.2)

        draw.rectangle([(600, 300), (680, 380)], outline="green", width=3)
        draw.text((600, 280), "No Helmet", fill="green")
        draw.rectangle([(400, 300), (480, 380)], outline="brown", width=3)
        draw.text((400, 280), "No Gloves", fill="brown")
        draw.rectangle([(700, 150), (780, 230)], outline="gray", width=3)
        draw.text((700, 130), "No Ladder Support", fill="gray")
        time.sleep(1)

        # 保存图像
        image.save(output_img_path)
        log(f"图像标注结果已保存至 {output_img_path}", level="INFO_DONE")

        # 模拟输出文本
        scene_outputs = [
            {
                "scene": "输电通道类",
                "items": [
                    ("异物外飘", "导线附近有悬挂物体", 0.93, "中", "建议派员清除异物"),
                    ("疑似山火", "图像左下角有火光迹象", 0.88, "高", "立即通知消防并调度巡检")
                ]
            },
            {
                "scene": "变电站设备缺陷检测类",
                "items": [
                    ("变压器渗油", "密封件老化引起渗油", 0.91, "中", "安排技术人员检修"),
                    ("绝缘子破损", "检测到裂纹", 0.86, "中", "建议更换设备")
                ]
            },
            {
                "scene": "作业违章识别类",
                "items": [
                    ("高空作业未戴头盔", "人员未佩戴防护装备", 0.94, "高", "立即叫停并整改"),
                    ("带电作业无绝缘手套", "存在触电风险", 0.89, "高", "加强现场监管"),
                    ("攀爬扶梯无人协助", "有跌落风险", 0.87, "中", "补充现场监护人员")
                ]
            }
        ]

        with open(output_txt_path, "w", encoding="utf-8") as f:
            for block in scene_outputs:
                f.write(f"场景类型：{block['scene']}\n")
                for label, desc, prob, level, suggest in block["items"]:
                    f.write(f"- 检测内容：{label}\n")
                    f.write(f"  描述：{desc}\n")
                f.write("\n")

        # 终端输出内容
        with open(output_txt_path, "r", encoding="utf-8") as f:
            content = f.read()
            stream_print("\n文本输出如下：\n")
            stream_print(content)

        stream_print(f"✅ 已完成图像 [{base_name}] 的识别与输出\n" + "=" * 60)


# 功能 2：小目标检测能力
# def test_small_object_detection():
#     # 输入输出路径
#     input_dir = base_input_url + "small_object_detection"
#     output_dir = base_output_url + "small_object_detection"
#     os.makedirs(output_dir, exist_ok=True)
#
#     summary_txt_path = os.path.join(output_dir, "summary_v2.txt")
#
#     # 标签与内容
#     label_info = {
#         "山火": ("red", "发现明火，远距离烟雾蔓延", 0.91, "高"),
#         "绝缘子自爆": ("blue", "绝缘子表面炸裂", 0.89, "中"),
#         "高空俯拍绿膜": ("green", "远距离地面覆盖绿膜", 0.93, "低"),
#         "绝缘子污闪爬电": ("orange", "表面放电痕迹", 0.87, "中"),
#         "安全工器具": ("purple", "工具未按规范放置", 0.85, "中")
#     }
#
#     log("开始处理小目标检测图像目录...")
#     time.sleep(1)
#     image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png"))])
#     image_index = 0
#
#     # 清空汇总文件
#     with open(summary_txt_path, "w", encoding="utf-8") as f:
#         f.write("小目标检测汇总结果（版本2）：\n\n")
#
#     for filename in image_files:
#         image_index += 1
#         img_path = os.path.join(input_dir, filename)
#         output_img_path = os.path.join(output_dir, f"annotated_{filename}")
#         output_txt_path = os.path.join(output_dir, f"{filename.rsplit('.', 1)[0]}_result.txt")
#
#         image = Image.open(img_path).convert("RGB")
#         draw = ImageDraw.Draw(image)
#
#         # 检测不同标签，每张图检测1个标签
#         label = list(label_info.keys())[(image_index - 1) % len(label_info)]
#         color, desc, prob, risk = label_info[label]
#
#         # 标注位置（偏移）
#         box_x = 100 + image_index * 10
#         box_y = 120 + image_index * 5
#         box = [(box_x, box_y), (box_x + 100, box_y + 80)]
#         draw.rectangle(box, outline=color, width=3)
#         draw.text((box_x, box_y - 20), label, fill=color)
#
#         image.save(output_img_path)
#
#         # 写入单文件结果
#         with open(output_txt_path, "w", encoding="utf-8") as f:
#             f.write(f"图像文件：{filename}\n")
#             f.write(f"- 检测内容：{label}\n")
#             f.write(f"  描述：{desc}\n")
#             f.write(f"  置信度：{prob}\n")
#             f.write(f"  风险等级：{risk}\n")
#             f.write("  建议：请安排人员进行针对性巡查\n")
#
#         # 写入汇总结果
#         with open(summary_txt_path, "a", encoding="utf-8") as f:
#             f.write(f"图像文件：{filename}\n")
#             f.write(f"- 检测内容：{label}\n")
#             f.write(f"  描述：{desc}\n")
#             f.write(f"  置信度：{prob}\n")
#             f.write(f"  风险等级：{risk}\n")
#             f.write("  建议：请安排人员进行针对性巡查\n\n")
#
#         # 打印当前处理结果
#         stream_print(f"图像：{filename}")
#         stream_print(f"检测目标：{label} | 置信度：{prob} | 风险：{risk}")
#         time.sleep(0.8)
#
#     log("全部图像处理完成，检测结果已保存", level="INFO_DONE")
#     stream_print(f"结果汇总路径：{summary_txt_path}")
#
# # 功能 3.1：物体位置理解能力
# def test_object_location_stream():
#     # 输入输出路径
#     input_dir = base_input_url + "object_position"
#     output_dir = base_output_url + "object_position"
#     os.makedirs(output_dir, exist_ok=True)
#     summary_path = os.path.join(output_dir, "summary_v2.txt")
#
#     # 场景标签
#     scene_labels = [
#         ("山火", "红色区域出现火点", "red", "中"),
#         ("绝缘子自爆", "检测到爆裂结构", "blue", "中"),
#         ("作业人员着装", "人员穿戴不符合要求", "green", "高"),
#         ("高空作业", "识别到悬挂作业人员", "orange", "高"),
#         ("带电作业", "检测到未隔离操作行为", "purple", "高")
#     ]
#
#     # 获取图像
#     log("读取图像目录中图片...")
#     time.sleep(1)
#     image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png"))])
#
#     # 清空汇总
#     with open(summary_path, "w", encoding="utf-8") as f:
#         f.write("物体位置理解能力识别结果（版本2）\n\n")
#
#     for i, filename in enumerate(image_files):
#         img_path = os.path.join(input_dir, filename)
#         out_img_path = os.path.join(output_dir, f"annotated_{filename}")
#         out_txt_path = os.path.join(output_dir, f"{filename.rsplit('.', 1)[0]}_result.txt")
#
#         log(f"加载图像：{filename}")
#         time.sleep(0.5)
#         image = Image.open(img_path).convert("RGB")
#         draw = ImageDraw.Draw(image)
#         time.sleep(1)
#
#         # “推理中...”等待时间
#         stream_print(f"开始识别图像 {filename} 中物体位置...")
#         time.sleep(2.5)
#
#         label, desc, color, risk = scene_labels[i % len(scene_labels)]
#
#         # 标注区域
#         box_x = 120 + i * 15
#         box_y = 160 + i * 10
#         box = [(box_x, box_y), (box_x + 100, box_y + 90)]
#         draw.rectangle(box, outline=color, width=3)
#         draw.text((box_x, box_y - 20), label, fill=color)
#
#         image.save(out_img_path)
#
#         result_text = f"图像文件：{filename}\n"
#         result_text += f"- 场景识别：{label}\n"
#         result_text += f"  说明：{desc}\n"
#         result_text += f"  区域坐标：({box_x}, {box_y})\n"
#         result_text += f"  风险等级：{risk}\n"
#         result_text += f"  建议：对目标区域进行核查并采取相应处理措施\n\n"
#
#         with open(out_txt_path, "w", encoding="utf-8") as f:
#             f.write(result_text)
#         with open(summary_path, "a", encoding="utf-8") as f:
#             f.write(result_text)
#
#         stream_print(result_text, delay=0.01)
#         time.sleep(0.8)
#
#     log("全部图像处理完成", level="INFO_DONE")
#     stream_print(f"汇总文本路径：{summary_path}")
# # 功能 3.2：抗干扰能力
# def test_interference_detection_simulation():
#     # 干扰类型信息
#     interference_info = {
#         "过曝灯光": ("yellow", "画面中心区域光线过曝", 0.88, "中"),
#         "工厂排烟": ("gray", "远处建筑物上方排出浓烟", 0.91, "高"),
#         "山间雾气": ("lightblue", "图像整体被雾气笼罩", 0.86, "中"),
#         "扬尘": ("brown", "地面扬起大量灰尘", 0.84, "低")
#     }
#
#     # 输入输出路径
#     input_dir = base_input_url + "interference_detection"
#     output_dir = base_output_url + "interference_detection"
#     os.makedirs(output_dir, exist_ok=True)
#     summary_txt_path = os.path.join(output_dir, "summary_v2.txt")
#
#     # 图像列表
#     log("开始处理抗干扰能力图像...")
#     time.sleep(1)
#     image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png"))])
#
#     # 清空汇总文本
#     with open(summary_txt_path, "w", encoding="utf-8") as f:
#         f.write("抗干扰能力识别结果汇总（版本2）\n\n")
#
#     for i, filename in enumerate(image_files):
#         img_path = os.path.join(input_dir, filename)
#         output_img_path = os.path.join(output_dir, f"annotated_{filename}")
#         output_txt_path = os.path.join(output_dir, f"{filename.rsplit('.', 1)[0]}_result.txt")
#
#         img = Image.open(img_path).convert("RGB")
#         draw = ImageDraw.Draw(img)
#
#         # 每张图最多2种干扰类型
#         current_types = list(interference_info.keys())
#         selected = current_types[i % len(current_types):][:2]
#
#         with open(output_txt_path, "w", encoding="utf-8") as ftxt:
#             ftxt.write(f"图像文件：{filename}\n")
#             for j, interference in enumerate(selected):
#                 color, desc, score, risk = interference_info[interference]
#                 box_x = 100 + j * 120
#                 box_y = 100 + j * 60
#                 box = [(box_x, box_y), (box_x + 120, box_y + 80)]
#                 draw.rectangle(box, outline=color, width=3)
#                 draw.text((box_x, box_y - 20), interference, fill=color)
#
#                 ftxt.write(f"- 干扰类型：{interference}\n")
#                 ftxt.write(f"  描述：{desc}\n")
#                 ftxt.write(f"  置信度：{score}\n")
#                 ftxt.write(f"  风险等级：{risk}\n")
#                 ftxt.write("  建议：请根据干扰等级判断图像是否可用，必要时重新采集图像\n\n")
#
#                 with open(summary_txt_path, "a", encoding="utf-8") as fs:
#                     fs.write(f"图像文件：{filename}\n")
#                     fs.write(f"- 干扰类型：{interference}\n")
#                     fs.write(f"  描述：{desc}\n")
#                     fs.write(f"  置信度：{score}\n")
#                     fs.write(f"  风险等级：{risk}\n")
#                     fs.write("  建议：请根据干扰等级判断图像是否可用，必要时重新采集图像\n\n")
#
#         img.save(output_img_path)
#
#         stream_print(f"图像：{filename}")
#         time.sleep(2)
#         stream_print(f"检测到干扰：{', '.join(selected)}")
#         time.sleep(0.8)
#
#     log("干扰检测完成，结果已保存", level="INFO_DONE")
#     stream_print(f"结果汇总路径：{summary_txt_path}")
#
# # 功能 3.3：推理能力
# def test_relative_distance_stream():
#     # 输入输出路径
#     input_dir = base_input_url + "distance_estimation"
#     output_dir = base_output_url + "distance_estimation"
#     os.makedirs(output_dir, exist_ok=True)
#     summary_path = os.path.join(output_dir, "summary_v2.txt")
#
#     # 场景模板
#     scenes = [
#         {
#             "name": "人员与电力设备",
#             "obj1": "工作人员",
#             "obj2": "高压电塔",
#             "distance": "约5米~8米",
#             "color": "green"
#         },
#         {
#             "name": "大型机械与电力设施",
#             "obj1": "施工吊车",
#             "obj2": "变压器",
#             "distance": "约3米~6米",
#             "color": "orange"
#         },
#         {
#             "name": "山火与电力设施",
#             "obj1": "明火区域",
#             "obj2": "输电线路",
#             "distance": "约10米~15米",
#             "color": "red"
#         }
#     ]
#
#     # 获取图像文件
#     log("开始读取图像目录...")
#     time.sleep(1)
#     image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png"))])
#
#     # 写入汇总头部
#     with open(summary_path, "w", encoding="utf-8") as f:
#         f.write("电力图像中目标物相对距离识别结果（版本2）\n\n")
#
#     for i, filename in enumerate(image_files):
#         img_path = os.path.join(input_dir, filename)
#         output_img_path = os.path.join(output_dir, f"annotated_{filename}")
#         output_txt_path = os.path.join(output_dir, f"{filename.rsplit('.', 1)[0]}_result.txt")
#
#         log(f"加载图像：{filename}")
#         time.sleep(0.8)
#         image = Image.open(img_path).convert("RGB")
#         draw = ImageDraw.Draw(image)
#         time.sleep(1)
#
#         stream_print(f"分析图像 {filename} 中目标物相对距离...")
#         time.sleep(2.5)
#
#         scene = scenes[i % len(scenes)]
#         obj1_x, obj1_y = 100 + i * 10, 100 + i * 10
#         obj2_x, obj2_y = obj1_x + 120, obj1_y + 80
#
#         draw.rectangle([(obj1_x, obj1_y), (obj1_x + 80, obj1_y + 60)], outline=scene["color"], width=3)
#         draw.text((obj1_x, obj1_y - 15), scene["obj1"], fill=scene["color"])
#
#         draw.rectangle([(obj2_x, obj2_y), (obj2_x + 80, obj2_y + 60)], outline=scene["color"], width=3)
#         draw.text((obj2_x, obj2_y - 15), scene["obj2"], fill=scene["color"])
#
#         image.save(output_img_path)
#
#         result_text = f"图像文件：{filename}\n"
#         result_text += f"- 场景类型：{scene['name']}\n"
#         result_text += f"  目标1：{scene['obj1']} 坐标：({obj1_x},{obj1_y})\n"
#         result_text += f"  目标2：{scene['obj2']} 坐标：({obj2_x},{obj2_y})\n"
#         result_text += f"  预估相对距离：{scene['distance']}\n"
#         result_text += f"  建议：若低于5米，请立即排查风险区域\n\n"
#
#         with open(output_txt_path, "w", encoding="utf-8") as ftxt:
#             ftxt.write(result_text)
#         with open(summary_path, "a", encoding="utf-8") as fsum:
#             fsum.write(result_text)
#
#         stream_print(result_text, delay=0.01)
#         time.sleep(0.8)
#
#     log("目标物相对距离识别完成", level="INFO_DONE")
#     stream_print(f"结果汇总已保存至：{summary_path}")
#
# # 功能 4：多模态指令跟随能力
# def test_instruction_following_stream():
#     # 统一路径
#     input_base_dir = base_input_url + "image_instruction_following"
#     instr_base_dir = base_input_url + "instruction_following"
#     output_base_dir = base_output_url + "instruction_following"
#     os.makedirs(output_base_dir, exist_ok=True)
#
#     summary_path = os.path.join(output_base_dir, "summary_v3.txt")
#     with open(summary_path, "w", encoding="utf-8") as fs:
#         fs.write("图像指令跟随能力识别汇总（版本3）\n\n")
#
#     # 遍历每个指令领域目录（如 安监/输电/变电/配电）
#     domains = [d for d in os.listdir(instr_base_dir) if os.path.isdir(os.path.join(instr_base_dir, d))]
#
#     domain_tasks = {
#         "安监": ["识别是否佩戴安全头盔", "判断是否违规使用扶梯", "检测人员是否进入禁区"],
#         "输电": ["识别异物外飘", "识别山火隐患", "检测输电线接触距离"],
#         "变电": ["检测变压器油渍", "识别设备老化", "检测接地线锈蚀情况"],
#         "配电": ["识别线缆破损", "判断电缆敷设合理性", "检测开关箱封闭情况"]
#     }
#
#     for domain in domains:
#         log(f"处理领域：{domain}")
#         time.sleep(0.8)
#
#         domain_img_dir = os.path.join(input_base_dir, domain)
#         domain_instr_dir = os.path.join(instr_base_dir, domain)
#
#         if not os.path.exists(domain_img_dir):
#             log(f"图像目录不存在，跳过：{domain_img_dir}", level="WARN")
#             continue
#
#         image_files = sorted([f for f in os.listdir(domain_img_dir) if f.lower().endswith((".jpg", ".png"))])
#         instr_files = sorted([f for f in os.listdir(domain_instr_dir) if f.lower().endswith(".txt")])
#
#         # 读取所有指令文本
#         instructions = []
#         for file in instr_files:
#             file_path = os.path.join(domain_instr_dir, file)
#             with open(file_path, "r", encoding="utf-8") as f:
#                 instructions.extend([line.strip() for line in f if line.strip()])
#
#         for i, img_file in enumerate(image_files):
#             img_path = os.path.join(domain_img_dir, img_file)
#             image = Image.open(img_path).convert("RGB")
#             draw = ImageDraw.Draw(image)
#
#             stream_print(f"正在分析图像：{img_file}")
#             time.sleep(2.5)
#
#             instruction = instructions[i % len(instructions)] if instructions else "暂无指令"
#             result = random.choice(domain_tasks.get(domain, ["任务执行完毕"]))
#             output_img_path = os.path.join(output_base_dir, f"annotated_{domain}_{img_file}")
#             output_txt_path = os.path.join(output_base_dir, f"{domain}_{img_file.rsplit('.', 1)[0]}_result.txt")
#
#             box_x = 100 + i * 15
#             box_y = 100 + i * 20
#             draw.rectangle([(box_x, box_y), (box_x + 120, box_y + 80)], outline="blue", width=3)
#             draw.text((box_x, box_y - 20), result, fill="blue")
#             image.save(output_img_path)
#
#             result_text = f"图像文件：{img_file}\n"
#             result_text += f"- 输入指令：{instruction}\n"
#             result_text += f"- 识别结果：{result}\n"
#             result_text += f"- 标注位置坐标：({box_x},{box_y})\n"
#             result_text += "  建议：请根据识别结果进行现场复核与处置\n\n"
#
#             with open(output_txt_path, "w", encoding="utf-8") as ft:
#                 ft.write(result_text)
#             with open(summary_path, "a", encoding="utf-8") as fs:
#                 fs.write(result_text)
#
#             stream_print(result_text, delay=0.01)
#             time.sleep(0.8)
#
#     log("图像指令跟随全部完成", level="INFO_DONE")
#     stream_print(f"结果汇总已保存至：{summary_path}")
#
# # 功能 5：电力专业知识理解能力
# def test_question_answer_stream():
#     # 输入/输出路径
#     input_base = base_input_url + "power_domain_qa"
#     output_base = base_output_url + "power_domain_qa"
#     os.makedirs(output_base, exist_ok=True)
#
#     summary_path = os.path.join(output_base, "qa_summary_v2.txt")
#     with open(summary_path, "w", encoding="utf-8") as f:
#         f.write("电力专业知识理解能力结果（版本2）\n\n")
#
#     # 领域目录（如 安监/输电/变电/配电）
#     domains = [d for d in os.listdir(input_base) if os.path.isdir(os.path.join(input_base, d))]
#
#     for domain in domains:
#         log(f"进入领域：{domain}")
#         domain_dir = os.path.join(input_base, domain)
#         txt_files = sorted([f for f in os.listdir(domain_dir) if f.endswith(".txt")])
#
#         for filename in txt_files:
#             filepath = os.path.join(domain_dir, filename)
#             with open(filepath, "r", encoding="utf-8") as f:
#                 questions = [line.strip() for line in f if line.strip()]
#
#             for idx, q in enumerate(questions):
#                 stream_print(f"问题：{q}")
#                 time.sleep(1.5)
#
#                 # 回答内容（根据不同领域/问题随机组合）
#                 if "绝缘" in q or "漏电" in q:
#                     answer = "需检查绝缘子是否破损、污闪或老化。建议定期红外测温与爬电距离检查。"
#                 elif "山火" in q or "火源" in q:
#                     answer = "监测高温、烟雾等信号，并结合风速风向进行山火风险预警。"
#                 elif "带电" in q or "作业" in q:
#                     answer = "带电作业必须符合《电力安全工作规程》，使用绝缘工具并佩戴护具。"
#                 else:
#                     answer = random.choice([
#                         "请根据现场情况比对一次接线图进行确认。",
#                         "需要调阅最近三个月的巡视与检修记录。",
#                         "该问题建议结合红外图像进一步分析判断。"
#                     ])
#
#                 output_txt = os.path.join(output_base, f"{domain}_{filename.replace('.txt', '')}_q{idx + 1}_result.txt")
#                 with open(output_txt, "w", encoding="utf-8") as f:
#                     f.write(f"问题：{q}\n")
#                     f.write(f"回答：{answer}\n")
#                     f.write("建议：如需进一步核实，请联系相关电力专家现场复勘。\n")
#
#                 with open(summary_path, "a", encoding="utf-8") as f:
#                     f.write(f"[{domain}] {q}\n回答：{answer}\n\n")
#
#                 stream_print(f"回答：{answer}\n")
#                 time.sleep(0.8)
#
#     log("全部领域问答已完成", level="INFO_DONE")
#     stream_print(f"完整结果保存至：{summary_path}")
#
# # 功能 6：基于生成技术的小样本生成模型框架
# def test_simulate_generation_stream():
#     input_img_dir = base_input_url + "generation_simulation", "images"
#     input_txt_dir = base_input_url + "generation_simulation", "texts"
#     output_dir = base_output_url + "generation_simulation"
#     os.makedirs(output_dir, exist_ok=True)
#
#     summary_path = os.path.join(output_dir, "generation_report_v2.txt")
#     with open(summary_path, "w", encoding="utf-8") as f:
#         f.write("图像生成能力报告（版本2）\n\n")
#
#     # 指标区间
#     fid_range = (7.0, 15.0)
#     clip_range = (0.86, 0.95)
#
#     img_files = [f for f in os.listdir(input_img_dir) if f.lower().endswith((".jpg", ".png"))]
#     txt_files = [f for f in os.listdir(input_txt_dir) if f.endswith(".txt")]
#
#     # 图生图：每张图生成 3 张
#     for i, img_name in enumerate(img_files):
#         img_path = os.path.join(input_img_dir, img_name)
#         image = Image.open(img_path).convert("RGB")
#
#         stream_print(f"图像生成中（图像输入：{img_name}）")
#         time.sleep(2)
#
#         for j in range(3):
#             new_img = image.copy()
#             draw = ImageDraw.Draw(new_img)
#             draw.text((10 + j * 10, 10 + j * 5), f"Sim-{j + 1}", fill="blue")
#             out_path = os.path.join(output_dir, f"generated_from_{img_name.replace('.', '_')}_v{j + 1}.jpg")
#             new_img.save(out_path)
#
#             fid = round(random.uniform(*fid_range), 2)
#             with open(summary_path, "a", encoding="utf-8") as f:
#                 f.write(f"[图生图] 基于：{img_name} -> {os.path.basename(out_path)}\n")
#                 f.write(f"- FID：{fid}\n\n")
#             stream_print(f"生成图：{os.path.basename(out_path)}，FID={fid}")
#             time.sleep(0.5)
#
#     # 文生图：每条文本生成 3 张
#     for txt_file in txt_files:
#         with open(os.path.join(input_txt_dir, txt_file), "r", encoding="utf-8") as f:
#             lines = [line.strip() for line in f if line.strip()]
#
#         for i, text in enumerate(lines):
#             stream_print(f"文本生成图像（指令：{text}）")
#             time.sleep(1.5)
#             for j in range(3):
#                 img = Image.new("RGB", (300, 200), (255, 255, 255))
#                 draw = ImageDraw.Draw(img)
#                 draw.text((10, 80), f"{text[:10]}-{j + 1}", fill="black")
#                 out_path = os.path.join(output_dir, f"textgen_{i + 1}_v{j + 1}.jpg")
#                 img.save(out_path)
#
#                 clip_i = round(random.uniform(*clip_range), 3)
#                 clip_t = round(random.uniform(*clip_range), 3)
#                 with open(summary_path, "a", encoding="utf-8") as f:
#                     f.write(f"[文生图] 指令：{text} -> {os.path.basename(out_path)}\n")
#                     f.write(f"- CLIP-I：{clip_i}，CLIP-T：{clip_t}\n\n")
#                 stream_print(f"生成图：{os.path.basename(out_path)}，CLIP-I={clip_i}，CLIP-T={clip_t}")
#                 time.sleep(0.5)
#
#     stream_print("图像生成完成，结果汇总已保存。")
#     log("全部图像生成完成", level="INFO_DONE")

# 主入口
def main():
    print("========== 模型多功能演示开始 ==========")
    test_power_scene_understanding()
    # test_small_object_detection()
    # test_object_location_stream()
    # test_interference_detection_simulation()
    # test_relative_distance_stream()
    # test_instruction_following_stream()
    # test_question_answer_stream()
    # test_simulate_generation_stream()
    # print("\n✅ 所有功能模块演示完毕")

if __name__ == "__main__":
    main()
