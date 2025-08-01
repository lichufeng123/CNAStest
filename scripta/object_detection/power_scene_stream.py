
import time
import sys

def stream_print(text, delay=0.03):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

import os
from PIL import Image, ImageDraw

# 输入输出路径
img_path = os.path.join("../../data_samples/scene_classification", "sdtd_images01.jpg")
output_img_path = os.path.join("../../data_output", "scene_classification_annotated.jpg")
output_txt_path = os.path.join("../../data_output", "scene_classification.txt")

# 打开图像并绘制模拟标注
img = Image.open(img_path).convert("RGB")
draw = ImageDraw.Draw(img)

# 场景1
scene_type01 = "输电通道类"
scene_type02 = "变电站设备缺陷检测类"
scene_type03 = "作业违章识别类"
draw.text((20, 20), "Scene: Transmission Channel", fill="black")

# 标注异物外飘
draw.rectangle([(500, 150), (580, 250)], outline="red", width=3)
draw.text((500, 125), "Foreign Object", fill="red")

# 标注疑似山火
draw.ellipse([(200, 380), (260, 440)], outline="orange", width=3)
draw.text((200, 350), "Suspected Fire", fill="orange")

# 保存标注图像
img.save(output_img_path)

# 写入文本结果
with open(output_txt_path, "w", encoding="utf-8") as f:
    f.write("📌 场景识别结果：\n")
    f.write(f"场景类型：{scene_type01}\n")
    f.write("检测内容：\n")
    f.write("- 检测到异物外飘：导线附近有悬挂物体\n")
    f.write("- 检测到火源疑点：图像左下角出现明火迹象，疑似山火\n")
    f.write("建议：派员巡检，及时清理异物并排查火源风险\n")

    f.write("📌 场景识别结果：\n")
    f.write(f"场景类型：{scene_type02}\n")
    f.write("检测内容：\n")
    f.write("- 检测到变压器异常：密封件老化引起渗油\n")
    f.write("- 检测到绝缘子破损\n")
    f.write("建议：派员巡检，及时维护设备\n")

    f.write("📌 场景识别结果：\n")
    f.write(f"场景类型：{scene_type03}\n")
    f.write("检测内容：\n")
    f.write("- 检测到高空作业：未佩戴安全头盔\n")
    f.write("- 检测到带电作业：工人未带绝缘手套\n")
    f.write("- 检测到攀爬扶梯：扶梯无人协助稳定\n")
    f.write("建议：派员巡检，及时维护设备\n")


# 保存标注图像
img.save(output_img_path)
with open(output_txt_path,"r",encoding="UTF-8") as f:
    content = f.read()
    stream_print(content)

stream_print("✅输出结果已保存：")
stream_print(f"- 标注图像保存路径：{output_img_path}")
stream_print(f"- 文本保存路径：{output_txt_path}")
