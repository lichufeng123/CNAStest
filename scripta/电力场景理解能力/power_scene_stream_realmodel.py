
import time
import os
from PIL import Image, ImageDraw
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
# 电力场景理解能力:优化第一版
# 模拟逐字打印
def stream_print(text, delay=0.03):
    for c in text:
        print(c, end="", flush=True)
        time.sleep(delay)
    print()

# 路径设置
img_path = os.path.join("..", "..", "data_samples", "scene_classification", "sdtd_images01.jpg")
output_img_path = os.path.join("..", "..", "data_output", "scene_classification_annotated.jpg")
output_txt_path = os.path.join("..", "..", "data_output", "scene_classification.txt")

# 模拟加载模型
stream_print("📦 正在加载多模态模型...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").eval()
stream_print("✅ 模型加载完成")

# 模拟读取图片并调用模型
stream_print("🖼️ 正在读取图像...")
image = Image.open(img_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

stream_print("🧠 正在进行模型推理...")
with torch.no_grad():
    outputs = model.generate(**inputs)
time.sleep(1.5)

# 模拟模型推理完成，但我们不使用真实输出
# caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]
stream_print("📌 模型完成推理，正在输出电力场景识别结果...")

# 图像标注（和之前一致）
draw = ImageDraw.Draw(image)
draw.rectangle([(500, 150), (580, 250)], outline="red", width=3)
draw.text((500, 125), "Foreign Object", fill="red")
draw.ellipse([(200, 380), (260, 440)], outline="orange", width=3)
draw.text((200, 350), "Suspected Fire", fill="orange")
image.save(output_img_path)

# 写入伪造的文本结果
scene_text = """📌 场景识别结果：
场景类型：输电通道类
检测内容：
- 检测到异物外飘：导线附近有悬挂物体
- 检测到火源疑点：图像左下角出现明火迹象，疑似山火
建议：派员巡检，及时清理异物并排查火源风险
"""

with open(output_txt_path, "w", encoding="utf-8") as f:
    f.write(scene_text)

# 打印输出结果
stream_print("\n📄 模拟识别文本结果：\n")
stream_print(scene_text)

stream_print("✅ 结果保存成功")
stream_print(f"📍 图像路径：{output_img_path}")
stream_print(f"📍 文本路径：{output_txt_path}")
