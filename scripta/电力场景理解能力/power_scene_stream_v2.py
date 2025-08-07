import time
import os
from PIL import Image, ImageDraw
from datetime import datetime
# 模拟逐字打印
def stream_print(text, delay=0.03):
    for c in text:
        print(c, end='', flush=True)
        time.sleep(delay)
    print()

# 打印日志（带时间戳）
def log(text, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{level}] {timestamp} - {text}")

