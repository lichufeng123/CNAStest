import os
import random
import shutil
from pathlib import Path
import argparse

def get_image_files(directory):
    """获取目录中的所有图片文件"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    return image_files

def copy_random_images(source_dir, target_dir, num_images=5):
    """从源目录随机复制指定数量的图片到目标目录"""
    
    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"错误：源目录 '{source_dir}' 不存在！")
        return
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    print(f"目标目录：{target_dir}")
    
    # 获取所有图片文件
    image_files = get_image_files(source_dir)
    
    if not image_files:
        print(f"在 '{source_dir}' 中没有找到图片文件！")
        return
    
    print(f"在源目录中找到 {len(image_files)} 张图片")
    
    # 随机选择图片
    num_to_copy = min(num_images, len(image_files))
    selected_images = random.sample(image_files, num_to_copy)
    
    print(f"将随机复制 {num_to_copy} 张图片...")
    
    # 复制图片
    for i, image_path in enumerate(selected_images, 1):
        image_name = os.path.basename(image_path)
        target_path = os.path.join(target_dir, image_name)
        
        # 如果目标文件已存在，添加数字后缀
        counter = 1
        original_target_path = target_path
        while os.path.exists(target_path):
            name, ext = os.path.splitext(original_target_path)
            target_path = f"{name}_{counter}{ext}"
            counter += 1
        
        try:
            shutil.copy2(image_path, target_path)
            print(f"  {i}. 复制: {image_name} -> {os.path.basename(target_path)}")
        except Exception as e:
            print(f"  错误：复制 {image_name} 时出错: {e}")
    
    print(f"\n完成！已将 {num_to_copy} 张图片复制到 '{target_dir}'")

def main():
    parser = argparse.ArgumentParser(description='从指定目录随机复制图片到test_images文件夹')
    parser.add_argument('source_dir', help='源图片目录路径')
    parser.add_argument('-n', '--num', type=int, default=5, help='要复制的图片数量 (默认: 5)')
    parser.add_argument('-t', '--target', default='test_images', help='目标目录名称 (默认: test_images)')
    
    args = parser.parse_args()
    
    print(f"源目录: {args.source_dir}")
    print(f"要复制的图片数量: {args.num}")
    print("-" * 50)
    
    copy_random_images(args.source_dir, args.target, args.num)

if __name__ == "__main__":
    main()
