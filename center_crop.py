import os
import threading
from PIL import Image
import argparse

def center_crop_image(image_path):
    """
    对单个图像进行中心裁剪
    确保裁剪后的图像长宽都是8的倍数
    """
    try:
        # 打开图像
        img = Image.open(image_path)
        width, height = img.size
        
        # 如果图像长宽已经是8的倍数，则直接返回
        if width % 8 == 0 and height % 8 == 0:
            return
        
        # 计算可以裁剪到8的倍数的最大区域
        new_width = width - (width % 8)
        new_height = height - (height % 8)
        
        # 计算裁剪的起始坐标
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        
        # 裁剪图像
        cropped_img = img.crop((left, top, right, bottom))
        
        # 保存图像（覆盖原文件）
        cropped_img.save(image_path)
        print(f"成功裁剪图像: {image_path}")
    except Exception as e:
        print(f"处理 {image_path} 时出错: {e}")

def center_crop_folder(folder_path):
    """
    多线程递归处理文件夹中的所有图像
    """
    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    
    # 收集所有图像文件（包括子文件夹）
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            if os.path.isfile(filepath) and os.path.splitext(filename)[1].lower() in image_extensions:
                image_paths.append(filepath)
    
    # 创建线程
    threads = []
    for image_path in image_paths:
        thread = threading.Thread(target=center_crop_image, args=(image_path,))
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    print("所有图像处理完成")

def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description='中心裁剪图像文件夹中的图像')
    
    # 添加文件夹路径参数
    parser.add_argument('folder', type=str, help='要处理的图像文件夹路径')
    
    # 解析参数
    args = parser.parse_args()
    
    # 检查文件夹是否存在
    if not os.path.isdir(args.folder):
        print("无效的文件夹路径")
        return
    
    # 执行中心裁剪
    center_crop_folder(args.folder)

if __name__ == "__main__":
    main()
