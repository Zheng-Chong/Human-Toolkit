import os
from PIL import Image
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import numpy as np
from SCHP import get_palette, dataset_settings

def rgb_to_label(rgb_image, num_classes):
    """将RGB格式的分割图像转换回标签图像"""
    if isinstance(rgb_image, Image.Image):
        rgb_array = np.array(rgb_image)
    else:
        rgb_array = rgb_image
        
    # 获取调色板映射
    palette = get_palette(num_classes)
    color_to_label = {}
    for label in range(num_classes):
        color = tuple(palette[label*3:(label+1)*3])
        color_to_label[color] = label
    
    # 创建输出图像
    h, w = rgb_array.shape[:2]
    label_image = np.zeros((h, w), dtype=np.uint8)
    
    # 逐像素转换
    for y in range(h):
        for x in range(w):
            color = tuple(rgb_array[y, x])
            label = color_to_label.get(color, 0)
            label_image[y, x] = label
            
    return Image.fromarray(label_image)

def convert_mask_to_label(file_path):
    """将RGB格式的SCHP掩码转换为标签图像并保存"""
    try:
        img = Image.open(file_path)
        if img.mode == 'RGB':
            # 根据文件路径判断是lip还是atr
            if 'schp_lip' in file_path:
                num_classes = dataset_settings['lip']['num_classes']
            else:  # schp_atr
                num_classes = dataset_settings['atr']['num_classes']
            
            # 转换为标签图像
            label_img = rgb_to_label(img, num_classes)
            label_img.save(file_path)
            
        img.close()
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")

def process_directory(base_dir):
    """处理目录下的所有SCHP掩码文件"""
    # 获取schp_lip和schp_atr目录
    lip_dir = os.path.join(base_dir, 'annotations', 'schp_lip')
    atr_dir = os.path.join(base_dir, 'annotations', 'schp_atr')
    
    all_files = []
    
    # 收集所有.png文件
    for directory in [lip_dir, atr_dir]:
        if os.path.exists(directory):
            all_files.extend([
                str(f) for f in Path(directory).rglob('*.png')
            ])
    
    print(f"找到 {len(all_files)} 个文件需要处理")
    
    # 使用线程池并行处理文件
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(
            executor.map(convert_mask_to_label, all_files),
            total=len(all_files),
            desc="转换掩码"
        ))

if __name__ == "__main__":
    # 设置基础目录路径
    base_directory = "Datasets/0-RawDatasets/Intern/Bean-Raw/upper"
    process_directory(base_directory) 