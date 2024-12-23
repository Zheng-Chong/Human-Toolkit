import PIL.Image as Image
from SCHP import LIP_MAPPING
from typing import Union, Tuple, List, Optional
import numpy as np
import cv2
from PIL import ImageOps
from sklearn.cluster import KMeans 
import os
import shutil
from tqdm import tqdm  # 添加这个导入
import json
import itertools
from math import sqrt


def gray_similarity(
    image1: Union[str, Image.Image], 
    image2: Union[str, Image.Image], 
    mask1: Union[str, Image.Image], 
    mask2: Union[str, Image.Image]
) -> float:
    """给定两张图片和两张mask，计算两张图片在 mask 区域内的平均灰度值的相似度

    Args:
        image1 (str | PIL.Image.Image): 第一张图片或图片路径
        image2 (str | PIL.Image.Image): 第二张图片或图片路径
        mask1 (str | PIL.Image.Image): 第一张 mask 或 mask 路径
        mask2 (str | PIL.Image.Image): 第二张 mask 或 mask 路径

    Returns:
        float: 两张图片在 mask 区域内的平均灰度值的相似度
    """
    # 转换输入为 PIL Image
    def convert_to_image(input_image):
        if isinstance(input_image, str):
            return Image.open(input_image)
        elif isinstance(input_image, Image.Image):
            return input_image
        else:
            raise TypeError("输入必须是图像路径或 PIL Image")

    # 转换图像和 mask
    image1 = convert_to_image(image1)
    image2 = convert_to_image(image2)
    mask1 = convert_to_image(mask1).convert('L')
    mask2 = convert_to_image(mask2).convert('L')
    
    # 转换为灰度图像
    gray1 = image1.convert('L')
    gray2 = image2.convert('L')
    
    # 转换为 NumPy 数组
    gray1_array = np.array(gray1)
    gray2_array = np.array(gray2)
    mask1_array = np.array(mask1)
    mask2_array = np.array(mask2)
    
    # 计算两张图片在 mask 区域内的平均灰度值
    gray1_mean = np.mean(gray1_array[mask1_array == 255])
    gray2_mean = np.mean(gray2_array[mask2_array == 255])
    
    # 计算两张图片在 mask 区域内的平均灰度值的相似度（绝对差值归一化）
    similarity = 1 - np.abs(gray1_mean - gray2_mean) / 255
    
    return similarity

# 计算两张图片的相似度(直方图)
def match_images(image1, image2, red_bins=4, green_bins=4, blue_bins=4, threshold=0.8):
    """
    Compare two images and return their similarity score using Bhattacharyya coefficient.

    :param image1: Path to the first image file or PIL Image object
    :param image2: Path to the second image file or PIL Image object
    :param red_bins: Number of bins for the red channel
    :param green_bins: Number of bins for the green channel
    :param blue_bins: Number of bins for the blue channel
    :param threshold: Threshold to determine if the images are similar
    :return: Bhattacharyya coefficient and similarity decision (True/False)
    """
    def calculate_histogram(image, red_bins, green_bins, blue_bins):
        """
        Calculate the normalized histogram for the given image.

        :param image: PIL Image object
        :param red_bins: Number of bins for the red channel
        :param green_bins: Number of bins for the green channel
        :param blue_bins: Number of bins for the blue channel
        :return: Normalized histogram as a 1D NumPy array
        """
        # Convert image to RGB if not already
        image = image.convert("RGB")
        
        # Get image dimensions
        width, height = image.size

        # Get pixel data as NumPy array
        pixels = np.array(image)

        # Initialize histogram
        histogram = np.zeros((red_bins, green_bins, blue_bins), dtype=float)

        # Define bin size
        bin_size = 256 / np.array([red_bins, green_bins, blue_bins])

        # Populate histogram
        for row in range(height):
            for col in range(width):
                r, g, b = pixels[row, col]
                red_idx = min(int(r / bin_size[0]), red_bins - 1)
                green_idx = min(int(g / bin_size[1]), green_bins - 1)
                blue_idx = min(int(b / bin_size[2]), blue_bins - 1)
                histogram[red_idx, green_idx, blue_idx] += 1

        # Normalize histogram
        histogram /= histogram.sum()

        return histogram.flatten()

    def bhattacharyya_coefficient(hist1, hist2):
        """
        Calculate the Bhattacharyya coefficient between two histograms.

        :param hist1: First normalized histogram as a NumPy array
        :param hist2: Second normalized histogram as a NumPy array
        :return: Bhattacharyya coefficient (0 to 1)
        """
        return np.sum(np.sqrt(hist1 * hist2))

    # Handle image inputs
    if isinstance(image1, str):
        image1 = Image.open(image1).resize((48, 64))
    if isinstance(image2, str):
        image2 = Image.open(image2).resize((48, 64))

    # Calculate histograms
    hist1 = calculate_histogram(image1, red_bins, green_bins, blue_bins)
    hist2 = calculate_histogram(image2, red_bins, green_bins, blue_bins)

    # Calculate similarity
    similarity = bhattacharyya_coefficient(hist1, hist2)

    return similarity, similarity >= threshold

# 根据 mask 裁剪图像到最大正方形区域
def crop_to_max_square_region(image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """
    根据 mask 裁剪图像到最大正方形区域
    
    :param image: 原始图像
    :param mask: 遮罩图像
    :return: 裁剪后的图像和遮罩
    """
    # 转换为灰度并获取非零区域
    mask_array = np.array(mask.convert('L'))
    rows = np.any(mask_array > 0, axis=1)
    cols = np.any(mask_array > 0, axis=0)
    
    # 获取非零区域边界
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # 计算最大正方形区域
    width = cmax - cmin + 1
    height = rmax - rmin + 1
    size = max(width, height)
    
    # 计算中心点和裁剪区域
    center_x = (cmin + cmax) // 2
    center_y = (rmin + rmax) // 2
    x1 = max(0, center_x - size // 2)
    y1 = max(0, center_y - size // 2)
    x2 = min(image.width, x1 + size)
    y2 = min(image.height, y1 + size)
    
    # 裁剪图像和 mask
    cropped_image = image.crop((x1, y1, x2, y2))
    cropped_mask = mask.crop((x1, y1, x2, y2))
    
    # 将 mask 反转，并合成到图像上
    cropped_mask = ImageOps.invert(cropped_mask.convert('L'))
    croped_masked_image = Image.composite(Image.new('RGB', cropped_image.size, 'white'), cropped_image, cropped_mask)
    
    return croped_masked_image, cropped_mask

# 判断是否是外套和上衣叠穿
def wear_upper_with_outer(
    person_lip_path_or_image: Union[str, Image.Image, np.ndarray]
):
    # 打开图片
    person_lip_image = person_lip_path_or_image
    if isinstance(person_lip_path_or_image, str):
        person_lip_image = Image.open(person_lip_path_or_image)
    if isinstance(person_lip_path_or_image, Image.Image):
        person_lip_image = person_lip_path_or_image
        assert person_lip_image.mode == 'P', f'[Wear Upper With Outer] person_lip_image 必须是 P 模式'
        person_lip_image = person_lip_image.getdata()
        
    # 上衣类别
    upper_index = [LIP_MAPPING[_] for _ in ['Upper-clothes', 'Dress', 'Jumpsuits']]
    # 外套类别
    outer_index = LIP_MAPPING['Coat']

    # 判断是否存在上衣和外套
    image_index = np.unique(person_lip_image)
    if outer_index in image_index and any([_ in image_index for _ in upper_index]):
        return True
    else:
        return False

# 获取目标服装区域的 Mask
def get_target_cloth_mask(
    person_lip_path_or_image: Union[str, Image.Image, np.ndarray],
    target_cloth_type: str
) -> Image.Image:
    """
    获取目标服装区域的 Mask
    
    参数:
        person_lip_path_or_image (str | Image.Image | np.ndarray): 输入的 person_lip 文件路径或图像数据。
        target_cloth_type (str): 目标服装类型，可选 'upper' 'lower' 或 'full'。
    
    返回:
        Image.Image: 目标服装区域的 Mask。
    """
    # 打开图片
    if isinstance(person_lip_path_or_image, str):
        person_lip_image = Image.open(person_lip_path_or_image)
    elif isinstance(person_lip_path_or_image, Image.Image):
        person_lip_image = person_lip_path_or_image
    elif isinstance(person_lip_path_or_image, np.ndarray):
        person_lip_image = Image.fromarray(person_lip_path_or_image)
    else:
        raise TypeError("输入必须是文件路径、PIL图像或NumPy数组")

    # 确保图像是调色板模式
    assert person_lip_image.mode == 'P', f'[Target Cloth Mask] person_lip_image 必须是 P 模式'

    # 获取图像数据
    lip_data = np.array(person_lip_image)

    # 定义目标类别索引
    if target_cloth_type == 'upper':
        target_indices = [LIP_MAPPING[_] for _ in ['Upper-clothes', 'Coat']]
    elif target_cloth_type == 'lower':
        target_indices = [LIP_MAPPING[_] for _ in ['Pants', 'Skirt']]
    elif target_cloth_type == 'full':
        target_indices = [LIP_MAPPING[_] for _ in ['Dress', 'Jumpsuits', 'Coat', 'Upper-clothes', "Skirt", "Pants"]]
    else:
        raise ValueError("target_cloth_type 必须是 'upper', 'lower' 或 'full'")

    # 创建 Mask
    mask = np.zeros_like(lip_data, dtype=np.uint8)
    for idx in target_indices:
        mask[lip_data == idx] = 255

    # 转换为 PIL 图像
    return Image.fromarray(mask, mode='L')

# 判断图像是否清晰
def is_image_clear(image_path, threshold=100):
    """
    判断图像是否为清晰高质量原图。

    参数:
        image_path (str): 图像文件路径。
        threshold (float): Laplacian变异数的阈值，值越高要求越清晰。默认值为100。

    返回:
        bool: True表示图像清晰度足够高，False表示图像模糊。
        float: 图像的Laplacian变异数值。
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图像，请检查路径是否正确。")
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算Laplacian变异数
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # 根据阈值判断是否清晰
    return laplacian_var > threshold, laplacian_var

# 利用LIP 判断是否所有服装的区域占了图像的 ratio 过大
def is_cloth_area_too_large(person_lip_path_or_image: Union[str, Image.Image, np.ndarray], threshold=0.5):
    """
    利用LIP判断是所有服装的区域占了图像的ratio过大。

    参数:
        person_lip_path_or_image (str | Image.Image | np.ndarray): 输入的 person_lip 文件路径或图像数据。
        threshold (float): 服装区域占图像的最大比例阈值，默认为0.5（50%）。

    返回:
        bool: 如果服装区域超过阈值，返回 True，否则返回 False。
    """
    # 打开图片
    if isinstance(person_lip_path_or_image, str):
        person_lip_image = Image.open(person_lip_path_or_image)
    elif isinstance(person_lip_path_or_image, Image.Image):
        person_lip_image = person_lip_path_or_image
    elif isinstance(person_lip_path_or_image, np.ndarray):
        person_lip_image = Image.fromarray(person_lip_path_or_image)
    else:
        raise TypeError("输入必须是文件路径、PIL图像或NumPy数组")

    # 确保图像是调色板模式
    assert person_lip_image.mode == 'P', f'[Is Cloth Area Too Large] person_lip_image 必须是 P 模式'

    # 获取图像数据
    lip_data = np.array(person_lip_image)

    # 定义所有服装类别的索引
    cloth_indices = [
        LIP_MAPPING[_] for _ in [
            'Upper-clothes', 'Dress', 'Coat', 'Jumpsuits',  # 上半身服装
            'Pants', 'Skirt'  # 下半身服装
        ]
    ]

    # 计算总像素数和服装像素数
    total_pixels = lip_data.size
    cloth_pixels = np.sum(np.isin(lip_data, cloth_indices))

    # 计算服装区域比例
    cloth_ratio = cloth_pixels / total_pixels
    
    # 判断是否超过阈值
    return cloth_ratio


# 判断一个 mask 文件读取后是否有大于3个边上有白色（Mask 区域）
def has_more_than_three_white_edges(mask_path_or_image: Union[str, Image.Image, np.ndarray]) -> bool:
    """
    判断一个 mask 文件读取后是否有大于3个边上有白色（Mask 区域）。

    参数:
        mask_path_or_image (str | Image.Image | np.ndarray): Mask 文件的路径或图像数据。

    返回:
        bool: 如果有大于3个边上有白色，返回 True，否则返回 False。
    """
    # 读取图片
    if isinstance(mask_path_or_image, str):
        mask_image = Image.open(mask_path_or_image).convert('L')
    elif isinstance(mask_path_or_image, Image.Image):
        mask_image = mask_path_or_image.convert('L')
    elif isinstance(mask_path_or_image, np.ndarray):
        if len(mask_path_or_image.shape) == 3:
            mask_image = cv2.cvtColor(mask_path_or_image, cv2.COLOR_BGR2GRAY)
        else:
            mask_image = mask_path_or_image
    else:
        raise TypeError("mask_path_or_image 必须是 str, Image.Image 或 numpy.ndarray 型")

    # 转换为二值图像
    _, binary_mask = cv2.threshold(np.array(mask_image), 127, 255, cv2.THRESH_BINARY)

    height, width = binary_mask.shape
    edges_with_white = 0

    # 检查上边缘
    if np.any(binary_mask[0, :] == 255):
        edges_with_white += 1

    # 检查下边缘
    if np.any(binary_mask[-1, :] == 255):
        edges_with_white += 1

    # 检查左边缘
    if np.any(binary_mask[:, 0] == 255):
        edges_with_white += 1

    # 检查右边缘
    if np.any(binary_mask[:, -1] == 255):
        edges_with_white += 1

    return edges_with_white >= 3


def are_dominant_colors_similar(image1, image2, threshold=50):
    """
    判断两个图像的主色调是否一致。

    参数:
        image1 (str | PIL.Image.Image | np.ndarray): 第一张图像的路径或图像。
        image2 (str | PIL.Image.Image | np.ndarray): 第二张图像的路径或图像。
        threshold (float): 颜色距离的阈值，默认为50。
        mask2 (str | PIL.Image.Image | np.ndarray, optional): 第二张图像的掩膜。默认为 None。
        threshold (float): 颜色距离的阈值，默认为50。

    返回:
        dict: 包括是否一致 (bool) 和每个图像的均值颜色及距离。
    """ 
    
    def extract_dominant_color(image, k=1):
        # 将图片转换到 HSV 色彩空间
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 将图片展平为 (n, 3) 的数据
        pixels = image_hsv.reshape(-1, 3)
        # 去掉纯白色背景
        pixels = pixels[~np.all(pixels == 255, axis=1)]
        # 使用 K-Means 聚类提取主色调
        kmeans = KMeans(n_clusters=k, random_state=42).fit(pixels)
        dominant_color = kmeans.cluster_centers_[0]  # 主色调
        return dominant_color

    def color_distance(color1, color2):
        """
        计算两个颜色之间的欧几里得距离。
        
        参数:
            color1 (tuple): 第一个颜色的BGR值 (B, G, R)。
            color2 (tuple): 第二个颜色的BGR值 (B, G, R)。
        
        返回:
            float: 欧几里得距离。
        """
        return np.linalg.norm(np.array(color1) - np.array(color2))

    # 转换图像为 OpenCV 格式
    def convert_to_cv2_image(img):
        if isinstance(img, str):
            return cv2.imread(img)
        elif isinstance(img, Image.Image):
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        elif isinstance(img, np.ndarray):
            return img
        else:
            raise TypeError("图像必须是文件路径、PIL图像或NumPy数组")

    # 读取图像
    image1 = convert_to_cv2_image(image1)
    image2 = convert_to_cv2_image(image2)
    if image1 is None or image2 is None:
        raise ValueError("无法读取图像，请检查输入是否正确。")
    
    color1 = extract_dominant_color(image1)
    color2 = extract_dominant_color(image2)
    dist = color_distance(color1, color2)
    is_similar = dist <= threshold
    return {
        "is_similar": is_similar,
        "color1": color1,
        "color2": color2,
        "distance": dist
    }

def merge_datasets(
    source_dataset_path: str, 
    target_dataset_path: str, 
    categories: Optional[List[str]] = None
):
    """
    将源数据集的所有图像文件复制到目标数据集，保持原有的子目录结构。

    参数:
        source_dataset_path (str): 源数据集的根目录路径
        target_dataset_path (str): 目标数据集的根目录路径
        categories (List[str], optional): 要合并的子类别列表，默认为 None 表示合并所有子类别

    返回:
        dict: 包含合并的文件数量和详细信息的字典
    """
    # 如果没有指定具体类别，则获取源数据集的所有子目录
    if categories is None:
        categories = [d for d in os.listdir(source_dataset_path) 
                      if os.path.isdir(os.path.join(source_dataset_path, d))]

    merge_summary = {
        "total_files_merged": 0,
        "categories_merged": {},
        "errors": []
    }

    # 首先计算总文件数，用于进度条
    total_files = 0
    for category in categories:
        source_category_path = os.path.join(source_dataset_path, category)
        if os.path.exists(source_category_path):
            for root, dirs, files in os.walk(source_category_path):
                total_files += len(files)

    # 使用 tqdm 创建总体进度条
    with tqdm(total=total_files, desc="合并数据集", unit="文件") as pbar:
        for category in categories:
            source_category_path = os.path.join(source_dataset_path, category)
            target_category_path = os.path.join(target_dataset_path, category)

            # 检查源类别路径是否存在
            if not os.path.exists(source_category_path):
                merge_summary["errors"].append(f"源类别路径不存在: {source_category_path}")
                continue

            # 遍历源类别目录下的所有文件和子目录
            for root, dirs, files in os.walk(source_category_path):
                # 计算相对路径
                relative_path = os.path.relpath(root, source_category_path)
                
                # 创建对应的目标路径
                target_root = os.path.join(target_category_path, relative_path)
                os.makedirs(target_root, exist_ok=True)

                # 复制所有文件
                for file in files:
                    source_file_path = os.path.join(root, file)
                    target_file_path = os.path.join(target_root, file)

                    try:
                        # 如果目标文件已存在，跳过
                        if not os.path.exists(target_file_path):
                            shutil.copy2(source_file_path, target_file_path)
                            merge_summary["total_files_merged"] += 1
                            pbar.update(1)  # 更新进度条
                    except Exception as e:
                        merge_summary["errors"].append(f"复制文件 {source_file_path} 失败: {str(e)}")
                        pbar.update(1)  # 即使出错也要更新进度条

            # 记录每个类别合并的文件数
            merge_summary["categories_merged"][category] = {
                "files_merged": merge_summary["total_files_merged"]
            }

    print(merge_summary)
    return merge_summary

def generate_paired_dataset(
    dataset_path: str, 
    output_path: Optional[str] = None, 
    pair_mode: str = 'cross'
) -> dict:
    """
    自动检测数据集类别并生成配对的 JSONL 数据文件。

    参数:
        dataset_path (str): 数据集根目录路径
        output_path (str, optional): 输出 JSONL 文件路径，默认为 None 时自动生成
        pair_mode (str, optional): 配对模式，可选 'cross'（跨图像配对）或 'same'（同 ID 配对）

    返回:
        dict: 包含所有类别配对信息的摘要字典
    """
    # 检查数据集目录是否存在
    if not os.path.exists(dataset_path):
        raise ValueError(f"数据集路径不存在: {dataset_path}")

    # 自动检测类别
    categories = [
        d for d in os.listdir(dataset_path) 
        if os.path.isdir(os.path.join(dataset_path, d)) 
        and all(os.path.exists(os.path.join(dataset_path, d, subdir)) for subdir in ['person', 'cloth'])
    ]
    if not categories:
        raise ValueError(f"在 {dataset_path} 中未找到符合要求的类别目录")

    # 总体摘要
    total_summary = {
        category: {
            'person_pairs': 0,
            'cloth_pairs': 0,
        } for category in categories
    }
    paired_data = []

    # 遍历每个类别
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        person_dir = os.path.join(category_path, 'person')
        cloth_dir = os.path.join(category_path, 'cloth')

        # 提取图片 ID（去掉括号和扩展名）
        def extract_id(filename):
            return filename.split('(')[0].split('.')[0]

        # 收集 person 和 cloth 图片信息
        person_images = {}
        cloth_images = {}
        for filename in os.listdir(person_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_id = extract_id(filename)
                if img_id not in person_images:
                    person_images[img_id] = [os.path.join(category, 'person', filename)]
                else:
                    person_images[img_id].append(os.path.join(category, 'person', filename))
        for filename in os.listdir(cloth_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_id = extract_id(filename)
                if img_id not in cloth_images:
                    cloth_images[img_id] = [os.path.join(category, 'cloth', filename)]
                else:
                    cloth_images[img_id].append(os.path.join(category, 'cloth', filename))

        # 生成配对数据
        pair_summary = {
            'total_pairs': 0,
            'cloth_pairs': 0,
            'person_pairs': 0,
        }
        
        # Cloth 和 Person 配对
        for cloth_id, cloth_paths in cloth_images.items():
            if cloth_id not in person_images:
                continue
            for cloth_path in cloth_paths:
                for person_paths in person_images[cloth_id]:
                    paired_data.append({
                        'person': person_paths,
                        'cloth': cloth_path,
                        'category': category
                    })
                    pair_summary['cloth_pairs'] += 1
                    pair_summary['total_pairs'] += 1
        
        if pair_mode == 'cross':
            # Person 和 Person 配对
            for person_id, person_paths in person_images.items():
                # person_paths 全排列
                person_pairs = list(itertools.combinations(person_paths, 2))
                for person_pair in person_pairs:
                    paired_data.append({
                        'person': person_pair[0],
                        'cloth': person_pair[1],
                        'category': category
                    })
                    paired_data.append({
                        'person': person_pair[1],
                        'cloth': person_pair[0],
                        'category': category
                    })
                    pair_summary['person_pairs'] += 2
                    pair_summary['total_pairs'] += 2
                    
        # 更新总体摘要
        total_summary[category] = pair_summary
        # total_summary['total_pairs'] += pair_summary['total_pairs']

    # 写入 JSONL 文件
    if output_path:
        # 如果提供了完整路径，直接使用
        output_file = output_path
    else:
        # 否则自动生成路径
        output_file = os.path.join(
            dataset_path, 
            f'dataset_pairs.jsonl'
        )

    with open(output_file, 'w') as f:
        for pair in paired_data:
            f.write(json.dumps(pair) + '\n')

    print(f"生成配对数据文件: {output_file}")

    # 打印和返回配对摘要
    print(json.dumps(total_summary, indent=2))
    print(f'total_pairs: {sum(total_summary[category]["total_pairs"] for category in categories)}')
    return total_summary

if __name__ == '__main__':
    pass