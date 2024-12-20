import PIL.Image as Image
from SCHP import LIP_MAPPING
from typing import Union
import numpy as np
import cv2


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
        assert person_lip_image.mode == 'P', 'person_lip_image 必须是 P 模式'
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
    assert person_lip_image.mode == 'P', 'person_lip_image 必须是 P 模式'

    # 获取图像数据
    lip_data = np.array(person_lip_image)

    # 定义目标类别索引
    if target_cloth_type == 'upper':
        target_indices = [LIP_MAPPING[_] for _ in ['Upper-clothes', 'Dress', 'Coat', 'Jumpsuits']]
    elif target_cloth_type == 'lower':
        target_indices = [LIP_MAPPING[_] for _ in ['Pants', 'Skirt']]
    elif target_cloth_type == 'full':
        target_indices = [LIP_MAPPING[_] for _ in ['Upper-clothes', 'Dress', 'Coat', 'Jumpsuits', 'Pants', 'Skirt']]
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
    利用LIP判断是否所有服装的区域占了图像的ratio过大。

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
    assert person_lip_image.mode == 'P', 'person_lip_image 必须是 P 模式'

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
    # print(f"cloth_ratio: {cloth_ratio}")
    # 判断是否超过阈值
    return cloth_ratio > threshold

import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_dominant_color(image, k=3):
    """
    提取图像的主色调（使用K均值聚类）。
    
    参数:
        image (numpy.ndarray): 输入的图像数据（BGR格式）。
        k (int): 聚类中心的个数，默认为3。

    返回:
        tuple: 主色调的BGR值 (B, G, R)。
    """
    # 将图像转换为二维数组 (像素数, 3)
    data = image.reshape((-1, 3))
    data = np.float32(data)
    # 使用K均值聚类
    kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    return tuple(map(int, dominant_color))

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

def are_dominant_colors_similar(
    image1, image2, 
    mask1=None, mask2=None,
    threshold=50):
    """
    判断两个图像的主色调是否一致。

    参数:
        image1 (str | PIL.Image.Image | np.ndarray): 第一张图像的路径或图像。
        image2 (str | PIL.Image.Image | np.ndarray): 第二张图像的路径或图像。
        mask1 (str | PIL.Image.Image | np.ndarray, optional): 第一张图像的掩膜。默认为 None。
        mask2 (str | PIL.Image.Image | np.ndarray, optional): 第二张图像的掩膜。默认为 None。
        threshold (float): 颜色距离的阈值，默认为50。

    返回:
        dict: 包括是否一致 (bool) 和每个图像的均值颜色及距离。
    """
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

    # 转换掩膜为灰度图
    def convert_to_mask(mask):
        if mask is None:
            return None
        if isinstance(mask, str):
            return cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        elif isinstance(mask, Image.Image):
            return np.array(mask.convert('L'))
        elif isinstance(mask, np.ndarray):
            if len(mask.shape) == 3:
                return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            return mask
        else:
            raise TypeError("掩膜必须是文件路径、PIL图像或NumPy数组")

    # 读取图像
    image1 = convert_to_cv2_image(image1)
    image2 = convert_to_cv2_image(image2)
    if image1 is None or image2 is None:
        raise ValueError("无法读取图像，请检查输入是否正确。")
    
    # 读取掩膜
    mask1 = convert_to_mask(mask1)
    mask2 = convert_to_mask(mask2)
    
    # 计算第一张图像的均值颜色
    if mask1 is not None:
        # 创建布尔掩膜
        mask1_bool = mask1 >= 128
        # 计算每个通道的均值
        mean_color1 = tuple([int(image1[:, :, c][mask1_bool].mean()) for c in range(3)])
    else:
        # 如果没有掩膜，计算整个图像的均值
        mean_color1 = tuple([int(image1[:, :, c].mean()) for c in range(3)])
    
    # 计算第二张图像的均值颜色
    if mask2 is not None:
        # 创建布尔掩膜
        mask2_bool = mask2 >= 128
        # 计算每个通道的均值
        mean_color2 = tuple([int(image2[:, :, c][mask2_bool].mean()) for c in range(3)])
    else:
        # 如果没有掩膜，计算整个图像的均值
        mean_color2 = tuple([int(image2[:, :, c].mean()) for c in range(3)])
    
    # 计算颜色距离
    dist = color_distance(mean_color1, mean_color2)
    is_similar = dist <= threshold

    # 返回结果
    return {
        "is_similar": is_similar,
        "mean_color1": mean_color1,
        "mean_color2": mean_color2,
        "distance": dist
    }

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
        raise TypeError("mask_path_or_image 必须是 str, Image.Image 或 numpy.ndarray 类型")

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

if __name__ == '__main__':
    pass