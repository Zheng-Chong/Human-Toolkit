import os
import clip
import torch
from PIL import Image, ImageOps
from huggingface_hub import snapshot_download
from typing import Union, List
import numpy as np
import cv2


class CLIPSimilarityCalculator:
    def __init__(self, model_path="zhengchong/Human-Toolkit", device="cuda"):
        """
        初始化CLIP模型
        
        :param model_path: HuggingFace模型仓库路径
        :param device: 计算设备，默认为cuda
        """
        self.device = device
        model_path = snapshot_download(repo_id=model_path)
        clip_path = os.path.join(model_path, "AestheticScoreV2", "ViT-L-14.pt")
        self.clip_model, self.preprocess = clip.load(clip_path, device=self.device)

    def _load_and_preprocess_image(
        self, 
        image_path: Union[str, Image.Image],
        mask_path: Union[str, Image.Image] = None,
    ) -> torch.Tensor:
        """
        加载并预处理图像，并裁剪到 mask 的最大正方形区域
        
        :param image_path: 图像路径或PIL图像对象
        :param mask_path: 服装遮罩路径或PIL图像对象
        :return: 预处理后的图像张量
        """
        # 统一图像输入类型
        image = image_path if isinstance(image_path, Image.Image) else Image.open(image_path)
        
        if mask_path is not None:
            # 统一 mask 输入类型
            mask = mask_path if isinstance(mask_path, Image.Image) else Image.open(mask_path)
            
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
            image = image.crop((x1, y1, x2, y2))
            mask = mask.crop((x1, y1, x2, y2))
            
            # 处理 mask
            mask = ImageOps.invert(mask.convert('L'))
            image = Image.composite(Image.new('RGB', image.size, 'white'), image, mask)
        
        # 预处理并返回图像张量
        return self.preprocess(image).unsqueeze(0).to(self.device)

    def calculate_similarity(
        self, 
        image_1: Union[str, Image.Image], 
        image_2: Union[str, Image.Image],
        mask_1: Union[str, Image.Image] = None,
        mask_2: Union[str, Image.Image] = None,
    ) -> float:
        """
        计算两张图像的余弦相似度
        
        :param image_1: 第一张图像的路径或PIL图像对象
        :param image_2: 第二张图像的路径或PIL图像对象
        :return: 图像特征的余弦相似度
        """
        # 预处理图像
        image_1_tensor = self._load_and_preprocess_image(image_1, mask_1)
        image_2_tensor = self._load_and_preprocess_image(image_2, mask_2)

        # 计算图像特征
        with torch.no_grad():
            image_features_1 = self.clip_model.encode_image(image_1_tensor)
            image_features_2 = self.clip_model.encode_image(image_2_tensor)

        # 计算余弦相似度
        similarity = torch.nn.functional.cosine_similarity(image_features_1, image_features_2, dim=1)
        return similarity.item()

    def batch_calculate_similarity(self, reference_image: Union[str, Image.Image], image_list: List[Union[str, Image.Image]]) -> List[float]:
        """
        批量计算参考图像与多张图像的相似度
        
        :param reference_image: 参考图像的路径或PIL图像对象
        :param image_list: 待比较图像列表
        :return: 相似度列表
        """
        reference_tensor = self._load_and_preprocess_image(reference_image)

        with torch.no_grad():
            reference_features = self.clip_model.encode_image(reference_tensor)
            
            similarities = []
            for image in image_list:
                image_tensor = self._load_and_preprocess_image(image)
                image_features = self.clip_model.encode_image(image_tensor)
                
                similarity = torch.nn.functional.cosine_similarity(reference_features, image_features, dim=1)
                similarities.append(similarity.item())

        return similarities




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
    LIP_MAPPING = {
        'Background': 0, 'Hat': 1, 'Hair': 2, 'Glove': 3, 
        'Sunglasses': 4, 'Upper-clothes': 5, 'Dress': 6, 'Coat': 7,
        'Socks': 8, 'Pants': 9, 'Jumpsuits': 10, 'Scarf': 11, 
        'Skirt': 12, 'Face': 13, 'Left-arm': 14, 'Right-arm': 15,
        'Left-leg': 16, 'Right-leg': 17, 'Left-shoe': 18, 'Right-shoe': 19
    }
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

 
# 示例使用
if __name__ == "__main__":
    calculator = CLIPSimilarityCalculator()
    
    
    while True:
        person_image_path = input("请输入 person 图像路径: ")
    # 单张图像相似度
    # person_image_path = "/home/chongzheng/data/Human-Toolkit/Datasets/Bean-Raw/upper/person/4397023(1).jpg"
    
        cloth_image_path = person_image_path[:person_image_path.rfind('(')].replace('person', 'cloth') + '.jpg'
        person_mask_path = person_image_path.replace('person', 'annotations/schp_lip').replace('.jpg', '.png')
        cloth_mask_path = cloth_image_path.replace('cloth', 'annotations/cloth_matting').replace('.jpg', '.png')
        # # mask_1_path = "/home/chongzheng/data/Human-Toolkit/Datasets/Bean-Raw/upper/annotations/mask_v1/4397022(1).png"
        # mask_2_path = "/home/chongzheng/data/Human-Toolkit/Datasets/Bean-Raw/upper/annotations/schp_lip/4400348(4).png"
        # mask_2 = get_target_cloth_mask(mask_2_path, 'upper')
        try:
            person_mask = get_target_cloth_mask(person_mask_path, 'upper')
            similarity = calculator.calculate_similarity(person_image_path, cloth_image_path, person_mask, cloth_mask_path)
            print(f"Similarity: {similarity}")
        except Exception as e:
            print(f"Error: {e}")
            continue

    # 批量计算相似度
    # image_list = [
    #     "/home/chongzheng/data/Human-Toolkit/Datasets/Bean-Raw/upper/person/4397019(4).jpg",
    #     "/home/chongzheng/data/Human-Toolkit/Datasets/Bean-Raw/upper/person/4397018(1).jpg"
    # ]
    # batch_similarities = calculator.batch_calculate_similarity(image_1_path, image_list)
    # print(f"Batch Similarities: {batch_similarities}")

