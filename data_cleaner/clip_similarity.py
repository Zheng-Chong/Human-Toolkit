import os
import clip
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from typing import Union, List
from data_cleaner.utils import crop_to_max_square_region, get_target_cloth_mask

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
            # 裁剪到最大正方形区域
            image  = crop_to_max_square_region(image, mask)
        
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

