import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

class LAIONAestheticScorer:
    def __init__(self, 
                 model_path='laion/CLIP-ViT-H-14-laion2B-s32B-b79K', 
                 rating_model_path='aesthetics_scorer/models/aesthetics_scorer_rating_openclip_vit_h_14.pth',
                 artifacts_model_path='aesthetics_scorer/models/aesthetics_scorer_artifacts_openclip_vit_h_14.pth',
                 device='cuda'):
        """
        初始化 LAION 审美分数计算器
        
        Args:
            model_path (str): CLIP模型路径
            rating_model_path (str): 评分模型路径
            artifacts_model_path (str): 伪影模型路径
            device (str): 计算设备，默认为 'cuda'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载 CLIP 模型和处理器
        self.model = CLIPModel.from_pretrained(model_path)
        self.vision_model = self.model.vision_model.to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        
        # 加载审美评分模型
        self.rating_model = self._load_aesthetic_model(rating_model_path)
        self.artifacts_model = self._load_aesthetic_model(artifacts_model_path)
    
    def _load_aesthetic_model(self, model_path):
        """
        加载预训练的审美评分模型
        
        Args:
            model_path (str): 模型路径
        
        Returns:
            torch.nn.Module: 加载的模型
        """
        from model import load_model  # 假设有一个 model.py 文件包含 load_model 函数
        model = load_model(model_path).to(self.device)
        return model
    
    def _preprocess_features(self, features):
        """
        预处理特征向量
        
        Args:
            features (torch.Tensor): 输入特征
        
        Returns:
            torch.Tensor: 预处理后的特征
        """
        from model import preprocess  # 假设有一个 model.py 文件包含 preprocess 函数
        return preprocess(features)
    
    def _extract_clip_features(self, image_path):
        """
        提取图像的 CLIP 特征
        
        Args:
             image_path (str): 图像路径
        
        Returns:
            torch.Tensor: 图像的 CLIP 特征向量
        """
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.vision_model(**inputs)
            image_features = outputs.pooler_output
        
        return image_features
    
    def calculate_aesthetic_score(self, image_path):
        """
        计算单张图像的审美分数
        
        Args:
            image_path (str): 图像路径
        
        Returns:
            tuple: (评分, 伪影分数)
        """
        image_features = self._extract_clip_features(image_path)
        preprocessed_features = self._preprocess_features(image_features)
        
        with torch.no_grad():
            rating = self.rating_model(preprocessed_features).item()
            artifacts = self.artifacts_model(preprocessed_features).item()
        
        return rating, artifacts
    
    def batch_calculate_aesthetic_scores(self, image_dir, output_path=None, batch_size=32):
        """
        批量计算目录中图像的审美分数
        
        Args:
            image_dir (str): 图像目录
            output_path (str, optional): 输出 JSON 文件路径
            batch_size (int, optional): 批处理大小
        
        Returns:
            dict: 图像路径与审美分数的映射
        """
        image_paths = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        
        aesthetic_scores = {}
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
            batch_paths = image_paths[i:i+batch_size]
            batch_features = torch.cat([
                self._extract_clip_features(path) for path in batch_paths
            ])
            
            preprocessed_features = self._preprocess_features(batch_features)
            
            with torch.no_grad():
                batch_ratings = self.rating_model(preprocessed_features).squeeze().cpu().numpy()
                batch_artifacts = self.artifacts_model(preprocessed_features).squeeze().cpu().numpy()
            
            for path, rating, artifact in zip(batch_paths, batch_ratings, batch_artifacts):
                aesthetic_scores[path] = {
                    'rating': float(rating),
                    'artifacts': float(artifact)
                }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(aesthetic_scores, f, indent=2)
        
        return aesthetic_scores

def main():
    scorer = LAIONAestheticScorer()
    
    # 示例：计算单张图像审美分数
    single_image_path = 'Datasets/Bean-Raw/upper/person/4397018(1).jpg'
    rating, artifacts = scorer.calculate_aesthetic_score(single_image_path)
    print(f"Aesthetic Score for {single_image_path}:")
    print(f"Rating: {rating}")
    print(f"Artifacts: {artifacts}")
    
    # 示例：批量计算目录中图像的审美分数
    # image_dir = 'path/to/image/directory'
    # output_path = 'aesthetic_scores.json'
    # scores = scorer.batch_calculate_aesthetic_scores(image_dir, output_path)
    # print(f"Processed {len(scores)} images. Results saved to {output_path}")

if __name__ == '__main__':
    main() 