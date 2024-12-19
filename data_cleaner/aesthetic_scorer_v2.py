import torch
import torch.nn as nn
import pytorch_lightning as pl
import clip
from PIL import Image
import numpy as np
from huggingface_hub import snapshot_download
import os
from tqdm import tqdm
import glob
import shutil

class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = nn.functional.mse_loss(x_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = nn.functional.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class AestheticScorer:
    def __init__(self, 
                 model_path="zhengchong/Human-Toolkit", 
                 device=None):
        """
        初始化美学评分器
        
        参数:
        - model_path: 预训练模型的路径
        - device: 计算设备，默认为自动检测
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path = snapshot_download(repo_id=model_path)
        
        # 加载CLIP模型
        clip_path = os.path.join(model_path, "AestheticScoreV2", "ViT-L-14.pt")
        self.clip_model, self.preprocess = clip.load(clip_path, device=self.device)

        # 初始化美学评分模型
        self.aesthetic_model = MLP(768)
        aesthetic_path = os.path.join(model_path, "AestheticScoreV2", "sac+logos+ava1-l14-linearMSE.pth")
        state_dict = torch.load(aesthetic_path, map_location=self.device, weights_only=False)
        self.aesthetic_model.load_state_dict(state_dict)
        self.aesthetic_model.to(self.device)
        self.aesthetic_model.eval()

    def _normalized(self, a, axis=-1, order=2):
        """
        标准化向量
        """
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)

    def score_image(self, image_path):
        """
        对图像进行美学评分
        
        参数:
        - image_path: 图像文件路径
        
        返回:
        - 美学评分
        """
        # 打开并预处理图像
        pil_image = Image.open(image_path)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        # 提取图像特征
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)

        # 标准化特征
        im_emb_arr = self._normalized(image_features.cpu().detach().numpy())

        # 预测美学评分
        prediction = self.aesthetic_model(
            torch.from_numpy(im_emb_arr).to(self.device).type(torch.cuda.FloatTensor)
        )

        return prediction.item()

# 使用示例
if __name__ == "__main__":
    # 创建美学评分器实例
    scorer = AestheticScorer()

    # 评分单张图片
    folder_path = "Datasets/Bean-Raw/full/cloth"
    
    for image_path in tqdm(glob.glob(os.path.join(folder_path, "*.jpg"))):
        score = scorer.score_image(image_path)
        if score < lowest_score:
            lowest_score = score
            lowest_image_path = image_path
            print(f"图像 {image_path} 的美学评分: {score}")
            # cp to./test
            shutil.copy(image_path, os.path.join("./test", os.path.basename(image_path)))
    print(f"最低分: {lowest_score}, 最低分图像: {lowest_image_path}")

