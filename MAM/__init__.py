import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from huggingface_hub import snapshot_download

import MAM.utils as utils
from GroundingDINO.groundingdino.util.inference import Model
from MAM.networks import m2ms
from SAM.segment_anything import sam_model_registry
from SAM.segment_anything.utils.transforms import ResizeLongestSide


class sam_m2m(nn.Module):
    def __init__(self, seg, m2m):
        super(sam_m2m, self).__init__()
        self.m2m = m2ms.__dict__[m2m](nc=256)
        if seg == 'sam_vit_b':
            self.seg_model = sam_model_registry['vit_b']()
        elif seg == 'sam_vit_l':
            self.seg_model = sam_model_registry['vit_l']()
        elif seg == 'sam_vit_h':
            self.seg_model = sam_model_registry['vit_h']()
        self.seg_model.eval()

    def forward(self, image, guidance):
        self.seg_model.eval()
        with torch.no_grad():
            feas, masks = self.seg_model.forward_m2m(image, guidance, multimask_output=True)
        pred = self.m2m(feas, image, masks)
        return pred

    def forward_inference(self, image_dict):
        self.seg_model.eval()
        with torch.no_grad():
            feas, masks, post_masks = self.seg_model.forward_m2m_inference(image_dict, multimask_output=True)
        pred = self.m2m(feas, image_dict["image"], masks)
        return feas, pred, post_masks


class MattingAnything:
    def __init__(self, ckpt_repo="zhengchong/Human-Toolkit", device='cuda', working_size=1024):
        ckpt_repo = snapshot_download(ckpt_repo)
        self.device = device
        self.working_size = working_size
        self.transform = ResizeLongestSide(working_size)    
        
        # initialize MAM
        print("Initializing MAM model...")
        self.mam_model = sam_m2m(seg='sam_vit_b', m2m='sam_decoder_deep')
        self.mam_model.to(device)
        sam_checkpoint = os.path.join(ckpt_repo, "MAM/sam_vit_b_01ec64.pth")
        self.mam_model.seg_model.load_state_dict(torch.load(sam_checkpoint, map_location=device, weights_only=False), strict=True)
        mam_checkpoint = torch.load(os.path.join(ckpt_repo, "MAM/mam_vitb.pth"), map_location=device, weights_only=False)
        self.mam_model.m2m.load_state_dict(utils.remove_prefix_state_dict(mam_checkpoint['state_dict']), strict=True)
        self.mam_model = self.mam_model.eval()
        
        # initialize GroundingDINO
        print("Initializing GroundingDINO model...")
        config_path = os.path.join(ckpt_repo, "GroundingDINO/GroundingDINO_SwinT_OGC.py")
        checkpoint_path = os.path.join(ckpt_repo, "GroundingDINO/groundingdino_swint_ogc.pth")
        self.grounding_dino_model = Model(model_config_path=config_path, model_checkpoint_path=checkpoint_path, device=device)
    
    def compose_image(self, image_ori, alpha_pred, background_color=[128, 128, 128]):
        background = np.zeros_like(image_ori)
        background[:, :, :] = background_color
        com_img = alpha_pred[..., None] * image_ori + (1 - alpha_pred[..., None]) * np.uint8(background)
        return com_img.astype(np.uint8)
    
    @torch.no_grad()
    def matting(self, image_ori, text_prompt, box_threshold=0.25, text_threshold=0.25, iou_threshold=0.5, guidance_mode='mask', return_pil=False):
        # 确保 image_ori 是 numpy 数组
        if isinstance(image_ori, Image.Image):
            image_ori = np.array(image_ori)
        
        # GroundingDINO to get the bbox
        detections, phrases = self.grounding_dino_model.predict_with_caption(
            image=cv2.cvtColor(image_ori, cv2.COLOR_RGB2BGR),
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        if len(detections.xyxy) > 1:
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy), 
                torch.from_numpy(detections.confidence), 
                iou_threshold,
            ).numpy().tolist()

            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]

        bbox = detections.xyxy[np.argmax(detections.confidence)]
        bbox = self.transform.apply_boxes(bbox, (image_ori.shape[0], image_ori.shape[1]))
        bbox = torch.as_tensor(bbox, dtype=torch.float).to(self.device)
        
        # Transform image
        if isinstance(image_ori, Image.Image):
            image_ori = np.array(image_ori)

        image = cv2.cvtColor(image_ori, cv2.COLOR_RGB2BGR)
        image = self.transform.apply_image(image)
        image = torch.as_tensor(image).to(self.device)
        image = image.permute(2, 0, 1).contiguous()

        pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3,1,1).to(self.device)
        pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3,1,1).to(self.device)

        image = (image - pixel_mean) / pixel_std

        h, w = image.shape[-2:]
        pad_size = image.shape[-2:]
        padh = self.working_size - h
        padw = self.working_size - w
        image = F.pad(image, (0, padw, 0, padh))

        sample = {
            'image': image.unsqueeze(0), 
            'bbox': bbox.unsqueeze(0), 
            'ori_shape': (image_ori.shape[0], image_ori.shape[1]), 
            'pad_shape': pad_size
        }

        feas, pred, post_mask = self.mam_model.forward_inference(sample)

        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
        alpha_pred_os8 = alpha_pred_os8[..., : sample['pad_shape'][0], : sample['pad_shape'][1]]
        alpha_pred_os4 = alpha_pred_os4[..., : sample['pad_shape'][0], : sample['pad_shape'][1]]
        alpha_pred_os1 = alpha_pred_os1[..., : sample['pad_shape'][0], : sample['pad_shape'][1]]

        alpha_pred_os8 = F.interpolate(alpha_pred_os8, sample['ori_shape'], mode="bilinear", align_corners=False)
        alpha_pred_os4 = F.interpolate(alpha_pred_os4, sample['ori_shape'], mode="bilinear", align_corners=False)
        alpha_pred_os1 = F.interpolate(alpha_pred_os1, sample['ori_shape'], mode="bilinear", align_corners=False)
        
        if guidance_mode == 'mask':
            weight_os8 = utils.get_unknown_tensor_from_mask_oneside(post_mask, rand_width=10, train_mode=False)
            post_mask[weight_os8>0] = alpha_pred_os8[weight_os8>0]
            alpha_pred = post_mask.clone().detach()
        else:
            weight_os8 = utils.get_unknown_box_from_mask(post_mask)
            alpha_pred_os8[weight_os8>0] = post_mask[weight_os8>0]
            alpha_pred = alpha_pred_os8.clone().detach()


        weight_os4 = utils.get_unknown_tensor_from_pred_oneside(alpha_pred, rand_width=20, train_mode=False)
        alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4>0]
        
        weight_os1 = utils.get_unknown_tensor_from_pred_oneside(alpha_pred, rand_width=10, train_mode=False)
        alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1>0]
       
        alpha_pred = alpha_pred[0][0].cpu().numpy()

        if return_pil:
            return Image.fromarray(alpha_pred * 255).convert("L")
        else:
            return alpha_pred
    
    
if __name__ == "__main__":
    from PIL import Image
    matting_anything = MattingAnything()
    image_ori = Image.open("/home/chongzheng/data/TryOnAnything/Datasets/Bean-Raw/lower/cloth/4406127.jpg")
    text_prompt = "pants"
    alpha_pred = matting_anything.matting(image_ori, text_prompt)
    image_compose = matting_anything.compose_image(image_ori, alpha_pred, background_color=[255, 255, 255])
    
    # 将浮点数数组转换为8位整数格式
    # image_compose = (image_compose * 255).astype(np.uint8)
    
    Image.fromarray(image_compose).save("alpha_pred.png")
