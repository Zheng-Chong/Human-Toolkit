import cv2
from torchvision.io import read_video
import torch
import torch.nn.functional as F
from einops import rearrange
def get_video_fps(video_path: str) -> float:
    """
    Get the frame rate (FPS) of the video file
    Args:
        video_path: The path of the video file
    Returns:
        float: The frame rate of the video (frames per second)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def get_video_frame_count(video_path: str) -> int:
    """
    Quickly get the total number of frames in a video file
    Args:
        video_path: Path to the video file
    Returns:
        total_frames: Total number of frames in the video
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def get_video_resolution(video_path: str) -> tuple[int, int]:
    """
    Quickly get the resolution (width and height) of a video file
    Args:
        video_path: Path to the video file
    Returns:
        tuple[int, int]: A tuple containing the video width and height (width, height)
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def read_video_frames(
    video_path, 
    start_frame=0, 
    end_frame=None, 
    to_float=True,
    normalize=True
    ):
    """Read video frames from video file.
    Args:
        video_path (str): Path to video file.
        start_frame (int, optional): Start frame index. Defaults to 0.
        end_frame (int, optional): End frame index. Defaults to None.
        to_float (bool, optional): Convert video frames to float32. Defaults to True.
        normalize (bool, optional): Normalize video frames to [-1, 1]. Defaults to True.
    Returns:
        torch.Tensor: Video frames in B(1)CTHW format.
    """
    video = read_video(video_path, pts_unit="sec", output_format="TCHW")[0]
    end_frame = min(end_frame, video.size(0)) if end_frame is not None else video.size(0)
    video = video[start_frame:end_frame].permute(1, 0, 2, 3).unsqueeze(0)
    if to_float:
        video = video.float() / 255.0
    if normalize:
        if to_float:
            video = video * 2 - 1
        else:
            raise ValueError("`to_float` must be True when `normalize` is True")
    return video

def crop_and_resize_video(
    video_input, 
    target_size: tuple[int, int], 
    resize_method: str = 'bilinear',
    to_float: bool = True,
    normalize: bool = True
    ) -> torch.Tensor:
    """
    Resize the video to the target size, first center cropping to match the target aspect ratio, then scaling
    
    Args:
        video_input: Can be a video file path (str) or a video tensor (torch.Tensor, format BCTHW)
        target_size: Target size tuple (width, height)
        resize_method: Method to use for resizing. Defaults to 'bilinear'. Options are 'nearest', 'bilinear', 'bicubic'.
        
    Returns:
        torch.Tensor: Resized video tensor, format BCTHW
    """
    # If input is a path, read the video first
    if isinstance(video_input, str):
        video = read_video_frames(video_input, to_float=to_float, normalize=normalize)  # BCTHW
    else:
        video = video_input
        
    _, _, _, curr_h, curr_w = video.shape
    target_w, target_h = target_size
    
    # If the size already matches, directly return
    if curr_w == target_w and curr_h == target_h:
        return video
        
    # Calculate the target aspect ratio
    target_ratio = target_w / target_h
    curr_ratio = curr_w / curr_h
    
    # Center cropping to match the target aspect ratio
    if curr_ratio > target_ratio:
        # The current video is too wide, needs to crop width
        new_w = int(curr_h * target_ratio)
        crop_w = (curr_w - new_w) // 2
        video = video[..., :, crop_w:crop_w + new_w]
    elif curr_ratio < target_ratio:
        # The current video is too high, needs to crop height
        new_h = int(curr_w / target_ratio)
        crop_h = (curr_h - new_h) // 2
        video = video[..., crop_h:crop_h + new_h, :]
    
    # 使用 enoip 的 rearrange 调整维度顺序
    b, c, t, h, w = video.shape
    video = rearrange(video, 'b c t h w -> (b t) c h w')
    video = F.interpolate(video, size=(target_h, target_w), mode=resize_method, align_corners=False)
    video = rearrange(video, '(b t) c h w -> b c t h w', b=b)
    return video



