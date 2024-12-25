import os
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from tqdm import tqdm

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

class ImageCaptioner:
    def __init__(self, device=None):
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.bfloat16
        
        # 初始化模型和处理器
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large", 
            torch_dtype=self.torch_dtype, 
            trust_remote_code=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large", 
            trust_remote_code=True
        )
        
        self.prompts = ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]

    def process_single_image(self, image_path):
        """处理单张图片，返回三种不同详细程度的描述"""
        try:
            image = Image.open(image_path)
            results = {}
            
            for prompt in self.prompts:
                inputs = self.processor(
                    text=prompt, 
                    images=image, 
                    return_tensors="pt"
                ).to(self.device, self.torch_dtype)
                
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3
                )
                
                generated_text = self.processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=False
                )[0]
                
                caption = self.processor.post_process_generation(
                    generated_text, 
                    task=prompt, 
                    image_size=(image.width, image.height)
                )
                
                # results[prompt] = caption
                results.update(caption)
                
            return {
                "image_path": os.path.basename(image_path),
                "captions": results
            }
            
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {str(e)}")
            return None

def process_images(args):
    # 读取已处理的图片记录
    processed_images = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed_images.add(data['image_path'])
                except:
                    continue
    
    # 初始化多个 ImageCaptioner
    num_workers = args.num_workers
    available_gpus = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    
    captioners = [
        ImageCaptioner(
            device=available_gpus[i % len(available_gpus)] if available_gpus else "cpu"
        ) for i in range(num_workers)
    ]
    
    # 获取所有图片文件
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    input_path = Path(args.input_dir)
    image_files = [
        img_path for img_path in input_path.glob('**/*')
        if img_path.suffix.lower() in img_extensions
        and img_path.name not in processed_images  # 只处理未处理过的图片
    ]
    
    if not image_files:
        print("所有图片都已处理完成！")
        return
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 使用线程池处理图片
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        progress_bar = tqdm(total=len(image_files), desc="生成图片描述")
        
        def process_and_save(result):
            if result:
                with open(args.output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            progress_bar.update(1)
        
        for i, img_path in enumerate(image_files):
            captioner = captioners[i % num_workers]
            future = executor.submit(captioner.process_single_image, img_path)
            future.add_done_callback(lambda f: process_and_save(f.result()))
            futures.append(future)
        
        concurrent.futures.wait(futures)
        progress_bar.close()

def main():
    parser = argparse.ArgumentParser(description='批量处理图片生成描述')
    parser.add_argument('--input_dir', type=str, required=True, help='输入图片目录路径')
    parser.add_argument('--output_file', type=str, required=True, help='输出JSONL文件路径')
    parser.add_argument('--num_workers', type=int, default=4, help='处理线程数量')
    args = parser.parse_args()
    
    # 不再删除已存在的输出文件
    process_images(args)

if __name__ == '__main__':
    main()