import os
import argparse
from huggingface_hub import snapshot_download
from cloth_masker import AutoMasker
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

def process_single_image(masker, img_path, output_dir, tools):
    """处理单张图片的函数"""
    try:
        results = masker.preprocess_image(
            str(img_path),
            tools=tools
        )
        base_name = img_path.stem
        # 保存mask
        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        results['mask'].save(mask_path)
        
        # 保存处理结果
        for tool in tools:
            tool_path = os.path.join(output_dir, tool, f"{base_name}.png")
            if not os.path.exists(tool_path):
                os.makedirs(os.path.dirname(tool_path), exist_ok=True)
                results[tool].save(tool_path)
        
    except Exception as e:
        print(f"处理 {img_path.name} 时发生错误: {str(e)}")

def process_images(args):
    # 初始化多个 AutoMasker
    model_zoo_root = snapshot_download(args.model_zoo_root)
    num_workers = args.num_workers
    maskers = [
        AutoMasker(
            model_zoo_root=model_zoo_root,
            device=args.device
        ) for _ in range(num_workers)
    ]
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取输入目录下的所有图片
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    input_path = Path(args.input_dir)
    image_files = [
        img_path for img_path in input_path.glob('*')
        if img_path.suffix.lower() in img_extensions
    ]
    
    # 使用线程池处理图片
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, img_path in enumerate(image_files):
            masker = maskers[i % num_workers]  # 循环使用 maskers
            future = executor.submit(
                process_single_image,
                masker,
                img_path,
                args.output_dir,
                args.mask_type,
                args.save_schp
            )
            futures.append(future)
        
        # 等待所有任务完成
        concurrent.futures.wait(futures)

def main():
    parser = argparse.ArgumentParser(description='批量处理图片生成衣物遮罩')
    parser.add_argument('--input_dir', type=str, required=True, help='输入图片目录路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出结果目录路径')
    parser.add_argument('--tools', type=str, nargs='+', default=['densepose', 'schp_atr', 'schp_lip', 'schp_pascal'], help='处理工具')
    parser.add_argument('--model_zoo_root', type=str, default='zhengchong/Human-Toolkit', help='Human-Toolkit 模型库')
    parser.add_argument('--num_workers', type=int, default=4, help='处理线程数量')
    args = parser.parse_args()
    process_images(args)

if __name__ == '__main__':
    main()
