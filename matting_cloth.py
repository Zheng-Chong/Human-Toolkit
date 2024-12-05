import json
from PIL import Image
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple, Union
import os
from MAM import MattingAnything

def process_single_cloth(
    cloth_path: str,
    matting_model: MattingAnything,
    category: str,
    output_dir: str = None
) -> Union[str, None]:
    """处理单件服装图像的matting"""
    # try:
    # 读取图像
    image = Image.open(cloth_path).convert('RGB')
    
    # 根据类别选择提示词
    text_prompt = "pants" if category == "lower" else "upper clothes"
    
    # 执行matting
    alpha_pred = matting_model.matting(
        image, 
        text_prompt=text_prompt,
        box_threshold=0.25,
        text_threshold=0.25,
        return_pil=True
    )
    
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_name = Path(cloth_path).stem + '.png'
        output_path = os.path.join(output_dir, output_name)
        alpha_pred.save(output_path)
        return output_path
        
    return alpha_pred
        
    # except Exception as e:
    #     print(f"处理{cloth_path}时发生错误: {str(e)}")
    #     return None

def process_cloth_batch(
    jsonl_path: str,
    output_dir: str = None,
    max_workers: int = 4,
    device: str = 'cuda',
    working_size: int = 1024,
    part_filter: list = None
) -> List[str]:
    """批量处理服装图像的matting"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 初始化MAM模型
    matting_model = MattingAnything(device=device, working_size=working_size)
    
    # 读取jsonl文件
    src_root = Path(jsonl_path).parent
    with open(jsonl_path, 'r') as f:
        items = [json.loads(line) for line in f]
    
    if part_filter:
        items = [item for item in items if item['category'] in part_filter]
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        # 预处理所有items的路径
        for item in items:
            cloth_path = str(src_root / item['cloth'])
            if output_dir is None:
                output_dir = os.path.dirname(cloth_path).replace('cloth', 'annotations/cloth_matting')
                
            if not os.path.exists(os.path.join(output_dir, os.path.basename(cloth_path).replace('.jpg', '.png'))):
                future = executor.submit(
                    process_single_cloth,
                    cloth_path,
                    matting_model,
                    item['category'],
                    output_dir
                )
                futures.append(future)
        
        # 使用tqdm显示进度
        for future in tqdm(futures, desc="生成matting遮罩中", total=len(futures)):
            try:
                result = future.result(timeout=60)  # 添加超时限制
                if result:
                    results.append(result)
            except Exception as e:
                print(f"处理失败: {str(e)}")
                continue
    
    return results

def main():
    """主函数：处理命令行参数并执行matting生成"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量生成服装matting遮罩')
    parser.add_argument('--jsonl_path', type=str, required=True,
                      help='输入的jsonl文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='输出目录路径')
    parser.add_argument('--max_workers', type=int, default=4,
                      help='最大线程数 (默认: 4)')
    parser.add_argument('--device', type=str, default='cuda',
                      help='运行设备 (默认: cuda)')
    parser.add_argument('--working_size', type=int, default=1024,
                      help='处理图像的工作尺寸 (默认: 1024)')
    parser.add_argument('--part_filter', type=str, nargs='+', default=None,
                      help='处理类别过滤，可指定多个类别 (例如: --part_filter upper lower)')
    
    args = parser.parse_args()
    
    results = process_cloth_batch(
        jsonl_path=args.jsonl_path,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        device=args.device,
        working_size=args.working_size,
        part_filter=args.part_filter
    )
    
    print(f"处理完成，共生成{len(results)}个matting遮罩文件")

if __name__ == '__main__':
    main()
