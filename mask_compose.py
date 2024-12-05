import json
from PIL import Image
import numpy as np
from tqdm import tqdm
from cloth_masker import AutoMasker
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple, Union
import os

def compose_single_mask_with_masker(
    paths: Tuple[str, str, str],
    masker: AutoMasker,
    part: str = 'upper',
    output_dir: str = None,
    width: int = None,
    height: int = None
) -> Union[Image.Image, str]:
    """与compose_single_mask类似，但接受预初始化的masker"""
    densepose_path, schp_lip_path, schp_atr_path = paths
    
    densepose_mask = Image.open(densepose_path).convert('L')
    schp_lip_mask = Image.open(schp_lip_path).convert('P')
    schp_atr_mask = Image.open(schp_atr_path).convert('P')
    
    # # 保证三个mask的尺寸一致
    # if width is None or height is None:
    #     width, height = densepose_mask.size
    # densepose_mask = densepose_mask.resize((width, height), Image.Resampling.NEAREST)
    # schp_lip_mask = schp_lip_mask.resize((width, height), Image.Resampling.NEAREST)
    # schp_atr_mask = schp_atr_mask.resize((width, height), Image.Resampling.NEAREST)
    
    mask = masker.cloth_agnostic_mask(
        densepose_mask=densepose_mask,
        schp_lip_mask=schp_lip_mask,
        schp_atr_mask=schp_atr_mask,
        part=part
    )
    
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_name = Path(densepose_path).stem + '.png'
        output_path = os.path.join(output_dir, output_name)
        mask.save(output_path)
        return output_path
    
    return mask

def process_item(item: dict, masker: AutoMasker, output_dir: str=None) -> str:
    """处理单个数据项的函数"""
    person_path = item['person']
    if output_dir is None:
        output_dir = os.path.dirname(person_path).replace('person', 'annotations/mask_v1')
        
    if not os.path.exists(os.path.join(output_dir, os.path.basename(person_path).replace('.jpg', '.png'))):
        # 获取 person 的尺寸
        person_img = Image.open(person_path)
        person_width, person_height = person_img.size
        # 根据person路径构造对应的预处理结果路径
        densepose_path = person_path.replace('person', 'annotations/densepose').replace('.jpg', '.png')
        schp_lip_path = person_path.replace('person', 'annotations/schp_lip').replace('.jpg', '.png')
        schp_atr_path = person_path.replace('person', 'annotations/schp_atr').replace('.jpg', '.png')
       
        return compose_single_mask_with_masker(
            paths=(densepose_path, schp_lip_path, schp_atr_path),
            masker=masker,
            part=item['category'],
            output_dir=output_dir,
            width=person_width,
            height=person_height
        )
    else:
        # 尝试读取，看是否有问题
        try:
            _ = Image.open(os.path.join(output_dir, os.path.basename(person_path).replace('.jpg', '.png')))
        except Exception as e:
            print(f"读取失败: {str(e)}")
            os.remove(os.path.join(output_dir, os.path.basename(person_path).replace('.jpg', '.png')))
            return None
        return os.path.join(output_dir, os.path.basename(person_path).replace('.jpg', '.png'))

def compose_masks(
    jsonl_path: str,
    output_dir: str = None,
    max_workers: int = None,
    part_filter: list = None
) -> List[str]:
    """从jsonl文件批量处理mask生成"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    src_root = Path(jsonl_path).parent
    with open(jsonl_path, 'r') as f:
        items = [json.loads(line) for line in f]
    
    if part_filter:
        items = [item for item in items if item['category'] in part_filter]
    
    masker = AutoMasker(model_zoo_root=None, load_models=False)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        # 修改：预处理所有items的路径
        for item in items:
            item['person'] = str(src_root / item['person'])
            future = executor.submit(process_item, item, masker, output_dir)
            futures.append(future)
        
        # 使用tqdm显示进度
        for future in tqdm(futures, desc="生成mask中", total=len(futures)):
            try:
                result = future.result(timeout=30)  # 添加超时限制
                results.append(result)
            except Exception as e:
                print(f"处理失败: {str(e)}")
                continue
    
    return results

def main():
    """主函数：处理命令行参数并执行mask生成"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量生成服装遮罩')
    parser.add_argument('--jsonl_path', type=str, required=True,
                      help='输入的jsonl文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='输出目录路径')
    parser.add_argument('--max_workers', type=int, default=4,
                      help='最大线程数 (默认: 4)')
    parser.add_argument('--part_filter', type=str, nargs='+', default=None,
                      help='处理类别过滤，可指定多个类别 (例如: --part_filter upper lower)')
    
    args = parser.parse_args()
    
    masks = compose_masks(
        jsonl_path=args.jsonl_path,
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
    print(f"处理完成，共生成{len(masks)}个mask文件")

if __name__ == '__main__':
    main()
