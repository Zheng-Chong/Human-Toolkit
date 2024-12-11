import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import List, Dict
from data_cleaner.utils import wear_upper_with_outer, are_dominant_colors_similar, has_more_than_three_white_edges
from PIL import Image

PROCESSED_LIP = dict()

def process_single_item(item: dict, src_root: Path) -> dict:
    """处理单个数据项，检查是否包含上衣和外套叠穿，以及person和cloth的颜色是否相似"""
    try:
        # 检查必要的键是否存在
        if 'person' not in item or 'cloth' not in item:
            return None

        # 构造完整的图片路径
        person_path = str(src_root / item['person'])
        cloth_path = str(src_root / item['cloth'])
        
        # 构造掩码路径
        person_mask_path = person_path.replace('person', 'annotations/mask_v1')
        person_mask_path = person_mask_path.replace('.jpg', '.png')
        if 'cloth' in cloth_path:
            cloth_mask_path = cloth_path.replace('cloth', 'annotations/cloth_matting')
            cloth_mask_path = cloth_mask_path.replace('.jpg', '.png')
        else:
            cloth_mask_path = cloth_path.replace('person', 'annotations/mask_v1')
            cloth_mask_path = cloth_mask_path.replace('.jpg', '.png')

        # 检查文件是否存在
        required_files = [person_path, cloth_path, person_mask_path, cloth_mask_path]
        if not all(os.path.exists(f) for f in required_files):
            print(f"部分文件不存在: {[f for f in required_files if not os.path.exists(f)]}")
            return None
        
        # 检查是否有大于3个边上有白色
        if has_more_than_three_white_edges(person_mask_path) or has_more_than_three_white_edges(cloth_mask_path):
            return None

        # 检查颜色相似度
        color_result = are_dominant_colors_similar(
            person_path, 
            cloth_path,
            person_mask_path,
            cloth_mask_path,
            threshold=50
        )
        if not color_result['is_similar']:
            return None

        # 检查是否是上衣和外套叠穿
        if item['category'] == 'upper' or item['category'] == 'full':
            schp_lip_path = person_path.replace('person', 'annotations/schp_lip').replace('.jpg', '.png')
            if not os.path.exists(schp_lip_path):
                print(f"SCHP LIP文件不存在: {schp_lip_path}")
                return None

            if schp_lip_path in PROCESSED_LIP:
                flag = PROCESSED_LIP[schp_lip_path]
            else:
                flag = wear_upper_with_outer(schp_lip_path)
                PROCESSED_LIP[schp_lip_path] = flag
            if flag:
                return None

        return item
    except Exception as e:
        print(f"处理失败: {str(e)}")
        return None

def clean_data(
    jsonl_path: str,
    output_jsonl_path: str = None,
    max_workers: int = 4
) -> List[Dict]:
    """清洗数据，去除上衣和外套叠穿的样本"""
    # 设置默认输出路径
    if output_jsonl_path is None:
        output_jsonl_path = jsonl_path.replace('.jsonl', '_cleaned.jsonl')
    
    # 获取源文件根目录
    src_root = Path(jsonl_path).parent
    
    # 读取jsonl文件
    with open(jsonl_path, 'r') as f:
        items = [json.loads(line) for line in f]
    
    cleaned_items = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        # 提交所有任务
        for item in items:
            future = executor.submit(process_single_item, item, src_root)
            futures.append(future)
        
        # 使用tqdm显示进度
        for future in tqdm(futures, desc="清洗数据中", total=len(futures)):
            try:
                result = future.result(timeout=30)  # 添加超时限制
                if result is not None:
                    cleaned_items.append(result)
                    print("len(cleaned_items): ", len(cleaned_items))
            except Exception as e:
                print(f"处理失败: {str(e)}")
                continue
    
    # 写入清洗后的数据
    with open(output_jsonl_path, 'w') as f:
        for item in cleaned_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"清洗完成！原始数据量: {len(items)}, 清洗后数据量: {len(cleaned_items)}")
    return cleaned_items

def main():
    """主函数：处理命令行参数并执行数据清洗"""
    import argparse
    
    parser = argparse.ArgumentParser(description='清洗包含上衣和外套叠穿的数据')
    parser.add_argument('--jsonl_path', type=str, required=True,
                      help='输入的jsonl文件路径')
    parser.add_argument('--output_jsonl_path', type=str, default=None,
                      help='输出的jsonl文件路径')
    parser.add_argument('--max_workers', type=int, default=16,
                      help='最大线程数 (默认: 4)')
    
    args = parser.parse_args()
    
    clean_data(
        jsonl_path=args.jsonl_path,
        output_jsonl_path=args.output_jsonl_path,
        max_workers=args.max_workers
    )

if __name__ == '__main__':
    main()
