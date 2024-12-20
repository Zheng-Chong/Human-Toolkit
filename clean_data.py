import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import List, Dict, Set
from data_cleaner.utils import is_image_clear, wear_upper_with_outer, are_dominant_colors_similar, has_more_than_three_white_edges, is_cloth_area_too_large
from data_cleaner.clip_similarity import CLIPSimilarityCalculator
from PIL import Image

# 全局变量：CLIP 相似度计算器和已处理的 LIP 文件
PROCESSED_LIP = dict()
CLIP_CALCULATOR = CLIPSimilarityCalculator()

def load_processed_items(output_jsonl_path: str) -> Set[str]:
    """
    加载已处理的条目哈希值
    
    :param output_jsonl_path: 输出的 JSONL 文件路径
    :return: 已处理条目的哈希集合
    """
    processed_items = set()
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                processed_items.add(json.dumps(item, sort_keys=True))
    return processed_items

def process_single_item(item: dict, src_root: Path, clip_similarity_threshold: float = 0.7, cloth_only: bool = False) -> dict:
    """
    处理单个数据项，检查是否包含上衣和外套叠穿，以及 person 和 cloth 的相似度
    
    :param item: 数据项
    :param src_root: 源文件根目录
    :param clip_similarity_threshold: CLIP 相似度阈值，默认为 0.7
    :return: 处理后的数据项，如果不符合条件则返回 None
    """
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
        
        # 计算 CLIP 相似度
        try:
            clip_similarity = CLIP_CALCULATOR.calculate_similarity(
                person_path, cloth_path, 
                person_mask_path, cloth_mask_path
            )
            # 如果 CLIP 相似度低于阈值，则过滤掉
            if clip_similarity < clip_similarity_threshold:
                return None
        except Exception as clip_error:
            print(f"CLIP 相似度计算失败: {clip_error}")
            return None
        
        # 检查图像是否清晰
        is_clear, _ = is_image_clear(person_path, threshold=100)
        if not is_clear:
            return None
        
        # # 检查是否有大于3个边上有白色
        # if has_more_than_three_white_edges(person_mask_path) or has_more_than_three_white_edges(cloth_mask_path):
        #     return None

        # # 检查颜色相似度
        # color_result = are_dominant_colors_similar(
        #     person_path, 
        #     cloth_path,
        #     person_mask_path,
        #     cloth_mask_path,
        #     threshold=50
        # )
        # if not color_result['is_similar']:
        #     return None
        
        # 检查服装区域是否过大
        schp_lip_path = person_path.replace('person', 'annotations/schp_lip').replace('.jpg', '.png')
        if is_cloth_area_too_large(schp_lip_path, threshold=0.5):
            return None

        # 检查是否是上衣和外套叠穿
        if item['category'] == 'upper' or item['category'] == 'full':
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
    max_workers: int = 4,
    clip_similarity_threshold: float = 0.7,
    cloth_only: bool = False
) -> List[Dict]:
    """
    清洗数据，去除上衣和外套叠穿的样本，并根据 CLIP 相似度过滤
    
    :param jsonl_path: 输入的 JSONL 文件路径
    :param output_jsonl_path: 输出的 JSONL 文件路径
    :param max_workers: 最大线程数
    :param clip_similarity_threshold: CLIP 相似度阈值
    :return: 清洗后的数据列表
    """
    # 设置默认输出路径
    if output_jsonl_path is None:
        output_jsonl_path = jsonl_path.replace('.jsonl', '_cleaned.jsonl')
    
    # 加载已处理的条目
    processed_items = load_processed_items(output_jsonl_path)
    
    # 获取源文件根目录
    src_root = Path(jsonl_path).parent
    
    # 读取 jsonl 文件
    with open(jsonl_path, 'r') as f:
        items = [json.loads(line) for line in f]
    
    # 打开输出文件以追加模式写入
    with open(output_jsonl_path, 'a') as output_file:
        cleaned_items = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            # 提交所有任务
            for item in items:
                # 检查是否已处理过
                item_hash = json.dumps(item, sort_keys=True)
                if item_hash in processed_items:
                    continue
                
                # 如果 cloth_only 为 True，则只保留服装和person配对的样本
                if cloth_only and 'cloth' not in item['cloth']:
                    continue
                
                future = executor.submit(
                    process_single_item, 
                    item, 
                    src_root, 
                    clip_similarity_threshold,
                    cloth_only
                )
                futures.append(future)
            
            # 创建进度条
            progress_bar = tqdm(total=len(futures), desc="清洗数据中")
            
            # 使用 concurrent.futures 处理任务
            for future in futures:
                try:
                    result = future.result(timeout=30)  # 添加超时限制
                    if result is not None:
                        # 实时写入文件
                        output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                        output_file.flush()  # 立即刷新缓冲区
                        cleaned_items.append(result)
                    
                    # 更新进度条
                    progress_bar.set_postfix_str(f"保留: {len(cleaned_items)}")
                    progress_bar.update(1)
                except Exception as e:
                    print(f"处理失败: {str(e)}")
                    progress_bar.update(1)
                    continue
            
            # 关闭进度条
            progress_bar.close()
    
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
                      help='最大线程数 (默认: 16)')
    parser.add_argument('--clip_similarity_threshold', type=float, default=0.8,
                      help='CLIP相似度阈值 (默认: 0.8)')
    parser.add_argument('--cloth_only', action='store_true',
                      help='是否只保留服装和person配对的样本')
    
    args = parser.parse_args()
    
    clean_data(
        jsonl_path=args.jsonl_path,
        output_jsonl_path=args.output_jsonl_path,
        max_workers=args.max_workers,
        clip_similarity_threshold=args.clip_similarity_threshold,
        cloth_only=args.cloth_only
    )

if __name__ == '__main__':
    main()
