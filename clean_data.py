import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import List, Dict, Set, Optional
from data_cleaner.utils import crop_to_max_square_region, get_target_cloth_mask, is_image_clear, wear_upper_with_outer, are_dominant_colors_similar, has_more_than_three_white_edges, is_cloth_area_too_large
from data_cleaner.clip_similarity import CLIPSimilarityCalculator
from PIL import Image
import shutil

# 全局变量：CLIP 相似度计算器和已处理的 LIP 文件
PROCESSED_LIP = dict()

def save_concat_image(input_list: List[str], output_path: str):
    """
    将输入的图片列表拼接后保存到指定路径
    """
    if isinstance(input_list[0  ], str):
        images = [Image.open(image_path) for image_path in input_list]
    elif isinstance(input_list[0], Image.Image):
        images = input_list
    else:
        raise ValueError(f"input_list 中的元素必须是文件路径或 PIL 图像，但当前类型为: {type(input_list[0])}")
    image_width = sum(image.width for image in images)
    image_height = max(image.height for image in images)
    combined_img = Image.new('RGB', (image_width, image_height))
    width_offset = 0
    for idx, image in enumerate(images):
        combined_img.paste(image, (width_offset, 0))
        width_offset += image.width
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    combined_img.save(output_path)

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

def load_similarity_cache(cache_path: str) -> Dict[str, Dict[str, float]]:
    """
    加载相似度缓存
    
    :param cache_path: 缓存 JSONL 文件路径
    :return: 相似度缓存字典
    """
    similarity_cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            for line in f:
                cache_item = json.loads(line)
                similarity_cache[cache_item['key']] = {
                    'color_similarity': cache_item.get('color_similarity'),
                    'clip_similarity': cache_item.get('clip_similarity')
                }
    return similarity_cache

def save_similarity_cache(cache_path: str, key: str, color_similarity: Optional[Dict], clip_similarity: Optional[float]):
    """
    保存相似度缓存
    
    :param cache_path: 缓存 JSONL 文件路径
    :param key: 缓存项的唯一键
    :param color_similarity: 颜色相似度信息
    :param clip_similarity: CLIP 相似度
    """
    with open(cache_path, 'a') as f:
        cache_item = {
            'key': key,
            'color_similarity': color_similarity,
            'clip_similarity': clip_similarity
        }
        f.write(json.dumps(cache_item, ensure_ascii=False) + '\n')

def process_single_item(
    item: dict, 
    src_root: Path, 
    clip_similarity_threshold: float = 0.7, 
    cloth_only: bool = False, 
    clip_calculator: CLIPSimilarityCalculator = None,
    similarity_cache: Dict[str, Dict[str, float]] = None,
    similarity_cache_path: str = None
) -> dict:
    """
    处理单个数据项，检查是否包含上衣和外套叠穿，以及 person 和 cloth 的相似度
    
    :param similarity_cache: 相似度缓存
    :param similarity_cache_path: 相似度缓存文件路径
    """
    # 如果没有传入 clip_calculator，则创建一个
    if clip_calculator is None:
        clip_calculator = CLIPSimilarityCalculator()

    # 检查必要的键是否存在
    if 'person' not in item or 'cloth' not in item:
        return None

    # 构造完整的图片路径
    person_path = str(src_root / item['person'])
    cloth_path = str(src_root / item['cloth'])
    
    # 生成缓存键
    cache_key = f"{item['person']}_{item['cloth']}"
    
    # 如果存在缓存，直接使用缓存的相似度
    if similarity_cache and cache_key in similarity_cache:
        cached_similarities = similarity_cache[cache_key]
        color_result = cached_similarities['color_similarity']
        clip_similarity = cached_similarities['clip_similarity']
    else:
        # 构造掩码路径
        person_mask_path = person_path.replace('person', 'annotations/mask_v1').replace('.jpg', '.png')
        if 'cloth' in cloth_path:
            cloth_mask_path = cloth_path.replace('cloth', 'annotations/cloth_matting').replace('.jpg', '.png')
            cloth_mask = Image.open(cloth_mask_path)
        else:
            cloth_mask_path = cloth_path.replace('person', 'annotations/mask_v1').replace('.jpg', '.png')
            # cloth_mask = get_target_cloth_mask(cloth_mask_path, item['category'])

        # 检查文件是否存在
        required_files = [person_path, cloth_path, person_mask_path, cloth_mask_path]
        if not all(os.path.exists(f) for f in required_files):
            print(f"部分文件不存在: {[f for f in required_files if not os.path.exists(f)]}")
            return None
        
        # 检查服装区域是否过大
        person_lip = Image.open(person_path.replace('person', 'annotations/schp_lip').replace('.jpg', '.png'))
        if is_cloth_area_too_large(person_lip, threshold=0.5):
            save_concat_image([person_path, cloth_path], f'test-cloth-area-too-large/{os.path.basename(person_path)}')
            return None

        # 检查是否是上衣和外套叠穿
        if item['category'] == 'upper' or item['category'] == 'full':
            if person_mask_path in PROCESSED_LIP:
                flag = PROCESSED_LIP[person_mask_path]
            else:
                flag = wear_upper_with_outer(person_lip)
                PROCESSED_LIP[person_mask_path] = flag
            if flag:
                save_concat_image([person_path, cloth_path], f'test-wear-upper-with-outer/{os.path.basename(person_path)}')
                return None
        
        # 检查主色调相似度
        # person_mask = get_target_cloth_mask(person_lip, item['category'])
        person_mask = Image.open(person_mask_path)
        try:
            masked_person = crop_to_max_square_region(Image.open(person_path), person_mask)
            masked_cloth = crop_to_max_square_region(Image.open(cloth_path), cloth_mask)
            # color_result = are_dominant_colors_similar(
            #     masked_person, 
            #     masked_cloth,
            #     threshold=50
            # )
            # 计算 CLIP 相似度
            clip_similarity = clip_calculator.calculate_similarity(masked_person, masked_cloth)
            # 保存相似度缓存
            if similarity_cache_path:
                save_similarity_cache(
                    similarity_cache_path, 
                    cache_key, 
                    color_result["distance"], 
                    clip_similarity
                )
            if clip_similarity < clip_similarity_threshold:
                save_concat_image([masked_person, masked_cloth], f'test-clip/{clip_similarity}-{os.path.basename(person_path)}')
                return None
        except Exception as e:
            return None
            
        
    
    # 检查主色调相似度和 CLIP 相似度
    # if not color_result['is_similar']:
    #     save_concat_image([masked_person, masked_cloth], f'test-color/{color_result["distance"]}-{os.path.basename(person_path)}')
    #     return None
    


    return item

def clean_data(
    jsonl_path: str,
    output_jsonl_path: str = None,
    similarity_cache_path: str = None,
    clip_similarity_threshold: float = 0.7,
    cloth_only: bool = False
) -> List[Dict]:
    """
    单线程版本的数据清洗函数，用于调试
    
    :param jsonl_path: 输入的 JSONL 文件路径
    :param output_jsonl_path: 输出的 JSONL 文件路径
    :param similarity_cache_path: 相似度缓存文件路径
    :param clip_similarity_threshold: CLIP 相似度阈值
    :param cloth_only: 是否只保留服装和 person 配对的样本
    :return: 清洗后的数据列表
    """
    src_root = Path(jsonl_path).parent
    with open(jsonl_path, 'r') as f:
        items = [json.loads(line) for line in f]
    
    # 加载相似度缓存
    similarity_cache = load_similarity_cache(similarity_cache_path) if similarity_cache_path else {}
    if not similarity_cache:
        similarity_cache_path = f'{jsonl_path.replace(".jsonl", "")}_similarity_cache.jsonl'
    
    # 创建 CLIP 计算器实例
    clip_calculator = CLIPSimilarityCalculator()
    
    # 设置输出文件路径
    if output_jsonl_path is None:
        output_jsonl_path = jsonl_path.replace('.jsonl', '_cleaned.jsonl')
        
    # 加载已处理的条目
    processed_items = load_processed_items(output_jsonl_path)
    print(f"已处理条目数量: {len(processed_items)}")
    
    # 打开输出文件以追加模式写入
    with open(output_jsonl_path, 'a') as output_file:
        cleaned_items = []
        
        # 创建进度条
        progress_bar = tqdm(total=len(items), desc="清洗数据中")
        
        # 逐个处理数据项
        for idx, item in enumerate(items):
            # 检查是否已处理过
            item_hash = json.dumps(item, sort_keys=True)
            if item_hash in processed_items:
                progress_bar.update(1)
                continue
            
            # 如果 cloth_only 为 True，则只保留服装和 person 配对的样本
            if cloth_only and 'cloth' not in item['cloth']:
                progress_bar.update(1)
                continue
            
            try:
                # 处理单个数据项
                result = process_single_item(
                    item, 
                    src_root, 
                    clip_similarity_threshold,
                    cloth_only,
                    clip_calculator,
                    similarity_cache,
                    similarity_cache_path
                )
                
                if result is not None:
                    # 实时写入文件
                    output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                    output_file.flush()  # 立即刷新缓冲区
                    cleaned_items.append(result)
                
                # 更新进度条
                progress_bar.set_postfix_str(f"保留: {len(cleaned_items)}")
                progress_bar.update(1)
            
            except Exception as e:
                # 记录详细的错误信息
                print(f"处理数据项时发生错误: {e}")
                # 可以选择记录到日志文件
                with open('error_log.txt', 'a') as log_file:
                    log_file.write(f"Error processing item {idx}: {e}\n")
                
                # 继续处理下一个数据项
                progress_bar.update(1)
        
        # 关闭进度条
        progress_bar.close()
    
    print(f"清洗完成！原始数据量: {len(items)}, 清洗后数据量: {len(cleaned_items)}")
    return cleaned_items


def clean_data_single_thread(
    jsonl_path: str,
    output_jsonl_path: str = None,
    similarity_cache_path: str = None,
    clip_similarity_threshold: float = 0.7,
    cloth_only: bool = False
) -> List[Dict]:
    """
    单线程版本的数据清洗函数，用于调试
    
    :param jsonl_path: 输入的 JSONL 文件路径
    :param output_jsonl_path: 输出的 JSONL 文件路径
    :param similarity_cache_path: 相似度缓存文件路径
    :param clip_similarity_threshold: CLIP 相似度阈值
    :param cloth_only: 是否只保留服装和 person 配对的样本
    :return: 清洗后的数据列表
    """
    src_root = Path(jsonl_path).parent
    with open(jsonl_path, 'r') as f:
        items = [json.loads(line) for line in f]
    
    # 加载相似度缓存
    similarity_cache = load_similarity_cache(similarity_cache_path) if similarity_cache_path else {}
    if not similarity_cache:
        similarity_cache_path = f'{jsonl_path.replace(".jsonl", "")}_similarity_cache.jsonl'
    
    # 创建 CLIP 计算器实例
    clip_calculator = CLIPSimilarityCalculator()
    
    # 设置输出文件路径
    if output_jsonl_path is None:
        output_jsonl_path = jsonl_path.replace('.jsonl', '_cleaned.jsonl')
        
    # 加载已处理的条目
    processed_items = load_processed_items(output_jsonl_path)
    print(f"已处理条目数量: {len(processed_items)}")
    
    # 打开输出文件以追加模式写入
    with open(output_jsonl_path, 'a') as output_file:
        cleaned_items = []
        
        # 创建进度条
        progress_bar = tqdm(total=len(items), desc="清洗数据中")
        
        # 逐个处理数据项
        for idx, item in enumerate(items):
            # 检查是否已处理过
            item_hash = json.dumps(item, sort_keys=True)
            if item_hash in processed_items:
                progress_bar.update(1)
                continue
            
            # 如果 cloth_only 为 True，则只保留服装和 person 配对的样本
            if cloth_only and 'cloth' not in item['cloth']:
                progress_bar.update(1)
                continue
            
            # try:
            # 处理单个数据项
            result = process_single_item(
                item, 
                src_root, 
                clip_similarity_threshold,
                cloth_only,
                clip_calculator,
                similarity_cache,
                similarity_cache_path
            )
            
            if result is not None:
                # 实时写入文件
                output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                output_file.flush()  # 立即刷新缓冲区
                cleaned_items.append(result)
            
            # 更新进度条
            progress_bar.set_postfix_str(f"保留: {len(cleaned_items)}")
            progress_bar.update(1)
            
            # except Exception as e:
            #     # 记录详细的错误信息
            #     print(f"处理数据项时发生错误: {e}")
            #     # 可以选择记录到日志文件
            #     with open('error_log.txt', 'a') as log_file:
            #         log_file.write(f"Error processing item {idx}: {e}\n")
                
            #     # 继续处理下一个数据项
            #     progress_bar.update(1)
        
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
    parser.add_argument('--similarity_cache_path', type=str, default=None,
                      help='相似度缓存文件路径')
    parser.add_argument('--max_workers', type=int, default=16,
                      help='最大线程数 (默认: 16)')
    parser.add_argument('--clip_similarity_threshold', type=float, default=0.78,
                      help='CLIP相似度阈值 (默认: 0.78)')
    parser.add_argument('--cloth_only', action='store_true',
                      help='是否只保留服装和person配对的样本')
    
    args = parser.parse_args()
    
    clean_data_single_thread(
        jsonl_path=args.jsonl_path,
        output_jsonl_path=args.output_jsonl_path,
        similarity_cache_path=args.similarity_cache_path,
        # max_workers=args.max_workers,
        clip_similarity_threshold=args.clip_similarity_threshold,
        cloth_only=args.cloth_only
    )

if __name__ == '__main__':
    main()
