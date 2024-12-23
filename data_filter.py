import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import List, Dict, Set, Optional
from data_cleaner.utils import crop_to_max_square_region, get_target_cloth_mask, is_image_clear, match_images, wear_upper_with_outer, are_dominant_colors_similar, has_more_than_three_white_edges, is_cloth_area_too_large, gray_similarity
from data_cleaner.clip_similarity import CLIPSimilarityCalculator
from PIL import Image
import numpy as np
from threading import Lock
import argparse  # 新增导入

# 全局变量
LINK_SYMBOL = '=='
CLIP_SIMILARITY_MODEL = CLIPSimilarityCalculator()
ITEM_CACHE_PATH = None
ITEM_CACHE = dict()
FILTERED_NUM = 0

def load_item_cache():
    """
    加载项目缓存，如果缓存文件不存在则创建空文件
    """
    global ITEM_CACHE_PATH, ITEM_CACHE
    if ITEM_CACHE_PATH is None:
        return
    
    # 如果缓存文件不存在，则创建一个空的 JSON 文件
    if not os.path.exists(ITEM_CACHE_PATH):
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(ITEM_CACHE_PATH), exist_ok=True)
            
            # 创建一个空的 JSON 文件
            with open(ITEM_CACHE_PATH, 'w') as f:
                json.dump({}, f)
        except Exception as e:
            print(f"创建缓存文件时出错: {e}")
            return
    
    # 尝试读取缓存文件
    try:
        with open(ITEM_CACHE_PATH, 'r') as f:
            # 使用 .get() 方法防止 JSON 解析错误
            ITEM_CACHE.update(json.load(f) or {})
    except json.JSONDecodeError:
        # 如果文件内容不是有效的 JSON，则重置为空字典
        ITEM_CACHE.clear()
    except Exception as e:
        print(f"读取缓存文件时出错: {e}")

def add_item_cache(key, value):
    global ITEM_CACHE_PATH, ITEM_CACHE
    p, c = key.split(LINK_SYMBOL)
    items = [
        {f"{c}{LINK_SYMBOL}{p}": value},
        {f"{p}{LINK_SYMBOL}{c}": value}
    ]
    with open(ITEM_CACHE_PATH, 'a') as f:
        for item in items:
            f.write(json.dumps(item) + '\n')
            ITEM_CACHE.update(item)
        
def save_concat_image(input_list: List[str], output_path: str):
    """
    将输入的图片列表拼接后保存到指定路径
    """
    return
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

def process_single_item(
    item: dict,
    src_root: Path,
    cloth_area_threshold: float = 0.5,
    clip_similarity_threshold: float = 0.78,
    hist_similarity_threshold: float = 0.8,
):
    """
    处理单个数据项，检查是否包含上衣和外套叠穿，以及 person 和 cloth 的相似度
    """
    # 检查必要的键是否存在
    if 'person' not in item or 'cloth' not in item or 'category' not in item:
        return None
    
    # 构造完整的图片路径
    pair_key = f"{os.path.splitext(os.path.basename(item['person']))[0]}{LINK_SYMBOL}{os.path.splitext(os.path.basename(item['cloth']))[0]}"
    item_cache = ITEM_CACHE[pair_key] if (is_cached := pair_key in ITEM_CACHE) else {}
    person_path = str(src_root / item['person'])
    cloth_path = str(src_root / item['cloth'])
    
    # 获取图片尺寸
    person_image = Image.open(person_path)
    cloth_image = Image.open(cloth_path)
    person_size = person_image.size
    cloth_size = cloth_image.size
    
    # 获取需要的 mask
    person_lip = Image.open(str(src_root / item['person']).replace('/person/', '/annotations/schp_lip/').replace('.jpg', '.png'))
    person_agnostic_mask = Image.open(str(src_root / item['person']).replace('/person/', '/annotations/mask_v1/').replace('.jpg', '.png'))
    if 'person' in item['cloth']:
        cloth_mask = Image.open(str(src_root / item['cloth']).replace('/person/', '/annotations/mask_v1/').replace('.jpg', '.png'))
        cloth_lip = Image.open(str(src_root / item['cloth']).replace('/person/', '/annotations/schp_lip/').replace('.jpg', '.png'))
    else:
        cloth_mask = Image.open(str(src_root / item['cloth']).replace('/cloth/', '/annotations/cloth_matting/').replace('.jpg', '.png'))

    # 1. Person 服装区域占比是否 > cloth_area_threshold
    if 'cloth_ratio' in item_cache:
        cloth_ratio = item_cache['cloth_ratio']
    else:
        cloth_ratio = is_cloth_area_too_large(person_lip, threshold=cloth_area_threshold)
        item_cache['cloth_ratio'] = cloth_ratio
    
    if cloth_ratio > cloth_area_threshold:
        if not is_cached:
            save_concat_image([person_path, cloth_path], f"test_filter/cloth_area_too_large/{pair_key}.jpg")
            add_item_cache(pair_key, item_cache)
        return None
    
    # 2. 上衣类型中存在上衣和外套叠穿
    if item['category'] == 'upper' or item['category'] == 'full':
        if 'wear_upper_with_outer' in item_cache:
            wear_upper_with_outer_flag = item_cache['wear_upper_with_outer']
        else:
            wear_upper_with_outer_flag = wear_upper_with_outer(person_lip)
            item_cache['wear_upper_with_outer'] = 1 if wear_upper_with_outer_flag else 0
        if wear_upper_with_outer_flag:
            if not is_cached:
                save_concat_image([person_path, cloth_path], f"test_filter/wear_upper_with_outer/{pair_key}.jpg")
                add_item_cache(pair_key, item_cache)
            return None
    
    # 3. 服装和人物服装区域的相似度 > clip_similarity_threshold
    # if 'clip_similarity' in item_cache:
    #     clip_similarity = item_cache['clip_similarity']
    # else:
    #     def compose_cloth_and_agnostic_mask(cloth_mask, agnostic_mask):
    #         cloth_mask = np.array(cloth_mask)
    #         composed_mask = np.array(agnostic_mask)
    #         composed_mask[cloth_mask == 0] = 0
    #         return Image.fromarray(composed_mask, mode='L')
        
    #     cloth_mask_of_person = get_target_cloth_mask(person_lip, "full")
    #     person_agnostic_mask_path = str(person_path).replace('/person/', '/annotations/mask_v1/').replace('.jpg', '.png')
    #     agnotic_mask_of_person = Image.open(person_agnostic_mask_path)
    #     composed_mask_of_person = compose_cloth_and_agnostic_mask(cloth_mask_of_person, agnotic_mask_of_person)
    #     person_size = composed_mask_of_person.size
        
    #     if 'cloth' in cloth_path:  # in-shop 服装图
    #         cloth_mask_of_cloth_path = str(cloth_path).replace('/cloth/', '/annotations/cloth_matting/').replace('.jpg', '.png')
    #         cloth_mask_of_cloth = Image.open(cloth_mask_of_cloth_path)
    #     else:  # 模特图
    #         cloth_mask_of_cloth_path = str(cloth_path).replace('/person/', '/annotations/mask_v1/').replace('.jpg', '.png')
    #         cloth_mask_of_cloth = Image.open(cloth_mask_of_cloth_path)
    #         agnostic_mask_of_cloth_path = str(cloth_path).replace('/person/', '/annotations/mask_v1/').replace('.jpg', '.png')
    #         agnostic_mask_of_cloth = Image.open(agnostic_mask_of_cloth_path)
    #         cloth_mask_of_cloth = compose_cloth_and_agnostic_mask(cloth_mask_of_cloth, agnostic_mask_of_cloth)
    #     cloth_size = cloth_mask_of_cloth.size
            
    #     masked_person = crop_to_max_square_region(Image.open(person_path), composed_mask_of_person)
    #     masked_cloth = crop_to_max_square_region(Image.open(cloth_path), cloth_mask_of_cloth)
        
    #     # try:    
    #     clip_similarity = CLIP_SIMILARITY_MODEL.calculate_similarity(
    #         masked_person, masked_cloth,
    #     )
    #     # except Exception as e:
    #     #     print(f"[Clip Similarity] 计算相似度时出错: {e}")
    #     #     clip_similarity = 0
    #     item_cache['clip_similarity'] = clip_similarity 
    # if clip_similarity < clip_similarity_threshold:
    #     if not is_cached:
    #         save_concat_image([masked_person, masked_cloth], f"test_filter/clip_similarity/{clip_similarity:.2f}_{pair_key}.jpg")
    #         add_item_cache(pair_key, item_cache)
    #     return None
    
    # 4. 双人物的情况下 Person mask_v1 占比 < cloth mask_v1 占比
    if 'person' in item['cloth']:
        person_agnostic_mask_np = np.array(person_agnostic_mask)
        person_mask_v1_ratio = np.sum(person_agnostic_mask_np == 255) / person_agnostic_mask_np.size
        cloth_mask_np = np.array(cloth_mask)
        cloth_mask_v1_ratio = np.sum(cloth_mask_np == 255) / cloth_mask_np.size
        if person_mask_v1_ratio < cloth_mask_v1_ratio:
            return None
        
    # 5. 灰度相似度
    person_mask = get_target_cloth_mask(person_lip, item['category'])
    cloth_mask = cloth_mask if 'cloth' in item['cloth'] else get_target_cloth_mask(cloth_lip, item['category'])
    s_gray = gray_similarity(Image.open(person_path), Image.open(cloth_path), person_mask, cloth_mask)
    if s_gray < hist_similarity_threshold:
        # # compose image and mask
        # person_mask = np.array(person_mask) / 255
        # cloth_mask = np.array(cloth_mask) / 255
        # person_image = np.array(person_image).astype(np.float32)
        # cloth_image = np.array(cloth_image).astype(np.float32)
        
        # # 正确地应用 mask
        # person_image_masked = person_image.copy()
        # cloth_image_masked = cloth_image.copy()
        
        # for i in range(3):  # 处理 RGB 三个通道
        #     person_image_masked[:,:,i] *= person_mask
        #     cloth_image_masked[:,:,i] *= cloth_mask
        
        # # 将图像转换回 uint8 类型
        # person_image_masked = (person_image_masked).clip(0, 255).astype(np.uint8)
        # cloth_image_masked = (cloth_image_masked).clip(0, 255).astype(np.uint8)
        
        # person_image_masked = Image.fromarray(person_image_masked)
        # cloth_image_masked = Image.fromarray(cloth_image_masked)
        
        # save_concat_image([person_image_masked, cloth_image_masked], f"test_filter/gray_similarity/{s_gray:.2f}_{pair_key}.jpg")
        return None
        
    
    if not is_cached:
        add_item_cache(pair_key, item_cache)
        
    if 'person_size' not in item:
        item['person_size'] = person_size
    if 'cloth_size' not in item:
        item['cloth_size'] = cloth_size
        
    return item

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
                processed_items.add(f"{item['person']}{LINK_SYMBOL}{item['cloth']}")
    return processed_items

def clean_data(
    jsonl_path: str,
    output_jsonl_path: str = None,
    cloth_area_threshold: float = 0.5,
    clip_similarity_threshold: float = 0.78,
    hist_similarity_threshold: float = 0.8,
    max_workers: int = None  # 新增线程数参数
):
    global ITEM_CACHE_PATH, FILTERED_NUM
    
    ITEM_CACHE_PATH = f"{jsonl_path.replace('.jsonl', '')}_item_cache.jsonl"
    load_item_cache()
    
    src_root = Path(jsonl_path).parent
    with open(jsonl_path, 'r') as f:
        items = [json.loads(line) for line in f]
    
    # 设置输出文件路径
    if output_jsonl_path is None:
        output_jsonl_path = jsonl_path.replace('.jsonl', '_cleaned.jsonl')

    # 加载已处理的条目
    processed_items = load_processed_items(output_jsonl_path)
    print(f"已处理条目数量: {len(processed_items)}")
    
    # 过滤已处理条目
    items = [item for item in items if f"{item['person']}{LINK_SYMBOL}{item['cloth']}" not in processed_items]
    
    # 线程安全的文件写入锁
    file_lock = Lock()
    
    with open(output_jsonl_path, 'a') as output_file:
        # 创建进度条
        progress_bar = tqdm(total=len(items), desc="清洗数据中", position=0, leave=True)
        
        # 使用线程池处理数据
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            def process_and_write(item):
                global FILTERED_NUM
                try:
                    result = process_single_item(
                        item, 
                        src_root, 
                        cloth_area_threshold, 
                        clip_similarity_threshold,
                        hist_similarity_threshold
                    )
                except Exception as e:
                    print(f"[Process Single Item] 处理单个数据项时出错: {e}")
                    return None
                
                # 使用锁确保线程安全地写入文件
                with file_lock:
                    if result is not None:
                        output_file.write(json.dumps(result) + '\n')
                        output_file.flush()  # 立即刷新缓冲区
                        FILTERED_NUM += 1
                        
                progress_bar.set_description(f"已过滤条目数量: {FILTERED_NUM}")
                progress_bar.update(1)
                return result

            # 提交所有任务并等待完成
            list(executor.map(process_and_write, items))
            
        progress_bar.close()

def parse_arguments():
    """
    解析命令行参数
    
    :return: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description='数据清洗工具')
    parser.add_argument(
        '--input_jsonl', 
        type=str, 
        help='输入的 JSONL 文件路径'
    )
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        default=None, 
        help='输出的 JSONL 文件路径（默认为输入文件名 + _cleaned.jsonl）'
    )
    parser.add_argument(
        '-c', '--cloth_area_threshold', 
        type=float, 
        default=0.45, 
        help='服装区域占比阈值（默认：0.45）'
    )
    parser.add_argument(
        '-s', '--clip_similarity_threshold', 
        type=float, 
        default=0.79, 
        help='CLIP 相似阈值（默认：0.79）'
    )
    parser.add_argument(
        '-hs', '--hist_similarity_threshold', 
        type=float, 
        default=0.85, 
        help='直方图相似度阈值（默认：0.85）'
    )
    parser.add_argument(
        '-w', '--workers', 
        type=int, 
        default=16, 
        help='线程池工作线程数（默认：CPU 核心数）'
    )
    
    return parser.parse_args()

def main():
    """
    主函数，处理命令行参数并用数据清洗函数
    """
    args = parse_arguments()
    
    clean_data(
        jsonl_path=args.input_jsonl,
        output_jsonl_path=args.output,
        cloth_area_threshold=args.cloth_area_threshold,
        clip_similarity_threshold=args.clip_similarity_threshold,
        hist_similarity_threshold=args.hist_similarity_threshold,
        max_workers=args.workers
    )

if __name__ == '__main__':
    main()

