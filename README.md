# Human-Toolkit

基于开源人体解析与姿态估计模型的批量预处理工具，可以自动处理大量图片并生成分割、姿态估计及服装遮罩的结果。

## 功能特点

- 支持批量处理图片
- 多线程并行处理，提高效率
- 支持多种分析工具：
  - DensePose：人体密集姿态估计
  - SCHP-ATR：人体语义分割（ATR数据集）
  - SCHP-LIP：人体语义分割（LIP数据集）
  - SCHP-Pascal：人体语义分割（Pascal数据集）
- 支持合成服装遮罩：
  - 自动合成上衣/下装遮罩
  - 基于多个预处理结果的智能融合
- 自动下载并使用预训练模型


## 安装

```bash
conda create -n human-toolkit python=3.10
conda activate human-toolkit
pip install -r requirments.txt
python -m pip install -e GroundingDINO  # C++ 扩展
```

## 基础预处理工具

### 使用方法

基本用法：
```bash
python main.py --input_dir 图片目录 --output_dir 输出目录 --tools 工具1 工具2 工具3 --model_zoo_root 模型库地址 --num_workers 线程数
```

### 参数说明

- `--input_dir`：输入图片所在目录（必需）
- `--output_dir`：处理结果保存目录（必需）
- `--tools`：选择使用的处理工具，可多选 [densepose, schp_atr, schp_lip, schp_pascal]
- `--model_zoo_root`：模型库地址，默认为 "zhengchong/Human-Toolkit"
- `--num_workers`：并行处理的线程数，默认为 4

### 示例

```bash
CUDA_VISIBLE_DEVICES=1 python main.py \
    --input_dir ./images \
    --output_dir ./annotations \
    --tools densepose schp_atr schp_lip schp_pascal \
    --num_workers 8
```

### 输出结果

程序会在 `output_dir`下生成以下文件：

- `densepose/{图片名}.png`：DensePose 处理结果
- `schp_atr/{图片名}.png`：SCHP-ATR 处理结果
- `schp_lip/{图片名}.png`：SCHP-LIP 处理结果
- `schp_pascal/{图片名}.png`：SCHP-Pascal 处理结果

## Mask 合成工具

### 使用方法

基本用法：
```bash
python mask_compose.py --jsonl_path 数据文件.jsonl --max_workers 线程数
```

### 数据文件层级  
`--jsonl_path` 所在目录需包含 person/ 和 cloth/ 两个目录，且与 annotations/ 同级
如：
```
ROOT/
├── annotations/
│   ├── densepose/
│   ├── schp_atr/
│   ├── schp_lip/
│   └── schp_pascal/
├── person/
├── cloth/
└── data.jsonl
```


### 参数说明

- `--jsonl_path`：输入的 JSONL 文件路径（必需），每行包含以下格式的 JSON：
  ```json
  {
    "person": "person/图片路径.jpg", # 与 --jsonl_path 同级目录
    "cloth": "cloth/图片路径.jpg", # 与 --jsonl_path 同级目录
    "category": "upper"  # 或 "lower", "full", 表示上衣、下装、全身
  }
  ```
  上面预处理的结果目录 annotations/ 需与 person/ 和 cloth/ 目录同级
- `--output_dir`：处理结果保存目录（必需），建议 `annotations/mask-v1`
- `--max_workers`：并行处理的线程数，默认为 4

### 输出结果

程序会在 `output_dir` 下生成与输入图片对应的 mask 文件（PNG格式）。

## Matting 工具

### 使用方法

基本用法：
```bash
python matting_cloth.py --jsonl_path 数据文件.jsonl --max_workers 线程数
```

### 数据文件层级
与mask合成工具类似，`--jsonl_path` 所在目录需包含 cloth/ 目录：
```
ROOT/
├── annotations/
│   ├── cloth_matting/  # matting结果保存目录
│   └── ...
├── cloth/
└── data.jsonl
```

### 参数说明

- `--jsonl_path`：输入的 JSONL 文件路径（必需），每行包含以下格式的 JSON：
  ```json
  {
    "cloth": "cloth/图片路径.jpg",  # 与 --jsonl_path 同级目录
    "category": "upper"  # 或 "lower"，表示上衣或下装
  }
  ```
- `--output_dir`：处理结果保存目录，默认为 `annotations/cloth_matting`
- `--max_workers`：并行处理的线程数，默认为 4
- `--device`：运行设备，默认为 'cuda'
- `--working_size`：处理图像的工作尺寸，默认为 1024
- `--part_filter`：处理类别过滤，可指定多个类别（例如：--part_filter upper lower）

### 输出结果

程序会在 `output_dir` 下生成与输入服装图片对应的matting遮罩文件（PNG格式）。matting遮罩是一个灰度图像，其中：
- 255表示完全属于服装的区域
- 0表示完全属于背景的区域
- 中间值表示半透明区域

### 示例

```bash
# 基本用法
python matting_cloth.py --jsonl_path ./data.jsonl

# 指定输出目录和线程数
python matting_cloth.py --jsonl_path ./data.jsonl --output_dir ./annotations/cloth_matting --max_workers 8

# 只处理上衣
python matting_cloth.py --jsonl_path ./data.jsonl --part_filter upper
```

## Image Caption 工具

### 使用方法

基本用法：
```bash
python image_caption.py --input_dir 图片目录 --output_file 输出文件.jsonl --num_workers 线程数
```

### 功能特点

- 支持批量处理图片目录
- 多GPU并行处理，提高效率
- 为每张图片生成三种不同详细程度的描述
- 支持断点续传，可以继续处理未完成的任务
- 使用 JSONL 格式保存结果，便于后续处理

### 参数说明

- `--input_dir`：输入图片所在目录（必需）
- `--output_file`：输出的 JSONL 文件路径（必需）
- `--num_workers`：并行处理的线程数，默认为 4

### 输出格式

程序会将处理结果保存为 JSONL 格式，每行包含一个图片的处理结果：
```json
{
    "image_path": "图片路径",
    "captions": {
        "<CAPTION>": "简单描述",
        "<DETAILED_CAPTION>": "详细描述",
        "<MORE_DETAILED_CAPTION>": "更详细的描述"
    }
}
```

### 示例

```bash
# 基本用法
python image_caption.py --input_dir ./images --output_file captions.jsonl

# 使用8个线程处理
python image_caption.py --input_dir ./images --output_file captions.jsonl --num_workers 8
```

### 注意事项

1. 首次运行时会自动下载 Florence-2-large 模型，需要保持网络连接
2. 程序支持断点续传，如果处理过程中断，再次运行时会自动跳过已处理的图片
3. 建议根据显卡数量和内存大小调整 num_workers 参数
4. 支持处理的图片格式：JPG、JPEG、PNG、BMP

## 注意事项

1. 支持的图片格式：JPG、JPEG、PNG、BMP
2. 请确保有足够的磁盘空间存储处理结果
3. 首次运行时会自动下载模型文件，需要保持网络连接
