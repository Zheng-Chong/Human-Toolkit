# Human-Toolkit

基于 开源人体解析与姿态估计模型 的批量预处理工具，可以自动处理大量图片并生成分割和姿态估计的结果。

## 功能特点

- 支持批量处理图片
- 多线程并行处理，提高效率
- 支持多种分析工具：
  - DensePose
  - SCHP-ATR
  - SCHP-LIP
  - SCHP-Pascal
- 自动下载并使用预训练模型

## 使用方法

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
python main.py \
    --input_dir ./images \
    --output_dir ./results \
    --tools densepose schp_atr \
    --num_workers 8
```

## 输出结果

程序会在 `output_dir`下生成以下文件：

- `densepose/{图片名}.png`：DensePose 处理结果
- `schp_atr/{图片名}.png`：SCHP-ATR 处理结果
- `schp_lip/{图片名}.png`：SCHP-LIP 处理结果
- `schp_pascal/{图片名}.png`：SCHP-Pascal 处理结果

## 注意事项

1. 支持的图片格式：JPG、JPEG、PNG、BMP
2. 请确保有足够的磁盘空间存储处理结果
3. 首次运行时会自动下载模型文件，需要保持网络连接
