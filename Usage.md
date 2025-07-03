# 3D-recon 项目脚本说明

## 1. main.py
批量或单个视频帧提取主入口。支持命令行和YAML配置，支持多种帧采样模式（间隔、均匀、随机），可并行处理多个视频。

- 输入：视频文件或视频目录
- 输出：帧图片目录
- 依赖：extractor.py, config.py

### 参数示例
```bash
python main.py --input_dir ./videos --output_dir ./frames --frame_interval 10 --mode interval --workers 8
```
或使用配置文件：
```yaml
# extract_config.yaml
input_dir: ./videos
output_dir: ./frames
frame_interval: 10
mode: interval
workers: 8
```

## 2. config.py
YAML配置文件加载与命令行参数合并工具。

- 提供load_config和merge_config_args两个函数，便于主流程灵活配置。

## 3. extractor.py
视频帧提取核心类VideoFrameExtractor，实现多种采样模式的帧提取。

- 支持间隔采样、均匀采样、随机采样
- 依赖OpenCV

## 4. extract_frames.py
命令行入口脚本，调用main.py实现基于配置文件的视频帧批量提取。

- 用法：
```bash
python extract_frames.py --config extract_config.yaml
```

## 5. extract_blendedmvs.py
BlendedMVS数据集提取脚本。批量提取图片、掩码、深度图和相机参数到指定输出目录，适配SAM+COLMAP+OpenMVS流程。

- 输入：BlendedMVS原始数据集目录
- 输出：每个场景一个子目录，包含images/、masks/、depths/、cams/、workspace/
- 支持通过YAML配置文件批量指定场景

### 参数示例
```yaml
# extract_blended.yaml
root: BlendedMVS
output: ./Reconstruction
scenes:
  - 5a3ca9cb270f0e3f14d0eddb
  - 5a3cb4e4270f0e3f14d12f43
copy_masked: false
```
用法：
```bash
python extract_blendedmvs.py --config extract_blended.yaml
```

## 6. reconstruct.py
COLMAP+OpenMVS 3D重建主流程脚本。

- 输入：图片目录、配置文件
- 步骤：COLMAP特征提取、匹配、稀疏重建，OpenMVS稠密重建
- 支持断点续跑和检查点
- 自动适配masks目录作为掩码输入

### 参数示例
```yaml
# reconstruct_config.yaml
image_dir: ./Reconstruction/scene_id/images
workspace: ./Reconstruction/scene_id/workspace
colmap_bin: colmap
openmvs_bin: /usr/local/bin
```
用法：
```bash
python reconstruct.py --config reconstruct_config.yaml
```

## 7. segment_and_reconstruct.py
主体分割+3D重建一体化脚本。集成SAM2模型自动为每个场景生成掩码，并调用COLMAP+OpenMVS完成重建。

- 输入：配置文件（支持指定sam2权重和配置）
- 步骤：
  1. 遍历所有场景，对images/生成masks/
  2. 为每个场景单独执行COLMAP+OpenMVS重建，结果输出到workspace/
- 支持跳过分割、自动适配目录结构

### 参数示例
```yaml
# segment_reconstruct_config.yaml
workspace: ./Reconstruction
sam2_model: hiera_t
sam2_checkpoint: ./sam2/checkpoints/sam2_hiera_t.pth
sam2_config: ./sam2/sam2/sam2_hiera_t.yaml
device: cuda
colmap_bin: colmap
openmvs_bin: /usr/local/bin
```
用法：
```bash
python segment_and_reconstruct.py --config segment_reconstruct_config.yaml
```

---

## 项目整体流程图

```mermaid
graph TD
    A[视频/数据集] --> B[extract_frames.py / extract_blendedmvs.py]
    B --> C[images/、masks/、cams/等]
    C --> D[segment_and_reconstruct.py]
    D --> E[分割生成masks/]
    E --> F[reconstruct.py (COLMAP+OpenMVS)]
    F --> G[workspace/（重建结果）]
```

---

如需详细参数说明或用法示例，请查阅各脚本开头注释或命令行 --help。
