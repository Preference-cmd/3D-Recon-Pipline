#!/usr/bin/env python3
"""
主体分割重建脚本
集成SAM2模型进行主体分割,然后执行COLMAP+OpenMVS 3D重建pipeline
适配新的目录结构：每个场景独立子目录
"""

import argparse
from cv2.typing import MatLike
import yaml
import os
import subprocess
import json
import shutil
import sys
import cv2
import numpy as np
from pathlib import Path
import torch
from typing import List, Dict, Tuple, Optional
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor


def norm_path(path):
    """标准化路径，统一使用正斜杠"""
    return os.path.abspath(path).replace("\\", "/")

def run(cmd, check_output=None, cwd=None):
    """执行命令并处理输出"""
    print(f"[RUN] {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        if check_output and os.path.exists(check_output) and os.path.getsize(check_output) > 0:
            print(f"[WARN] Command failed (code {result.returncode}) but output file exists: {check_output}")
        else:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

def load_checkpoints(path):
    """加载检查点文件"""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_checkpoints(path, checkpoints):
    """保存检查点文件"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(checkpoints, f, ensure_ascii=False, indent=2)

def ensure_images_link(workspace, img_dir):
    """确保images软链接存在"""
    images_link = os.path.join(workspace, 'images')
    if os.path.exists(images_link):
        return
    try:
        os.symlink(os.path.abspath(img_dir), images_link)
        print(f"[INFO] Created symlink: {images_link} -> {img_dir}")
    except (AttributeError, NotImplementedError, OSError):
        print(f"[INFO] Symlink not supported, copying images to {images_link}")
        shutil.copytree(img_dir, images_link)

def load_and_update_config(config_path):
    """加载并更新配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    workspace = config['workspace']
    # 自动建立model_map - 适配新结构
    if 'model_map' not in config:
        # 查找workspace下的所有场景子目录
        scene_dirs = []
        for item in os.listdir(workspace):
            item_path = os.path.join(workspace, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'images')):
                scene_dirs.append(item)
        scene_dirs = sorted(scene_dirs)
        model_map = {i: name for i, name in enumerate(scene_dirs)}
        config['model_map'] = model_map
        # 回写到配置文件
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
    else:
        model_map = config['model_map']
    return config, model_map

class SAM2Segmenter:
    """SAM2分割器类"""
    
    def __init__(
        self, 
        model_type: str = "hiera_b+", 
        device: str = "cuda", 
        checkpoint: Optional[str] = None, 
        mode: str = "auto", 
        config_path: Optional[str] = None,
        params: Dict = {},
    ):
        """
        初始化SAM2分割器
        
        Args:
            model_type: 模型类型 (hiera_t, hiera_s, hiera_b+, hiera_l)
            device: 设备类型 (cuda, cpu)
            checkpoint: 模型权重路径
            mode: 分割模式 (auto, prompt)
            config_path: 模型配置文件路径
        """
        self.device = device
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.config_path = config_path
        self.mode = mode
        self.params = params


        if mode == 'auto':
            self.mask_generator = self.init_automatic_mask_generator()
            print(f"[INFO] SAM2 Automatic Mask Generator loaded successfully on {device}")
        elif mode == 'prompt':
            self.mask_generator = self.init_prompt_mask_generator()
            print(f"[INFO] SAM2 Prompt Mask Generator loaded successfully on {device}")
        else:
            raise ValueError(f"Invalid segmentation mode: {mode}")
        
    def init_automatic_mask_generator(self):
        """
        初始化自动掩码生成器
        
        """
        # 初始化SAM2自动分割模型

        print(f"[INFO] Loading SAM2 Automatic Mask Generator: {self.model_type}")
        predictor = build_sam2(self.config_path, self.checkpoint, self.device, apply_postprocessing=False)
        
        # 初始化自动掩码生成器
        return SAM2AutomaticMaskGenerator(
            predictor,
            points_per_side=self.params.get('points_per_side', 32),
            pred_iou_thresh=self.params.get('pred_iou_thresh', 0.86),
            stability_score_thresh=self.params.get('stability_score_thresh', 0.92),
            crop_n_layers=self.params.get('crop_n_layers', 1),
            crop_n_points_downscale_factor=self.params.get('crop_n_points_downscale_factor', 2),
            min_mask_region_area=self.params.get('min_mask_region_area', 100),
        )

    def init_prompt_mask_generator(self):
        """
        初始化提示掩码生成器
        
        """

        print(f"[INFO] Loading SAM2 Prompt Mask Generator: {self.model_type}")
        predictor = build_sam2(self.config_path, self.checkpoint, self.device, apply_postprocessing=False)
        
        # 初始化提示掩码生成器
        return SAM2ImagePredictor(
            predictor,
        )

    def segment_image(self, image_path: str, mask_dir: str) -> str:
        """
        对单张图片进行分割，生成mask
        
        Args:
            image_path: 输入图片路径
            mask_dir: mask输出目录
            
        Returns:
            生成的mask路径
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print(f"[INFO] Generating masks for {os.path.basename(image_path)}")
        
        if type(self.mask_generator) == SAM2AutomaticMaskGenerator:
            # 自动分割模式生成掩码
            masks = self.mask_generator.generate(image_rgb)
        elif type(self.mask_generator) == SAM2ImagePredictor:
            # 提示分割模式生成掩码
            pass
        else:
            raise ValueError(f"Invalid mask generator type: {type(self.mask_generator)}")




        
        if not masks:
            print(f"[WARN] No masks generated for {image_path}")
            # 生成全白mask
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        else:
            # 找到最大的掩码（通常是主体）
            largest_mask = max(masks, key=lambda x: x['area'])
            mask = largest_mask['segmentation'].astype(np.uint8) * 255
        
        # 保存mask为灰度PNG，文件名为原图名+.png
        mask_filename = os.path.basename(image_path) + ".png"
        mask_path = os.path.join(mask_dir, mask_filename)
        cv2.imwrite(mask_path, mask)
        
        print(f"[INFO] Mask saved: {mask_path}")
        return mask_path
    
    def segment_image_prompt(self, image_rgb: MatLike):
        pass
    
    def segment_scene(self, scene_dir: str) -> List[str]:
        """
        对单个场景进行分割
        
        Args:
            scene_dir: 场景目录路径
            
        Returns:
            生成的mask路径列表
        """
        images_dir = os.path.join(scene_dir, 'images')
        masks_dir = os.path.join(scene_dir, 'masks')
        
        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory not found: {images_dir}")
        
        # 保证masks目录结构与images一致
        for root, dirs, files in os.walk(images_dir):
            rel_dir = os.path.relpath(root, images_dir)
            target_mask_dir = os.path.join(masks_dir, rel_dir) if rel_dir != '.' else masks_dir
            os.makedirs(target_mask_dir, exist_ok=True)
            for filename in files:
                if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']):
                    image_path = os.path.join(root, filename)
                    try:
                        self.segment_image(image_path, target_mask_dir)
                    except Exception as e:
                        print(f"[ERROR] Failed to segment {filename}: {e}")
                        # 如果分割失败，生成全白mask
                        img = cv2.imread(image_path)
                        if img is not None:
                            mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
                            mask_filename = filename + ".png"
                            mask_path = os.path.join(target_mask_dir, mask_filename)
                            cv2.imwrite(mask_path, mask)
        return []

def prepare_openmvs_input(workspace, img_dir, sparse_i, subdir):
    """准备OpenMVS输入"""
    tmp_dir = os.path.join(workspace, f'tmp_openmvs_{subdir}')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    # 复制 images
    shutil.copytree(img_dir, os.path.join(tmp_dir, 'images'))
    # 复制 sparse_i 到 tmp_dir/sparse
    shutil.copytree(sparse_i, os.path.join(tmp_dir, 'sparse'))
    return tmp_dir

def check_file_exists(filepath, description=None):
    """检查文件是否存在且非空"""
    if filepath and os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        msg = f"[SKIP] {filepath} 已存在，跳过该步。"
        if description:
            msg = f"[SKIP] {description} ({filepath}) 已存在，跳过该步。"
        print(msg)
        return True
    return False

def check_and_fix_masks(images_dir, masks_dir):
    for root, dirs, files in os.walk(images_dir):
        rel_dir = os.path.relpath(root, images_dir)
        mask_subdir = os.path.join(masks_dir, rel_dir) if rel_dir != '.' else masks_dir
        os.makedirs(mask_subdir, exist_ok=True)
        for filename in files:
            if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']):
                mask_expected = os.path.join(mask_subdir, filename + ".png")
                if not os.path.exists(mask_expected):
                    # 查找可能的旧掩码名
                    candidates = [
                        os.path.join(mask_subdir, filename),
                        os.path.join(mask_subdir, os.path.splitext(filename)[0] + ".png"),
                    ]
                    for cand in candidates:
                        if os.path.exists(cand):
                            print(f"[FIX] Renaming {cand} -> {mask_expected}")
                            os.rename(cand, mask_expected)
                            break

def main():
    parser = argparse.ArgumentParser(description="主体分割重建pipeline - 集成SAM2 + COLMAP + OpenMVS")
    parser.add_argument('--config', type=str, default='segment_reconstruct_config.yaml', help='Path to reconstruction config YAML')
    parser.add_argument('--sam2-model', type=str, default='hiera_t', 
                       choices=['hiera_t', 'hiera_s', 'hiera_b+', 'hiera_l'],
                       help='SAM2 model type')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device for SAM2 inference')
    parser.add_argument('--skip-segmentation', action='store_true',
                       help='Skip segmentation step if masks already exist')
    parser.add_argument('--force', action='store_true', help='强制重新分割，覆盖已有masks')
    args = parser.parse_args()
    
    # 加载配置
    config_path = args.config
    config, model_map = load_and_update_config(config_path)
    workspace = norm_path(config['workspace'])
    colmap = config.get('colmap_bin', 'colmap')
    openmvs_bin = config.get('openmvs_bin', '')
    force = args.force or config.get('force', False)

    # 设置默认参数
    default_params = {
        "points_per_side" : 32,
        "pred_iou_thresh" : 0.86,
        "stability_score_thresh" : 0.92,
        "crop_n_layers" : 1,
        "crop_n_points_downscale_factor" : 2,
        "min_mask_region_area" : 100,
    }
    
    # 检查点文件
    checkpoint_path = norm_path(os.path.join(workspace, 'segment_recon_checkpoints.json'))
    
    # 初始化SAM2分割器
    if not args.skip_segmentation:
        print(f"[INFO] Initializing SAM2 segmenter with model: {args.sam2_model}")
        segmenter = SAM2Segmenter(
            model_type=args.sam2_model,
            device=args.device,
            checkpoint=config.get('sam2_checkpoint'),
            mode=config.get('segmentation_mode', 'auto'),
            config_path=config.get('sam2_config'),
            params=config.get('segmentation_params', default_params),
        )
    
    # 遍历所有场景进行分割
    scene_names = list(model_map.values())
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "{task.completed}/{task.total}",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True
    ) as progress:
        seg_task = progress.add_task("[分割] Scenes", total=len(scene_names))
        for idx, scene_name in enumerate(scene_names):
            progress.update(seg_task, description=f"[分割] {scene_name}")
            scene_dir = norm_path(os.path.join(workspace, scene_name))
            masks_dir = norm_path(os.path.join(scene_dir, 'masks'))
            images_dir = norm_path(os.path.join(scene_dir, 'images'))
            # 强制分割：如force为True，先清空masks目录
            if force and os.path.exists(masks_dir):
                print(f"[FORCE] Removing existing masks for {scene_name}")
                shutil.rmtree(masks_dir)
            # 不强制分割时，自动修正掩码文件名
            if not force and os.path.exists(masks_dir):
                check_and_fix_masks(images_dir, masks_dir)
            # 检查是否已存在masks
            if os.path.exists(masks_dir) and len(os.listdir(masks_dir)) > 0 and not force:
                print(f"[SKIP] Masks for {scene_name} already exist")
            elif not args.skip_segmentation:
                print(f"[INFO] Segmenting scene {scene_name}")
                try:
                    segmenter.segment_scene(scene_dir)
                    print(f"[INFO] Segmentation completed for {scene_name}")
                except Exception as e:
                    print(f"[ERROR] Segmentation failed for {scene_name}: {e}")
                    # 如果分割失败，生成全白masks
                    print(f"[INFO] Generating white masks for {scene_name}")
                    for img_file in os.listdir(images_dir):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(images_dir, img_file)
                            img = cv2.imread(img_path)
                            if img is not None:
                                mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
                                mask_path = os.path.join(masks_dir, img_file + ".png")
                                cv2.imwrite(mask_path, mask)
            else:
                print(f"[INFO] Skipping segmentation for {scene_name}")
            progress.update(seg_task, advance=1)

    # 执行COLMAP+OpenMVS重建流程
    print(f"\n[INFO] Starting 3D reconstruction with masks")
    
    # 为每个场景执行重建
    with Progress(
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "{task.completed}/{task.total}",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True
    ) as progress:
        recon_task = progress.add_task("[重建] Scenes", total=len(scene_names))
        for idx, scene_name in enumerate(scene_names):
            progress.update(recon_task, description=f"[重建] {scene_name}")
            scene_dir = norm_path(os.path.join(workspace, scene_name))
            scene_workspace = norm_path(os.path.join(scene_dir, 'workspace'))
            # 创建场景特定的配置文件
            scene_config = {
                'image_dir': os.path.join(scene_dir, 'images'),
                'workspace': scene_workspace,
                'colmap_bin': colmap,
                'openmvs_bin': openmvs_bin,
                'model_map': {0: scene_name}
            }
            # 保存临时配置文件
            temp_config_path = os.path.join(scene_workspace, 'temp_reconstruct_config.yaml')
            os.makedirs(scene_workspace, exist_ok=True)
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(scene_config, f, allow_unicode=True)
            # 调用重建脚本
            reconstruct_cmd = f'python reconstruct.py --config "{temp_config_path}"'
            try:
                run(reconstruct_cmd)
                print(f"[INFO] Reconstruction completed for {scene_name}")
            except Exception as e:
                print(f"[ERROR] Reconstruction failed for {scene_name}: {e}")
                continue
            # 清理临时配置文件
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            progress.update(recon_task, advance=1)

    print(f"\n[INFO] Pipeline completed successfully!")
    print(f"[INFO] Workspace: {workspace}")

if __name__ == '__main__':
    main() 