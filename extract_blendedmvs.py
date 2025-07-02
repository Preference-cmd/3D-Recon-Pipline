#!/usr/bin/env python3
"""
BlendedMVS数据集提取脚本
批量提取图片、掩码、深度图和相机参数到指定输出目录
"""
import os
import shutil
import argparse
from glob import glob
import yaml
from PIL import Image

def extract_scene(scene_dir, output_dir, copy_masked=True):
    """提取单个场景的数据到输出目录，适配SAM+COLMAP+OpenMVS流程"""
    img_dir = os.path.join(scene_dir, 'blended_images')
    depth_dir = os.path.join(scene_dir, 'rendered_depth_maps')
    cam_dir = os.path.join(scene_dir, 'cams')
    
    out_img_dir = os.path.join(output_dir, 'images')
    out_mask_dir = os.path.join(output_dir, 'masks')
    out_depth_dir = os.path.join(output_dir, 'depths')
    out_cam_dir = os.path.join(output_dir, 'cams')
    out_workspace_dir = os.path.join(output_dir, 'workspace')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)
    os.makedirs(out_cam_dir, exist_ok=True)
    os.makedirs(out_workspace_dir, exist_ok=True)

    # 提取图片和掩码，images和masks严格一一对应
    for img_path in glob(os.path.join(img_dir, '*.jpg')):
        fname = os.path.basename(img_path)
        if fname.endswith('_masked.jpg'):
            if copy_masked:
                # mask文件名与原图一致（去掉_masked）
                mask_name = fname.replace('_masked', '')
                shutil.copy2(img_path, os.path.join(out_mask_dir, mask_name))
        else:
            shutil.copy2(img_path, os.path.join(out_img_dir, fname))
            # 若未提取mask，自动生成全白mask（可选，便于后续流程健壮性）
            mask_path = os.path.join(out_mask_dir, fname)
            if not os.path.exists(mask_path) and not copy_masked:
                img = Image.open(img_path)
                mask = Image.new('L', img.size, 255)
                mask.save(mask_path)
    # 提取深度图
    for depth_path in glob(os.path.join(depth_dir, '*.pfm')):
        shutil.copy2(depth_path, os.path.join(out_depth_dir, os.path.basename(depth_path)))
    # 提取相机参数
    for cam_path in glob(os.path.join(cam_dir, '*_cam.txt')):
        shutil.copy2(cam_path, os.path.join(out_cam_dir, os.path.basename(cam_path)))
    # 提取pair.txt
    pair_txt = os.path.join(cam_dir, 'pair.txt')
    if os.path.exists(pair_txt):
        shutil.copy2(pair_txt, os.path.join(out_cam_dir, 'pair.txt'))
    print(f"[INFO] 提取完成: {scene_dir} -> {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="提取BlendedMVS数据集指定场景")
    parser.add_argument('--config', type=str, help='YAML配置文件路径')
    parser.add_argument('--root', type=str, help='BlendedMVS数据集根目录')
    parser.add_argument('--output', type=str, help='输出目录')
    parser.add_argument('--scenes', type=str, nargs='*', default=None, help='要提取的场景ID列表（默认全部）')
    parser.add_argument('--copy-masked', action='store_true', help='同时提取_masked.jpg为mask')
    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

    # 合并参数，命令行优先
    root = args.root or config.get('root')
    output = args.output or config.get('output')
    scenes = args.scenes if args.scenes is not None else config.get('scenes', None)
    copy_masked = args.copy_masked or config.get('copy_masked', False)

    if not root or not output:
        parser.error('必须指定 --root 和 --output，或在配置文件中提供')

    if scenes is None or len(scenes) == 0:
        # 默认全部场景
        scene_ids = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    else:
        scene_ids = scenes
    print(f"[INFO] 待提取场景: {scene_ids}")
    for scene_id in scene_ids:
        scene_dir = os.path.join(root, scene_id)
        out_dir = os.path.join(output, scene_id)
        extract_scene(scene_dir, out_dir, copy_masked=copy_masked)

if __name__ == '__main__':
    main() 