import argparse
import yaml
import os
import subprocess
import json
import shutil
import sys

def norm_path(path):
    return os.path.abspath(path).replace("\\", "/")

def run(cmd, check_output=None, cwd=None):
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
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_checkpoints(path, checkpoints):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(checkpoints, f, ensure_ascii=False, indent=2)

def ensure_images_link(workspace, img_dir):
    images_link = os.path.join(workspace, 'images')
    if os.path.exists(images_link):
        return
    try:
        os.symlink(os.path.abspath(img_dir), images_link)
        print(f"[INFO] Created symlink: {images_link} -> {img_dir}")
    except (AttributeError, NotImplementedError, OSError):
        # Windows无管理员权限时无法创建软链接，直接复制
        print(f"[INFO] Symlink not supported, copying images to {images_link}")
        shutil.copytree(img_dir, images_link)

def prepare_openmvs_input(workspace, img_dir, sparse_i, subdir):
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
    if filepath and os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        msg = f"[SKIP] {filepath} 已存在，跳过该步。"
        if description:
            msg = f"[SKIP] {description} ({filepath}) 已存在，跳过该步。"
        print(msg)
        return True
    return False

def load_and_update_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    image_dir = config['image_dir']
    # 自动建立model_map
    if 'model_map' not in config:
        subdirs = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
        model_map = {i: name for i, name in enumerate(subdirs)}
        config['model_map'] = model_map
        # 回写到配置文件
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
    else:
        model_map = config['model_map']
    return config, model_map

def main():
    parser = argparse.ArgumentParser(description="3D Reconstruction pipeline using COLMAP + OpenMVS")
    parser.add_argument('--config', type=str, default='reconstruct_config.yaml', help='Path to reconstruction config YAML')
    args = parser.parse_args()
    config_path = args.config
    config, model_map = load_and_update_config(config_path)
    workspace = norm_path(config['workspace'])
    image_dir = norm_path(config['image_dir'])
    colmap = config.get('colmap_bin', 'colmap')
    openmvs_bin = config.get('openmvs_bin', '')
    db = norm_path(os.path.join(workspace, 'database.db'))
    sparse_dir = norm_path(os.path.join(workspace, 'sparse'))
    dense_dir = norm_path(os.path.join(workspace, 'dense'))
    checkpoint_path = norm_path(os.path.join(workspace, 'recon_checkpoints.json'))

    # 自动创建所有用到的目录
    os.makedirs(os.path.dirname(db), exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(dense_dir, exist_ok=True)
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    ensure_images_link(workspace, image_dir)

    densify = norm_path(os.path.join(openmvs_bin, 'DensifyPointCloud')) if openmvs_bin else 'DensifyPointCloud'
    reconstruct = norm_path(os.path.join(openmvs_bin, 'ReconstructMesh')) if openmvs_bin else 'ReconstructMesh'
    refine = norm_path(os.path.join(openmvs_bin, 'RefineMesh')) if openmvs_bin else 'RefineMesh'
    texture = norm_path(os.path.join(openmvs_bin, 'TextureMesh')) if openmvs_bin else 'TextureMesh'
    interface_colmap = norm_path(os.path.join(openmvs_bin, 'InterfaceCOLMAP')) if openmvs_bin else 'InterfaceCOLMAP'

    steps = [
        ("feature_extractor", f'"{colmap}" feature_extractor --database_path "{db}" --image_path "{image_dir}" --ImageReader.camera_model PINHOLE --ImageReader.mask_path "{os.path.join(os.path.dirname(image_dir), "masks")}"'),
        ("exhaustive_matcher", f'"{colmap}" exhaustive_matcher --database_path "{db}"'),
        ("mapper", f'"{colmap}" mapper --database_path "{db}" --image_path "{image_dir}" --output_path "{sparse_dir}"'),
    ]

    checkpoints = load_checkpoints(checkpoint_path)
    # 先执行COLMAP三步
    for step, cmd in steps:
        if checkpoints.get(step):
            print(f"[SKIP] {step} 已完成，跳过。")
            continue
        try:
            run(cmd)
            checkpoints[step] = True
            save_checkpoints(checkpoint_path, checkpoints)
        except Exception as e:
            print(f"[ERROR] {step} 执行失败: {e}")
            return

    # 遍历image_dir下所有子目录，分别执行OpenMVS流程
    for idx, model_name in model_map.items():
        img_subdir = norm_path(os.path.join(image_dir, model_name))
        sparse_i = norm_path(os.path.join(sparse_dir, str(idx)))
        dense_i = norm_path(os.path.join(dense_dir, str(idx)))
        os.makedirs(dense_i, exist_ok=True)
        ply = norm_path(os.path.join(dense_i, f'scene_{model_name}.ply'))
        mvs = norm_path(os.path.join(dense_i, f'scene_{model_name}.mvs'))
        mvs_dense = mvs.replace('.mvs', '_dense.mvs')
        mvs_dense_mesh = mvs.replace('.mvs', '_dense_mesh.ply')
        mvs_dense_mesh_refine = mvs.replace('.mvs', '_dense_mesh_refine.mvs')
        mvs_dense_mesh_refine_ply = mvs.replace('.mvs', '_dense_mesh_refine.ply')
        mvs_dense_mesh_refine_textured = mvs.replace('.mvs', '_dense_mesh_refine_textured.mvs')
        mvs_dense_mesh_refine_textured_ply = mvs.replace('.mvs', '_dense_mesh_refine_textured.ply')
        dmap_dir = norm_path(os.path.join(dense_i, "dmap"))
        log_file = norm_path(os.path.join(dense_i, "densify.log"))
        # OpenMVS相关命令全部在dense_i目录下运行，mvs路径只用文件名
        mvs_name = os.path.basename(mvs)
        mvs_dense_name = os.path.basename(mvs_dense)
        mvs_dense_mesh_name = os.path.basename(mvs_dense_mesh)
        mvs_dense_mesh_refine_name = os.path.basename(mvs_dense_mesh_refine)
        mvs_dense_mesh_refine_textured_name = os.path.basename(mvs_dense_mesh_refine_textured)
        mvs_dense_mesh_refine_textured_ply_name = os.path.basename(mvs_dense_mesh_refine_textured_ply)
        mvs_dense_mesh_refine_ply_name = os.path.basename(mvs_dense_mesh_refine_ply)
        # densify命令
        densify_cmd = f'"{densify}" "{mvs_name}" > "densify.log" 2>&1'
        # 检查点key带编号
        step_keys = [
            f"model_converter_{model_name}",
            f"interface_colmap_{model_name}",
            f"densify_{model_name}",
            f"reconstruct_{model_name}",
            f"refine_{model_name}",
            f"texture_{model_name}"
        ]
        cmds = [
            (f'"{colmap}" model_converter --input_path "{sparse_i}" --output_path "{ply}" --output_type PLY', ply),
            (None, mvs),  # interface_colmap命令后面动态生成
            (densify_cmd, mvs_dense),
            (f'"{reconstruct}" "{mvs_dense_name}"', mvs_dense_mesh),
            (f'"{refine}" "{mvs_dense_name}" -m "{mvs_dense_mesh_name}" -o "{mvs_dense_mesh_refine_name}"', mvs_dense_mesh_refine),
            (f'"{texture}" "{mvs_dense_name}" -m "{mvs_dense_mesh_refine_ply_name}" -o "{mvs_dense_mesh_refine_textured_name}" --export-type obj', mvs_dense_mesh_refine_textured)
        ]
        skip_check_files = [
            ply,  # model_converter
            mvs,  # interface_colmap
            mvs_dense,  # densify
            mvs_dense_mesh,  # reconstruct
            mvs_dense_mesh_refine_ply,  # refine（只检查ply）
            mvs_dense_mesh_refine_textured  # texture
        ]
        # --- 临时拷贝sparse bin文件 ---
        sparse_bin_dir = os.path.join(sparse_i, 'sparse')
        need_cleanup_sparse = False
        if not os.path.exists(sparse_bin_dir):
            os.makedirs(sparse_bin_dir, exist_ok=True)
            need_cleanup_sparse = True
        copied_files = []
        for fname in ['cameras.bin', 'images.bin', 'points3D.bin']:
            src = os.path.join(sparse_i, fname)
            dst = os.path.join(sparse_bin_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied_files.append(dst)
        # ---
        try:
            for i, (cmd, out_file) in enumerate(cmds):
                key = step_keys[i]
                skip_file = skip_check_files[i]
                if key in checkpoints and os.path.exists(skip_file):
                    print(f"[SKIP] {key} 已完成，跳过。")
                    continue
                if key == f"interface_colmap_{model_name}":
                    interface_colmap_cmd = f'"{interface_colmap}" -i "{sparse_i}" -o "{mvs}"'
                    run(interface_colmap_cmd, check_output=mvs)
                elif cmd:
                    # OpenMVS相关命令全部在dense_i目录下运行
                    run(cmd, check_output=out_file, cwd=dense_i)
                checkpoints[key] = True
                save_checkpoints(checkpoint_path, checkpoints)
        finally:
            # --- 删除临时拷贝的sparse bin文件和目录 ---
            for f in copied_files:
                if os.path.exists(f):
                    os.remove(f)
            if need_cleanup_sparse and os.path.exists(sparse_bin_dir) and not os.listdir(sparse_bin_dir):
                os.rmdir(sparse_bin_dir)
        # ---
        # 在 dense_i 下创建 images 软链或拷贝，指向当前模型图片子目录
        images_link = os.path.join(dense_i, 'images')
        if not (os.path.islink(images_link) or os.path.exists(images_link)):
            try:
                os.symlink(os.path.abspath(img_subdir), images_link)
                print(f"[INFO] Created symlink: {images_link} -> {img_subdir}")
            except (AttributeError, NotImplementedError, OSError):
                print(f"[INFO] Symlink not supported, copying images to {images_link}")
                shutil.copytree(img_subdir, images_link)

if __name__ == '__main__':
    main()
