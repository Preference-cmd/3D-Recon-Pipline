import argparse
import os
import json
from config import load_config, merge_config_args
from extractor import VideoFrameExtractor
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from concurrent.futures import ProcessPoolExecutor, as_completed

CACHE_FILE = 'batch_cache.json'

def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from video for SfM pipeline.")
    parser.add_argument('--video_path', type=str, help='Path to input video file')
    parser.add_argument('--input_dir', type=str, help='Directory containing multiple video files for batch processing')
    parser.add_argument('--output_dir', type=str, help='Directory to save extracted frames')
    parser.add_argument('--frame_interval', type=int, help='Interval between saved frames (default: 1)')
    parser.add_argument('--mode', type=str, choices=['interval', 'uniform', 'random'], default=None, help='Extraction mode: interval, uniform, random')
    parser.add_argument('--num_frames', type=int, help='Number of frames to extract (for uniform/random mode)')
    parser.add_argument('--config', type=str, help='Path to YAML config file', default=None)
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers for batch processing')
    return parser.parse_args()

def is_video_file(filename):
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    return any(filename.lower().endswith(ext) for ext in video_exts)

def get_expected_num_frames(video_path, mode, frame_interval, num_frames):
    import cv2
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if mode == 'interval':
        return len(list(range(0, total_frames, frame_interval)))
    elif mode == 'uniform' or mode == 'random':
        n = num_frames or 1
        return min(n, total_frames)
    else:
        return 0

def process_single_video(video_path, output_dir, frame_interval, mode, num_frames):
    extractor = VideoFrameExtractor(
        video_path=video_path,
        output_dir=output_dir,
        frame_interval=frame_interval,
        mode=mode,
        num_frames=num_frames
    )
    extractor.extract_frames()
    return output_dir

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def main():
    args = parse_args()
    config = {}
    if args.config:
        config = load_config(args.config)
    args_dict = vars(args)
    merged = merge_config_args(config, args_dict)
    if not merged.get('output_dir'):
        raise ValueError('output_dir 必须指定（命令行或配置文件）')
    frame_interval = merged.get('frame_interval', 1)
    mode = merged.get('mode', 'interval')
    num_frames = merged.get('num_frames', None)
    workers = merged.get('workers', 4)
    cache = load_cache()
    updated_cache = cache.copy()
    if merged.get('input_dir'):
        input_dir = merged['input_dir']
        video_files = [fname for fname in os.listdir(input_dir) if is_video_file(fname)]
        tasks = []
        skipped = 0
        submit_tasks = []
        for fname in video_files:
            video_path = os.path.join(input_dir, fname)
            video_name = os.path.splitext(os.path.basename(fname))[0]
            out_dir = os.path.join(merged['output_dir'], video_name)
            expected = get_expected_num_frames(video_path, mode, frame_interval, num_frames)
            # 检查缓存和输出文件夹
            cache_entry = cache.get(video_name)
            actual = len([f for f in os.listdir(out_dir)]) if os.path.exists(out_dir) else 0
            if cache_entry and cache_entry.get('num_frames') == expected and actual == expected:
                print(f"[SKIP] {video_name}: 已完成，输出张数={expected}")
                skipped += 1
                continue
            if actual == expected and expected > 0:
                print(f"[SKIP] {video_name}: 输出文件夹已存在且张数一致，自动标记为已完成")
                updated_cache[video_name] = {"output_dir": out_dir, "num_frames": expected}
                skipped += 1
                continue
            submit_tasks.append((video_path, out_dir, frame_interval, mode, num_frames, video_name, expected))
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "{task.completed}/{task.total}",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True
        ) as progress:
            task = progress.add_task("Batch Processing", total=len(submit_tasks))
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {}
                for t in submit_tasks:
                    video_path, out_dir, frame_interval, mode, num_frames, video_name, expected = t
                    futures[executor.submit(process_single_video, video_path, out_dir, frame_interval, mode, num_frames)] = (video_name, out_dir, expected)
                for future in as_completed(futures):
                    video_name, out_dir, expected = futures[future]
                    try:
                        _ = future.result()
                        print(f"Frames extracted to {out_dir}")
                        updated_cache[video_name] = {"output_dir": out_dir, "num_frames": expected}
                    except Exception as e:
                        print(f"Error processing {video_name}: {e}")
                    progress.update(task, advance=1)
        save_cache(updated_cache)
        print(f"跳过任务数: {skipped}, 新处理任务数: {len(submit_tasks)}")
    elif merged.get('video_path'):
        process_single_video(
            video_path=merged['video_path'],
            output_dir=merged['output_dir'],
            frame_interval=frame_interval,
            mode=mode,
            num_frames=num_frames
        )
    else:
        raise ValueError('video_path 或 input_dir 必须指定其一（命令行或配置文件）')

if __name__ == "__main__":
    main()
