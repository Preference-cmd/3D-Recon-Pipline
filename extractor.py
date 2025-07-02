import cv2
import os
import random
from typing import Optional, Literal
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

class VideoFrameExtractor:
    def __init__(self, video_path: str, output_dir: str, frame_interval: int = 1, mode: str = 'interval', num_frames: Optional[int] = None):
        self.video_path = video_path
        self.output_dir = output_dir
        self.frame_interval = frame_interval
        self.mode = mode
        self.num_frames = num_frames
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {self.video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        selected_indices = []
        if self.mode == 'interval':
            selected_indices = list(range(0, total_frames, self.frame_interval))
        elif self.mode == 'uniform':
            n = self.num_frames or 1
            if n > total_frames:
                n = total_frames
            step = total_frames / n
            selected_indices = [int(i * step) for i in range(n)]
        elif self.mode == 'random':
            n = self.num_frames or 1
            if n > total_frames:
                n = total_frames
            selected_indices = sorted(random.sample(range(total_frames), n))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        frame_id = 0
        saved_count = 0
        selected_set = set(selected_indices)
        max_index = max(selected_indices, default=-1)
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "{task.completed}/{task.total}",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True
        ) as progress:
            task = progress.add_task(f"Extracting {os.path.basename(self.video_path)}", total=len(selected_indices))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_id in selected_set:
                    filename = os.path.join(self.output_dir, f"frame_{saved_count:05d}.jpg")
                    cv2.imwrite(filename, frame)
                    saved_count += 1
                    progress.update(task, advance=1)
                frame_id += 1
                if frame_id > max_index:
                    break
        cap.release() 