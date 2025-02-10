import cv2
import numpy as np
from typing import List, Dict, Any
from PIL import Image
import os
from datetime import datetime

class FrameExtractor:
    def __init__(self, config: Dict[str, Any] = None):
        """フレーム抽出クラスの初期化"""
        self.config = config or {}
        self.interval = self.config.get('interval', 1.0)  # デフォルトは1秒間隔
        self.quality = self.config.get('quality', 95)

    def extract_frames(self, video_path: str) -> List[Dict[str, Any]]:
        """動画からフレームを抽出する"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"動画ファイルを開けません: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps

            frames = []
            frame_interval = int(fps * self.interval)
            current_frame = 0

            while current_frame < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
                
                if not ret:
                    break

                # フレームをPIL Imageに変換
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                # タイムスタンプを計算
                timestamp = current_frame / fps

                frames.append({
                    'timestamp': timestamp,
                    'frame_number': current_frame,
                    'image': pil_image,
                    'metadata': {
                        'fps': fps,
                        'total_frames': total_frames,
                        'duration': duration,
                        'quality': self.quality
                    }
                })

                current_frame += frame_interval

            return frames

        finally:
            cap.release()

    def save_frames(self, frames: List[Dict[str, Any]], output_dir: str) -> List[str]:
        """抽出したフレームを保存する"""
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []

        for i, frame in enumerate(frames):
            image_path = os.path.join(output_dir, f'frame_{i:04d}.jpg')
            frame['image'].save(image_path, 'JPEG', quality=self.quality)
            saved_paths.append(image_path)

        return saved_paths
