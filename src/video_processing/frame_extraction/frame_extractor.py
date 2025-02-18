import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from PIL import Image
import os
from datetime import datetime
import logging

class FrameExtractionError(Exception):
    """フレーム抽出に関するエラー"""
    pass

class FrameExtractor:
    def __init__(self, config: Dict[str, Any] = None):
        """FrameExtractorを初期化します。"""
        self._config = {
            'frame_interval': 1.0,
            'quality': 90,
            'target_frames_per_hour': 1000,
            'important_frames_ratio': 0.05,
            'min_frames': 100,
            'max_frames': 5000,
            'min_scene_change': 0.3
        }
        
        if config:
            print(f"Debug: Updating config with: {config}")
            self._config.update(config)
            print(f"Debug: Final configuration: {self._config}")

    @property
    def interval(self) -> float:
        """フレーム間隔を取得します。"""
        return self._config.get('interval', 1.0)
    
    @property
    def quality(self) -> int:
        """JPEG品質を取得します。"""
        return self._config.get('quality', 90)

    def _calculate_scene_change_score(self, current_frame: np.ndarray, prev_frame: Optional[np.ndarray] = None) -> float:
        """シーン変更スコアを計算します"""
        if prev_frame is None:
            return 0.0
        
        # フレーム間の差分を計算
        diff = cv2.absdiff(current_frame, prev_frame)
        score = np.mean(diff) / 255.0
        return score

    def _calculate_text_quality_score(self, text: str) -> float:
        """テキストの品質スコアを計算します"""
        if not text or len(text.strip()) < 3:
            return 0.0

        # 基本スコアの初期化
        score = 1.0

        # 1. 文字種類の評価
        chars = Counter(text)
        unique_ratio = len(chars) / len(text)
        score *= min(1.0, unique_ratio * 2)

        # 2. 意味のある文字の割合
        meaningful_chars = sum(1 for c in text if c.isalnum() or 0x3000 <= ord(c) <= 0x9FFF)
        meaningful_ratio = meaningful_chars / len(text)
        score *= meaningful_ratio

        # 3. 記号の割合評価
        symbol_ratio = sum(1 for c in text if not c.isalnum() and not 0x3000 <= ord(c) <= 0x9FFF) / len(text)
        score *= (1.0 - min(1.0, symbol_ratio * 2))

        # 4. 日本語文字の評価
        jp_ratio = sum(1 for c in text if 0x3000 <= ord(c) <= 0x9FFF) / len(text)
        if jp_ratio > 0:
            score *= (1.0 + jp_ratio)

        return min(1.0, score)

    def _calculate_importance_score(self, scene_change_score: float, text_quality_score: float, ocr_confidence: float) -> float:
        """重要度スコアを計算します
        
        Args:
            scene_change_score (float): シーン変更スコア (0.0-1.0)
            text_quality_score (float): テキスト品質スコア (0.0-1.0)
            ocr_confidence (float): OCRの信頼度スコア (0.0-1.0)
            
        Returns:
            float: 重要度スコア (0.0-20.0)
        """
        # 各要素の重み付け
        weights = {
            'scene_change': 8.0,  # シーン変更の重要性を強調
            'text_quality': 6.0,  # テキスト品質も重視
            'ocr_confidence': 6.0  # OCR信頼度も同様に重視
        }
        
        # 重要度スコアの計算
        importance_score = (
            scene_change_score * weights['scene_change'] +
            text_quality_score * weights['text_quality'] +
            ocr_confidence * weights['ocr_confidence']
        )
        
        return importance_score

    def _process_frame(self, frame: np.ndarray, frame_number: int, timestamp: float, fps: float, 
                      total_frames: int, prev_frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """フレームを処理し、必要な情報を付加します"""
        # フレームをPIL Imageに変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # シーン変更スコアを計算
        scene_change_score = self._calculate_scene_change_score(frame, prev_frame)
        
        # メタデータの作成
        metadata = {
            'quality': self.quality,
            'frame_interval': self.interval,
            'resolution': (frame.shape[1], frame.shape[0]),
            'fps': fps,
            'total_frames': total_frames,
            'duration': total_frames / fps  # 動画の長さを追加（秒）
        }
        
        # 重要度スコアの初期計算
        importance_score = self._calculate_importance_score(
            scene_change_score=scene_change_score,
            text_quality_score=0.0,  # OCR処理後に更新
            ocr_confidence=0.0       # OCR処理後に更新
        )

        return {
            'image': pil_image,
            'frame_number': frame_number,
            'timestamp': timestamp,
            'metadata': metadata,
            'scene_change_score': scene_change_score,
            'importance_score': importance_score,
            'ocr_confidence': 0.0,    # OCR処理後に更新
            'text_quality_score': 0.0, # OCR処理後に更新
            'is_important': False      # 後で更新
        }

    def extract_frames(self, video_path: str) -> List[Dict[str, Any]]:
        """動画からフレームを抽出します"""
        if not os.path.exists(video_path):
            raise FrameExtractionError(f"ビデオファイルが存在しません: {video_path}")

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise FrameExtractionError(f"ビデオファイルを開けません: {video_path}")

            # ビデオの基本情報を取得
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps

            # フレーム間隔を正確に計算
            frames_to_skip = int(round(fps * self.interval))
            actual_interval = frames_to_skip / fps

            logging.debug(f"Starting frame extraction with interval: {self.interval}s (frames to skip: {frames_to_skip})")
            logging.debug(f"Video duration: {duration:.2f}s, FPS: {fps}, Total frames: {total_frames}")
            logging.debug(f"Actual interval: {actual_interval:.3f}s")

            extracted_frames = []
            frame_number = 0
            prev_frame = None
            prev_timestamp = None

            while frame_number < total_frames:
                # フレーム番号を設定して読み込み
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if not ret:
                    break

                actual_frame = frame_number + 1
                actual_pos = frame_number / fps

                # タイムスタンプの差分を確認
                if prev_timestamp is not None:
                    time_diff = actual_pos - prev_timestamp
                    logging.debug(f"Time difference from previous frame: {time_diff:.3f}s")

                logging.debug(f"Frame {frame_number}, Actual frame: {actual_frame}, Position: {actual_pos:.3f}s")

                frame_info = self._process_frame(
                    frame, actual_frame, actual_pos, fps, total_frames, prev_frame
                )
                extracted_frames.append(frame_info)
                prev_frame = frame.copy()
                prev_timestamp = actual_pos

                # 次のフレーム位置を計算
                frame_number += frames_to_skip
                logging.debug(f"Next frame will be: {frame_number}")

            cap.release()

            # 重要フレームの選定
            if extracted_frames:
                important_count = max(1, int(len(extracted_frames) * self._config.get('important_frames_ratio', 0.05)))
                sorted_frames = sorted(extracted_frames, key=lambda x: x['scene_change_score'], reverse=True)
                for frame in sorted_frames[:important_count]:
                    frame['is_important'] = True

            logging.debug(f"Extracted {len(extracted_frames)} frames with interval {actual_interval:.3f}s")
            
            # フレーム間隔の検証
            for i in range(1, len(extracted_frames)):
                time_diff = extracted_frames[i]['timestamp'] - extracted_frames[i-1]['timestamp']
                logging.debug(f"Frame {i} interval: {time_diff:.3f}s")

            return extracted_frames

        except Exception as e:
            raise FrameExtractionError(f"フレーム抽出中にエラーが発生: {str(e)}")

    def _detect_scene_change(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """2つのフレーム間のシーン変更スコアを計算
        
        Args:
            frame1 (np.ndarray): 1つ目のフレーム
            frame2 (np.ndarray): 2つ目のフレーム
            
        Returns:
            float: シーン変更スコア（0-1）
        """
        if frame1 is None or frame2 is None:
            return 0.0
        
        # グレースケールに変換
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # 差分を計算
        diff = cv2.absdiff(gray1, gray2)
        non_zero = cv2.countNonZero(diff)
        total_pixels = frame1.shape[0] * frame1.shape[1]
        
        return non_zero / total_pixels
    def save_frames(self, frames: List[Dict[str, Any]], output_dir: str) -> List[str]:
        """抽出したフレームを保存する"""
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []

        for i, frame in enumerate(frames):
            # タイムスタンプをファイル名に含める
            timestamp = frame['timestamp']
            is_important = frame.get('is_important', False)
            
            # 重要フレームは特別な接頭辞を付ける
            prefix = 'important_' if is_important else ''
            filename = f'{prefix}frame_{timestamp:.3f}.jpg'
            image_path = os.path.join(output_dir, filename)
            
            frame['image'].save(image_path, 'JPEG', quality=self.quality)
            saved_paths.append(image_path)

            # メタデータをJSONファイルとして保存
            metadata = {
                'timestamp': timestamp,
                'frame_number': frame['frame_number'],
                'is_important': is_important,
                'scene_change_score': frame.get('scene_change_score', 0.0),
                'importance_score': frame.get('importance_score', 0.0),
                'metadata': frame['metadata']
            }
            
            json_path = os.path.join(output_dir, f'{prefix}frame_{timestamp:.3f}.json')
            with open(json_path, 'w') as f:
                import json
                json.dump(metadata, f, indent=2)

        return saved_paths

