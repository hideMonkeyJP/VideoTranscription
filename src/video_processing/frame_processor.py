import cv2
import numpy as np
from PIL import Image
import logging
from typing import Dict, Any, List
from pathlib import Path

class FrameProcessingError(Exception):
    """フレーム処理に関するエラー"""
    pass

class FrameProcessor:
    """動画フレームの処理を担当するクラス"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初期化
        
        Args:
            config (Dict[str, Any], optional): 設定辞書
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # フレーム抽出の設定
        self.frame_interval = self.config.get('frame_interval', 1)  # 秒単位
        self.min_brightness = self.config.get('min_brightness', 0.1)
        self.output_dir = Path(self.config.get('output_dir', 'output/frames'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_frames(self, video_path: str) -> List[Dict[str, Any]]:
        """動画からフレームを抽出
        
        Args:
            video_path (str): 動画ファイルのパス
            
        Returns:
            List[Dict[str, Any]]: 抽出されたフレーム情報のリスト
            
        Raises:
            FrameProcessingError: フレーム抽出に失敗した場合
        """
        try:
            self.logger.info("フレーム抽出を開始します")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise FrameProcessingError(f"動画ファイルを開けません: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames_to_skip = int(fps * self.frame_interval)
            
            frames = []
            frame_number = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 指定間隔でフレームを抽出
                if frame_number % frames_to_skip == 0:
                    frame_info = self._process_frame(frame, frame_number, fps)
                    if frame_info:
                        frames.append(frame_info)
                        
                        # フレーム画像を保存
                        frame_path = self.output_dir / f"frame_{frame_number}.jpg"
                        frame_info['image'].save(frame_path, 'JPEG')
                        frame_info['image_path'] = str(frame_path)
                
                frame_number += 1
            
            cap.release()
            self.logger.info(f"{len(frames)}フレームを抽出しました")
            return frames
            
        except Exception as e:
            self.logger.error(f"フレーム抽出中にエラーが発生: {str(e)}")
            raise FrameProcessingError(f"フレーム抽出に失敗しました: {str(e)}")
    
    def _process_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> Dict[str, Any]:
        """個別フレームの処理
        
        Args:
            frame (np.ndarray): OpenCVフレームデータ
            frame_number (int): フレーム番号
            fps (float): フレームレート
            
        Returns:
            Dict[str, Any]: 処理済みフレーム情報
        """
        # フレームをPIL Imageに変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # 重要度スコアを計算
        importance_score = self._calculate_importance_score(frame)
        
        # 明るさが閾値以下のフレームは除外
        if importance_score < self.min_brightness:
            return None
        
        return {
            'image': pil_image,
            'timestamp': frame_number / fps,
            'frame_number': frame_number,
            'importance_score': importance_score
        }
    
    def _calculate_importance_score(self, frame: np.ndarray) -> float:
        """フレームの重要度スコアを計算
        
        Args:
            frame (np.ndarray): OpenCVフレームデータ
            
        Returns:
            float: 重要度スコア（0.0 - 1.0）
        """
        # 輝度ベースのスコア計算
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        
        # エッジ検出ベースのスコア計算
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)
        
        # スコアの組み合わせ
        return float((brightness + edge_density) / 2) 