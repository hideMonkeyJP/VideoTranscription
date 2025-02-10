import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

class OCRError(Exception):
    """OCR処理に関するエラー"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}

class OCRProcessor:
    def __init__(self, config: Dict[str, Any] = None):
        """OCRプロセッサーを初期化します
        
        Args:
            config (Dict[str, Any], optional): 設定辞書
                - lang (str): OCRの言語設定 (default: 'jpn')
                - psm (int): ページセグメンテーションモード (default: 3)
                - oem (int): OCRエンジンモード (default: 3)
                - min_confidence (float): 最小信頼度スコア (default: 0.5)
        """
        self.config = config or {}
        self.lang = self.config.get('lang', 'jpn')
        self.psm = self.config.get('psm', 3)
        self.oem = self.config.get('oem', 3)
        self.min_confidence = self.config.get('min_confidence', 0.5)
        
        # カスタムの設定を適用
        if 'tesseract_cmd' in self.config:
            pytesseract.pytesseract.tesseract_cmd = self.config['tesseract_cmd']
            
        self.logger = logging.getLogger(__name__)
        
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """画像を前処理します
        
        Args:
            image (Image.Image): 入力画像
            
        Returns:
            np.ndarray: 前処理済みの画像
        """
        try:
            # PILからOpenCVの形式に変換
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # グレースケール変換
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # ノイズ除去
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # コントラスト強調
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # 二値化
            _, binary = cv2.threshold(
                enhanced,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
            return binary
            
        except Exception as e:
            self.logger.error(f"画像の前処理中にエラーが発生: {str(e)}")
            raise OCRError("画像の前処理に失敗しました", {"error": str(e)})
            
    def extract_text(self, image: Image.Image, timestamp: float = 0.0) -> Dict[str, Any]:
        """画像からテキストを抽出します
        
        Args:
            image (Image.Image): 入力画像
            timestamp (float, optional): フレームのタイムスタンプ
            
        Returns:
            Dict[str, Any]: 抽出されたテキスト情報
        """
        try:
            # 画像の前処理
            processed_img = self._preprocess_image(image)
            
            # OCR設定
            config = f'--oem {self.oem} --psm {self.psm}'
            
            # OCRの実行
            result = pytesseract.image_to_data(
                processed_img,
                lang=self.lang,
                config=config,
                output_type=pytesseract.Output.DICT
            )
            
            # 結果の解析と信頼度フィルタリング
            texts = []
            for i in range(len(result['text'])):
                confidence = float(result['conf'][i])
                if confidence >= self.min_confidence and result['text'][i].strip():
                    text_info = {
                        'text': result['text'][i],
                        'confidence': confidence,
                        'position': {
                            'left': result['left'][i],
                            'top': result['top'][i],
                            'width': result['width'][i],
                            'height': result['height'][i]
                        },
                        'timestamp': timestamp
                    }
                    texts.append(text_info)
            
            return {
                'texts': texts,
                'timestamp': timestamp,
                'language': self.lang,
                'preprocessing_info': {
                    'original_size': image.size,
                    'processed_size': processed_img.shape[:2]
                }
            }
            
        except Exception as e:
            self.logger.error(f"テキスト抽出中にエラーが発生: {str(e)}")
            raise OCRError("テキスト抽出に失敗しました", {
                "timestamp": timestamp,
                "error": str(e)
            })
            
    def process_frames(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """複数のフレームに対してOCR処理を実行します
        
        Args:
            frames (List[Dict[str, Any]]): フレーム情報のリスト
            
        Returns:
            List[Dict[str, Any]]: OCR結果のリスト
        """
        results = []
        for frame in frames:
            try:
                image = frame.get('image')
                timestamp = frame.get('timestamp', 0.0)
                
                if not image:
                    continue
                    
                result = self.extract_text(image, timestamp)
                result['frame_number'] = frame.get('frame_number')
                results.append(result)
                
            except Exception as e:
                self.logger.warning(
                    f"フレーム {frame.get('frame_number')} の処理中にエラー: {str(e)}"
                )
                continue
                
        return results
