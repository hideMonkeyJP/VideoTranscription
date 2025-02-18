import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
from collections import Counter
import itertools

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
            # PILでの前処理
            # 1. グレースケール変換
            image = image.convert('L')
            
            # 2. コントラスト強調
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.5)
            
            # 3. シャープネス強調
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.5)
            
            # 4. ノイズ除去
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # OpenCVの形式に変換
            img = np.array(image)
            
            # グレースケール変換
            if len(img.shape) == 3:  # カラー画像の場合
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:  # すでにグレースケールの場合
                gray = img
            
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
                text = result['text'][i].strip()
                
                if confidence >= self.min_confidence and text:
                    texts.append({
                        'text': text,
                        'confidence': confidence,
                        'position': {
                            'left': result['left'][i],
                            'top': result['top'][i],
                            'width': result['width'][i],
                            'height': result['height'][i]
                        }
                    })
            
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
            
    def _calculate_text_quality(self, text: str) -> float:
        """テキストの品質スコアを計算します"""
        if not text or len(text.strip()) < 3:
            return 0.0

        # 基本スコアの初期化
        score = 1.0

        # 1. 文字種類の評価
        chars = Counter(text)
        unique_ratio = len(chars) / len(text)
        score *= min(1.0, unique_ratio * 2)  # 文字の多様性を評価

        # 2. 意味のある文字の割合
        meaningful_chars = sum(1 for c in text if c.isalnum() or 0x3000 <= ord(c) <= 0x9FFF)
        meaningful_ratio = meaningful_chars / len(text)
        score *= meaningful_ratio

        # 3. 記号の割合評価
        symbol_ratio = sum(1 for c in text if not c.isalnum() and not 0x3000 <= ord(c) <= 0x9FFF) / len(text)
        score *= (1.0 - min(1.0, symbol_ratio * 2))

        # 4. パターン検出
        # 連続する同じ文字
        max_repeat = max(len(list(g)) for _, g in itertools.groupby(text))
        if max_repeat > 3:
            score *= 0.5

        # 5. 日本語文字の評価
        jp_ratio = sum(1 for c in text if 0x3000 <= ord(c) <= 0x9FFF) / len(text)
        if jp_ratio > 0:
            score *= (1.0 + jp_ratio)  # 日本語文字が含まれる場合はスコアを上げる

        # 6. アルファベットの評価
        if text.isascii():
            # 母音の存在確認
            vowel_ratio = sum(1 for c in text.lower() if c in 'aeiou') / len(text)
            if vowel_ratio < 0.1:  # 母音が少なすぎる場合
                score *= 0.5

        return min(1.0, score)

    def process_frames_detailed(self, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """フレームを処理してOCR結果を返します（詳細バージョン）"""
        try:
            screenshots = []
            for frame in frames:
                result = self.extract_text(frame['image'], frame.get('timestamp', 0))
                if result and result.get('texts'):
                    # OCRの信頼度評価
                    confidences = [text_info['confidence'] / 100.0 for text_info in result['texts']]
                    ocr_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                    # テキスト品質評価
                    all_text = ' '.join(text_info['text'] for text_info in result['texts'])
                    text_quality_score = self._calculate_text_quality(all_text)

                    # 重要度スコアの計算
                    importance_score = (
                        ocr_confidence * 0.4 +          # OCR信頼度(40%)
                        text_quality_score * 0.4 +      # テキスト品質(40%)
                        frame.get('scene_change_score', 0.0) * 0.2  # シーン変化(20%)
                    )

                    # スクリーンショットの追加
                    screenshots.append({
                        'timestamp': frame.get('timestamp', 0),
                        'frame_number': frame.get('frame_number', 0),
                        'importance_score': min(importance_score, 1.0),
                        'ocr_confidence': ocr_confidence,
                        'text_quality_score': text_quality_score,
                        'texts': result['texts']
                    })
            
            return {
                'screenshots': screenshots
            }
            
        except Exception as e:
            self.logger.error(f"フレーム処理中にエラーが発生: {str(e)}")
            raise OCRError(f"フレーム処理に失敗しました: {str(e)}")

    def process_frames(self, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """フレームを処理してOCR結果を返します（シンプルバージョン）"""
        try:
            screenshots = []
            for frame in frames:
                result = self.extract_text(frame['image'], frame.get('timestamp', 0))
                if result and result.get('texts'):
                    # OCRの信頼度評価
                    confidences = [text_info['confidence'] / 100.0 for text_info in result['texts']]
                    ocr_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                    # テキスト品質評価
                    all_text = ' '.join(text_info['text'] for text_info in result['texts'])
                    text_quality_score = self._calculate_text_quality(all_text)

                    # 重要度スコアの計算
                    importance_score = (
                        ocr_confidence * 0.4 +          # OCR信頼度(40%)
                        text_quality_score * 0.4 +      # テキスト品質(40%)
                        frame.get('scene_change_score', 0.0) * 0.2  # シーン変化(20%)
                    )

                    # スクリーンショットの追加（シンプルな形式）
                    screenshots.append({
                        'timestamp': frame.get('timestamp', 0),
                        'frame_number': frame.get('frame_number', 0),
                        'importance_score': min(importance_score, 1.0),
                        'text': all_text,
                        'image_path': frame.get('path', '')  # 画像パスを追加
                    })
            
            return {
                'screenshots': screenshots
            }
            
        except Exception as e:
            self.logger.error(f"フレーム処理中にエラーが発生: {str(e)}")
            raise OCRError(f"フレーム処理に失敗しました: {str(e)}")
