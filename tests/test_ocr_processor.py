import unittest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image, ImageDraw
import os
from pathlib import Path
from collections import Counter

from src.analysis.ocr.ocr_processor import OCRProcessor, OCRError

class TestOCRProcessor(unittest.TestCase):
    def setUp(self):
        """テストの前準備"""
        self.config = {
            'lang': 'jpn',
            'psm': 3,
            'oem': 3,
            'min_confidence': 0.5
        }
        self.processor = OCRProcessor(self.config)
        
        # テスト用画像の作成
        self.test_image = Image.new('RGB', (100, 30), color='white')
        # テキストを描画
        draw = ImageDraw.Draw(self.test_image)
        draw.text((10, 10), "test123", fill='black')  # テスト用のテキスト
        
    def test_initialization(self):
        """初期化のテスト"""
        self.assertEqual(self.processor.lang, 'jpn')
        self.assertEqual(self.processor.psm, 3)
        self.assertEqual(self.processor.oem, 3)
        self.assertEqual(self.processor.min_confidence, 0.5)
        
    @patch('cv2.cvtColor')
    @patch('cv2.fastNlMeansDenoising')
    @patch('cv2.createCLAHE')
    @patch('cv2.threshold')
    def test_preprocess_image(self, mock_threshold, mock_clahe, mock_denoise, mock_cvtcolor):
        """画像前処理のテスト"""
        # モックの設定
        mock_cvtcolor.return_value = np.zeros((30, 100))
        mock_denoise.return_value = np.zeros((30, 100))
        mock_clahe.return_value.apply.return_value = np.zeros((30, 100))
        mock_threshold.return_value = (None, np.zeros((30, 100)))
        
        # 前処理の実行
        result = self.processor._preprocess_image(self.test_image)
        
        # アサーション
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (30, 100))
        
    @patch('pytesseract.image_to_data')
    def test_extract_text(self, mock_image_to_data):
        """テキスト抽出のテスト"""
        # モックの設定
        mock_result = {
            'text': ['テスト', 'テキスト'],
            'conf': [90.0, 85.0],
            'left': [10, 50],
            'top': [5, 5],
            'width': [30, 40],
            'height': [20, 20]
        }
        mock_image_to_data.return_value = mock_result
        
        # テキスト抽出の実行
        result = self.processor.extract_text(self.test_image, timestamp=1.0)
        
        # アサーション
        self.assertIn('texts', result)
        self.assertEqual(len(result['texts']), 2)
        self.assertEqual(result['timestamp'], 1.0)
        self.assertEqual(result['language'], 'jpn')
        
        # テキスト情報の検証
        text_info = result['texts'][0]
        self.assertEqual(text_info['text'], 'テスト')
        self.assertEqual(text_info['confidence'], 90.0)
        self.assertEqual(text_info['position']['left'], 10)
        self.assertEqual(text_info['position']['top'], 5)
        
    def test_calculate_text_quality(self):
        """テキスト品質スコアの計算テスト"""
        test_cases = [
            ("", 0.0),  # 空文字列
            ("a", 0.0),  # 短すぎるテキスト
            ("テスト", 1.0),  # 日本語テキスト
            ("test123", 1.0),  # 英数字
            ("!!!???", 0.0),  # 記号のみ
            ("XYZXYZ", 0.5),  # 英語（母音なし）
            ("aaabbbccc", 0.67),  # 繰り返しパターン
        ]
        
        for text, expected_score in test_cases:
            score = self.processor._calculate_text_quality(text)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            if expected_score > 0:
                self.assertAlmostEqual(score, expected_score, places=2,
                                     msg=f"Text: {text}, Expected: {expected_score}, Got: {score}")

    def test_process_frames(self):
        """フレーム処理のテスト"""
        # テストデータの準備
        frames = [
            {
                'image': self.test_image,
                'timestamp': 0.52,
                'frame_number': 13,
                'scene_change_score': 0.8,  # シーン変化スコア (20%)
                'importance_score': 0.0,
                'ocr_confidence': 0.0,
                'text_quality_score': 0.0
            },
            {
                'image': self.test_image,
                'timestamp': 1.04,
                'frame_number': 26,
                'scene_change_score': 0.4,  # シーン変化スコア (20%)
                'importance_score': 0.0,
                'ocr_confidence': 0.0,
                'text_quality_score': 0.0
            }
        ]
        
        # フレーム処理の実行
        with patch.object(self.processor, 'extract_text') as mock_extract:
            # OCR結果のモック（高品質なケース）
            mock_extract.return_value = {
                'texts': [
                    {'text': 'テスト', 'confidence': 90.0},  # OCR信頼度 0.9 (40%)
                    {'text': 'データ', 'confidence': 85.0}   # OCR信頼度 0.85 (40%)
                ],
                'timestamp': 0.52,
                'language': 'jpn'
            }
            
            results = self.processor.process_frames(frames)
            
            # アサーション
            self.assertIsInstance(results, dict)
            self.assertIn('screenshots', results)
            self.assertEqual(len(results['screenshots']), 2)
            
            # 最初のスクリーンショットの検証
            screenshot = results['screenshots'][0]
            self.assertEqual(screenshot['timestamp'], 0.52)
            self.assertEqual(screenshot['frame_number'], 13)
            
            # 重要度スコアの検証
            # OCR信頼度: (0.9 + 0.85) / 2 = 0.875 * 0.4 = 0.35
            # テキスト品質: 1.0 * 0.4 = 0.4 (日本語テキスト)
            # シーン変化: 0.8 * 0.2 = 0.16
            # 合計: 0.35 + 0.4 + 0.16 = 0.91
            expected_importance = 0.91
            self.assertAlmostEqual(screenshot['importance_score'], expected_importance, places=2)
            
            # テキスト内容の検証
            self.assertIn('text', screenshot)
            self.assertEqual(screenshot['text'], 'テスト データ')

    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        # 無効な画像でのテスト
        invalid_image = None
        frames = [{'image': invalid_image, 'timestamp': 1.0, 'frame_number': 1}]
        
        with self.assertRaises(OCRError):
            self.processor.process_frames(frames)
        
        # OCRエラーのテスト
        with patch('pytesseract.image_to_data', side_effect=Exception('OCRエラー')):
            with self.assertRaises(OCRError):
                self.processor.extract_text(self.test_image)
                
if __name__ == '__main__':
    unittest.main() 