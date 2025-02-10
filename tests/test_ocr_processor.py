import unittest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
import os
from pathlib import Path

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
        
    def test_process_frames(self):
        """フレーム処理のテスト"""
        # テストデータの準備
        frames = [
            {'image': self.test_image, 'timestamp': 1.0, 'frame_number': 1},
            {'image': self.test_image, 'timestamp': 2.0, 'frame_number': 2}
        ]
        
        # フレーム処理の実行
        with patch.object(self.processor, 'extract_text') as mock_extract:
            mock_extract.return_value = {
                'texts': [{'text': 'テスト', 'confidence': 90.0}],
                'timestamp': 1.0,
                'language': 'jpn'
            }
            
            results = self.processor.process_frames(frames)
            
            # アサーション
            self.assertEqual(len(results), 2)
            self.assertIn('texts', results[0])
            self.assertIn('frame_number', results[0])
            
    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        # 無効な画像でのテスト
        invalid_image = None
        frames = [{'image': invalid_image, 'timestamp': 1.0, 'frame_number': 1}]
        
        results = self.processor.process_frames(frames)
        self.assertEqual(len(results), 0)
        
        # OCRエラーのテスト
        with patch('pytesseract.image_to_data', side_effect=Exception('OCRエラー')):
            with self.assertRaises(OCRError):
                self.processor.extract_text(self.test_image)
                
if __name__ == '__main__':
    unittest.main() 