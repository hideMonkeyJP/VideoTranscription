import unittest
import os
from pathlib import Path
import shutil
from PIL import Image
import json

from src.video_processing.frame_extraction.frame_extractor import FrameExtractor, FrameExtractionError

class TestFrameExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """テストクラスの前準備"""
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_output_dir = os.path.join(cls.test_dir, 'test_output')
        cls.test_video = os.path.join(cls.test_dir, '..', '..', '..', 'videos', 'Sample.mp4')
        
        # テスト用の出力ディレクトリを作成
        os.makedirs(cls.test_output_dir, exist_ok=True)

    def setUp(self):
        """各テストの前準備"""
        self.config = {
            'interval': 0.5,  # 0.5秒間隔
            'quality': 90
        }
        self.extractor = FrameExtractor(config=self.config)

    def test_init(self):
        """初期化のテスト"""
        self.assertEqual(self.extractor.interval, 0.5)
        self.assertEqual(self.extractor.quality, 90)

    def test_extract_frames(self):
        """フレーム抽出のテスト"""
        frames = self.extractor.extract_frames(self.test_video)
        
        # フレームが抽出されていることを確認
        self.assertGreater(len(frames), 0)
        
        # 各フレームの形式を確認
        for frame in frames:
            self.assertIn('timestamp', frame)
            self.assertIn('frame_number', frame)
            self.assertIn('image', frame)
            self.assertIn('metadata', frame)
            
            # タイムスタンプが数値であることを確認
            self.assertIsInstance(frame['timestamp'], (int, float))
            
            # フレーム番号が整数であることを確認
            self.assertIsInstance(frame['frame_number'], int)
            
            # 画像がPIL Imageオブジェクトであることを確認
            self.assertIsInstance(frame['image'], Image.Image)
            
            # メタデータの形式を確認
            metadata = frame['metadata']
            self.assertIn('fps', metadata)
            self.assertIn('total_frames', metadata)
            self.assertIn('duration', metadata)
            self.assertIn('quality', metadata)

    def test_save_frames(self):
        """フレーム保存のテスト"""
        # フレームを抽出
        frames = self.extractor.extract_frames(self.test_video)
        
        # フレームを保存
        saved_paths = self.extractor.save_frames(frames, self.test_output_dir)
        
        # 保存されたファイルの確認
        self.assertEqual(len(saved_paths), len(frames))
        for path in saved_paths:
            self.assertTrue(os.path.exists(path))
            self.assertTrue(path.endswith('.jpg'))

    def test_nonexistent_video(self):
        """存在しない動画ファイルのテスト"""
        with self.assertRaises(FileNotFoundError):
            self.extractor.extract_frames('nonexistent.mp4')

    def tearDown(self):
        """各テストのクリーンアップ"""
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)
        os.makedirs(self.test_output_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """テストクラスのクリーンアップ"""
        if os.path.exists(cls.test_output_dir):
            shutil.rmtree(cls.test_output_dir)

if __name__ == '__main__':
    unittest.main(verbosity=2) 