import unittest
import os
from pathlib import Path
import shutil
from PIL import Image
import json

from src.video_processing.frame_extraction import FrameExtractor, FrameExtractionError

class TestFrameExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """テストクラスの前準備"""
        cls.test_dir = Path("test_output")
        cls.frames_dir = cls.test_dir / "frames"
        cls.test_video = Path("videos/sample.mp4")
        
        # テスト用のディレクトリ構造を作成
        cls.test_dir.mkdir(exist_ok=True)
        cls.frames_dir.mkdir(exist_ok=True, parents=True)

    def setUp(self):
        """各テストの前準備"""
        self.config = {
            'interval': 1.0,  # frame_intervalからintervalに変更
            'quality': 90,
            'target_frames_per_hour': 1000,
            'important_frames_ratio': 0.05
        }
        self.extractor = FrameExtractor(config=self.config)

    def test_initialization(self):
        """初期化のテスト"""
        self.assertEqual(self.extractor.interval, 1.0)
        self.assertEqual(self.extractor.quality, 90)
        self.assertEqual(self.extractor._config.get('target_frames_per_hour'), 1000)
        self.assertEqual(self.extractor._config.get('important_frames_ratio'), 0.05)

    def test_config_update(self):
        """設定更新のテスト"""
        new_config = {
            'interval': 2.5,
            'quality': 85,
            'target_frames_per_hour': 500
        }
        extractor = FrameExtractor(new_config)
        
        self.assertEqual(extractor.interval, 2.5)
        self.assertEqual(extractor.quality, 85)
        self.assertEqual(extractor._config.get('target_frames_per_hour'), 500)
        # 更新されていない設定は元の値を保持
        self.assertEqual(extractor._config.get('important_frames_ratio'), 0.05)

    def test_scene_change_detection(self):
        """シーン変更検出のテスト"""
        if not self.test_video.exists():
            self.skipTest(f"テスト用動画が見つかりません: {self.test_video}")

        frames = self.extractor.extract_frames(str(self.test_video))
        
        # シーン変更スコアが計算されていることを確認
        for frame in frames:
            self.assertIn('scene_change_score', frame)
            self.assertIsInstance(frame['scene_change_score'], float)
            self.assertGreaterEqual(frame['scene_change_score'], 0.0)
            self.assertLessEqual(frame['scene_change_score'], 1.0)

    def test_important_frames_selection(self):
        """重要フレーム選定のテスト"""
        if not self.test_video.exists():
            self.skipTest(f"テスト用動画が見つかりません: {self.test_video}")

        frames = self.extractor.extract_frames(str(self.test_video))
        
        # 重要フレームの数を確認
        important_frames = [f for f in frames if f.get('is_important', False)]
        expected_count = max(1, int(len(frames) * self.config['important_frames_ratio']))
        self.assertEqual(len(important_frames), expected_count)

    def test_metadata_json_creation(self):
        """メタデータJSONファイルの作成テスト"""
        if not self.test_video.exists():
            self.skipTest(f"テスト用動画が見つかりません: {self.test_video}")

        frames = self.extractor.extract_frames(str(self.test_video))
        saved_paths = self.extractor.save_frames(frames, str(self.frames_dir))
        
        # 各フレームに対応するJSONファイルを確認
        for path in saved_paths:
            json_path = path.replace('.jpg', '.json')
            self.assertTrue(os.path.exists(json_path))
            
            # JSONファイルの内容を検証
            with open(json_path, 'r') as f:
                metadata = json.load(f)
                self.assertIn('timestamp', metadata)
                self.assertIn('frame_number', metadata)
                self.assertIn('is_important', metadata)
                self.assertIn('scene_change_score', metadata)
                self.assertIn('metadata', metadata)

    def test_extract_frames(self):
        """フレーム抽出の基本機能テスト"""
        if not self.test_video.exists():
            self.skipTest(f"テスト用動画が見つかりません: {self.test_video}")

        frames = self.extractor.extract_frames(str(self.test_video))
        
        self.assertGreater(len(frames), 0)
        
        for frame in frames:
            self.assertIn('timestamp', frame)
            self.assertIn('image', frame)
            self.assertIsInstance(frame['image'], Image.Image)
            self.assertIn('frame_number', frame)
            self.assertIn('scene_change_score', frame)
            
            if frames.index(frame) > 0:
                prev_frame = frames[frames.index(frame) - 1]
                time_diff = frame['timestamp'] - prev_frame['timestamp']
                # 許容誤差を0.001秒に設定
                self.assertGreaterEqual(time_diff, 0.999,
                    f"フレーム間隔が1.0秒未満です: {time_diff}秒")

    def test_extract_frames_with_custom_interval(self):
        """カスタム間隔でのフレーム抽出テスト"""
        # カスタム設定（0.5秒間隔）
        config = {
            'interval': 0.5,
            'quality': 90,
            'target_frames_per_hour': 1000,
            'important_frames_ratio': 0.05
        }
        self.extractor = FrameExtractor(config=config)

        # フレーム抽出の実行
        frames = self.extractor.extract_frames(str(self.test_video))

        # 基本的な検証
        self.assertGreater(len(frames), 0, "フレームが抽出されていません")

        # 各フレームの検証
        for i, frame in enumerate(frames):
            # 必須フィールドの存在確認
            self.assertIn('timestamp', frame, "タイムスタンプがありません")
            self.assertIn('frame_number', frame, "フレーム番号がありません")
            self.assertIn('image', frame, "画像データがありません")
            self.assertIn('metadata', frame, "メタデータがありません")

            # メタデータの検証
            metadata = frame['metadata']
            self.assertIn('fps', metadata, "FPSがありません")
            self.assertIn('total_frames', metadata, "総フレーム数がありません")
            self.assertIn('duration', metadata, "動画長がありません")
            self.assertIn('quality', metadata, "画質設定がありません")
            self.assertEqual(metadata['quality'], 90, "画質設定が正しくありません")

            # 時間間隔の検証（2フレーム目以降）
            if i > 0:
                time_diff = frame['timestamp'] - frames[i-1]['timestamp']
                self.assertGreaterEqual(time_diff, 0.45, 
                    f"フレーム間隔が0.45秒未満です: {time_diff}")
                self.assertLessEqual(time_diff, 0.55, 
                    f"フレーム間隔が0.55秒を超えています: {time_diff}")

            # 画像データの検証
            self.assertIsInstance(frame['image'], Image.Image, 
                "画像データが正しい形式ではありません")
            self.assertGreater(frame['image'].width, 0, "画像幅が0以下です")
            self.assertGreater(frame['image'].height, 0, "画像高さが0以下です")

    def test_nonexistent_video(self):
        """存在しない動画ファイルのテスト"""
        with self.assertRaises(FrameExtractionError):
            self.extractor.extract_frames("nonexistent.mp4")

    def test_save_frames(self):
        """フレーム保存のテスト"""
        if not self.test_video.exists():
            self.skipTest(f"テスト用動画が見つかりません: {self.test_video}")

        frames = self.extractor.extract_frames(str(self.test_video))
        saved_paths = self.extractor.save_frames(frames, str(self.frames_dir))
        
        self.assertGreater(len(saved_paths), 0)
        for path in saved_paths:
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.getsize(path) > 0)
            
            img = Image.open(path)
            self.assertIsInstance(img, Image.Image)

    def tearDown(self):
        """各テストのクリーンアップ"""
        if hasattr(self, 'frames_dir') and self.frames_dir.exists():
            shutil.rmtree(self.frames_dir)
            self.frames_dir.mkdir(exist_ok=True, parents=True)

    @classmethod
    def tearDownClass(cls):
        """テストクラスのクリーンアップ"""
        if hasattr(cls, 'test_dir') and cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

if __name__ == '__main__':
    unittest.main() 