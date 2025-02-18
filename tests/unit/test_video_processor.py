import os
import unittest
from pathlib import Path
from datetime import datetime

from src.video_processor import VideoProcessor, VideoProcessingError

class TestVideoProcessor(unittest.TestCase):
    """VideoProcessorのテストケース - フレーム抽出フェーズ"""
    
    def setUp(self):
        """テストの前準備"""
        self.test_dir = Path("test_output")
        self.test_dir.mkdir(exist_ok=True)
        
        self.config = {
            "output_dir": str(self.test_dir),
            "frame_extractor_config": {
                "frame_interval": 1.0,  # 1秒ごとにフレームを抽出
                "min_scene_duration": 2.0  # 最小シーン長2秒
            }
        }
        
        self.processor = VideoProcessor(self.config)
        
        # テスト用の動画ファイル
        self.test_video = "videos/sample.mp4"
        
        # テスト用の動画ファイルが存在しない場合はスキップ
        if not os.path.exists(self.test_video):
            self.skipTest(f"テスト用の動画ファイルが見つかりません: {self.test_video}")
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        # テストで生成されたファイルを削除
        if self.test_dir.exists():
            for file in self.test_dir.glob("**/*"):
                if file.is_file():
                    file.unlink()
            for dir in reversed(list(self.test_dir.glob("**/*"))):
                if dir.is_dir():
                    dir.rmdir()
            self.test_dir.rmdir()
    
    def test_video_file_not_found(self):
        """存在しない動画ファイルのテスト"""
        with self.assertRaises(VideoProcessingError) as context:
            self.processor.process_video("non_existent.mp4")
        
        self.assertIn("動画ファイルが見つかりません", str(context.exception))
    
    def test_frame_extraction(self):
        """フレーム抽出の基本機能テスト"""
        # 動画処理の実行
        result = self.processor.process_video(self.test_video)
        
        # 基本的な結果の検証
        self.assertIsInstance(result, dict)
        self.assertIn("timestamp", result)
        self.assertIn("video_path", result)
        self.assertIn("frames", result)
        self.assertIn("total_frames", result)
        
        # タイムスタンプの検証
        timestamp = datetime.fromisoformat(result["timestamp"])
        self.assertIsInstance(timestamp, datetime)
        
        # フレーム情報の検証
        frames = result["frames"]
        self.assertGreater(len(frames), 0)
        
        # フレームディレクトリの検証
        frames_dir = self.test_dir / "frames"
        self.assertTrue(frames_dir.exists())
        
        # 保存されたフレーム画像の検証
        saved_frames = list(frames_dir.glob("*.jpg"))
        self.assertEqual(len(saved_frames), len(frames))
        
        # 各フレームの内容検証
        for frame in frames:
            self.assertIn("timestamp", frame)
            self.assertIn("saved_path", frame)
            self.assertTrue(os.path.exists(frame["saved_path"]))
    
    def test_frame_extraction_with_config(self):
        """設定を変更してのフレーム抽出テスト"""
        # フレーム間隔を2秒に変更
        self.processor.config["frame_extractor_config"]["frame_interval"] = 2.0
        
        # 動画処理の実行
        result = self.processor.process_video(self.test_video)
        
        # 結果の検証
        self.assertIsInstance(result, dict)
        self.assertGreater(result["total_frames"], 0)
        
        # 2秒間隔でのフレーム数を検証
        frames = result["frames"]
        if len(frames) > 1:
            time_diff = frames[1]["timestamp"] - frames[0]["timestamp"]
            self.assertAlmostEqual(time_diff, 2.0, places=1)

if __name__ == '__main__':
    unittest.main() 