import os
import unittest
from video_processor import VideoProcessor

class TestVideoProcessor(unittest.TestCase):
    def setUp(self):
        """テストの前準備"""
        self.processor = VideoProcessor(output_dir='test_output')
        self.test_video = 'videos/Sample_.mp4'  # 小さいサイズのテストビデオを使用
        os.makedirs('test_output', exist_ok=True)

    def test_audio_extraction(self):
        """音声抽出のテスト"""
        print("\n=== 音声抽出テスト ===")
        audio_path = self.processor.extract_audio(self.test_video)
        self.assertIsNotNone(audio_path)
        self.assertTrue(os.path.exists(audio_path))
        print("✅ 音声抽出テスト完了")

    def test_transcription(self):
        """音声認識のテスト"""
        print("\n=== 音声認識テスト ===")
        audio_path = self.processor.extract_audio(self.test_video)
        transcription = self.processor.transcribe_audio(audio_path)
        self.assertIsNotNone(transcription)
        self.assertGreater(len(transcription), 0)
        print(f"認識されたセグメント数: {len(transcription)}")
        print("✅ 音声認識テスト完了")

    def test_screenshot_capture(self):
        """スクリーンショット取得のテスト"""
        print("\n=== スクリーンショット取得テスト ===")
        screenshots = self.processor.capture_screenshots(self.test_video)
        self.assertIsNotNone(screenshots)
        self.assertGreater(len(screenshots), 0)
        print(f"取得されたスクリーンショット数: {len(screenshots)}")
        print("✅ スクリーンショット取得テスト完了")

    def test_ocr(self):
        """OCRのテスト"""
        print("\n=== OCRテスト ===")
        screenshots = self.processor.capture_screenshots(self.test_video)
        processed_screenshots = self.processor.process_screenshots(screenshots)
        self.assertIsNotNone(processed_screenshots)
        text_count = sum(1 for ss in processed_screenshots if ss['text'].strip())
        print(f"テキストが検出されたスクリーンショット数: {text_count}/{len(processed_screenshots)}")
        print("✅ OCRテスト完了")

    def test_full_process(self):
        """全体処理のテスト"""
        print("\n=== 全体処理テスト ===")
        result = self.processor.process_video(self.test_video)
        self.assertIsNotNone(result)
        self.assertIn('segments', result)
        self.assertGreater(len(result['segments']), 0)
        print(f"生成されたセグメント数: {len(result['segments'])}")
        print("✅ 全体処理テスト完了")

    def test_nonexistent_video(self):
        """存在しないビデオファイルのテスト"""
        print("\n=== 存在しないビデオファイルテスト ===")
        result = self.processor.process_video('nonexistent.mp4')
        self.assertIsNotNone(result)
        self.assertIn('metadata', result)
        self.assertIn('error', result['metadata'])
        self.assertFalse(result['metadata']['success'])
        print("✅ 存在しないビデオファイルテスト完了")

    def test_empty_video(self):
        """空のビデオファイルのテスト"""
        print("\n=== 空のビデオファイルテスト ===")
        # 空のファイルを作成
        empty_video = 'test_output/empty.mp4'
        with open(empty_video, 'w') as f:
            pass
        
        result = self.processor.process_video(empty_video)
        self.assertIsNotNone(result)
        self.assertIn('metadata', result)
        self.assertFalse(result['metadata']['success'])
        print("✅ 空のビデオファイルテスト完了")

    def test_process_segment_error(self):
        """セグメント処理エラーのテスト"""
        print("\n=== セグメント処理エラーテスト ===")
        # 無効なセグメントを作成
        invalid_segment = {
            'start': 0,
            'end': 1,
            'text': None  # 無効なテキスト
        }
        
        result = self.processor.process_segment(invalid_segment, 0, 1)
        self.assertIsNotNone(result)
        self.assertIn('text', result)
        self.assertIn('heading', result)
        self.assertIn('summary', result)
        self.assertIn('key_points', result)
        print("✅ セグメント処理エラーテスト完了")

    def tearDown(self):
        """テスト後のクリーンアップ"""
        import shutil
        if os.path.exists('test_output'):
            shutil.rmtree('test_output')

if __name__ == '__main__':
    unittest.main(verbosity=2) 