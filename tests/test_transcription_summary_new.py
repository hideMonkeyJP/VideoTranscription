import os
import sys
import unittest
from pathlib import Path

# プロジェクトルートへのパスを追加
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.video_processor import VideoProcessor

class TestTranscriptionAndSummaryNew(unittest.TestCase):
    def setUp(self):
        """テストの前準備"""
        self.video_processor = VideoProcessor()
        self.test_video_path = os.path.join(project_root, 'videos', 'Sample.mp4')
        self.output_dir = os.path.join(project_root, 'output', 'test_output_new')
        os.makedirs(self.output_dir, exist_ok=True)

    def test_audio_extraction_and_transcription(self):
        """音声抽出と文字起こしのテスト"""
        try:
            # 音声の抽出
            audio_path = self.video_processor.extract_audio(self.test_video_path)
            self.assertTrue(os.path.exists(audio_path))
            
            # 音声認識
            transcription = self.video_processor.transcribe_audio(audio_path)
            self.assertIsNotNone(transcription)
            self.assertGreater(len(transcription), 0)
            
            # 文字起こし結果の検証
            for segment in transcription:
                self.assertIn('text', segment)
                self.assertIn('start', segment)
                self.assertIn('end', segment)
                self.assertGreater(len(segment['text']), 0)
                
        except Exception as e:
            self.fail(f"テストが失敗しました: {str(e)}")

    def test_text_summarization(self):
        """テキスト要約のテスト"""
        try:
            # テスト用のテキスト
            test_text = """
            人工知能技術の進化により、画像認識や自然言語処理などの分野で
            革新的な進歩が見られています。特に、大規模言語モデルの登場により、
            テキストの生成や要約、対話などのタスクで高い性能を発揮しています。
            また、深層学習の発展により、より複雑なパターンの認識や予測が可能になり、
            様々な産業分野での応用が進んでいます。
            """
            
            # 要約の生成
            summary = self.video_processor._generate_summary(test_text)
            self.assertIsNotNone(summary)
            self.assertGreater(len(summary), 0)
            self.assertLess(len(summary), len(test_text))
            
        except Exception as e:
            self.fail(f"テストが失敗しました: {str(e)}")

    @unittest.skip("エンドツーエンドテストは現在スキップします")
    def test_end_to_end_processing(self):
        """エンドツーエンドの処理テスト"""
        try:
            # 音声の抽出と文字起こし
            audio_path = self.video_processor.extract_audio(self.test_video_path)
            transcription = self.video_processor.transcribe_audio(audio_path)
            
            # スクリーンショットの取得と処理
            screenshots = self.video_processor.extract_frames(self.test_video_path)
            processed_screenshots = self.video_processor.process_screenshots(screenshots)
            
            # コンテンツ分析(要約を含む)
            analysis = self.video_processor.analyze_content(transcription, processed_screenshots)
            
            # 結果の検証
            self.assertIn('scene_analyses', analysis)
            self.assertIn('keywords', analysis)
            self.assertIn('topics', analysis)
            
            # 各シーンの分析結果を検証
            for scene in analysis['scene_analyses']:
                self.assertIn('timestamp', scene)
                self.assertIn('summary', scene)
                self.assertGreater(len(scene['summary']), 0)
            
        except Exception as e:
            self.fail(f"テストが失敗しました: {str(e)}")

    def tearDown(self):
        """テスト後のクリーンアップ"""
        # 一時ファイルの削除
        import shutil
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

if __name__ == '__main__':
    unittest.main() 