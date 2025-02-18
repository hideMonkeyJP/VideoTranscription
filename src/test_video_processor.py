import os
import unittest
from video_processor import VideoProcessor
import cv2
import numpy as np
import json
import shutil
import yaml
from pathlib import Path
from PIL import Image
from src.utils.config import Config

class TestVideoProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """テストクラスの前準備"""
        # プロジェクトのルートディレクトリを取得
        cls.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.test_output_dir = os.path.join(cls.root_dir, 'test_output')
        cls.test_video = os.path.join(cls.root_dir, 'videos', 'Sample.mp4')
        cls.test_config = os.path.join(cls.root_dir, 'config', 'test_config.yaml')
        
        # テスト用の出力ディレクトリを作成
        os.makedirs(cls.test_output_dir, exist_ok=True)
        
        # テスト用の設定ファイルを作成
        cls._create_test_config()

    @classmethod
    def _create_test_config(cls):
        """テスト用の設定ファイルを作成"""
        config = {
            'models': {
                'whisper_model': 'base',
                'tesseract_path': '/usr/local/bin/tesseract'
            },
            'speech_recognition': {
                'language': 'ja',
                'whisper_model': 'base',
                'temperature': 0.0,
                'beam_size': 5
            },
            'screenshot': {
                'interval': 1.0,
                'quality': 95
            },
            'ocr': {
                'language': 'jpn',
                'min_quality': 0.5
            },
            'audio_extraction': {
                'format': 'wav',
                'sample_rate': 16000
            },
            'output': {
                'image_format': 'PNG',
                'image_quality': 95
            }
        }
        
        os.makedirs(os.path.dirname(cls.test_config), exist_ok=True)
        with open(cls.test_config, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, allow_unicode=True, default_flow_style=False)

    def setUp(self):
        """各テストの前準備"""
        self.processor = VideoProcessor(config_path=self.test_config)

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
        
        # スクリーンショットの形式を確認
        for screenshot in screenshots:
            self.assertIn('timestamp', screenshot)
            self.assertIn('frame_number', screenshot)
            self.assertIn('image', screenshot)
            self.assertIsInstance(screenshot['timestamp'], (int, float))
            self.assertIsInstance(screenshot['frame_number'], int)
            self.assertTrue(hasattr(screenshot['image'], 'save'))  # PIL Imageオブジェクトの確認
        
        print(f"取得されたスクリーンショット数: {len(screenshots)}")
        print("✅ スクリーンショット取得テスト完了")

    def test_save_results(self):
        """結果保存のテスト"""
        print("\n=== 結果保存テスト ===")
        # テスト用のダミーデータを作成
        dummy_transcription = [
            {'text': 'テストセグメント1', 'start': 0, 'end': 1},
            {'text': 'テストセグメント2', 'start': 1, 'end': 2}
        ]
        
        dummy_screenshots = [
            {'timestamp': 0.5, 'frame_number': 15, 'image': self._create_dummy_image()},
            {'timestamp': 1.5, 'frame_number': 45, 'image': self._create_dummy_image()}
        ]
        
        dummy_analysis = {
            'summary': 'テスト用の要約です',
            'keywords': ['テスト', 'キーワード'],
            'topics': ['トピック1', 'トピック2']
        }
        
        # 結果を保存
        result = self.processor.save_results(
            self.test_output_dir,
            dummy_transcription,
            dummy_screenshots,
            dummy_analysis
        )
        
        self.assertIsNotNone(result)
        self.assertIn('json_path', result)
        self.assertIn('screenshots_dir', result)
        
        # JSONファイルの存在を確認
        self.assertTrue(os.path.exists(result['json_path']))
        
        # スクリーンショットディレクトリの存在を確認
        self.assertTrue(os.path.exists(result['screenshots_dir']))
        
        # JSONファイルの内容を確認
        with open(result['json_path'], 'r', encoding='utf-8') as f:
            saved_result = json.load(f)
        
        self.assertIn('transcription', saved_result)
        self.assertIn('analysis', saved_result)
        self.assertEqual(len(saved_result['transcription']), 2)
        
        print("✅ 結果保存テスト完了")

    def test_full_process(self):
        """全体処理のテスト"""
        print("\n=== 全体処理テスト ===")
        result = self.processor.process_video(self.test_video, self.test_output_dir)
        self.assertIsNotNone(result)
        self.assertIn('json_path', result)
        self.assertIn('screenshots_dir', result)
        self.assertTrue(os.path.exists(result['json_path']))
        self.assertTrue(os.path.exists(result['screenshots_dir']))
        print("✅ 全体処理テスト完了")

    def test_nonexistent_video(self):
        """存在しないビデオファイルのテスト"""
        print("\n=== 存在しないビデオファイルテスト ===")
        with self.assertRaises(Exception):
            self.processor.process_video('nonexistent.mp4', self.test_output_dir)
        print("✅ 存在しないビデオファイルテスト完了")

    def test_empty_video(self):
        """空のビデオファイルのテスト"""
        print("\n=== 空のビデオファイルテスト ===")
        # 空のファイルを作成
        empty_video = os.path.join(self.test_output_dir, 'empty.mp4')
        with open(empty_video, 'w') as f:
            pass
        
        with self.assertRaises(Exception):
            self.processor.process_video(empty_video, self.test_output_dir)
        print("✅ 空のビデオファイルテスト完了")

    def _create_dummy_image(self):
        """テスト用のダミー画像を作成"""
        image = Image.new('RGB', (100, 100), color='white')
        return image

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
        if os.path.exists(cls.test_config):
            os.remove(cls.test_config)

def main():
    """VideoProcessorを使用してOCR処理を実行"""
    # 入力ディレクトリのパス
    input_dir = "/Users/takayanagihidenori/Cursor/VideoTranscription/output_test/notion_test/screenshots_20250209_160200"
    
    if not os.path.exists(input_dir):
        print(f"エラー: 入力ディレクトリが見つかりません: {input_dir}")
        return
        
    # 出力ディレクトリの作成
    output_dir = Path(__file__).parent.parent / "output" / "video_processor_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("画像の読み込みを開始...")
        # 画像ファイルの読み込み
        frames = []
        for file in sorted(os.listdir(input_dir)):
            if file.endswith('.png'):
                image_path = os.path.join(input_dir, file)
                image = Image.open(image_path)
                frame_number = int(file.split('_')[1].split('.')[0])
                frames.append({
                    'image': image,
                    'frame_number': frame_number,
                    'timestamp': frame_number * 0.52,  # 0.52秒間隔
                    'importance_score': 1.0  # デフォルトスコア
                })
        print(f"読み込まれた画像数: {len(frames)}")
        
        # VideoProcessorの設定
        config = {
            'ocr_processor': {
                'lang': 'jpn+eng',  # 日本語と英語
                'psm': 3,
                'oem': 3,
                'min_confidence': 60.0  # 信頼度閾値
            },
            'video_processor': {
                'output_dir': str(output_dir),
                'temp_dir': str(output_dir / 'temp')
            }
        }
        processor = VideoProcessor(config)
        
        print("\nOCR処理を開始...")
        # OCR処理の実行
        results = processor.ocr_processor.process_frames(frames)
        
        # 結果の表示と保存
        print("\n=== OCR処理結果 ===")
        for screenshot in results['screenshots']:
            frame_number = screenshot.get('frame_number', 'unknown')
            timestamp = screenshot.get('timestamp', 0.0)
            texts = screenshot.get('texts', [])
            
            if texts:
                print(f"\nフレーム {frame_number} (タイムスタンプ: {timestamp:.1f}秒):")
                for text_info in texts:
                    confidence = text_info.get('confidence', 0.0)
                    text = text_info.get('text', '')
                    print(f"- {text} (信頼度: {confidence:.1f}%)")
        
        # 結果をJSONファイルとして保存
        output_file = output_dir / "ocr_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n詳細な結果をJSONファイルとして保存しました: {output_file}")
            
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
    main() 