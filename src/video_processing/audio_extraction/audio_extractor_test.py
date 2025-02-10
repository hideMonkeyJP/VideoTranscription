import unittest
import os
import shutil
from audio_extractor import AudioExtractor

class TestAudioExtractor(unittest.TestCase):
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
            'format': 'wav',
            'sample_rate': 16000
        }
        self.extractor = AudioExtractor(config=self.config)

    def test_init(self):
        """初期化のテスト"""
        self.assertEqual(self.extractor.format, 'wav')
        self.assertEqual(self.extractor.sample_rate, 16000)

    def test_extract_audio(self):
        """音声抽出のテスト"""
        output_path = os.path.join(self.test_output_dir, 'test_output.wav')
        result_path = self.extractor.extract_audio(self.test_video, output_path)
        
        # 出力ファイルが生成されていることを確認
        self.assertTrue(os.path.exists(result_path))
        self.assertEqual(result_path, output_path)
        
        # 出力ファイルが音声ファイルとして有効であることを確認
        audio_info = self.extractor.get_audio_info(result_path)
        self.assertIn('duration', audio_info)
        self.assertIn('format', audio_info)
        self.assertEqual(audio_info['format'], 'wav')

    def test_extract_audio_without_output_path(self):
        """出力パスを指定しない場合のテスト"""
        result_path = self.extractor.extract_audio(self.test_video)
        
        # 出力ファイルが生成されていることを確認
        self.assertTrue(os.path.exists(result_path))
        self.assertTrue(result_path.endswith('.wav'))

    def test_nonexistent_video(self):
        """存在しない動画ファイルのテスト"""
        with self.assertRaises(FileNotFoundError):
            self.extractor.extract_audio('nonexistent.mp4')

    def test_get_audio_info_wav(self):
        """WAVファイルの情報取得テスト"""
        # まず音声を抽出
        output_path = os.path.join(self.test_output_dir, 'test_output.wav')
        self.extractor.extract_audio(self.test_video, output_path)
        
        # 音声情報を取得
        audio_info = self.extractor.get_audio_info(output_path)
        
        # 必要な情報が含まれていることを確認
        self.assertIn('channels', audio_info)
        self.assertIn('sample_width', audio_info)
        self.assertIn('frame_rate', audio_info)
        self.assertIn('frames', audio_info)
        self.assertIn('duration', audio_info)
        self.assertIn('format', audio_info)

    def test_get_audio_info_nonexistent(self):
        """存在しない音声ファイルの情報取得テスト"""
        with self.assertRaises(FileNotFoundError):
            self.extractor.get_audio_info('nonexistent.wav')

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