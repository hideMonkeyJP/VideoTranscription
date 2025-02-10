import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path
from .transcription_processor import TranscriptionProcessor, TranscriptionError

class TestTranscriptionProcessor(unittest.TestCase):
    def setUp(self):
        """テストの前準備"""
        self.patcher = patch("whisper.load_model")
        self.mock_whisper_load = self.patcher.start()
        
        # Whisperモデルのモックを設定
        self.mock_model = MagicMock()
        self.mock_model.transcribe.return_value = {
            "text": "テストの音声です",
            "segments": [
                {
                    "text": "テストの音声です",
                    "start": 0.0,
                    "end": 2.0,
                    "words": [
                        {"word": "テスト", "start": 0.0, "end": 1.0, "probability": 0.9},
                        {"word": "の", "start": 1.0, "end": 1.2, "probability": 0.8},
                        {"word": "音声", "start": 1.2, "end": 1.8, "probability": 0.95},
                        {"word": "です", "start": 1.8, "end": 2.0, "probability": 0.85}
                    ]
                }
            ]
        }
        self.mock_whisper_load.return_value = self.mock_model
        
        self.processor = TranscriptionProcessor(model_name="tiny")
        self.test_audio_path = str(Path(__file__).parent / "test_data" / "test_audio.wav")
        
    def tearDown(self):
        """テストの後片付け"""
        self.patcher.stop()
        
    def test_init(self):
        """初期化のテスト"""
        self.assertIsNotNone(self.processor.model)
        self.assertIsNotNone(self.processor.logger)
        
    @patch("librosa.load")
    def test_transcribe(self, mock_librosa_load):
        """文字起こしのテスト"""
        # モックの設定
        mock_audio = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
        mock_librosa_load.return_value = (mock_audio, 16000)
        
        # テストの実行
        result = self.processor.transcribe_audio(self.test_audio_path)
        
        # 検証
        self.assertEqual(result["text"], "テストの音声です")
        self.assertEqual(len(result["segments"]), 1)
        self.assertEqual(len(result["segments"][0]["words"]), 4)
        
        # モックの呼び出し確認
        mock_librosa_load.assert_called_once_with(self.test_audio_path, sr=16000, dtype=np.float32)
        self.mock_model.transcribe.assert_called_once()
        transcribe_args = self.mock_model.transcribe.call_args[1]
        self.assertEqual(transcribe_args["language"], "ja")
        self.assertEqual(transcribe_args["task"], "transcribe")
        self.assertTrue(transcribe_args["condition_on_previous_text"])
        self.assertTrue(transcribe_args["word_timestamps"])
        
    def test_get_word_timestamps(self):
        """単語タイムスタンプ取得のテスト"""
        # テストデータ
        test_result = {
            "segments": [
                {
                    "words": [
                        {"word": "テスト", "start": 0.0, "end": 1.0, "probability": 0.9},
                        {"word": "です", "start": 1.0, "end": 1.5, "probability": 0.8}
                    ]
                }
            ]
        }
        
        # テストの実行
        words = self.processor.get_word_timestamps(test_result)
        
        # 検証
        self.assertEqual(len(words), 2)
        self.assertEqual(words[0]["text"], "テスト")
        self.assertEqual(words[0]["start"], 0.0)
        self.assertEqual(words[0]["end"], 1.0)
        self.assertEqual(words[0]["probability"], 0.9)
        
    def test_nonexistent_audio(self):
        """存在しない音声ファイルのテスト"""
        with self.assertRaises(TranscriptionError):
            self.processor.transcribe_audio("nonexistent.wav")
            
if __name__ == "__main__":
    unittest.main() 