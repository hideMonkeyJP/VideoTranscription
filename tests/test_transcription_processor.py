import unittest
import torch
import platform
from src.analysis.transcription.transcription_processor import TranscriptionProcessor, TranscriptionError

class TestTranscriptionProcessor(unittest.TestCase):
    """TranscriptionProcessorのテストクラス"""

    def setUp(self):
        """テストの前準備"""
        self.config = {
            'model_name': 'tiny',
            'language': 'ja'
        }

    @unittest.skipIf(not torch.cuda.is_available() or platform.system() != 'Windows',
                    "CUDAが利用できない環境、またはWindows以外の環境ではスキップ")
    def test_device_selection_with_cuda(self):
        """CUDA GPUが利用可能な場合のデバイス選択テスト"""
        processor = TranscriptionProcessor(self.config)
        self.assertEqual(processor.device, 'cuda')

    @unittest.skipIf(not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available() or platform.system() != 'Darwin',
                    "MPSが利用できない環境、またはMac以外の環境ではスキップ")
    def test_device_selection_with_mps(self):
        """Apple Silicon (MPS)が利用可能な場合のデバイス選択テスト"""
        processor = TranscriptionProcessor(self.config)
        self.assertEqual(processor.device, 'mps')

    def test_device_selection_cpu_fallback(self):
        """GPUが利用できない場合のCPUフォールバックテスト"""
        # GPUが利用できない場合のテスト
        config = self.config.copy()
        config['device'] = 'cpu'
        processor = TranscriptionProcessor(config)
        self.assertEqual(processor.device, 'cpu')

    def test_device_selection_from_config(self):
        """設定からのデバイス指定テスト"""
        # 設定からデバイスを指定する場合のテスト
        config = self.config.copy()
        config['device'] = 'cpu'
        processor = TranscriptionProcessor(config)
        self.assertEqual(processor.device, 'cpu')

    def test_device_selection_error_handling(self):
        """デバイス選択時のエラーハンドリングテスト"""
        # エラーハンドリングのテスト
        config = self.config.copy()
        config['device'] = 'invalid_device'
        with self.assertRaises(TranscriptionError):
            processor = TranscriptionProcessor(config)

if __name__ == '__main__':
    unittest.main() 