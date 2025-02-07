import os
import pytest
from src.video_processor import VideoProcessor, AudioExtractionError
from moviepy.editor import AudioFileClip

class TestAudioExtraction:
    @pytest.fixture
    def video_processor(self):
        return VideoProcessor()

    @pytest.fixture
    def sample_video_path(self):
        return os.path.join('videos', 'Sample.mp4')

    def test_audio_extraction(self, video_processor, sample_video_path):
        """音声抽出の基本機能をテスト"""
        audio_path = video_processor.extract_audio(sample_video_path)
        
        # 基本的な検証
        assert audio_path is not None
        assert os.path.exists(audio_path)
        assert audio_path.endswith('.wav')
        
        # 音声ファイルの検証
        audio = AudioFileClip(audio_path)
        assert audio.duration > 0
        assert audio.fps == 44100  # デフォルトのサンプルレート
        
        # クリーンアップ
        audio.close()
        if os.path.exists(audio_path):
            os.remove(audio_path)

    def test_audio_extraction_with_invalid_path(self, video_processor):
        """無効なパスでの音声抽出をテスト"""
        with pytest.raises(AudioExtractionError):
            video_processor.extract_audio('invalid/path/to/video.mp4')

    def test_audio_extraction_with_config(self, video_processor, sample_video_path):
        """設定に基づく音声抽出をテスト"""
        # 設定から期待値を取得
        audio_config = video_processor.config.get('audio_extraction', {})
        expected_format = audio_config.get('format', 'wav')
        expected_sample_rate = audio_config.get('sample_rate', 44100)  # デフォルトを44100Hzに変更
        
        audio_path = video_processor.extract_audio(sample_video_path)
        
        # 音声ファイルの検証
        audio = AudioFileClip(audio_path)
        assert audio.fps == expected_sample_rate
        
        # クリーンアップ
        audio.close()
        if os.path.exists(audio_path):
            os.remove(audio_path)

    def test_audio_extraction_temp_directory(self, video_processor, sample_video_path):
        """一時ディレクトリの動作をテスト"""
        # 一時ディレクトリの存在を確認
        assert os.path.exists(video_processor.temp_dir)
        
        audio_path = video_processor.extract_audio(sample_video_path)
        
        # 音声ファイルが一時ディレクトリに保存されていることを確認
        assert os.path.dirname(audio_path) == video_processor.temp_dir
        
        # クリーンアップ
        if os.path.exists(audio_path):
            os.remove(audio_path)

    def test_audio_extraction_memory_usage(self, video_processor, sample_video_path):
        """メモリ使用量をモニタリング"""
        initial_memory = video_processor.performance_monitor.get_memory_usage()
        audio_path = video_processor.extract_audio(sample_video_path)
        final_memory = video_processor.performance_monitor.get_memory_usage()
        
        # メモリリークがないことを確認
        assert final_memory < initial_memory * 2
        
        # クリーンアップ
        if os.path.exists(audio_path):
            os.remove(audio_path)

    def test_audio_extraction_error_handling(self, video_processor, sample_video_path, monkeypatch):
        """エラーハンドリングをテスト"""
        class MockVideoClip:
            def __init__(self, *args, **kwargs):
                self.reader = None
            @property
            def audio(self):
                raise AudioExtractionError("動画に音声トラックが含まれていません")
            def close(self):
                pass
        
        # VideoFileClipのモック
        monkeypatch.setattr("moviepy.editor.VideoFileClip", MockVideoClip)
        
        with pytest.raises(AudioExtractionError) as exc_info:
            video_processor.extract_audio(sample_video_path)
        
        assert "動画に音声トラックが含まれていません" in str(exc_info.value)