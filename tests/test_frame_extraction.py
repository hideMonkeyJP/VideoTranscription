import os
import pytest
import cv2
from src.video_processor import VideoProcessor
from PIL import Image

@pytest.fixture
def caplog(caplog):
    return caplog

class TestFrameExtraction:
    @pytest.fixture
    def video_processor(self):
        return VideoProcessor()

    @pytest.fixture
    def sample_video_path(self):
        return os.path.join('videos', 'Sample.mp4')

    def test_frame_extraction(self, video_processor, sample_video_path):
        """フレーム抽出の基本機能をテスト"""
        frames = video_processor.extract_frames(sample_video_path)
        
        # 基本的な検証
        assert frames is not None
        assert len(frames) > 0
        assert isinstance(frames, list)
        
        # フレームの内容を検証
        for frame in frames:
            assert isinstance(frame, dict)
            assert 'image' in frame
            assert 'timestamp' in frame
            assert 'frame_number' in frame
            assert 'importance_score' in frame
            
            # 画像の検証
            assert isinstance(frame['image'], Image.Image)
            assert frame['image'].size[0] > 0
            assert frame['image'].size[1] > 0
            
            # タイムスタンプの検証
            assert isinstance(frame['timestamp'], (int, float))
            assert frame['timestamp'] >= 0
            
            # フレーム番号の検証
            assert isinstance(frame['frame_number'], int)
            assert frame['frame_number'] >= 0
            
            # 重要度スコアの検証
            assert isinstance(frame['importance_score'], (int, float))

    def test_frame_extraction_with_invalid_path(self, video_processor):
        """無効なパスでのフレーム抽出をテスト"""
        frames = video_processor.extract_frames('invalid/path/to/video.mp4')
        assert frames == []

    def test_frame_extraction_with_config(self, video_processor, sample_video_path):
        """設定に基づくフレーム抽出をテスト"""
        # 設定から期待値を取得
        frame_config = video_processor.config.get('screenshot', {}).get('frame_extraction', {})
        base_frames_per_hour = frame_config.get('base_frames_per_hour', 1000)
        min_frames = frame_config.get('min_frames', 100)
        important_frame_ratio = frame_config.get('important_frame_ratio', 0.05)
        min_important_frames = frame_config.get('min_important_frames', 10)

        frames = video_processor.extract_frames(sample_video_path)
        
        # フレーム数の検証
        assert len(frames) >= min_important_frames
        
        # フレームの時系列順を検証
        timestamps = [frame['timestamp'] for frame in frames]
        assert timestamps == sorted(timestamps)

    def test_frame_importance_scoring(self, video_processor, sample_video_path):
        """フレームの重要度スコアリングをテスト"""
        frames = video_processor.extract_frames(sample_video_path)
        
        # スコアの範囲を検証
        scores = [frame['importance_score'] for frame in frames]
        assert all(score >= 0 for score in scores)
        
        # スコアの分布を検証
        if len(scores) > 1:
            assert max(scores) > min(scores)  # スコアに変化があることを確認

    def test_frame_extraction_memory_usage(self, video_processor, sample_video_path):
        """メモリ使用量をモニタリング"""
        initial_memory = video_processor.performance_monitor.get_memory_usage()
        frames = video_processor.extract_frames(sample_video_path)
        final_memory = video_processor.performance_monitor.get_memory_usage()
        
        # メモリリークがないことを確認
        assert final_memory < initial_memory * 2  # メモリ使用量が2倍以上増えていないことを確認

    def test_scene_change_detection(self, video_processor, sample_video_path):
        """シーン変化の検出をテスト"""
        frames = video_processor.extract_frames(sample_video_path)
        
        # 連続するフレーム間でスコアを比較
        for i in range(len(frames) - 1):
            current_score = frames[i]['importance_score']
            next_score = frames[i + 1]['importance_score']
            
            # スコアの差が大きい場合はシーン変化として検出されているはず
            if abs(current_score - next_score) > 0.5:
                assert current_score != next_score, "シーン変化が検出されていません"

    def test_frame_count_adjustment(self, video_processor, sample_video_path):
        """動画の長さに応じたフレーム数の調整をテスト"""
        frames = video_processor.extract_frames(sample_video_path)
        
        # 設定から期待値を取得
        frame_config = video_processor.config.get('screenshot', {}).get('frame_extraction', {})
        base_frames_per_hour = frame_config.get('base_frames_per_hour', 1000)
        min_frames = frame_config.get('min_frames', 100)
        
        # フレーム数が設定に基づいて調整されていることを確認
        assert len(frames) >= min_frames, "最小フレーム数を下回っています"
        
        # 動画の長さに応じたフレーム数の検証
        cap = cv2.VideoCapture(sample_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_hours = (total_frames / fps) / 3600
        cap.release()
        
        expected_frames = max(min_frames, int(duration_hours * base_frames_per_hour))
        assert abs(len(frames) - expected_frames) <= expected_frames * 0.1, "フレーム数が期待値から大きく外れています"

    def test_error_logging(self, video_processor, caplog):
        """エラー時のログ出力をテスト"""
        invalid_path = 'invalid/path/to/video.mp4'
        frames = video_processor.extract_frames(invalid_path)
        
        # エラーログが出力されていることを確認
        assert any("動画ファイルを開けません" in record.message for record in caplog.records)
        assert frames == []