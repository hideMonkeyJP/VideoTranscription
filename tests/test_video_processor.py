import os
import unittest
import pytest
from unittest.mock import patch, MagicMock
from src.video_processor import VideoProcessor, VideoProcessingError
from pathlib import Path
from datetime import datetime
import json
import shutil
import tempfile

from src.utils.config import Config

@pytest.fixture(autouse=True)
def setup_environment():
    """テスト環境のセットアップ"""
    with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_api_key'}):
        yield

@pytest.fixture
def mock_gemini():
    """Geminiモデルのモック"""
    with patch('google.generativeai.GenerativeModel') as mock:
        mock_model = MagicMock()
        mock_model.generate_content.return_value.text = '{"heading": "テスト", "summary": "要約", "key_points": ["ポイント1"]}'
        mock.return_value = mock_model
        yield mock

@pytest.fixture
def video_processor(mock_gemini):
    """VideoProcessorのインスタンスを作成"""
    config = {
        'video_processor': {
            'output_dir': 'test_output',
            'temp_dir': 'temp',
            'log_dir': 'logs'
        },
        'frame_extractor': {
            'output_dir': 'test_output/frames',
            'frame_interval': 1
        },
        'audio_extractor': {
            'output_dir': 'test_output/audio'
        },
        'ocr_processor': {
            'min_confidence': 0.6
        },
        'text_analyzer': {
            'model_name': 'ja_core_news_lg'
        }
    }
    return VideoProcessor(config)

def test_process_segment(video_processor):
    # テスト用のセグメントデータ
    test_segment = {
        'text': 'これはテストのテキストです。重要なポイントが含まれています。',
        'start': 0,
        'end': 10
    }
    
    # セグメントを処理
    result = video_processor.process_segment(test_segment, 0, 1)
    
    # 結果の検証
    assert isinstance(result, dict)
    assert 'start_time' in result
    assert 'end_time' in result
    assert 'text' in result
    assert 'heading' in result
    assert 'summary' in result
    assert 'key_points' in result
    assert 'screenshot' in result
    
    # 値の検証
    assert result['start_time'] == 0
    assert result['end_time'] == 10
    assert result['text'] == test_segment['text']
    assert isinstance(result['heading'], str)
    assert isinstance(result['summary'], str)
    assert isinstance(result['key_points'], list)
    assert result['screenshot'] == 'screenshot_0.jpg'

def test_clean_llm_response(video_processor):
    # 辞書形式のレスポンス
    dict_response = {'generated_text': '見出し：テストの見出し'}
    assert video_processor._clean_llm_response(dict_response) == 'テストの見出し'
    
    # リスト形式のレスポンス
    list_response = [{'generated_text': '要約：テストの要約'}]
    assert video_processor._clean_llm_response(list_response) == 'テストの要約'
    
    # 文字列形式のレスポンス
    str_response = 'ポイント：テストのポイント'
    assert video_processor._clean_llm_response(str_response) == 'テストのポイント'

def test_extract_key_points(video_processor):
    # 箇条書き形式のレスポンス
    response = """
    ・1つ目のポイント
    ・2つ目のポイント
    ・3つ目のポイント
    ・4つ目のポイント
    """
    result = video_processor._extract_key_points(response)
    
    assert isinstance(result, list)
    assert len(result) <= 3  # 最大3つまで
    assert '1つ目のポイント' in result[0]
    assert '2つ目のポイント' in result[1]
    assert '3つ目のポイント' in result[2]

def test_generate_html_report(video_processor):
    # テスト用のデータ
    test_data = {
        "metadata": {
            "processed_at": "2024-03-20T12:00:00",
            "video_duration": 100,
            "segment_count": 2,
            "screenshot_count": 2
        },
        "segments": [
            {
                "start_time": 0,
                "end_time": 10,
                "text": "最初のセグメントです。",
                "heading": "セグメント1",
                "summary": "最初のセグメントの要約です。",
                "key_points": ["ポイント1", "ポイント2", "ポイント3"],
                "screenshot": "screenshot_0.jpg"
            },
            {
                "start_time": 10,
                "end_time": 20,
                "text": "2番目のセグメントです。",
                "heading": "セグメント2",
                "summary": "2番目のセグメントの要約です。",
                "key_points": ["ポイント1", "ポイント2", "ポイント3"],
                "screenshot": "screenshot_10.jpg"
            }
        ]
    }
    
    # 出力ディレクトリの作成
    os.makedirs('test_output', exist_ok=True)
    output_path = os.path.join('test_output', 'test_report.html')
    
    # HTMLレポートを生成
    result = video_processor.generate_html_report(test_data, output_path)
    
    # 結果の検証
    assert result == True
    assert os.path.exists(output_path)
    
    # ファイルの内容を確認
    with open(output_path, 'r', encoding='utf-8') as f:
        content = f.read()
        assert '<!DOCTYPE html>' in content
        assert '<html lang="ja">' in content
        assert 'セグメント1' in content
        assert 'セグメント2' in content
        assert 'screenshot_0.jpg' in content
        assert 'screenshot_10.jpg' in content

def teardown_module(module):
    """テスト終了後のクリーンアップ"""
    import shutil
    if os.path.exists('test_output'):
        shutil.rmtree('test_output')

class TestVideoProcessor(unittest.TestCase):
    """VideoProcessorのテストケース"""
    
    def setUp(self):
        """テストの前準備"""
        # テスト用の一時ディレクトリを作成
        self.test_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.test_dir / "output"
        self.temp_dir = self.test_dir / "temp"
        
        # テスト用の設定を作成
        self.config = {
            "video_processor": {
                "output_dir": str(self.output_dir),
                "temp_dir": str(self.temp_dir)
            },
            "frame_extractor": {},
            "audio_extractor": {},
            "transcription": {},
            "ocr_processor": {},
            "text_analyzer": {},
            "notion_sync": {
                "auth_token": "test_token",
                "database_id": "test_db"
            }
        }
        
        self.processor = VideoProcessor(self.config)
        self.test_video = "test_video.mp4"

    def test_intermediate_files_creation(self):
        """中間ファイルの生成テスト"""
        # モックデータの準備
        mock_frames = [{"timestamp": 0, "path": "frame_0.jpg"}]
        mock_ocr_results = {"text": "test OCR"}
        mock_transcription = {"text": "test transcription"}
        mock_analysis = {"summary": "test analysis"}
        
        # 各メソッドをモック化
        with patch.object(self.processor.frame_extractor, 'extract_frames', return_value=mock_frames), \
             patch.object(self.processor.audio_extractor, 'extract_audio', return_value="test.wav"), \
             patch.object(self.processor.transcription_processor, 'transcribe_audio', return_value=mock_transcription), \
             patch.object(self.processor.ocr_processor, 'process_frames', return_value=mock_ocr_results), \
             patch.object(self.processor.text_analyzer, 'analyze_content', return_value=mock_analysis), \
             patch.object(self.processor.report_generator, 'generate_report'), \
             patch.object(self.processor.notion_sync, 'sync_results', return_value={"url": "test_url"}):
            
            # テスト実行
            result = self.processor.process_video(self.test_video)
            
            # 中間ファイルの存在確認
            self.assertTrue((self.temp_dir / "frames.json").exists())
            self.assertTrue((self.temp_dir / "ocr_results.json").exists())
            self.assertTrue((self.temp_dir / "transcription.json").exists())
            self.assertTrue((self.temp_dir / "analysis.json").exists())
            self.assertTrue((self.output_dir / "final_result.json").exists())
            self.assertTrue((self.output_dir / "report.html").exists())
            
            # 中間ファイルの内容検証
            with open(self.temp_dir / "ocr_results.json", "r") as f:
                ocr_data = json.load(f)
                self.assertEqual(ocr_data, mock_ocr_results)
            
            with open(self.temp_dir / "transcription.json", "r") as f:
                transcription_data = json.load(f)
                self.assertEqual(transcription_data, mock_transcription)
            
            # 結果の検証
            self.assertEqual(result["status"], "success")
            self.assertIn("timestamp", result)
            self.assertEqual(result["video_path"], self.test_video)
            self.assertEqual(result["output_dir"], str(self.output_dir))
            self.assertEqual(result["notion_page_url"], "test_url")
    
    def tearDown(self):
        """テストのクリーンアップ"""
        # テストディレクトリの削除
        shutil.rmtree(self.test_dir)

if __name__ == '__main__':
    unittest.main() 