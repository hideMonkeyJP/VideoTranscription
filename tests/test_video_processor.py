import os
import pytest
from src.video_processor import VideoProcessor

@pytest.fixture
def video_processor():
    return VideoProcessor(output_dir='test_output')

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