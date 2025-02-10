import os
import json
import pytest
from src.utils.data_generator import VideoContextDataGenerator

@pytest.fixture
def sample_json_path(tmp_path):
    """テスト用のJSONファイルを作成"""
    data = {
        "timestamp": "20250209_154640",
        "contexts": [
            {
                "time_range": {"start": 0.0, "end": 7.9},
                "summary": "テストサマリー1",
                "scenes": [
                    {
                        "keywords": ["テスト", "キーワード", "その1"],
                        "timestamp": 0.52
                    }
                ],
                "screenshots": [
                    {"timestamp": 0.52, "frame_number": 13}
                ]
            },
            {
                "time_range": {"start": 7.9, "end": 19.42},
                "summary": "テストサマリー2",
                "scenes": [
                    {
                        "keywords": ["テスト", "キーワード", "その2"],
                        "timestamp": 8.32
                    }
                ],
                "screenshots": [
                    {"timestamp": 8.32, "frame_number": 208}
                ]
            }
        ]
    }
    
    json_path = tmp_path / "test_final_result.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return str(json_path)

def test_generate_context_data(sample_json_path):
    """文脈データの生成をテスト"""
    generator = VideoContextDataGenerator(sample_json_path)
    contexts = generator.generate_context_data()
    
    assert len(contexts) == 2
    
    # 最初の文脈をチェック
    context1 = contexts[0]
    assert context1["id"] == 1
    assert context1["title"] == "文脈1: 0.0秒 - 7.9秒"
    assert context1["summary"] == "テストサマリー1"
    assert context1["timestamp"] == "0.0秒 - 7.9秒"
    assert context1["keywords"] == ["テスト", "キーワード", "その1"]
    assert context1["screenshot"]["filename"] == "screenshot_000.png"
    
    # 2番目の文脈をチェック
    context2 = contexts[1]
    assert context2["id"] == 2
    assert context2["title"] == "文脈2: 7.9秒 - 19.4秒"
    assert context2["summary"] == "テストサマリー2"
    assert context2["timestamp"] == "7.9秒 - 19.4秒"
    assert context2["keywords"] == ["テスト", "キーワード", "その2"]
    assert context2["screenshot"]["filename"] == "screenshot_020.png"

def test_save_to_json(sample_json_path, tmp_path):
    """JSONファイルへの保存をテスト"""
    output_path = tmp_path / "test_output" / "contexts_data.json"
    
    generator = VideoContextDataGenerator(sample_json_path)
    generator.save_to_json(str(output_path))
    
    # 出力ファイルが作成されたことを確認
    assert os.path.exists(output_path)
    
    # 出力内容を確認
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    assert "timestamp" in data
    assert "source_file" in data
    assert data["source_file"] == sample_json_path
    assert len(data["contexts"]) == 2