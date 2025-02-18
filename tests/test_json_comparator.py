import json
import copy
import pytest
from pathlib import Path
from src.utils.json_comparator import (
    normalize_text,
    compare_json_files,
    compare_summaries,
    compare_contexts,
    compare_scenes,
    compare_generic_json
)

@pytest.fixture
def sample_summaries():
    return [
        {
            "timestamp": 0.52,
            "screenshot_text": "テストテキスト1",
            "summary": {
                "main_points": ["要点1", "要点2"],
                "full_text": "完全なテキスト1"
            },
            "keywords": ["キーワード1", "キーワード2"],
            "segments": ["セグメント1"],
            "importance_score": 0.8,
            "metadata": {
                "segment_count": 1,
                "has_screenshot_text": True,
                "summary_points": 2,
                "keyword_count": 2
            }
        }
    ]

@pytest.fixture
def sample_contexts():
    return [
        {
            "time_range": {
                "start": 0.0,
                "end": 1.0
            },
            "summary": "テスト要約",
            "scenes": [
                {
                    "timestamp": 0.5,
                    "screenshot_text": "シーンテキスト",
                    "summary": {
                        "main_points": ["シーン要点"],
                        "full_text": "シーン完全テキスト"
                    },
                    "keywords": ["シーンキーワード"],
                    "segments": ["シーンセグメント"],
                    "importance_score": 0.7,
                    "metadata": {
                        "segment_count": 1
                    }
                }
            ]
        }
    ]

def test_normalize_text():
    """テキスト正規化のテスト"""
    # 基本的な正規化
    assert normalize_text("テスト です。") == "テスト"
    assert normalize_text("Test TEXT") == "testtext"
    
    # 記号の正規化
    assert normalize_text("「こんにちは」、世界!") == "こんにちは世界"
    
    # 空白の除去
    assert normalize_text("  スペース  テスト  ") == "スペーステスト"

def test_compare_summaries(sample_summaries):
    """summaries.jsonの比較テスト"""
    # 同一データの比較
    assert compare_summaries(sample_summaries, sample_summaries) == True
    
    # タイムスタンプが異なる場合(許容範囲内)
    modified = copy.deepcopy(sample_summaries)
    modified[0]["timestamp"] = 0.57  # 0.05秒の差
    assert compare_summaries(sample_summaries, modified) == True
    
    # タイムスタンプが異なる場合(許容範囲外)
    modified = copy.deepcopy(sample_summaries)
    modified[0]["timestamp"] = 0.82  # 0.3秒の差
    assert compare_summaries(sample_summaries, modified) == False
    
    # テキストが異なる場合
    modified = copy.deepcopy(sample_summaries)
    modified[0]["screenshot_text"] = "全く異なるテキスト"
    assert compare_summaries(sample_summaries, modified) == False

def test_compare_contexts(sample_contexts):
    """contextsの比較テスト"""
    # 同一データの比較
    assert compare_contexts(sample_contexts, sample_contexts) == True
    
    # time_rangeが異なる場合
    modified = copy.deepcopy(sample_contexts)
    modified[0]["time_range"]["start"] = 0.3  # 0.3秒の差
    assert compare_contexts(sample_contexts, modified) == False
    
    # シーン数が異なる場合
    modified = copy.deepcopy(sample_contexts)
    modified[0]["scenes"].append(copy.deepcopy(modified[0]["scenes"][0]))
    assert compare_contexts(sample_contexts, modified) == False
    
    # シーンの内容が異なる場合
    modified = copy.deepcopy(sample_contexts)
    modified[0]["scenes"][0]["screenshot_text"] = "異なるシーンテキスト"
    assert compare_contexts(sample_contexts, modified) == False

def test_compare_scenes(sample_contexts):
    """シーンの比較テスト"""
    scene = sample_contexts[0]["scenes"][0]
    
    # 同一データの比較
    assert compare_scenes(scene, scene) == True
    
    # タイムスタンプが異なる場合
    modified = copy.deepcopy(scene)
    modified["timestamp"] = 0.8  # 0.3秒の差
    assert compare_scenes(scene, modified) == False
    
    # テキストが異なる場合
    modified = copy.deepcopy(scene)
    modified["screenshot_text"] = "全く異なるテキスト"
    assert compare_scenes(scene, modified) == False
    
    # メタデータが異なる場合
    modified = copy.deepcopy(scene)
    modified["metadata"]["segment_count"] = 2
    assert compare_scenes(scene, modified) == False

def test_compare_generic_json():
    """汎用的なJSON比較のテスト"""
    # 単純な値の比較
    assert compare_generic_json(1, 1) == True
    assert compare_generic_json("test", "test") == True
    assert compare_generic_json(1, 2) == False
    
    # 配列の比較
    assert compare_generic_json([1, 2, 3], [1, 2, 3]) == True
    assert compare_generic_json([1, 2], [1, 2, 3]) == False
    
    # オブジェクトの比較
    obj1 = {"a": 1, "b": 2}
    obj2 = {"b": 2, "a": 1}
    assert compare_generic_json(obj1, obj2) == True
    
    obj3 = {"a": 1, "b": 3}
    assert compare_generic_json(obj1, obj3) == False

def test_compare_json_files(tmp_path):
    """JSONファイル比較の統合テスト"""
    # テスト用のJSONファイルを作成
    file1 = tmp_path / "test1.json"
    file2 = tmp_path / "test2.json"
    
    # 基本的なJSONデータ
    data = {
        "key": "value",
        "number": 42,
        "array": [1, 2, 3]
    }
    
    # 同じデータを両方のファイルに書き込む
    with open(file1, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    with open(file2, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    
    # ファイル比較のテスト
    assert compare_json_files(str(file1), str(file2)) == True
    
    # 異なるデータでテスト
    modified_data = copy.deepcopy(data)
    modified_data["key"] = "different"
    with open(file2, 'w', encoding='utf-8') as f:
        json.dump(modified_data, f)
    
    assert compare_json_files(str(file1), str(file2)) == False

def test_error_cases():
    """エラーケースのテスト"""
    # 無効なJSONデータ
    with pytest.raises(Exception):
        compare_generic_json(None, {"key": "value"})
    
    # 型の不一致
    assert compare_generic_json([1, 2, 3], {"key": "value"}) == False
    
    # キーの不一致
    obj1 = {"a": 1, "b": 2}
    obj2 = {"a": 1, "c": 2}
    assert compare_generic_json(obj1, obj2) == False