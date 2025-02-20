"""
Supabaseデータ登録機能のテスト
"""

import json
import pytest
from pathlib import Path
from src.tools.supabase_register import register_to_supabase

def test_register_to_supabase_file_not_found(tmp_path):
    """存在しないファイルを指定した場合のテスト"""
    result = register_to_supabase(
        json_path="not_exists.json",
        video_title="テスト",
        video_path="not_exists.mp4",
        duration=60
    )
    assert result is False

def test_register_to_supabase_invalid_json(tmp_path):
    """不正なJSONファイルの場合のテスト"""
    # 不正なJSONファイルを作成
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{invalid json")
    
    result = register_to_supabase(
        json_path=str(invalid_json),
        video_title="テスト",
        video_path="test.mp4",
        duration=60
    )
    assert result is False

def test_register_to_supabase_success(tmp_path):
    """正常系のテスト"""
    # テスト用のJSONファイルを作成
    test_json = {
        "segments": [
            {
                "No": 1,
                "Summary": "テストセグメント1",
                "Timestamp": "0.0秒 - 10.0秒",
                "Thumbnail": "https://example.com/thumb1.jpg"
            }
        ]
    }
    json_path = tmp_path / "test.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(test_json, f, ensure_ascii=False)
    
    # テスト用の動画ファイルを作成（空ファイル）
    video_path = tmp_path / "test.mp4"
    video_path.touch()
    
    result = register_to_supabase(
        json_path=str(json_path),
        video_title="テスト動画",
        video_path=str(video_path),
        duration=60
    )
    assert result is True  # 注: 実際のSupabase接続はモック化が必要 