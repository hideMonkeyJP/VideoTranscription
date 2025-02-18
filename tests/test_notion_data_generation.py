import json
import csv
from pathlib import Path
import pytest
from src.video_processor import VideoProcessor

def test_generate_notion_data(tmp_path):
    # テスト用のfinal_result.jsonを作成
    final_result = {
        "metadata": {
            "processed_at": "2025-02-15T12:23:21.819942",
            "version": "1.0.0",
            "screenshot_count": 104
        },
        "analysis": {
            "segments": [
                {
                    "time_range": {
                        "start": 0.0,
                        "end": 8.0
                    },
                    "text": "テストセグメント1",
                    "analysis": {
                        "heading": "テスト見出し1",
                        "summary": "テスト要約1",
                        "key_points": ["ポイント1", "ポイント2", "ポイント3"]
                    }
                },
                {
                    "time_range": {
                        "start": 8.0,
                        "end": 15.5
                    },
                    "text": "テストセグメント2",
                    "analysis": {
                        "heading": "テスト見出し2",
                        "summary": "テスト要約2",
                        "key_points": ["ポイント4", "ポイント5", "ポイント6"]
                    }
                }
            ]
        }
    }

    # テストデータの保存
    test_dir = tmp_path / "test_output"
    test_dir.mkdir()
    final_result_path = test_dir / "final_result.json"
    with open(final_result_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    # VideoProcessorの初期化（Notion同期を無効化）
    config = {
        'notion': {
            'enabled': False  # Notion同期を無効化
        }
    }
    processor = VideoProcessor(config)
    output_path = test_dir / "Regist.csv"
    processor.generate_notion_data(str(final_result_path), str(output_path))

    # 生成されたCSVファイルの検証
    assert output_path.exists()

    with open(output_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        rows = list(reader)

        # ヘッダー行の検証
        assert rows[0] == ['Title', 'Summary', 'Keywords', 'Timestamp', 'Thumbnail', 'FileName']

        # データ行の検証
        assert len(rows) == 3  # ヘッダー + 2セグメント
        
        # 1つ目のセグメントの検証
        assert rows[1][0] == "テスト見出し1: 0.0秒 - 8.0秒"  # Title
        assert rows[1][1] == "テスト要約1"  # Summary
        assert rows[1][2] == "ポイント1, ポイント2, ポイント3"  # Keywords
        assert rows[1][3] == "0.0秒 - 8.0秒"  # Timestamp
        assert rows[1][4].startswith("https://i.gyazo.com/placeholder_")  # Thumbnail
        assert rows[1][5].startswith("screenshot_")  # FileName

        # 2つ目のセグメントの検証
        assert rows[2][0] == "テスト見出し2: 8.0秒 - 15.5秒"  # Title
        assert rows[2][1] == "テスト要約2"  # Summary
        assert rows[2][2] == "ポイント4, ポイント5, ポイント6"  # Keywords
        assert rows[2][3] == "8.0秒 - 15.5秒"  # Timestamp
        assert rows[2][4].startswith("https://i.gyazo.com/placeholder_")  # Thumbnail
        assert rows[2][5].startswith("screenshot_")  # FileName