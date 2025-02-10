import os
import json
import pytest
from unittest.mock import Mock, patch
from src.utils.notion_sync import NotionContextSync

@pytest.fixture
def sample_context_data():
    """テスト用の文脈データを作成"""
    return {
        "id": 1,
        "title": "文脈1: 0.0秒 - 7.9秒",
        "summary": "テストサマリー",
        "timestamp": "0.0秒 - 7.9秒",
        "keywords": ["テスト", "キーワード", "その1"],
        "screenshot": {
            "filename": "screenshot_000.png",
            "path": "screenshots_20250209_154640/screenshot_000.png",
            "timestamp": 0.52
        },
        "time_range": {"start": 0.0, "end": 7.9}
    }

@pytest.fixture
def sample_json_data(sample_context_data, tmp_path):
    """テスト用のJSONファイルを作成"""
    data = {
        "timestamp": "20250209_154640",
        "source_file": "test_final_result.json",
        "contexts": [sample_context_data]
    }
    
    json_path = tmp_path / "contexts_data.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return str(json_path)

@pytest.fixture
def mock_notion_client():
    """NotionClientのモック"""
    with patch('src.utils.notion_sync.NotionClient') as mock:
        client = Mock()
        client.create_page.return_value = {
            "id": "test-page-id",
            "url": "https://notion.so/test-page"
        }
        mock.return_value = client
        yield client

@pytest.fixture
def mock_gyazo_client():
    """GyazoClientのモック"""
    with patch('src.utils.notion_sync.GyazoClient') as mock:
        client = Mock()
        client.upload_image.return_value = "https://gyazo.com/test-image"
        mock.return_value = client
        yield client

def test_sync_context(sample_context_data, mock_notion_client, mock_gyazo_client, tmp_path):
    """1つの文脈の同期をテスト"""
    # スクリーンショットのテストファイルを作成
    screenshot_dir = tmp_path / "screenshots_20250209_154640"
    os.makedirs(screenshot_dir)
    screenshot_path = screenshot_dir / "screenshot_000.png"
    with open(screenshot_path, 'w') as f:
        f.write("dummy image data")
    
    syncer = NotionContextSync("test-token", "test-db-id", "test-gyazo-token")
    result = syncer.sync_context(sample_context_data, str(tmp_path))
    
    # Gyazoへのアップロードが呼ばれたことを確認
    mock_gyazo_client.upload_image.assert_called_once()
    upload_args = mock_gyazo_client.upload_image.call_args[0]
    assert str(screenshot_path) == upload_args[0]
    
    # Notionのページ作成が呼ばれたことを確認
    mock_notion_client.create_page.assert_called_once()
    page_props = mock_notion_client.create_page.call_args[0][0]
    
    assert page_props["Title"]["title"][0]["text"]["content"] == sample_context_data["title"]
    assert page_props["Summary"]["rich_text"][0]["text"]["content"] == sample_context_data["summary"]
    assert page_props["Timestamp"]["rich_text"][0]["text"]["content"] == sample_context_data["timestamp"]
    assert len(page_props["Keywords"]["multi_select"]) == len(sample_context_data["keywords"])
    assert page_props["FileName"]["rich_text"][0]["text"]["content"] == sample_context_data["screenshot"]["filename"]
    assert page_props["Thumbnail"]["files"][0]["external"]["url"] == "https://gyazo.com/test-image"

def test_sync_from_json(sample_json_data, mock_notion_client, mock_gyazo_client, tmp_path):
    """JSONファイルからの同期をテスト"""
    # スクリーンショットのテストファイルを作成
    screenshot_dir = tmp_path / "screenshots_20250209_154640"
    os.makedirs(screenshot_dir)
    screenshot_path = screenshot_dir / "screenshot_000.png"
    with open(screenshot_path, 'w') as f:
        f.write("dummy image data")
    
    syncer = NotionContextSync("test-token", "test-db-id", "test-gyazo-token")
    results = syncer.sync_from_json(sample_json_data, str(tmp_path))
    
    assert len(results) == 1
    assert results[0]["id"] == "test-page-id"
    assert results[0]["url"] == "https://notion.so/test-page"
    
    # Gyazoへのアップロードが呼ばれたことを確認
    mock_gyazo_client.upload_image.assert_called_once()
    
    # Notionのページ作成が呼ばれたことを確認
    mock_notion_client.create_page.assert_called_once()

def test_sync_missing_screenshot(sample_context_data, mock_notion_client, mock_gyazo_client, tmp_path):
    """スクリーンショットが見つからない場合のテスト"""
    syncer = NotionContextSync("test-token", "test-db-id", "test-gyazo-token")
    result = syncer.sync_context(sample_context_data, str(tmp_path))
    
    # Gyazoへのアップロードが呼ばれないことを確認
    mock_gyazo_client.upload_image.assert_not_called()
    
    # Notionのページ作成が呼ばれたことを確認
    mock_notion_client.create_page.assert_called_once()
    page_props = mock_notion_client.create_page.call_args[0][0]
    
    # Thumbnailプロパティが設定されていないことを確認
    assert "Thumbnail" not in page_props