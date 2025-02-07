import os
import pytest
from unittest.mock import Mock, patch
from src.utils.notion_client import NotionClient
from src.video_processor import VideoProcessor

@pytest.fixture
def mock_notion_client():
    return Mock(spec=NotionClient)

@pytest.fixture
def video_processor_with_notion(mock_notion_client):
    processor = VideoProcessor()
    processor.notion_client = mock_notion_client
    return processor

def test_notion_client_initialization():
    """NotionClientの初期化テスト"""
    auth_token = "test_token"
    database_id = "test_db_id"
    
    client = NotionClient(auth_token, database_id)
    assert client.auth_token == auth_token
    assert client.database_id == database_id
    assert client.base_url == "https://api.notion.com/v1"

def test_format_video_data():
    """動画データのフォーマットテスト"""
    client = NotionClient("test_token", "test_db_id")
    
    test_data = {
        "metadata": {
            "title": "Test Video",
            "duration": 120
        },
        "summary": {
            "keywords": ["test", "video", "content"],
            "key_scenes": [
                {"timestamp": 10, "summary": "Scene 1"},
                {"timestamp": 20, "summary": "Scene 2"}
            ]
        }
    }
    
    formatted = client.format_video_data(test_data)
    
    assert formatted["title"] == "Test Video"
    assert formatted["duration"] == 120
    assert len(formatted["keywords"]) == 3
    assert "test" in formatted["keywords"]

@patch('requests.post')
def test_create_video_entry(mock_post):
    """動画エントリの作成テスト"""
    # Mockレスポンスの設定
    mock_response = Mock()
    mock_response.json.return_value = {"id": "test_page_id", "url": "https://notion.so/test"}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    # NotionClientの初期化
    client = NotionClient("test_token", "test_db_id")
    
    test_data = {
        "title": "Test Video",
        "summary": "Test summary",
        "keywords": ["test", "video"],
        "duration": 120
    }
    
    result = client.create_video_entry(test_data)
    
    # 結果の検証
    assert result["id"] == "test_page_id"
    
    # APIコールの検証
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[0][0] == "https://api.notion.com/v1/pages"
    
    # リクエストボディの検証
    request_body = call_args[1]["json"]
    assert request_body["properties"]["Name"]["title"][0]["text"]["content"] == "Test Video"
    assert request_body["properties"]["Duration"]["number"] == 120.0

def test_sync_to_notion(video_processor_with_notion):
    """Notionとの同期テスト"""
    mock_notion_client = video_processor_with_notion.notion_client
    mock_notion_client.format_video_data.return_value = {
        "title": "Test Video",
        "summary": "Test summary",
        "keywords": ["test", "video"],
        "duration": 120
    }
    mock_notion_client.create_video_entry.return_value = {"id": "test_page_id"}
    
    test_analysis = {
        "metadata": {"title": "Test Video"},
        "summary": {
            "keywords": ["test", "video"],
            "key_scenes": []
        }
    }
    
    result = video_processor_with_notion.sync_to_notion(test_analysis)
    assert result["id"] == "test_page_id"
    mock_notion_client.format_video_data.assert_called_once()
    mock_notion_client.create_video_entry.assert_called_once()

def test_notion_integration_disabled(video_processor_with_notion):
    """Notion連携が無効な場合のテスト"""
    video_processor_with_notion.notion_client = None
    result = video_processor_with_notion.sync_to_notion({})
    assert result is None

@patch('requests.post')
def test_full_notion_integration(mock_post):
    """完全なNotion統合テスト"""
    # Mockレスポンスの設定
    mock_response = Mock()
    mock_response.json.return_value = {
        "id": "test_page_id",
        "url": "https://notion.so/test",
        "properties": {
            "Name": {"title": [{"text": {"content": "Integration Test Video"}}]},
            "Duration": {"number": 60}
        }
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    client = NotionClient("test_token", "test_db_id")
    test_data = {
        "title": "Integration Test Video",
        "summary": "Test summary for integration",
        "keywords": ["integration", "test"],
        "duration": 60
    }

    result = client.create_video_entry(test_data)
    
    # 結果の検証
    assert result["id"] == "test_page_id"
    assert "properties" in result
    
    # APIコールの検証
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[0][0] == "https://api.notion.com/v1/pages"
    
    # リクエストボディの検証
    request_body = call_args[1]["json"]
    assert request_body["properties"]["Name"]["title"][0]["text"]["content"] == "Integration Test Video"
    assert request_body["properties"]["Duration"]["number"] == 60.0