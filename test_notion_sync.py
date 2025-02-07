import os
from datetime import datetime
from src.utils.notion_client import NotionClient

def test_notion_sync():
    # テストデータの作成
    test_data = {
        "title": "テスト動画",
        "summary": "これはテスト用の要約です。",
        "keywords": ["テスト", "動画", "サンプル"],
        "duration": 120.5,
        "thumbnail_url": "https://example.com/thumbnail.jpg"
    }

    # NotionClientの初期化
    # 環境変数の読み込み
    from dotenv import load_dotenv
    load_dotenv(override=True)  # 既存の環境変数を上書き
    
    auth_token = os.getenv("NOTION_AUTH_TOKEN")
    database_id = os.getenv("NOTION_DATABASE_ID")
    
    print("\n環境変数の値:")
    print(f"Database ID from env: {database_id}")
    
    if not auth_token or not database_id:
        print("Error: NOTION_AUTH_TOKEN or NOTION_DATABASE_ID is not set")
        return

    client = NotionClient(auth_token, database_id)
    
    try:
        # Notionにデータを同期
        # リクエストの詳細をデバッグ出力
        print("\nNotionへのリクエスト:")
        print("- Auth Token:", auth_token[:10] + "..." if auth_token else "Not set")
        print("- Database ID:", database_id)
        print("\nテストデータ:")
        print(test_data)

        # リクエストペイロードを表示
        properties = client.create_video_entry(test_data)
        print("\nリクエストペイロード:")
        import json
        print(json.dumps(properties, indent=2, ensure_ascii=False))

        result = client.create_page(properties)
        print("\nNotion同期結果:")
        print(f"Page ID: {result.get('id')}")
        print(f"URL: {result.get('url')}")
        return result
    except Exception as e:
        print(f"\nError: {str(e)}")
        if hasattr(e, 'response'):
            print("\nエラーレスポンス:")
            print(e.response.json())
        return None

if __name__ == "__main__":
    test_notion_sync()