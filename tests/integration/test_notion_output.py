import os
from dotenv import load_dotenv
import json
from src.utils.notion_client import NotionClient

def load_test_data():
    with open('output_test/test3/results_20250206_054207.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # データの整形
    summary = ""
    if data.get("analysis") and data["analysis"].get("topics"):
        summary = "\n".join(data["analysis"]["topics"])
    
    keywords = []
    if data.get("analysis") and data["analysis"].get("keywords"):
        keywords = data["analysis"]["keywords"][:10]  # 最大10個まで
    
    # 動画の長さを計算(最後のセグメントのend時間)
    duration = 0
    if data.get("transcription"):
        duration = data["transcription"][-1]["end"]
    
    return {
        "title": "タスク管理についての動画",
        "summary": summary,
        "keywords": keywords,
        "duration": duration,
        "thumbnail_url": "https://example.com/thumbnail.jpg"  # サンプル用
    }

def main():
    # 環境変数の読み込み
    load_dotenv(override=True)
    
    auth_token = os.getenv("NOTION_AUTH_TOKEN")
    database_id = os.getenv("NOTION_DATABASE_ID")
    
    print("\n環境変数の値:")
    print(f"Database ID: {database_id}")
    
    # テストデータの読み込み
    test_data = load_test_data()
    print("\nテストデータ:")
    print(json.dumps(test_data, indent=2, ensure_ascii=False))
    
    # Notionクライアントの初期化とデータの送信
    try:
        client = NotionClient(auth_token, database_id)
        result = client.create_video_entry(test_data)
        print("\nNotion同期結果:")
        print(f"Page ID: {result.get('id')}")
        print(f"URL: {result.get('url')}")
        return result
    except Exception as e:
        print(f"\nError: {str(e)}")
        return None

if __name__ == "__main__":
    main()