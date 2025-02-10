import os
from dotenv import load_dotenv
from src.utils.notion_client import NotionClient

def test_notion_connection():
    """Notionデータベースへの接続テスト"""
    # .envファイルを読み込む
    load_dotenv()
    
    # 環境変数から認証情報を取得
    notion_auth_token = os.getenv('NOTION_AUTH_TOKEN')
    notion_database_id = os.getenv('NOTION_DATABASE_ID')
    
    print("=== 認証情報 ===")
    print(f"トークン: {notion_auth_token}")
    print(f"データベースID: {notion_database_id}")
    
    try:
        # NotionClientを初期化
        notion_client = NotionClient(notion_auth_token, notion_database_id)
        
        # データベースのプロパティを表示
        print("\n=== データベースのプロパティ ===")
        for name, prop in notion_client.properties.items():
            print(f"{name}: {prop.get('type')}")
            
        print("\n接続テスト成功！")
        return True
        
    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")
        return False

if __name__ == "__main__":
    test_notion_connection() 