import os
import json
from datetime import datetime
from dotenv import load_dotenv
from src.utils.data_generator import VideoContextDataGenerator
from src.utils.notion_sync import NotionContextSync

def test_new_notion_workflow():
    """データ生成からNotion同期までの新しいワークフローをテスト"""
    try:
        # .envファイルを読み込む
        load_dotenv()
        
        # 入力JSONファイルのパス (元のコードと同じパスを使用)
        input_json = "output_test/json_test/final_result.json"
        
        # 中間データの出力先
        output_dir = "output_test/notion_test"
        os.makedirs(output_dir, exist_ok=True)
        contexts_json = os.path.join(output_dir, "contexts_data.json")
        
        print("\n=== データ生成フェーズ ===")
        # VideoContextDataGeneratorを使用してデータを生成
        generator = VideoContextDataGenerator(input_json)
        generator.save_to_json(contexts_json)
        print(f"コンテキストデータを生成: {contexts_json}")
        
        print("\n=== Notion同期フェーズ ===")
        # 環境変数から認証情報を取得 (元のコードと同じ)
        notion_auth_token = os.getenv('NOTION_AUTH_TOKEN')
        notion_database_id = os.getenv('NOTION_DATABASE_ID')
        gyazo_access_token = "8bAQUk5x4GrEqT6-1xmIMClIRt2F6QGpWds_LY3kDGs"
        
        if not notion_auth_token or not notion_database_id:
            print("Notion認証情報が設定されていません")
            print(f"NOTION_AUTH_TOKEN: {notion_auth_token}")
            print(f"NOTION_DATABASE_ID: {notion_database_id}")
            return
        
        # NotionContextSyncを使用してデータを同期
        syncer = NotionContextSync(notion_auth_token, notion_database_id, gyazo_access_token)
        results = syncer.sync_from_json(
            input_json,  # 元のJSONファイルを直接使用
            "output_test/json_test"  # スクリーンショットの基準ディレクトリ
        )
        
        print("\n=== 同期結果 ===")
        print(f"同期された文脈数: {len(results)}")
        for i, result in enumerate(results, 1):
            print(f"\n文脈{i}:")
            print(f"ページID: {result.get('id')}")
            print(f"URL: {result.get('url')}")
            
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    test_new_notion_workflow()