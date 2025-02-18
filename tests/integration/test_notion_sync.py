import os
import json
from datetime import datetime
from dotenv import load_dotenv
from src.utils.notion_client import NotionClient
from src.utils.gyazo_client import GyazoClient

def test_notion_sync():
    """既存のJSONファイルを使用してNotionとの同期をテスト"""
    # .envファイルを読み込む
    load_dotenv()
    
    # 環境変数から認証情報を取得
    notion_auth_token = os.getenv('NOTION_AUTH_TOKEN')
    notion_database_id = os.getenv('NOTION_DATABASE_ID')
    gyazo_access_token = "8bAQUk5x4GrEqT6-1xmIMClIRt2F6QGpWds_LY3kDGs"
    
    if not notion_auth_token or not notion_database_id:
        print("Notion認証情報が設定されていません")
        print(f"NOTION_AUTH_TOKEN: {notion_auth_token}")
        print(f"NOTION_DATABASE_ID: {notion_database_id}")
        return
    
    try:
        # NotionClientとGyazoClientを初期化
        notion_client = NotionClient(notion_auth_token, notion_database_id)
        gyazo_client = GyazoClient(gyazo_access_token)
        
        # データベースのプロパティを表示
        notion_client.print_database_schema()
        
        # 既存のJSONファイルを読み込む
        json_path = "output_test/json_test/final_result.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        print("\n=== 同期するデータ ===")
        print(f"文脈数: {len(result.get('contexts', []))}")
        print(f"総シーン数: {result.get('metadata', {}).get('total_scenes', 0)}")
        print(f"総セグメント数: {result.get('metadata', {}).get('total_segments', 0)}")
        
        # 各文脈をNotionに同期
        contexts = result.get('contexts', [])
        for i, context in enumerate(contexts, 1):
            # 文脈内の最後のスクリーンショットを使用
            screenshots = context.get('screenshots', [])
            image_url = None
            screenshot_filename = None
            if screenshots:
                last_screenshot = screenshots[-1]
                # スクリーンショットのインデックスを計算
                screenshot_index = (i - 1) * 20  # 各文脈20枚のスクリーンショット
                screenshot_filename = f"screenshot_{screenshot_index:03d}.png"
                screenshot_path = f"output_test/json_test/screenshots_{result['timestamp']}/{screenshot_filename}"
                
                print(f"\nスクリーンショットをアップロード: {screenshot_path}")
                if os.path.exists(screenshot_path):
                    # Gyazoに画像をアップロード
                    image_url = gyazo_client.upload_image(
                        screenshot_path,
                        description=f"文脈{i}のスクリーンショット ({context['time_range']['start']:.1f}秒 - {context['time_range']['end']:.1f}秒)"
                    )
                    if image_url:
                        print(f"Gyazoアップロード成功: {image_url}")
                else:
                    print(f"スクリーンショットが見つかりません: {screenshot_path}")
            
            # キーワードを収集(各シーンから最大3つずつ)
            keywords = []
            for scene in context.get('scenes', []):
                scene_keywords = scene.get('keywords', [])[:3]  # 各シーンから最大3つ
                keywords.extend(scene_keywords)
            # 重複を除去して最大10個に制限
            unique_keywords = list(dict.fromkeys(keywords))[:10]
            
            # Notionのプロパティを設定
            properties = {
                "Title": {
                    "title": [{"text": {"content": f"文脈{i}: {context['time_range']['start']:.1f}秒 - {context['time_range']['end']:.1f}秒"}}]
                },
                "Summary": {
                    "rich_text": [{"text": {"content": context['summary']}}]
                },
                "Timestamp": {
                    "rich_text": [{"text": {"content": f"{context['time_range']['start']:.1f}秒 - {context['time_range']['end']:.1f}秒"}}]
                },
                "Keywords": {
                    "multi_select": [{"name": kw} for kw in unique_keywords]
                }
            }
            
            # スクリーンショットのファイル名を設定
            if screenshot_filename:
                properties["FileName"] = {
                    "rich_text": [{"text": {"content": screenshot_filename}}]
                }
            
            # スクリーンショットのURLが取得できた場合は追加
            if image_url:
                properties["Thumbnail"] = {
                    "files": [
                        {
                            "type": "external",
                            "name": screenshot_filename,
                            "external": {
                                "url": image_url
                            }
                        }
                    ]
                }
            
            # ページを作成
            notion_result = notion_client.create_page(properties)
            print(f"\n=== Notion同期結果 (文脈{i}) ===")
            print(f"ページID: {notion_result.get('id')}")
            print(f"URL: {notion_result.get('url')}")
            if image_url:
                print(f"スクリーンショットURL: {image_url}")
            
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    test_notion_sync()