import os
import json
from typing import Dict, List, Any, Optional
from .notion_client import NotionClient
from .gyazo_client import GyazoClient

class NotionContextSync:
    """生成されたコンテキストデータをNotionと同期するクラス"""
    
    def __init__(self, notion_auth_token: str, notion_database_id: str, gyazo_access_token: str):
        """
        Args:
            notion_auth_token (str): NotionのAPIトークン
            notion_database_id (str): NotionデータベースのID
            gyazo_access_token (str): GyazoのAPIトークン
        """
        self.notion_client = NotionClient(notion_auth_token, notion_database_id)
        self.gyazo_client = GyazoClient(gyazo_access_token)
    
    def sync_from_json(self, json_path: str, base_screenshot_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """JSONファイルからデータを読み込んでNotionと同期する

        Args:
            json_path (str): 同期するJSONファイルのパス
            base_screenshot_dir (Optional[str]): スクリーンショットの基準ディレクトリ
                                               指定がない場合はJSONファイルのディレクトリを使用

        Returns:
            List[Dict[str, Any]]: 各文脈の同期結果
        """
        # JSONファイルを読み込む
        with open(json_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        # スクリーンショットの基準ディレクトリを設定
        if base_screenshot_dir is None:
            base_screenshot_dir = os.path.dirname(json_path)
        
        # データベースのプロパティを表示
        self.notion_client.print_database_schema()
        
        print("\n=== 同期するデータ ===")
        print(f"文脈数: {len(result.get('contexts', []))}")
        print(f"総シーン数: {result.get('metadata', {}).get('total_scenes', 0)}")
        print(f"総セグメント数: {result.get('metadata', {}).get('total_segments', 0)}")
        
        # 各文脈を同期
        results = []
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
                screenshot_path = f"{base_screenshot_dir}/screenshots_{result['timestamp']}/{screenshot_filename}"
                
                print(f"\nスクリーンショットをアップロード: {screenshot_path}")
                if os.path.exists(screenshot_path):
                    # Gyazoに画像をアップロード
                    image_url = self.gyazo_client.upload_image(
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
            notion_result = self.notion_client.create_page(properties)
            print(f"\n=== Notion同期結果 (文脈{i}) ===")
            print(f"ページID: {notion_result.get('id')}")
            print(f"URL: {notion_result.get('url')}")
            if image_url:
                print(f"スクリーンショットURL: {image_url}")
            
            results.append(notion_result)
        
        return results

def main():
    """使用例"""
    # 環境変数から認証情報を取得
    notion_auth_token = os.getenv('NOTION_AUTH_TOKEN')
    notion_database_id = os.getenv('NOTION_DATABASE_ID')
    gyazo_access_token = "8bAQUk5x4GrEqT6-1xmIMClIRt2F6QGpWds_LY3kDGs"
    
    if not notion_auth_token or not notion_database_id:
        print("Notion認証情報が設定されていません")
        return
    
    # NotionContextSyncを初期化
    syncer = NotionContextSync(notion_auth_token, notion_database_id, gyazo_access_token)
    
    # JSONファイルからデータを同期
    results = syncer.sync_from_json(
        "output_test/notion_test/contexts_data.json",
        "output_test/json_test"
    )
    
    print(f"\n同期完了: {len(results)}個の文脈を同期しました")

if __name__ == "__main__":
    main()