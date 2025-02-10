import os
import json
import pytest
from datetime import datetime
from dotenv import load_dotenv
from src.video_processor import VideoProcessor
from src.utils.data_generator import VideoContextDataGenerator
from src.utils.notion_sync import NotionContextSync

def test_video_to_notion_integration():
    """動画処理からNotion同期までの統合テスト"""
    try:
        # .envファイルを読み込む
        load_dotenv()
        
        # テスト用の出力ディレクトリ
        output_dir = "output_test/integration_test"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 動画処理フェーズ
        print("\n=== 動画処理フェーズ ===")
        video_path = "videos/Sample.mp4"
        processor = VideoProcessor()  # configは省略してデフォルトを使用
        result = processor.process_video(video_path, output_dir)
        
        # 中間ファイルの存在を確認
        assert os.path.exists(os.path.join(output_dir, "final_result.json")), "final_result.jsonが生成されていません"
        assert os.path.exists(os.path.join(output_dir, "ocr_results.json")), "ocr_results.jsonが生成されていません"
        assert os.path.exists(os.path.join(output_dir, "transcription.json")), "transcription.jsonが生成されていません"
        assert os.path.exists(os.path.join(output_dir, "summaries.json")), "summaries.jsonが生成されていません"
        
        # スクリーンショットディレクトリの存在を確認
        screenshot_dirs = [d for d in os.listdir(output_dir) if d.startswith("screenshots_")]
        assert len(screenshot_dirs) > 0, "スクリーンショットディレクトリが生成されていません"
        screenshot_dir = screenshot_dirs[-1]  # 最新のディレクトリを使用
        
        # スクリーンショットの存在を確認
        screenshot_path = os.path.join(output_dir, screenshot_dir)
        screenshots = [f for f in os.listdir(screenshot_path) if f.endswith(".png")]
        assert len(screenshots) > 0, "スクリーンショットが生成されていません"
        
        # 2. データ生成フェーズ
        print("\n=== データ生成フェーズ ===")
        input_json = os.path.join(output_dir, "final_result.json")
        contexts_json = os.path.join(output_dir, "contexts_data.json")
        
        generator = VideoContextDataGenerator(input_json)
        generator.save_to_json(contexts_json)
        print(f"コンテキストデータを生成: {contexts_json}")
        
        # 生成されたJSONの検証
        with open(contexts_json, 'r', encoding='utf-8') as f:
            contexts_data = json.load(f)
        assert 'contexts' in contexts_data, "生成されたJSONにcontextsが含まれていません"
        assert len(contexts_data['contexts']) > 0, "生成されたcontextsが空です"
        
        # 3. Notion同期フェーズ
        print("\n=== Notion同期フェーズ ===")
        notion_auth_token = os.getenv('NOTION_AUTH_TOKEN')
        notion_database_id = os.getenv('NOTION_DATABASE_ID')
        gyazo_access_token = "8bAQUk5x4GrEqT6-1xmIMClIRt2F6QGpWds_LY3kDGs"
        
        if not notion_auth_token or not notion_database_id:
            pytest.skip("Notion認証情報が設定されていません")
        
        syncer = NotionContextSync(notion_auth_token, notion_database_id, gyazo_access_token)
        results = syncer.sync_from_json(
            contexts_json,
            output_dir
        )
        
        # 同期結果の検証
        assert len(results) > 0, "Notionページが作成されていません"
        for result in results:
            assert 'id' in result, "NotionページIDが含まれていません"
            assert 'url' in result, "NotionページURLが含まれていません"
        
        print("\n=== 同期結果 ===")
        print(f"同期された文脈数: {len(results)}")
        for i, result in enumerate(results, 1):
            print(f"\n文脈{i}:")
            print(f"ページID: {result.get('id')}")
            print(f"URL: {result.get('url')}")
        
        print("\n統合テストが正常に完了しました!")
        
    except Exception as e:
        pytest.fail(f"統合テストでエラーが発生しました: {str(e)}")

if __name__ == "__main__":
    test_video_to_notion_integration()