import os
import json
from datetime import datetime
from typing import Dict, List, Any

class VideoContextDataGenerator:
    """動画文脈データの生成を担当するクラス"""
    
    def __init__(self, json_path: str):
        """
        Args:
            json_path (str): 入力JSONファイルのパス
        """
        self.json_path = json_path
        
    def generate_context_data(self) -> Dict[str, Any]:
        """文脈データを生成する

        Returns:
            Dict[str, Any]: 生成されたデータ
        """
        # JSONファイルを読み込む
        with open(self.json_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        # 元のタイムスタンプを保持
        timestamp = result.get('timestamp', '')
        metadata = result.get('metadata', {})
        
        # 各文脈のデータを生成
        contexts = []
        for i, context in enumerate(result.get('contexts', []), 1):
            # スクリーンショット情報を取得
            screenshots = context.get('screenshots', [])
            if screenshots:
                last_screenshot = screenshots[-1]
                screenshot_index = (i - 1) * 20  # 各文脈20枚のスクリーンショット
                screenshot_filename = f"screenshot_{screenshot_index:03d}.png"
                screenshot_path = f"screenshots_{timestamp}/{screenshot_filename}"
                screenshot_data = {
                    "filename": screenshot_filename,
                    "path": screenshot_path,
                    "timestamp": last_screenshot['timestamp']
                }
            else:
                screenshot_data = None
            
            # キーワードを収集(各シーンから最大3つずつ)
            keywords = []
            for scene in context.get('scenes', []):
                scene_keywords = scene.get('keywords', [])[:3]  # 各シーンから最大3つ
                keywords.extend(scene_keywords)
            # 重複を除去して最大10個に制限
            unique_keywords = list(dict.fromkeys(keywords))[:10]
            
            # 文脈データを構築
            context_data = {
                "id": i,
                "title": f"文脈{i}: {context['time_range']['start']:.1f}秒 - {context['time_range']['end']:.1f}秒",
                "summary": context['summary'],
                "timestamp": f"{context['time_range']['start']:.1f}秒 - {context['time_range']['end']:.1f}秒",
                "keywords": unique_keywords,
                "screenshot": screenshot_data,
                "time_range": context['time_range'],
                "scenes": context.get('scenes', [])  # シーン情報も保持
            }
            
            contexts.append(context_data)
        
        return {
            "timestamp": timestamp,  # 元のタイムスタンプを使用
            "metadata": metadata,    # メタデータを保持
            "contexts": contexts
        }
    
    def save_to_json(self, output_path: str) -> None:
        """生成したデータをJSONファイルとして保存する

        Args:
            output_path (str): 出力JSONファイルのパス
        """
        data = self.generate_context_data()
        
        # 出力ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # JSONファイルとして保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 使用例
    generator = VideoContextDataGenerator("output_test/json_test/final_result.json")
    generator.save_to_json("output_test/notion_test/contexts_data.json")