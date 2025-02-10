import os
import json
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional

class NotionClient:
    def __init__(self, auth_token: str, database_id: str):
        """NotionClientの初期化"""
        self.auth_token = auth_token
        self.database_id = database_id
        self.base_url = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }
        # データベースのプロパティを取得
        self.properties = self.get_database_properties()

    def get_database_properties(self) -> Dict[str, Any]:
        """データベースのプロパティ情報を取得"""
        url = f"{self.base_url}/databases/{self.database_id}"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            database = response.json()
            return database.get("properties", {})
        else:
            print(f"Error getting database: {response.json()}")
            return {}

    def create_page(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """データベースに新しいページを作成"""
        url = f"{self.base_url}/pages"
        
        payload = {
            "parent": {
                "database_id": self.database_id
            },
            "properties": properties
        }
        
        response = requests.post(url, headers=self.headers, json=payload)
        response_json = response.json()
        
        if response.status_code in [200, 201]:  # 200と201の両方を正常なレスポンスとして扱う
            return {
                "id": response_json.get("id"),
                "url": response_json.get("url"),
                "properties": response_json.get("properties", {})
            }
        else:
            error_message = response_json
            print(f"Error creating page: {error_message}")
            raise requests.exceptions.HTTPError(
                f"{response.status_code} {response.reason} for url: {url}",
                response=response
            )

    def format_video_data(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """分析結果をNotion用にフォーマット"""
        metadata = analysis_result.get("metadata", {})
        contexts = analysis_result.get("contexts", [])
        
        # 文脈ごとの要約を構築
        context_summaries = []
        for i, context in enumerate(contexts, 1):
            time_range = context.get('time_range', {})
            start_time = time_range.get('start', 0)
            end_time = time_range.get('end', 0)
            
            # 重要シーンを選択
            scenes = context.get('scenes', [])
            important_scenes = sorted(
                scenes,
                key=lambda x: x.get('importance_score', 0),
                reverse=True
            )[:5]
            
            # 要約を生成
            summary_lines = [
                f"### {start_time:.1f}秒 - {end_time:.1f}秒",
                f"要約: {context.get('summary', '')}",
                "重要シーン:"
            ]
            
            for scene in important_scenes[:2]:
                summary_lines.append(
                    f"- {scene.get('timestamp', 0):.1f}秒: {scene.get('summary', {}).get('main_points', [''])[0]}"
                )
            
            context_summaries.append("\n".join(summary_lines))
        
        # キーワードの収集と重要度によるソート
        keyword_scores = {}
        for context in contexts:
            for scene in context.get('scenes', []):
                for keyword in scene.get('keywords', []):
                    keyword_scores[keyword] = keyword_scores.get(keyword, 0) + scene.get('importance_score', 0)
        
        sorted_keywords = sorted(
            keyword_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "title": f"動画分析レポート {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "summary": "\n\n".join(context_summaries),
            "keywords": [kw for kw, _ in sorted_keywords],
            "duration": metadata.get("total_duration", 0),
            "processed_at": datetime.now().isoformat()
        }

    def print_database_schema(self):
        """データベースのスキーマ情報を表示"""
        print("\n=== Notionデータベースのプロパティ ===")
        for name, prop in self.properties.items():
            print(f"{name}: {prop.get('type')}")