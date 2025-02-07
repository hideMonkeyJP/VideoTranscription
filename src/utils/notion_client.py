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

    def create_page(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """データベースに新しいページを作成"""
        url = f"{self.base_url}/pages"
        
        # ページ作成用のペイロードを構築
        # 必要なプロパティのみを含むペイロードを作成
        filtered_properties = {
            k: v for k, v in properties.items()
            if k in ["Title", "Summary", "Keywords ", "Duration ", "ProcessedDate ", "Thumbnail "]
        }
        
        payload = {
            "parent": {
                "database_id": self.database_id
            },
            "properties": filtered_properties
        }
        
        response = requests.post(url, headers=self.headers, json=payload)
        response_json = response.json()
        
        if response.status_code == 200:
            # 必要なフィールドのみを抽出
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

    def update_page(self, page_id: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """既存のページを更新"""
        url = f"{self.base_url}/pages/{page_id}"
        
        payload = {
            "properties": properties
        }
        
        response = requests.patch(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def create_video_entry(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """動画データをNotionデータベースに登録"""
        properties = {
            "Title": {
                "title": [
                    {
                        "type": "text",
                        "text": {
                            "content": video_data.get("title", "Untitled Video")
                        }
                    }
                ]
            },
            "Summary": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": video_data.get("summary", "")[:2000]
                        }
                    }
                ]
            },
            "Keywords ": {
                "multi_select": [
                    {"name": keyword} for keyword in video_data.get("keywords", [])[:10]
                ]
            },
            "Duration ": {
                "number": float(video_data.get("duration", 0))
            },
            "ProcessedDate ": {
                "date": {
                    "start": datetime.now().isoformat()
                }
            }
        }

        # サムネイル画像のURLがある場合
        if "thumbnail_url" in video_data:
            properties["Thumbnail "] = {
                "files": [
                    {
                        "type": "external",
                        "name": "thumbnail.jpg",
                        "external": {
                            "url": video_data["thumbnail_url"]
                        }
                    }
                ]
            }

        try:
            return self.create_page(properties)
        except Exception as e:
            print(f"Error creating video entry: {str(e)}")
            raise

    def format_video_data(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """分析結果をNotionに適した形式に変換"""
        # 基本情報の取得
        metadata = analysis_result.get("metadata", {})
        title = metadata.get("title", "Untitled Video")
        duration = float(metadata.get("duration", 0))  # 数値型に変換

        # 要約の取得
        summary = ""
        if isinstance(analysis_result.get("summary"), dict):
            # 新しい形式: summaryオブジェクトから直接取得
            summary_points = analysis_result["summary"].get("key_scenes", [])
            summary = "\n".join(f"• {scene['summary']}" for scene in summary_points[:5])
        elif "scene_analyses" in analysis_result:
            # 従来の形式: scene_analysesから取得
            summaries = []
            for scene in analysis_result["scene_analyses"]:
                if scene.get("summary", {}).get("main_points"):
                    summaries.extend(scene["summary"]["main_points"])
            summary = "\n".join(f"• {point}" for point in summaries[:5])

        # キーワードの取得
        keywords = []
        if isinstance(analysis_result.get("summary"), dict):
            # 新しい形式: summaryオブジェクトから直接取得
            keywords = analysis_result["summary"].get("keywords", [])
        elif "scene_analyses" in analysis_result:
            # 従来の形式: scene_analysesから集約
            keyword_set = set()
            for scene in analysis_result["scene_analyses"]:
                keyword_set.update(scene.get("keywords", []))
            keywords = list(keyword_set)

        # durationが0の場合は最後のシーンのタイムスタンプを使用
        if duration == 0 and "scene_analyses" in analysis_result and analysis_result["scene_analyses"]:
            last_scene = analysis_result["scene_analyses"][-1]
            duration = float(last_scene.get("timestamp", 0))

        # テスト用のデータ形式に対応
        if isinstance(analysis_result, dict) and "duration" in analysis_result:
            duration = float(analysis_result["duration"])

        return {
            "title": title,
            "summary": summary,
            "keywords": keywords[:10],  # 最大10個のキーワード
            "duration": duration,
            "thumbnail_url": metadata.get("thumbnail_url", "")
        }

    def sync_analysis_results(self, analysis_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """複数の分析結果をNotionと同期"""
        synced_results = []
        for result in analysis_results:
            formatted_data = self.format_video_data(result)
            notion_entry = self.create_video_entry(formatted_data)
            synced_results.append(notion_entry)
        return synced_results