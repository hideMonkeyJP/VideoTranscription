"""
Supabaseとの連携を行うモジュール
"""

import os
from pathlib import Path
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from supabase import create_client, Client
import urllib3

# SSL証明書の警告を無効化（開発環境用）
urllib3.disable_warnings()

class SupabaseClient:
    """Supabaseとの連携を行うクラス"""
    
    def __init__(self):
        """SupabaseClientの初期化"""
        load_dotenv()
        
        # Supabase認証情報の取得
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        
        if not self.url or not self.key:
            raise ValueError("Supabase URL or key is not set in .env file")
            
        print(f"Connecting to Supabase at: {self.url}")
        
        try:
            self.client = create_client(self.url, self.key)
            # 認証ヘッダーの設定
            self.client.postgrest.auth(self.key)
        except Exception as e:
            raise ConnectionError(f"Supabaseクライアントの初期化に失敗: {str(e)}")
        
    def test_connection(self) -> bool:
        """Supabaseとの接続をテスト

        Returns:
            bool: 接続が成功したかどうか
        """
        try:
            # テーブルの存在を確認（シンプルなクエリに変更）
            self.client.table('videos').select("id").limit(1).execute()
            print("Connection test successful")
            return True
        except Exception as e:
            print(f"Connection test failed: {str(e)}")
            return False

    def insert_video(self, title: str, file_path: str, duration: int) -> Optional[str]:
        """動画情報をvideosテーブルに挿入

        Args:
            title (str): 動画のタイトル
            file_path (str): 動画ファイルのパス
            duration (int): 動画の長さ（秒）

        Returns:
            Optional[str]: 挿入された動画のID（失敗時はNone）
        """
        try:
            # videosテーブルにデータを挿入
            result = self.client.table('videos').insert({
                'title': title,
                'file_path': file_path,
                'duration': duration
            }).execute()
            
            # 挿入されたレコードのIDを返す
            if result.data and len(result.data) > 0:
                video_id = result.data[0]['id']
                print(f"動画情報を登録しました: {video_id}")
                return video_id
            return None
            
        except Exception as e:
            print(f"Failed to insert video: {str(e)}")
            return None
            
    def insert_segments(self, video_id: str, segments: List[Dict[str, Any]]) -> bool:
        """セグメント情報をsegmentsテーブルに挿入

        Args:
            video_id (str): 動画のID
            segments (List[Dict[str, Any]]): セグメント情報のリスト

        Returns:
            bool: 挿入が成功したかどうか
        """
        try:
            print(f"\n入力セグメント数: {len(segments)}")
            # セグメントデータの整形
            formatted_segments = []
            for i, segment in enumerate(segments, 1):
                try:
                    print(f"\nセグメント {i} の処理:")
                    print(f"  - 入力データ: {segment}")
                    
                    # タイムスタンプの分割
                    time_parts = segment['Timestamp'].replace('秒', '').split(' - ')
                    print(f"  - タイムスタンプ分割: {time_parts}")
                    
                    start_time = float(time_parts[0])
                    end_time = float(time_parts[1])
                    print(f"  - 変換後の時間: start={start_time}, end={end_time}")
                    
                    # レコードの作成
                    record = {
                        'video_id': video_id,
                        'segment_no': segment['No'],
                        'summary': segment['Summary'],
                        'start_time': start_time,
                        'end_time': end_time,
                        'thumbnail_url': segment['Thumbnail']
                    }
                    print(f"  - 作成されたレコード: {record}")
                    formatted_segments.append(record)
                    
                except Exception as e:
                    print(f"  - エラー: セグメント {i} の処理中にエラーが発生: {str(e)}")
                    raise
                
            print(f"\n変換後のセグメント数: {len(formatted_segments)}")
            result = self.client.table('segments').insert(formatted_segments).execute()
            
            if result.data:
                print(f"セグメント情報を登録しました: {len(result.data)}件")
                return True
                
            print("セグメントの登録に失敗: result.dataが空です")
            return False
            
        except Exception as e:
            print(f"セグメントの登録中にエラーが発生: {str(e)}")
            return False
            
    def process_regist_json(self, json_path: str, video_title: str, video_path: str, duration: int) -> bool:
        """regist.jsonの内容をデータベースに登録

        Args:
            json_path (str): regist.jsonのパス
            video_title (str): 動画のタイトル
            video_path (str): 動画ファイルのパス
            duration (int): 動画の長さ（秒）

        Returns:
            bool: 処理が成功したかどうか
        """
        try:
            # 接続テスト
            if not self.test_connection():
                return False

            # JSONファイルの読み込み
            with open(json_path, 'r', encoding='utf-8') as f:
                segments_data = json.load(f)
            
            print(f"JSONファイルを読み込みました: {len(segments_data)}件のセグメント")
            
            # 動画情報の挿入
            video_id = self.insert_video(video_title, video_path, duration)
            if not video_id:
                return False
            
            # セグメント情報の挿入
            return self.insert_segments(video_id, segments_data)
            
        except json.JSONDecodeError:
            print(f"Error: 不正なJSONファイルです: {json_path}")
            return False
        except Exception as e:
            print(f"Failed to process regist.json: {str(e)}")
            return False 