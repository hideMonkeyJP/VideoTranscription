"""
Supabaseへのデータ登録を行うユーティリティモジュール
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union
import json
from ..database.supabase_client import SupabaseClient

def register_to_supabase(
    data: Union[str, Dict[str, Any], Path],
    table_name: str,
    **kwargs
) -> bool:
    """Supabaseにデータを登録する

    Args:
        data: 以下のいずれかを指定:
            - JSONファイルのパス（str or Path）
            - 登録するデータの辞書（Dict）
        table_name: 登録先のテーブル名
        **kwargs: テーブルに応じた追加のパラメータ
            例：videos テーブルの場合
            - title: 動画のタイトル
            - file_path: 動画ファイルのパス
            - duration: 動画の長さ（秒）

    Returns:
        bool: 登録が成功したかどうか
    """
    try:
        # データの準備
        if isinstance(data, (str, Path)):
            data_path = Path(data).resolve()
            if not data_path.exists():
                print(f"Error: ファイルが見つかりません: {data_path}")
                return False
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: 不正なJSONファイルです: {data_path}")
                return False
        else:
            json_data = data

        # ファイルパスの検証（指定されている場合）
        if 'file_path' in kwargs:
            file_path = Path(kwargs['file_path']).resolve()
            if not file_path.exists():
                print(f"Error: ファイルが見つかりません: {file_path}")
                return False
            kwargs['file_path'] = str(file_path)

        # Supabaseクライアントの初期化
        client = SupabaseClient()

        # テーブルに応じたデータ登録処理
        if table_name == 'videos':
            if not all(k in kwargs for k in ['title', 'file_path', 'duration']):
                print("Error: videos テーブルには title, file_path, duration が必要です")
                return False

            # 動画情報の登録
            video_id = client.insert_video(
                kwargs['title'],
                kwargs['file_path'],
                kwargs['duration']
            )
            if not video_id:
                return False

            # セグメント情報の登録
            if isinstance(json_data, list):
                print(f"セグメント情報を登録します: {len(json_data)}件")
                return client.insert_segments(video_id, json_data)
            else:
                print("Error: セグメント情報が不正な形式です")
                return False

        else:
            # 汎用的なデータ登録
            try:
                result = client.client.table(table_name).insert(json_data).execute()
                return bool(result.data)
            except Exception as e:
                print(f"Error: データの登録に失敗: {str(e)}")
                return False

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        return False

# コマンドライン実行用
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='データをSupabaseに登録します')
    parser.add_argument('data', help='登録するデータのJSONファイルパス')
    parser.add_argument('table', help='登録先のテーブル名')
    parser.add_argument('--title', help='動画のタイトル（videosテーブルの場合）')
    parser.add_argument('--file-path', help='動画ファイルのパス（videosテーブルの場合）')
    parser.add_argument('--duration', type=int, help='動画の長さ（秒）（videosテーブルの場合）')
    
    args = parser.parse_args()
    
    # 引数の検証
    if args.table == 'videos':
        if not all([args.title, args.file_path, args.duration]):
            print("Error: videos テーブルには --title, --file-path, --duration が必要です")
            exit(1)
    
    # キーワード引数の準備
    kwargs = {}
    if args.title:
        kwargs['title'] = args.title
    if args.file_path:
        kwargs['file_path'] = args.file_path
    if args.duration is not None:  # 0も有効な値として扱う
        kwargs['duration'] = args.duration
    
    # データの登録
    success = register_to_supabase(args.data, args.table, **kwargs)
    
    # 結果に応じて終了コードを設定
    exit(0 if success else 1) 