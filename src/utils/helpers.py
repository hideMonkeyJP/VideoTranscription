import os
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

def create_directory(path: Union[str, Path]) -> Path:
    """
    ディレクトリを作成します。
    
    Args:
        path (Union[str, Path]): 作成するディレクトリのパス
        
    Returns:
        Path: 作成されたディレクトリのパス
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def generate_file_hash(file_path: Union[str, Path]) -> str:
    """
    ファイルのハッシュ値を生成します。
    
    Args:
        file_path (Union[str, Path]): ハッシュを生成するファイルのパス
        
    Returns:
        str: SHA-256ハッシュ値
    """
    file_path = Path(file_path)
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()

def format_timestamp(seconds: float) -> str:
    """
    秒数を時:分:秒形式にフォーマットします。
    
    Args:
        seconds (float): フォーマットする秒数
        
    Returns:
        str: フォーマットされた時間文字列
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def clean_filename(filename: str) -> str:
    """
    ファイル名から無効な文字を除去します。
    
    Args:
        filename (str): クリーニングするファイル名
        
    Returns:
        str: クリーニングされたファイル名
    """
    # 無効な文字を除去
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # 先頭と末尾の空白を除去
    filename = filename.strip()
    # 連続する空白を1つに
    filename = re.sub(r'\s+', ' ', filename)
    
    return filename

def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    JSONファイルを読み込みます。
    
    Args:
        file_path (Union[str, Path]): 読み込むJSONファイルのパス
        
    Returns:
        Dict[str, Any]: 読み込まれたJSONデータ
    """
    file_path = Path(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 4):
    """
    データをJSONファイルとして保存します。
    
    Args:
        data (Dict[str, Any]): 保存するデータ
        file_path (Union[str, Path]): 保存先のパス
        indent (int, optional): インデントのスペース数
    """
    file_path = Path(file_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    ファイルの情報を取得します。
    
    Args:
        file_path (Union[str, Path]): 情報を取得するファイルのパス
        
    Returns:
        Dict[str, Any]: ファイル情報
    """
    file_path = Path(file_path)
    stats = file_path.stat()
    
    return {
        'name': file_path.name,
        'extension': file_path.suffix,
        'size': stats.st_size,
        'created_at': datetime.fromtimestamp(stats.st_ctime),
        'modified_at': datetime.fromtimestamp(stats.st_mtime),
        'hash': generate_file_hash(file_path)
    }

def find_files(directory: Union[str, Path], pattern: str) -> List[Path]:
    """
    指定したパターンに一致するファイルを検索します。
    
    Args:
        directory (Union[str, Path]): 検索するディレクトリ
        pattern (str): 検索パターン（glob形式）
        
    Returns:
        List[Path]: 一致したファイルのパスのリスト
    """
    directory = Path(directory)
    return list(directory.glob(pattern))

def ensure_suffix(path: Union[str, Path], suffix: str) -> Path:
    """
    パスに指定した拡張子が付いていない場合は追加します。
    
    Args:
        path (Union[str, Path]): 対象のパス
        suffix (str): 追加する拡張子
        
    Returns:
        Path: 拡張子が追加されたパス
    """
    path = Path(path)
    if not suffix.startswith('.'):
        suffix = '.' + suffix
    
    if path.suffix != suffix:
        path = path.with_suffix(suffix)
    
    return path

def get_unique_path(path: Union[str, Path]) -> Path:
    """
    重複しないファイルパスを生成します。
    
    Args:
        path (Union[str, Path]): 元のパス
        
    Returns:
        Path: 重複しないパス
    """
    path = Path(path)
    if not path.exists():
        return path
    
    counter = 1
    while True:
        new_path = path.with_stem(f"{path.stem}_{counter}")
        if not new_path.exists():
            return new_path
        counter += 1

def cleanup_directory(directory: Union[str, Path], pattern: str = "*"):
    """
    ディレクトリ内のファイルを削除します。
    
    Args:
        directory (Union[str, Path]): クリーンアップするディレクトリ
        pattern (str, optional): 削除対象のパターン
    """
    directory = Path(directory)
    for file in directory.glob(pattern):
        if file.is_file():
            file.unlink()

def get_relative_path(path: Union[str, Path], base: Union[str, Path]) -> Path:
    """
    ベースパスからの相対パスを取得します。
    
    Args:
        path (Union[str, Path]): 対象のパス
        base (Union[str, Path]): ベースパス
        
    Returns:
        Path: 相対パス
    """
    return Path(path).relative_to(Path(base))

def is_video_file(file_path: Union[str, Path]) -> bool:
    """
    ファイルが動画ファイルかどうかを判定します。
    
    Args:
        file_path (Union[str, Path]): 判定するファイルのパス
        
    Returns:
        bool: 動画ファイルの場合はTrue
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv'}
    return Path(file_path).suffix.lower() in video_extensions

def is_image_file(file_path: Union[str, Path]) -> bool:
    """
    ファイルが画像ファイルかどうかを判定します。
    
    Args:
        file_path (Union[str, Path]): 判定するファイルのパス
        
    Returns:
        bool: 画像ファイルの場合はTrue
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    return Path(file_path).suffix.lower() in image_extensions 