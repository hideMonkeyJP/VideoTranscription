import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, List
import time

from video_processor import VideoProcessor
from utils.config import Config
from utils.exceptions import VideoProcessorError, ConfigurationError

# 中間ファイルのパス定義
PATHS = {
    'frames': Path('temp/frames.json'),
    'ocr': Path('temp/ocr_results.json'),
    'transcription': Path('temp/transcription.json'),
    'screenshots': Path('screenshots'),
    'audio': Path('audio'),
    'analysis': Path('temp/analysis.json'),
    'notion_data': Path('regist.json')
}

def setup_logging(output_dir: Path) -> logging.Logger:
    """ロギングの設定を行います"""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'video_processing.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s - (%(filename)s:%(lineno)d)',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

def validate_json_file(path: Path, required_keys: List[str]) -> bool:
    """JSONファイルの妥当性確認"""
    try:
        if not path.exists():
            return False
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return all(all(key in item for key in required_keys) for item in data if item)
            return all(key in data for key in required_keys)
    except Exception:
        return False

def ensure_dir(path: Path) -> Path:
    """ディレクトリの存在確認と作成"""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def check_intermediate_files(output_dir: Path, logger: logging.Logger) -> Dict[str, bool]:
    """中間ファイルの存在と有効性を確認"""
    status = {}
    required_keys = {
        'frames': ['timestamp', 'frame_number', 'scene_change_score', 'path'],
        'ocr': ['screenshots'],
        'transcription': ['text', 'start', 'end'],
        'analysis': ['segments', 'total_segments']
    }
    
    for key, path in PATHS.items():
        full_path = output_dir / path
        if key in required_keys:
            status[key] = validate_json_file(full_path, required_keys[key])
            logger.info(f"中間ファイル {key}: {'有効' if status[key] else '無効または未存在'}")
        else:
            status[key] = full_path.exists()
            logger.info(f"ディレクトリ {key}: {'存在' if status[key] else '未存在'}")
    
    return status

def main():
    start_time = time.time()
    
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='動画から文字起こしと要約を生成します')
    parser.add_argument('video_path', help='処理する動画ファイルのパス')
    parser.add_argument('--output', '-o', default='output', help='出力ディレクトリ')
    parser.add_argument('--config', '-c', default='config/config.yaml', help='設定ファイルのパス')
    parser.add_argument('--force', '-f', action='store_true', help='中間ファイルを再生成')
    args = parser.parse_args()

    try:
        # 出力ディレクトリの設定
        output_dir = Path(args.output)
        logger = setup_logging(output_dir)
        logger.info(f"処理を開始します - 出力ディレクトリ: {output_dir}")

        # 入力ファイルの検証
        video_path = Path(args.video_path)
        if not video_path.exists():
            logger.error(f"エラー: 指定された動画ファイル '{video_path}' が見つかりません。")
            sys.exit(1)
        
        # 設定ファイルの検証
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"エラー: 設定ファイル '{config_path}' が見つかりません。")
            sys.exit(1)

        # 出力ディレクトリの作成
        for path in PATHS.values():
            ensure_dir(output_dir / path)
        logger.info("出力ディレクトリを作成しました")

        # 中間ファイルの確認
        if not args.force:
            file_status = check_intermediate_files(output_dir, logger)
            logger.info("中間ファイルの確認が完了しました")
        else:
            logger.info("--force オプションが指定されたため、中間ファイルを再生成します")
            file_status = {k: False for k in PATHS.keys()}

        # 設定の読み込みと更新
        config = Config(args.config)
        config_data = config.get_all()
        config_data.update({
            'video_processor': {
                'output_dir': str(output_dir),
                'temp_dir': str(output_dir / 'temp'),
                'reuse_intermediate_files': not args.force
            }
        })

        # VideoProcessorの初期化
        logger.info("VideoProcessorを初期化します...")
        processor = VideoProcessor(config=config_data)

        # 動画の処理
        logger.info(f"動画の処理を開始します: {video_path}")
        result = processor.process_video(str(video_path), str(output_dir))

        # 処理結果の出力
        logger.info("\n処理が完了しました")
        logger.info("\n出力ファイル:")
        for key, path in result['output_files'].items():
            logger.info(f"- {key}: {path}")

        # パフォーマンス情報の出力
        end_time = time.time()
        duration = end_time - start_time
        logger.info("\nパフォーマンス情報:")
        logger.info(f"- 総処理時間: {duration:.2f}秒")
        if result.get('performance'):
            for key, value in result['performance'].items():
                logger.info(f"- {key}: {value:.2f}秒")

        # 中間ファイルのサイズ情報
        logger.info("\n中間ファイルサイズ:")
        for key, path in PATHS.items():
            full_path = output_dir / path
            if full_path.is_file():
                size = full_path.stat().st_size / (1024 * 1024)  # MB単位
                logger.info(f"- {key}: {size:.2f}MB")

    except VideoProcessorError as e:
        logger.error(f"処理エラー: {str(e)}")
        if hasattr(e, 'context'):
            logger.error(f"エラーコンテキスト: {e.context}")
        sys.exit(1)
    except ConfigurationError as e:
        logger.error(f"設定エラー: {str(e)}")
        if hasattr(e, 'context'):
            logger.error(f"エラーコンテキスト: {e.context}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 