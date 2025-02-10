import sys
import os
import argparse
import os
import sys
import argparse

# srcディレクトリをPythonパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.video_processor import VideoProcessor
from utils.exceptions import VideoProcessorError, ConfigurationError
import logging

def setup_logging():
    """ロギングの設定を行います"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('video_processing.log', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='動画から文字起こしと要約を生成します')
    parser.add_argument('video_path', help='処理する動画ファイルのパス')
    parser.add_argument('--output', '-o', default='output', help='出力ディレクトリ')
    parser.add_argument('--config', '-c', help='設定ファイルのパス')
    args = parser.parse_args()

    try:
        # 入力ファイルの検証
        if not os.path.exists(args.video_path):
            logger.error(f"エラー: 指定された動画ファイル '{args.video_path}' が見つかりません。")
            sys.exit(1)

        # 出力ディレクトリの作成
        os.makedirs(args.output, exist_ok=True)
        logger.info(f"出力ディレクトリを作成/確認しました: {args.output}")

        # VideoProcessorの初期化
        logger.info("VideoProcessorを初期化します...")
        processor = VideoProcessor(config_path=args.config)

        # 動画の処理
        logger.info(f"動画の処理を開始します: {args.video_path}")
        result = processor.process_video(args.video_path, args.output)

        # 処理結果の出力
        logger.info("処理が完了しました。")
        logger.info(f"結果ファイル:")
        logger.info(f"- JSON: {result['json_path']}")
        logger.info(f"- スクリーンショット: {result['screenshots_dir']}")

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