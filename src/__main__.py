import os
import sys
import argparse
from pathlib import Path
from typing import Optional

from .utils import Config, Logger
from .video_processor import VideoProcessor

def parse_args():
    """コマンドライン引数をパースします"""
    parser = argparse.ArgumentParser(
        description="動画から文字起こしと分析を行い、構造化されたレポートを生成します。"
    )
    
    parser.add_argument(
        "video_path",
        type=str,
        help="処理する動画ファイルのパス"
    )
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="設定ファイルのパス"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        help="出力ディレクトリのパス"
    )
    
    parser.add_argument(
        "--frame-interval",
        type=float,
        help="フレーム抽出間隔（秒）"
    )
    
    parser.add_argument(
        "--min-scene-change",
        type=float,
        help="シーン変更の閾値（0-1）"
    )
    
    parser.add_argument(
        "--min-text-quality",
        type=float,
        help="テキスト品質の閾値（0-1）"
    )
    
    parser.add_argument(
        "--whisper-model",
        type=str,
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisperモデルの種類"
    )
    
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="GPUを使用する"
    )
    
    return parser.parse_args()

def main(args: Optional[argparse.Namespace] = None):
    """メイン処理を実行します"""
    if args is None:
        args = parse_args()
    
    try:
        # 設定の読み込み
        config = Config(args.config if args.config else None)
        
        # コマンドライン引数による設定の上書き
        if args.output_dir:
            config.config['video_processor']['output_dir'] = args.output_dir
        if args.frame_interval:
            config.config['frame_extractor']['frame_interval'] = args.frame_interval
        if args.min_scene_change:
            config.config['frame_extractor']['min_scene_change'] = args.min_scene_change
        if args.min_text_quality:
            config.config['ocr_processor']['min_confidence'] = args.min_text_quality
        if args.whisper_model:
            config.config['audio_extractor']['model_name'] = args.whisper_model
        if args.use_gpu:
            config.config['performance']['use_gpu'] = True
        
        # 設定の検証
        config.validate()
        
        # ロガーの初期化
        logger = Logger(config)
        logger.info("処理を開始します")
        
        # 動画パスの検証
        video_path = Path(args.video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")
        
        # VideoProcessorの初期化と実行
        processor = VideoProcessor(config.get_all())
        result = processor.process_video(str(video_path))
        
        # 結果の出力
        logger.info(f"処理が完了しました: {result['report_path']}")
        return 0
        
    except Exception as e:
        logger.exception(f"エラーが発生しました: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 