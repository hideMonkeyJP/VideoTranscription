import logging
import logging.handlers
import os
import yaml
from datetime import datetime
from typing import Optional, Dict, Any

class VideoProcessorLogger:
    def __init__(self, config_path: str = None):
        self.logger = logging.getLogger('VideoProcessor')
        if config_path is None:
            # プロジェクトのルートディレクトリを取得
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = os.path.join(root_dir, 'config', 'config.yaml')
        self.setup_logger(config_path)

    def setup_logger(self, config_path: str) -> None:
        """ロガーの設定を行います"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            log_config = config.get('logging', {})
            log_level = getattr(logging, log_config.get('level', 'INFO'))
            max_files = log_config.get('max_files', 7)
            
            # ログディレクトリの作成
            log_dir = 'logs'
            os.makedirs(log_dir, exist_ok=True)
            
            # ログフォーマットの設定
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # ファイルハンドラの設定
            log_file = os.path.join(log_dir, 'video_processor.log')
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=max_files,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            
            # コンソールハンドラの設定
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            
            # ロガーの設定
            self.logger.setLevel(log_level)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
        except Exception as e:
            print(f"ロガーの設定中にエラーが発生しました: {str(e)}")
            raise

    def log_process_start(self, video_path: str) -> None:
        """動画処理開始時のログを記録"""
        self.logger.info(f"動画処理を開始します: {video_path}")

    def log_process_complete(self, video_path: str, stats: Dict[str, Any]) -> None:
        """動画処理完了時のログを記録"""
        self.logger.info(
            f"動画処理が完了しました: {video_path}\n"
            f"統計情報:\n"
            f"- 処理時間: {stats.get('processing_time', 'N/A')}秒\n"
            f"- 抽出フレーム数: {stats.get('frame_count', 0)}\n"
            f"- 音声セグメント数: {stats.get('segment_count', 0)}"
        )

    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """エラー情報をログに記録"""
        error_msg = f"エラーが発生しました: {str(error)}"
        if context:
            error_msg += f"\nコンテキスト: {context}"
        self.logger.error(error_msg, exc_info=True)

    def log_warning(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """警告情報をログに記録"""
        warning_msg = message
        if context:
            warning_msg += f"\nコンテキスト: {context}"
        self.logger.warning(warning_msg)

    def log_frame_extraction(self, frame_count: int, duration: float) -> None:
        """フレーム抽出情報をログに記録"""
        self.logger.info(
            f"フレーム抽出完了:\n"
            f"- 抽出フレーム数: {frame_count}\n"
            f"- 動画長: {duration:.2f}秒\n"
            f"- 平均フレームレート: {frame_count/duration:.2f}fps"
        )

    def log_transcription(self, segment_count: int, total_duration: float) -> None:
        """音声認識情報をログに記録"""
        self.logger.info(
            f"音声認識完了:\n"
            f"- セグメント数: {segment_count}\n"
            f"- 総時間: {total_duration:.2f}秒"
        )

    def log_performance_stats(self, stats: Dict[str, Any]) -> None:
        """パフォーマンス統計情報をログに記録"""
        self.logger.info(
            f"パフォーマンス統計:\n"
            f"- メモリ使用量: {stats.get('memory_usage', 'N/A')}MB\n"
            f"- CPU使用率: {stats.get('cpu_usage', 'N/A')}%\n"
            f"- 処理時間: {stats.get('processing_time', 'N/A')}秒"
        )

    def log_step_start(self, step_name: str) -> None:
        """処理ステップの開始をログに記録"""
        self.logger.info(f"ステップ開始: {step_name}")

    def log_step_complete(self, message: str) -> None:
        """処理ステップの完了をログに記録"""
        self.logger.info(f"ステップ完了: {message}") 