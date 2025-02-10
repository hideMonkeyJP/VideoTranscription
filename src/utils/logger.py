"""
ロギング設定とユーティリティ
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

class CustomLogger:
    """カスタムロガークラス"""
    
    _instances = {}
    
    def __new__(cls, name: str, *args, **kwargs):
        if name not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[name] = instance
            return instance
        return cls._instances[name]
    
    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        log_level: int = logging.INFO,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        if hasattr(self, 'logger'):
            return
            
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # 既存のハンドラをクリア
        self.logger.handlers = []
        
        # ログディレクトリの作成
        os.makedirs(log_dir, exist_ok=True)
        
        # ファイル名の設定
        log_file = os.path.join(log_dir, f"{name}.log")
        
        # ファイルハンドラの設定
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        
        # コンソールハンドラの設定
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # フォーマッタの設定
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # ハンドラの追加
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str) -> None:
        """
        INFO レベルのログを記録
        
        Args:
            message: ログメッセージ
        """
        self.logger.info(message)
    
    def error(self, message: str, exc_info: Optional[Exception] = None) -> None:
        """
        ERROR レベルのログを記録
        
        Args:
            message: ログメッセージ
            exc_info: 例外情報
        """
        self.logger.error(message, exc_info=exc_info)
    
    def warning(self, message: str) -> None:
        """
        WARNING レベルのログを記録
        
        Args:
            message: ログメッセージ
        """
        self.logger.warning(message)
    
    def debug(self, message: str) -> None:
        """
        DEBUG レベルのログを記録
        
        Args:
            message: ログメッセージ
        """
        self.logger.debug(message)
    
    def critical(self, message: str, exc_info: Optional[Exception] = None) -> None:
        """
        CRITICAL レベルのログを記録
        
        Args:
            message: ログメッセージ
            exc_info: 例外情報
        """
        self.logger.critical(message, exc_info=exc_info)

def get_logger(
    name: str,
    log_dir: str = "logs",
    log_level: int = logging.INFO
) -> CustomLogger:
    """
    ロガーインスタンスを取得
    
    Args:
        name: ロガー名
        log_dir: ログディレクトリ
        log_level: ログレベル
    
    Returns:
        CustomLogger: カスタムロガーインスタンス
    """
    return CustomLogger(name, log_dir, log_level)

class VideoProcessorLogger:
    """VideoProcessor専用のロガークラス"""
    
    def __init__(self, config_path=None):
        """
        Args:
            config_path: 設定ファイルのパス
        """
        self.logger = get_logger("video_processor")
        self.config_path = config_path
    
    def log_error(self, error: Exception, context: dict = None):
        """エラーログを記録"""
        error_message = f"エラー発生: {str(error)}"
        if context:
            error_message += f"\nコンテキスト: {context}"
        self.logger.error(error_message, exc_info=error)
    
    def log_warning(self, message: str, context: dict = None):
        """警告ログを記録"""
        warning_message = message
        if context:
            warning_message += f"\nコンテキスト: {context}"
        self.logger.warning(warning_message)

# デフォルトロガーの設定
default_logger = get_logger("video_processor")