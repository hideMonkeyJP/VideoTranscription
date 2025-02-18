"""
ロギング設定とユーティリティ
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any
from pathlib import Path

class Logger:
    """ロガークラス"""
    
    _instances = {}
    
    def __new__(cls, config: 'Config', name: str, *args, **kwargs):
        if name not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[name] = instance
            return instance
        return cls._instances[name]
    
    def __init__(self, config: 'Config', name: str):
        """初期化
        
        Args:
            config (Config): 設定オブジェクト
            name (str): ロガー名
        """
        self.name = name
        self.logger = logging.getLogger(name)
        
        # 設定から各種パラメータを取得
        log_config = config.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_dir = Path(log_config.get('log_dir', 'logs'))
        log_file = log_dir / 'video_processing.log'
        
        # ログレベルの設定
        self.logger.setLevel(log_level)
        
        # 既存のハンドラをクリア
        self.logger.handlers = []
        
        # フォーマッタの作成
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # ディレクトリの作成
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイルハンドラの設定
        file_handler = RotatingFileHandler(
            str(log_file),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # コンソールハンドラの設定
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    @classmethod
    def get_logger(cls, name: str, config: 'Config') -> 'Logger':
        """ロガーインスタンスを取得
        
        Args:
            name (str): ロガー名
            config (Config): 設定オブジェクト
        
        Returns:
            Logger: ロガーインスタンス
        """
        return cls(config, name)
    
    def debug(self, message: str):
        """デバッグログを出力"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """情報ログを出力"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """警告ログを出力"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """エラーログを出力"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """重大エラーログを出力"""
        self.logger.critical(message)
    
    def exception(self, message: str):
        """例外ログを出力（スタックトレース付き）"""
        self.logger.exception(message)

class VideoProcessorLogger:
    """VideoProcessor専用のロガークラス"""
    
    def __init__(self, config: 'Config'):
        """初期化
        
        Args:
            config (Config): 設定オブジェクト
        """
        self.logger = Logger.get_logger("video_processor", config)
    
    def log_error(self, error: Exception, context: dict = None):
        """エラーログを記録
        
        Args:
            error (Exception): 発生したエラー
            context (dict, optional): エラーのコンテキスト
        """
        error_message = f"エラー発生: {str(error)}"
        if context:
            error_message += f"\nコンテキスト: {context}"
        self.logger.exception(error_message)
    
    def log_warning(self, message: str, context: dict = None):
        """警告ログを記録
        
        Args:
            message (str): 警告メッセージ
            context (dict, optional): 警告のコンテキスト
        """
        warning_message = message
        if context:
            warning_message += f"\nコンテキスト: {context}"
        self.logger.warning(warning_message)