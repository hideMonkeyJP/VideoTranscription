import os
import logging
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from src.utils.config import Config
from src.utils.logger import Logger

class TestLogger:
    """Loggerクラスのテスト"""
    
    @pytest.fixture
    def temp_log_dir(self):
        """一時的なログディレクトリを作成します"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def config(self, temp_log_dir):
        """テスト用の設定を作成します"""
        return Config()
    
    @pytest.fixture
    def logger(self, config):
        """テスト用のロガーを作成します"""
        return Logger(config, 'test_logger')
    
    def test_initialization(self, logger):
        """初期化のテスト"""
        assert logger.name == 'test_logger'
        assert isinstance(logger.logger, logging.Logger)
        assert logger.logger.level == logging.INFO
        
        # ハンドラーの確認
        handlers = logger.logger.handlers
        assert len(handlers) == 2  # ファイルハンドラーとコンソールハンドラー
        assert any(isinstance(h, logging.FileHandler) for h in handlers)
        assert any(isinstance(h, logging.StreamHandler) for h in handlers)
    
    def test_log_levels(self, logger, temp_log_dir):
        """各ログレベルのテスト"""
        test_message = "テストメッセージ"
        
        # 各レベルでログを出力
        logger.debug(test_message)
        logger.info(test_message)
        logger.warning(test_message)
        logger.error(test_message)
        logger.critical(test_message)
        
        # ログファイルの内容を確認
        log_file = Path(temp_log_dir) / 'video_processing.log'
        assert log_file.exists()
        
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # INFOレベル以上のメッセージが含まれていることを確認
            assert "INFO" in content
            assert "WARNING" in content
            assert "ERROR" in content
            assert "CRITICAL" in content
            # DEBUGレベルのメッセージは含まれていないことを確認
            assert "DEBUG" not in content
    
    def test_custom_log_level(self, temp_log_dir):
        """カスタムログレベルのテスト"""
        config = Config()
        config.config['logging']['level'] = 'DEBUG'
        
        logger = Logger(config, 'test_debug_logger')
        assert logger.logger.level == logging.DEBUG
        
        test_message = "デバッグメッセージ"
        logger.debug(test_message)
        
        # ログファイルの内容を確認
        log_file = Path(temp_log_dir) / 'video_processing.log'
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "DEBUG" in content
            assert test_message in content
    
    def test_exception_logging(self, logger, temp_log_dir):
        """例外ログのテスト"""
        try:
            raise ValueError("テストエラー")
        except ValueError:
            logger.exception("エラーが発生しました")
        
        # ログファイルの内容を確認
        log_file = Path(temp_log_dir) / 'video_processing.log'
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "ERROR" in content
            assert "エラーが発生しました" in content
            assert "ValueError: テストエラー" in content
            assert "Traceback" in content
    
    def test_get_logger(self, config):
        """get_loggerメソッドのテスト"""
        logger1 = Logger.get_logger('test_logger1', config)
        logger2 = Logger.get_logger('test_logger1', config)
        
        # 同じ名前のロガーは同じインスタンスを返す
        assert logger1.name == logger2.name
        assert logger1.logger == logger2.logger
        
        # 異なる名前のロガーは異なるインスタンスを返す
        logger3 = Logger.get_logger('test_logger2', config)
        assert logger1.name != logger3.name
        assert logger1.logger != logger3.logger
    
    def test_log_rotation(self, temp_log_dir):
        """ログローテーションのテスト"""
        config = Config()
        logger = Logger(config, 'test_rotation_logger')
        
        # 大きなメッセージを生成（10MB以上）
        large_message = "x" * 1024 * 1024  # 1MB
        for _ in range(11):  # 11MB分のログを書き込む
            logger.info(large_message)
        
        # ログファイルが複数生成されていることを確認
        log_files = list(Path(temp_log_dir).glob('video_processing.log*'))
        assert len(log_files) > 1
    
    def test_log_format(self, logger, temp_log_dir):
        """ログフォーマットのテスト"""
        test_message = "フォーマットテスト"
        logger.info(test_message)
        
        # ログファイルの内容を確認
        log_file = Path(temp_log_dir) / 'video_processing.log'
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # タイムスタンプ、ロガー名、レベル、メッセージが含まれていることを確認
            assert all(x in content for x in ['test_logger', 'INFO', test_message])
            # デフォルトフォーマットの区切り文字が含まれていることを確認
            assert ' - ' in content
    
    def teardown_method(self, method):
        """テスト後のクリーンアップ"""
        # ロガーのハンドラーをクリア
        logging.getLogger().handlers.clear() 