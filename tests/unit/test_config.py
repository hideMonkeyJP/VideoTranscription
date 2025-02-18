import os
import json
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch

from src.utils.config import Config, ConfigError

class TestConfig:
    """Configクラスのテスト"""
    
    @pytest.fixture
    def temp_config_file(self):
        """一時的な設定ファイルを作成します"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'video_processor': {
                    'output_dir': 'custom_output',
                    'temp_dir': 'custom_temp'
                },
                'frame_extractor': {
                    'frame_interval': 2
                }
            }, f)
            return Path(f.name)
    
    @pytest.fixture
    def temp_env_file(self):
        """一時的な環境変数ファイルを作成します"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write('OUTPUT_DIR=env_output\n')
            f.write('FRAME_INTERVAL=3\n')
            return Path(f.name)
    
    def test_initialization(self):
        """初期化のテスト"""
        config = Config()
        assert isinstance(config.config, dict)
        assert 'video_processor' in config.config
        assert 'frame_extractor' in config.config
        assert 'audio_extractor' in config.config
    
    def test_load_custom_config(self, temp_config_file):
        """カスタム設定の読み込みテスト"""
        config = Config(str(temp_config_file))
        assert config.get('video_processor.output_dir') == 'custom_output'
        assert config.get('frame_extractor.frame_interval') == 2
    
    def test_env_override(self, temp_env_file):
        """環境変数による上書きテスト"""
        with patch.dict(os.environ, {
            'OUTPUT_DIR': 'env_output',
            'FRAME_INTERVAL': '3'
        }):
            config = Config()
            assert config.get('video_processor.output_dir') == 'env_output'
            assert config.get('frame_extractor.frame_interval') == 3.0
    
    def test_get_with_default(self):
        """デフォルト値を指定したget()のテスト"""
        config = Config()
        assert config.get('nonexistent.key', 'default') == 'default'
        assert config.get('video_processor.nonexistent', 123) == 123
    
    def test_get_nested_key(self):
        """ネストされたキーの取得テスト"""
        config = Config()
        value = config.get('frame_extractor.quality.jpeg_quality')
        assert isinstance(value, int)
        assert 0 <= value <= 100
    
    def test_validate_valid_config(self):
        """有効な設定の検証テスト"""
        config = Config()
        assert config.validate() is True
    
    def test_validate_invalid_config(self):
        """無効な設定の検証テスト"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'frame_extractor': {
                    'frame_interval': -1  # 無効な値
                }
            }, f)
            
        with pytest.raises(ConfigError):
            config = Config(f.name)
            config.validate()
    
    def test_save_config(self):
        """設定の保存テスト"""
        config = Config()
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config.save(f.name)
            
            # 保存された設定を読み込んで検証
            with open(f.name, 'r') as f2:
                saved_config = json.load(f2)
                assert saved_config == config.get_all()
    
    def test_merge_config(self, temp_config_file):
        """設定のマージテスト"""
        base_config = {
            'video_processor': {
                'output_dir': 'base_output',
                'temp_dir': 'base_temp',
                'log_dir': 'base_logs'
            },
            'frame_extractor': {
                'frame_interval': 1,
                'quality': {
                    'jpeg_quality': 90
                }
            }
        }
        
        custom_config = {
            'video_processor': {
                'output_dir': 'custom_output'
            },
            'frame_extractor': {
                'quality': {
                    'jpeg_quality': 95
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(custom_config, f)
            
        config = Config(f.name)
        
        # マージ結果の検証
        assert config.get('video_processor.output_dir') == 'custom_output'
        assert config.get('video_processor.temp_dir') == 'temp'  # デフォルト値が保持される
        assert config.get('frame_extractor.quality.jpeg_quality') == 95
    
    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        # 存在しないファイルを指定
        with pytest.raises(ConfigError):
            Config('nonexistent.json')
        
        # 無効なJSONファイル
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('invalid json')
            
        with pytest.raises(ConfigError):
            Config(f.name)
    
    def test_type_conversion(self):
        """型変換のテスト"""
        with patch.dict(os.environ, {
            'BATCH_SIZE': '20',
            'USE_GPU': 'true',
            'MIN_SCENE_CHANGE': '0.5'
        }):
            config = Config()
            assert isinstance(config.get('performance.batch_size'), int)
            assert isinstance(config.get('performance.use_gpu'), bool)
            assert isinstance(config.get('frame_extractor.min_scene_change'), float)
    
    def teardown_method(self, method):
        """テスト後のクリーンアップ"""
        # 一時ファイルの削除
        for file in Path('.').glob('*.json'):
            if file.is_file() and 'temp' in file.name:
                file.unlink() 