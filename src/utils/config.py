import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv

class ConfigError(Exception):
    """設定関連のエラーを表すカスタム例外クラス"""
    pass

class Config:
    """設定を管理するクラス"""
    
    def __init__(self, config_path_or_dict: Optional[Union[str, Dict[str, Any]]] = None):
        """
        設定を初期化します。
        
        Args:
            config_path_or_dict (Union[str, Dict[str, Any]], optional): 設定ファイルのパスまたは設定辞書
        """
        self.config = {}
        
        # デフォルト設定の読み込み
        self.config = self._load_default_config()
        
        # カスタム設定の読み込み
        if config_path_or_dict is not None:
            if isinstance(config_path_or_dict, dict):
                self._merge_config(config_path_or_dict)
            elif isinstance(config_path_or_dict, str):
                custom_config = self._load_config(config_path_or_dict)
                self._merge_config(custom_config)
        
        # 環境変数による上書き
        self._override_from_env()
    
    def __getitem__(self, key: str) -> Any:
        """辞書のように値を取得します"""
        return self.config[key]
    
    def __setitem__(self, key: str, value: Any):
        """辞書のように値を設定します"""
        self.config[key] = value
    
    def __contains__(self, key: str) -> bool:
        """辞書のようにキーの存在を確認します"""
        return key in self.config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        指定されたキーの値を取得します。キーが存在しない場合はデフォルト値を返します。
        ドット記法でネストされた値にアクセスできます（例: 'video_processor.output_dir'）。
        
        Args:
            key (str): 設定キー（ドット区切りでネストされた値にアクセス可能）
            default (Any, optional): デフォルト値
            
        Returns:
            Any: 設定値またはデフォルト値
        """
        return self.config.get(key, default)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を読み込みます"""
        return {
            "video_processor": {
                "output_dir": "output",
                "temp_dir": "temp"
            },
            "frame_extractor": {
                "interval": 1.0,
                "quality": 95
            },
            "audio_extractor": {
                "format": "wav",
                "sample_rate": 16000
            },
            "ocr_processor": {
                "lang": "jpn",
                "min_confidence": 0.6
            },
            "text_analyzer": {
                "min_segment_length": 50,
                "similarity_threshold": 0.7
            }
        }
    
    def _load_config(self, path: str) -> Dict[str, Any]:
        """設定ファイルを読み込みます"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ConfigError(f"設定ファイルの読み込みに失敗しました: {str(e)}")
    
    def _merge_config(self, custom_config: Dict[str, Any]):
        """カスタム設定をマージします"""
        for key, value in custom_config.items():
            if isinstance(value, dict) and key in self.config:
                self.config[key].update(value)
            else:
                self.config[key] = value
    
    def _override_from_env(self):
        """環境変数で設定を上書きします"""
        # パス設定
        self.config['video_processor']['output_dir'] = os.getenv('OUTPUT_DIR', self.config['video_processor']['output_dir'])
        self.config['video_processor']['temp_dir'] = os.getenv('TEMP_DIR', self.config['video_processor']['temp_dir'])
        
        # 処理設定
        if os.getenv('FRAME_INTERVAL'):
            self.config['frame_extractor']['interval'] = float(os.getenv('FRAME_INTERVAL'))
        if os.getenv('MIN_TEXT_QUALITY'):
            self.config['ocr_processor']['min_confidence'] = float(os.getenv('MIN_TEXT_QUALITY'))
        
        # モデル設定
        self.config['audio_extractor']['format'] = os.getenv('WHISPER_MODEL', self.config['audio_extractor']['format'])
        self.config['text_analyzer']['min_segment_length'] = int(os.getenv('SPACY_MODEL', self.config['text_analyzer']['min_segment_length']))
        self.config['ocr_processor']['lang'] = os.getenv('OCR_LANGUAGE', self.config['ocr_processor']['lang'])
        
        # ロギング設定
        if os.getenv('LOG_LEVEL'):
            self.config['logging']['level'] = os.getenv('LOG_LEVEL')
        if os.getenv('LOG_FORMAT'):
            self.config['logging']['format'] = os.getenv('LOG_FORMAT')
        
        # パフォーマンス設定
        if os.getenv('BATCH_SIZE'):
            self.config['performance']['batch_size'] = int(os.getenv('BATCH_SIZE'))
        if os.getenv('MAX_WORKERS'):
            self.config['performance']['max_workers'] = int(os.getenv('MAX_WORKERS'))
        if os.getenv('USE_GPU'):
            self.config['performance']['use_gpu'] = os.getenv('USE_GPU').lower() == 'true'
    
    def get_all(self) -> Dict[str, Any]:
        """
        全ての設定を取得します。
        
        Returns:
            Dict[str, Any]: 設定辞書
        """
        return self.config.copy()
    
    def validate(self) -> bool:
        """
        設定値を検証します。
        
        Returns:
            bool: 検証結果
        """
        try:
            # 必須パスの存在確認
            required_paths = [
                self.config['video_processor']['output_dir'],
                self.config['video_processor']['temp_dir']
            ]
            for path in required_paths:
                os.makedirs(path, exist_ok=True)
            
            # 数値パラメータの範囲チェック
            if not 0 < self.config['frame_extractor']['interval'] <= 10:
                raise ConfigError("frame_intervalは0より大きく10以下である必要があります")
            
            if not 0 < self.config['frame_extractor']['quality'] < 100:
                raise ConfigError("qualityは0より大きく100未満である必要があります")
            
            if not 0 < self.config['ocr_processor']['min_confidence'] <= 1:
                raise ConfigError("min_confidenceは0より大きく1以下である必要があります")
            
            # パフォーマンス設定のチェック
            if self.config['performance']['batch_size'] < 1:
                raise ConfigError("batch_sizeは1以上である必要があります")
            
            if self.config['performance']['max_workers'] < 1:
                raise ConfigError("max_workersは1以上である必要があります")
            
            return True
            
        except Exception as e:
            raise ConfigError(f"設定の検証に失敗しました: {str(e)}")
    
    def save(self, path: str):
        """
        現在の設定を保存します。
        
        Args:
            path (str): 保存先のパス
        """
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            raise ConfigError(f"設定の保存に失敗しました: {str(e)}") 