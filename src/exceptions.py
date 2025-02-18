from typing import Dict, Any, Optional

class VideoProcessingError(Exception):
    """動画処理中のエラーを表す例外クラス"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        
    def __str__(self):
        error_msg = super().__str__()
        if self.context:
            error_msg += f"\nコンテキスト: {self.context}"
        return error_msg

class AudioExtractionError(VideoProcessingError):
    """音声抽出に関するエラー"""
    pass

class TextAnalysisError(Exception):
    """テキスト分析時のエラーを表すカスタム例外クラス"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {} 