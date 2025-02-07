from typing import Optional, Dict, Any
from datetime import datetime

class VideoProcessorError(Exception):
    """ビデオ処理の基本例外クラス"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.context = context or {}
        super().__init__(self.message)

class VideoProcessingError(VideoProcessorError):
    """ビデオ処理全般に関する例外"""
    pass

class VideoInputError(VideoProcessorError):
    """ビデオ入力に関する例外"""
    pass

class AudioExtractionError(VideoProcessorError):
    """音声抽出に関する例外"""
    pass

class TranscriptionError(VideoProcessorError):
    """音声認識に関する例外"""
    pass

class FrameExtractionError(VideoProcessorError):
    """フレーム抽出に関する例外"""
    pass

class OCRError(VideoProcessorError):
    """OCR処理に関する例外"""
    pass

class SummaryGenerationError(VideoProcessorError):
    """要約生成に関する例外"""
    pass

class OutputError(VideoProcessorError):
    """出力処理に関する例外"""
    pass

class ConfigurationError(VideoProcessorError):
    """設定に関する例外"""
    pass

class ResourceError(VideoProcessorError):
    """リソース（メモリ、CPU等）に関する例外"""
    pass

def create_error_context(
    operation: str,
    input_data: Optional[Dict[str, Any]] = None,
    error_details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """エラーコンテキストを生成します"""
    context = {
        'operation': operation,
        'timestamp': str(datetime.now()),
    }
    
    if input_data:
        context['input_data'] = input_data
    if error_details:
        context['error_details'] = error_details
        
    return context 