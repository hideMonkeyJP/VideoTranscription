"""
カスタム例外クラスの定義
"""
from datetime import datetime

class VideoProcessorError(Exception):
    """VideoProcessor全般に関連する基本的な例外"""
    def __init__(self, message: str, context: dict = None):
        super().__init__(message)
        self.context = context

class VideoProcessingError(VideoProcessorError):
    """動画処理に関連する基本的な例外"""
    pass

class VideoInputError(VideoProcessingError):
    """動画入力に関連するエラー"""
    pass

class FileNotFoundError(VideoInputError):
    """ファイルが見つからない場合のエラー"""
    pass

class InvalidFileFormatError(VideoInputError):
    """無効なファイル形式の場合のエラー"""
    pass

class AudioExtractionError(VideoProcessingError):
    """音声抽出に関連するエラー"""
    pass

class ProcessingError(VideoProcessingError):
    """処理中のエラー"""
    pass

class FrameExtractionError(ProcessingError):
    """フレーム抽出時のエラー"""
    pass

class OCRError(ProcessingError):
    """OCR処理時のエラー"""
    pass

class TranscriptionError(ProcessingError):
    """音声書き起こし時のエラー"""
    pass

class SummaryError(ProcessingError):
    """要約処理時のエラー"""
    pass

class OutputError(VideoProcessingError):
    """出力処理に関連するエラー"""
    pass

class DirectoryCreationError(OutputError):
    """ディレクトリ作成時のエラー"""
    pass

class FileWriteError(OutputError):
    """ファイル書き込み時のエラー"""
    pass

class NotionError(Exception):
    """Notion APIに関連する基本的な例外"""
    pass

class NotionAuthenticationError(NotionError):
    """認証エラー"""
    pass

class NotionDatabaseError(NotionError):
    """データベース操作に関連するエラー"""
    pass

class NotionPropertyError(NotionError):
    """プロパティ設定に関連するエラー"""
    pass

class ConfigurationError(VideoProcessorError):
    """設定に関連するエラー"""
    pass

class ContentAnalysisError(VideoProcessorError):
    """コンテンツ分析に関連するエラー"""
    pass

def create_error_context(operation: str, details: dict = None) -> dict:
    """
    エラーコンテキストを生成する
    
    Args:
        operation: エラーが発生した操作の名前
        details: エラーの詳細情報
    
    Returns:
        dict: エラーコンテキスト
    """
    context = {
        'operation': operation,
        'timestamp': datetime.now().isoformat(),
    }
    
    if details:
        context.update(details)
    
    return context

def handle_video_processing_error(error: Exception) -> str:
    """
    エラーメッセージを生成する
    
    Args:
        error: 発生した例外
    
    Returns:
        str: ユーザーフレンドリーなエラーメッセージ
    """
    if isinstance(error, FileNotFoundError):
        return f"指定されたファイルが見つかりません: {str(error)}"
    elif isinstance(error, InvalidFileFormatError):
        return f"無効なファイル形式です: {str(error)}"
    elif isinstance(error, FrameExtractionError):
        return f"フレーム抽出中にエラーが発生しました: {str(error)}"
    elif isinstance(error, OCRError):
        return f"OCR処理中にエラーが発生しました: {str(error)}"
    elif isinstance(error, TranscriptionError):
        return f"音声書き起こし中にエラーが発生しました: {str(error)}"
    elif isinstance(error, SummaryError):
        return f"要約処理中にエラーが発生しました: {str(error)}"
    elif isinstance(error, OutputError):
        return f"出力処理中にエラーが発生しました: {str(error)}"
    elif isinstance(error, NotionError):
        return f"Notion API処理中にエラーが発生しました: {str(error)}"
    else:
        return f"予期せぬエラーが発生しました: {str(error)}"