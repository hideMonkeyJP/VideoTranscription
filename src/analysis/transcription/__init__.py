"""
音声認識モジュール

このパッケージは、音声認識に関連する機能を提供します。
主な機能：
- 音声ファイルの文字起こし
- 単語単位のタイムスタンプ取得
- 音声の前処理
"""

from .transcription_processor import TranscriptionProcessor, TranscriptionError

__all__ = ['TranscriptionProcessor', 'TranscriptionError']
