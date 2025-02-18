"""
フレーム抽出モジュール

このパッケージは、動画からのフレーム抽出に関連する機能を提供します。
主な機能：
- 動画からのフレーム抽出
- フレーム間隔の制御
- 画質設定
- メタデータの管理
"""

from .frame_extractor import FrameExtractor, FrameExtractionError

__all__ = ['FrameExtractor', 'FrameExtractionError']
