"""
OCRモジュール

このパッケージは、画像からのテキスト抽出に関連する機能を提供します。
主な機能：
- 画像の前処理
- テキスト抽出
- 位置情報の取得
- 信頼度スコアの計算
"""

from .ocr_processor import OCRProcessor, OCRError

__all__ = ['OCRProcessor', 'OCRError']
