# VideoTranscription 開発者ドキュメント

## 開発環境のセットアップ

### 必要なツール

- Python 3.9以上
- FFmpeg
- Tesseract OCR
- Git
- Visual Studio Code（推奨）

### 環境構築手順

1. システム要件の確認:
```bash
python --version  # Python 3.9以上
ffmpeg -version  # FFmpeg
tesseract --version  # Tesseract OCR
```

2. 開発用の依存関係のインストール:
```bash
pip install -r requirements-dev.txt
```

3. pre-commitフックの設定:
```bash
pre-commit install
```

## コーディング規約

### Pythonコーディングスタイル

- [PEP 8](https://www.python.org/dev/peps/pep-0008/)に準拠
- 行の最大長は88文字（blackの設定に合わせる）
- docstringはGoogle形式を使用

### 命名規則

- クラス名: UpperCamelCase
- メソッド名: snake_case
- 変数名: snake_case
- 定数: UPPER_SNAKE_CASE
- プライベートメンバー: _leading_underscore

### インポート順序

1. 標準ライブラリ
2. サードパーティライブラリ
3. ローカルアプリケーション/ライブラリ

```python
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import cv2

from src.utils import config
from src.video_processing import frame_extractor
```

## プロジェクト構造

### ディレクトリ構造

```
VideoTranscription/
├── src/                    # ソースコード
│   ├── video_processing/   # 動画処理関連
│   ├── analysis/          # 分析関連
│   └── output/            # 出力関連
├── tests/                 # テストコード
│   ├── unit/             # ユニットテスト
│   ├── integration/      # 統合テスト
│   └── test_data/        # テストデータ
├── docs/                  # ドキュメント
├── config/               # 設定ファイル
└── templates/            # テンプレート
```

### モジュール構成

- `video_processing/`: 動画処理の基本機能
  - `frame_extractor.py`: フレーム抽出
  - `audio_extractor.py`: 音声抽出

- `analysis/`: 分析機能
  - `ocr_processor.py`: OCR処理
  - `text_analyzer.py`: テキスト分析

- `output/`: 出力機能
  - `report_generator.py`: レポート生成

## テスト

### テストの種類

1. ユニットテスト
   - 各クラスの個別機能をテスト
   - モックを使用して外部依存を分離

2. 統合テスト
   - 複数のコンポーネントの連携をテスト
   - 実際のファイルシステムとの相互作用を含む

3. エンドツーエンドテスト
   - 完全な処理パイプラインのテスト
   - 実際の動画ファイルを使用

### テストの実行

```bash
# 全テストの実行
pytest

# 特定のテストの実行
pytest tests/unit/test_frame_extractor.py

# カバレッジレポートの生成
pytest --cov=src --cov-report=html
```

## エラーハンドリング

### 例外の階層

```python
class VideoProcessingError(Exception):
    """ビデオ処理時のエラーを表すベース例外クラス"""
    pass

class FrameExtractionError(VideoProcessingError):
    """フレーム抽出時のエラー"""
    pass

class AudioExtractionError(VideoProcessingError):
    """音声抽出時のエラー"""
    pass

class TextAnalysisError(VideoProcessingError):
    """テキスト分析時のエラー"""
    pass
```

### ロギング

```python
import logging

logger = logging.getLogger(__name__)

try:
    # 処理
    logger.info("処理を開始します")
except Exception as e:
    logger.error(f"エラーが発生しました: {str(e)}")
    raise VideoProcessingError("処理に失敗しました") from e
```

## パフォーマンス最適化

### メモリ使用量の最適化

1. ジェネレータの使用
```python
def process_frames():
    for frame in frame_generator():
        yield process_frame(frame)
```

2. 一時ファイルの管理
```python
def cleanup():
    """一時ファイルの削除"""
    for file in temp_dir.glob("*.tmp"):
        file.unlink()
```

### 処理速度の最適化

1. マルチプロセシング
```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor() as executor:
    results = list(executor.map(process_frame, frames))
```

2. バッチ処理
```python
def process_batch(frames, batch_size=10):
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        yield process_frames_batch(batch)
```

## デプロイメント

### パッケージング

```bash
# wheelの作成
python setup.py bdist_wheel

# パッケージのインストール
pip install dist/video_transcription-1.0.0-py3-none-any.whl
```

### 環境変数

必要な環境変数:
- `OPENAI_API_KEY`: OpenAI APIキー
- `TESSERACT_PATH`: Tesseract OCRのパス
- `OUTPUT_DIR`: 出力ディレクトリ
- `LOG_LEVEL`: ログレベル

### Dockerサポート

```dockerfile
FROM python:3.9-slim

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-jpn

# アプリケーションのインストール
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

# 実行
CMD ["python", "-m", "src.main"]
```

## 貢献ガイドライン

### プルリクエストのプロセス

1. 新しいブランチの作成
```bash
git checkout -b feature/new-feature
```

2. コードの変更とコミット
```bash
git add .
git commit -m "Add new feature"
```

3. テストの実行
```bash
pytest
```

4. コードの品質チェック
```bash
black src/ tests/
flake8 src/ tests/
mypy src/ tests/
```

5. プルリクエストの作成
- 変更内容の説明
- テスト結果の添付
- レビュアーの指定

### コードレビュー

レビューのチェックポイント:
- コーディング規約の遵守
- テストの網羅性
- エラーハンドリング
- パフォーマンスへの影響
- ドキュメントの更新 