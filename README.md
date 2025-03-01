# VideoTranscription

動画から文字起こしと分析を行い、構造化されたレポートを生成するPythonプロジェクト。

## 機能

- 動画からの音声抽出と文字起こし
- フレーム抽出とOCR処理
- テキスト分析（要約、キーポイント抽出）
- HTMLレポート生成
- Notion連携
- デバイス最適化（CPU/GPU自動選択）

## 必要条件

- Python 3.9以上
- FFmpeg
- Tesseract OCR
- CUDA対応GPUを推奨（文字起こし処理の高速化）
- Apple Silicon Mac対応（MPS機能）

## インストール

1. リポジトリのクローン:
```bash
git clone https://github.com/yourusername/VideoTranscription.git
cd VideoTranscription
```

2. 仮想環境の作成と有効化:
```bash
python -m venv venv
source venv/bin/activate  # Unix
# または
.\venv\Scripts\activate  # Windows
```

3. 依存関係のインストール:
```bash
pip install -r requirements.txt
```

4. 環境変数の設定:
```bash
cp .env.example .env
# .envファイルを編集して必要なAPIキーを設定
```

## 使用方法

1. 基本的な使用方法:
```python
from src.video_processor import VideoProcessor

processor = VideoProcessor()
result = processor.process_video("path/to/video.mp4")
```

2. カスタム設定での使用:
```python
config = {
    'frame_extractor_config': {
        'frame_interval': 1,
        'min_scene_change': 0.3
    },
    'ocr_config': {
        'min_confidence': 0.6
    },
    'text_analyzer_config': {
        'model_name': 'ja_core_news_lg'
    }
}

processor = VideoProcessor(config)
result = processor.process_video("path/to/video.mp4")
```

## プロジェクト構造

```
VideoTranscription/
├── src/
│   ├── video_processing/
│   │   ├── frame_extractor.py
│   │   └── audio_extractor.py
│   ├── analysis/
│   │   ├── ocr_processor.py
│   │   └── text_analyzer.py
│   └── output/
│       └── report_generator.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── test_data/
├── templates/
│   └── report_template.html
├── config/
│   └── default_config.json
└── docs/
    ├── API.md
    └── DEVELOPMENT.md
```

## テスト

テストの実行:
```bash
pytest tests/
```

カバレッジレポートの生成:
```bash
pytest --cov=src tests/
```

## 開発

1. 開発環境のセットアップ:
```bash
pip install -r requirements-dev.txt
```

2. コード品質チェック:
```bash
black src/ tests/
flake8 src/ tests/
mypy src/ tests/
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 貢献

1. このリポジトリをフォーク
2. 新しいブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチをプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 作者

- 作者名
- メールアドレス
- プロジェクトURL

## 最近の更新

### 2023-03-01: Whisperモデルのデバイス最適化

- モデルサイズに応じた最適なデバイス（CPU/GPU）選択機能を追加
- 小さいモデル（tiny, base）はCPUで高速に動作
- 大きいモデル（small, medium, large）はGPU（MPS/CUDA）で高速に動作
- 詳細は[ドキュメント](docs/troubleshooting/whisper_device_optimization.md)を参照