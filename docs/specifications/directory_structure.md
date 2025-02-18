# ディレクトリ構造仕様

## 出力ディレクトリ構成

```
output/
├── {timestamp}/                      # 処理実行時のタイムスタンプ（YYYYMMDD_HHMMSS形式）
│   ├── ocr_results.json             # OCR処理結果
│   ├── transcription.json           # 音声認識結果
│   ├── summaries.json               # 要約生成結果
│   ├── final_result.json            # 統合結果
│   ├── contexts_data.json           # Notion用データ
│   └── screenshots_{timestamp}/      # スクリーンショット画像保存ディレクトリ
│       └── screenshot_{index}.png    # スクリーンショット画像（3桁の連番）
└── sample/                          # サンプルデータ（参照用）
    ├── ocr_results.json             # OCR処理結果のサンプル
    ├── transcription.json           # 音声認識結果のサンプル
    ├── summaries.json               # 要約生成結果のサンプル
    ├── final_result.json            # 統合結果のサンプル
    ├── contexts_data.json           # Notion用データのサンプル
    └── screenshots_20250209_154640/ # スクリーンショットのサンプル
        └── screenshot_000.png       # スクリーンショット画像サンプル

```

## ディレクトリの説明

### frames/
- スクリーンショットの保存ディレクトリ
- タイムスタンプ付きのサブディレクトリに画像を格納
- 形式: PNG
- 命名規則: `screenshot_[連番].png`

### audio/
- 抽出された音声ファイルの保存ディレクトリ
- 形式: WAV
- サンプリングレート: 16kHz
- チャンネル: モノラル

### json/
- 全ての処理結果JSONファイルの保存ディレクトリ
- 各ファイルの役割:
  - ocr_results.json: OCR処理結果
  - transcription.json: 音声文字起こし結果
  - summaries.json: 要約生成結果
  - final_result.json: 全結果の統合
  - contexts_data.json: Notion用フォーマットデータ

## テスト用ディレクトリ

```
/Users/takayanagihidenori/Cursor/VideoTranscription/output_test/
├── integration_test/  # 統合テスト用
└── notion_test/      # Notion連携テスト用
```

## サンプル出力ディレクトリ

```
/Users/takayanagihidenori/Cursor/VideoTranscription/output/sample/
├── frames/
│   └── screenshots_20240211_120000/
│       ├── screenshot_001.png  # 開始シーン
│       ├── screenshot_002.png  # 重要な変化点
│       └── screenshot_003.png  # 終了シーン
├── audio/
│   └── sample_video_audio.wav  # 抽出された音声
└── json/
    ├── ocr_results.json       # テキスト認識結果
    ├── transcription.json     # 音声文字起こし
    ├── summaries.json         # 要約データ
    └── final_result.json      # 統合結果
```

### サンプルデータの説明

#### frames/
- 代表的なシーンのスクリーンショット
- 開始・重要・終了シーンを含む
- 画質: 1280x720, PNG形式

#### audio/
- 高品質音声サンプル
- 形式: 16bit/16kHz WAV
- 長さ: 約3分

#### json/
- 処理結果のサンプルデータ
- 日本語・英語混在のテキスト
- 実際の使用例を示す構造化データ

## JSONファイルフォーマット

### 1. final_result.json
```json
{
  "timestamp": "20250209_160200",
  "contexts": [
    {
      "time_range": {
        "start": 0.0,
        "end": 7.9
      },
      "summary": "タスク管理が、稼げる人と稼げない人、できる人とできない人の大きな違いであることが最初に強調される。",
      "scenes": [
        {
          "timestamp": 0.52,
          "screenshot_text": "...",
          "summary": {
            "main_points": ["..."],
            "full_text": "..."
          },
          "keywords": ["タスク", "管理", "時間"],
          "segments": ["まずタスク管理ですね"],
          "importance_score": 0,
          "metadata": {
            "segment_count": 1,
            "has_screenshot_text": true,
            "summary_points": 1,
            "keyword_count": 10
          }
        }
      ]
    }
  ]
}
```

### 2. contexts_data.json
```json
{
  "timestamp": "20250209_154640",
  "metadata": {
    "total_duration": 53.72,
    "context_count": 5,
    "total_scenes": 71,
    "total_segments": 29,
    "total_screenshots": 100,
    "processing_time": 209.28084015846252,
    "intermediate_files": {
      "transcription": "output_test/json_test/transcription.json",
      "ocr": "output_test/json_test/ocr_results.json",
      "summaries": "output_test/json_test/summaries.json"
    }
  },
  "contexts": [
    {
      "id": 1,
      "title": "文脈1: 0.0秒 - 7.9秒",
      "summary": "タスク管理が、稼げる人と稼げない人、できる人とできない人の大きな違いであることが最初に強調される。",
      "timestamp": "0.0秒 - 7.9秒",
      "keywords": ["タスク", "管理", "時間", "ある", "示示"],
      "screenshot": {
        "filename": "screenshot_000.png",
        "path": "screenshots_20250209_154640/screenshot_000.png",
        "timestamp": 7.8
      },
      "time_range": {
        "start": 0.0,
        "end": 7.9
      },
      "scenes": [
        {
          "timestamp": 0.52,
          "screenshot_text": "...",
          "summary": {
            "main_points": ["..."],
            "full_text": "..."
          },
          "keywords": ["タスク", "管理", "時間"],
          "segments": ["まずタスク管理ですね"],
          "importance_score": 0,
          "metadata": {
            "segment_count": 1,
            "has_screenshot_text": true,
            "summary_points": 1,
            "keyword_count": 10
          }
        }
      ]
    }
  ]
}
```

### 3. ocr_results.json
```json
{
  "screenshots": [
    {
      "timestamp": 0.52,
      "frame_number": 13,
      "importance_score": 11.79566796875
    }
  ]
}
```

### 4. transcription.json
```json
[
  {
    "text": "まずタスク管理ですね",
    "start": 0.0,
    "end": 1.4,
    "confidence": 0.0
  }
]
```

### 5. summaries.json
```json
[
  {
    "timestamp": 0.52,
    "screenshot_text": "...",
    "summary": {
      "main_points": ["..."],
      "full_text": "..."
    },
    "keywords": ["タスク", "管理", "時間"],
    "segments": ["まずタスク管理ですね"],
    "importance_score": 0,
    "metadata": {
      "segment_count": 1,
      "has_screenshot_text": true,
      "summary_points": 1,
      "keyword_count": 10
    }
  }
]
```

各JSONファイルは、処理の各段階で生成される構造化データを保持します。
タイムスタンプは秒単位で記録され、信頼度スコアは0-100の範囲で表されます。 