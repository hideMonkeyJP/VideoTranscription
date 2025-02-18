# 中間ファイルフォーマット仕様書

## 1. 概要

このドキュメントでは、動画処理システムで使用される中間ファイルのフォーマット仕様について説明します。

## 2. フレーム抽出結果 (frames.json)

### 2.1 基本構造
```json
[
  {
    "timestamp": 0.52,
    "frame_number": 13,
    "scene_change_score": 0.8,
    "path": "output/screenshots/frame_0013.jpg",
    "importance_score": 0.91
  }
]
```

### 2.2 フィールド説明
| フィールド | 型 | 説明 | 必須 |
|------------|------|------------|------|
| timestamp | number | フレームのタイムスタンプ（秒） | ✓ |
| frame_number | integer | フレーム番号 | ✓ |
| scene_change_score | number | シーン変更スコア（0-1） | ✓ |
| path | string | スクリーンショットの保存パス | ✓ |
| importance_score | number | 重要度スコア（0-1） | ✓ |

## 3. OCR処理結果 (ocr_results.json)

### 3.1 基本構造
```json
{
  "screenshots": [
    {
      "timestamp": 0.52,
      "frame_number": 13,
      "importance_score": 0.91,
      "text": "スクリーンショットのテキスト",
      "ocr_confidence": 0.85,
      "text_regions": [
        {
          "text": "部分テキスト",
          "confidence": 0.92,
          "bbox": [10, 20, 100, 50]
        }
      ]
    }
  ]
}
```

### 3.2 フィールド説明
| フィールド | 型 | 説明 | 必須 |
|------------|------|------------|------|
| timestamp | number | フレームのタイムスタンプ（秒） | ✓ |
| frame_number | integer | フレーム番号 | ✓ |
| importance_score | number | 重要度スコア（0-1） | ✓ |
| text | string | 抽出されたテキスト | ✓ |
| ocr_confidence | number | OCR全体の信頼度（0-1） | ✓ |
| text_regions | array | テキスト領域の配列 | - |

## 4. 音声認識結果 (transcription.json)

### 4.1 基本構造
```json
[
  {
    "text": "文字起こしテキスト",
    "start": 0.0,
    "end": 2.5,
    "confidence": 0.95,
    "speaker": "Speaker 1",
    "words": [
      {
        "word": "文字",
        "start": 0.0,
        "end": 0.8,
        "confidence": 0.97
      }
    ]
  }
]
```

### 4.2 フィールド説明
| フィールド | 型 | 説明 | 必須 |
|------------|------|------------|------|
| text | string | セグメントのテキスト | ✓ |
| start | number | 開始時間（秒） | ✓ |
| end | number | 終了時間（秒） | ✓ |
| confidence | number | 信頼度（0-1） | ✓ |
| speaker | string | 話者識別 | - |
| words | array | 単語レベルの情報 | - |

## 5. テキスト分析結果 (analysis.json)

### 5.1 基本構造
```json
{
  "segments": [
    {
      "time_range": {
        "start": 0.0,
        "end": 2.5
      },
      "summary": "セグメントの要約",
      "importance_score": 0.8,
      "metadata": {
        "segment_count": 1,
        "has_screenshot_text": true,
        "summary_points": 3,
        "keyword_count": 10
      },
      "screenshot": {
        "timestamp": 0.52,
        "frame_number": 13,
        "text": "スクリーンショットのテキスト",
        "ocr_confidence": 0.85
      },
      "keywords": ["キーワード1", "キーワード2"],
      "key_points": [
        "重要ポイント1",
        "重要ポイント2"
      ]
    }
  ],
  "total_segments": 10,
  "total_duration": 53.72,
  "processing_info": {
    "start_time": "2025-02-18T20:48:00",
    "end_time": "2025-02-18T20:52:30",
    "processing_duration": 270.0
  }
}
```

### 5.2 フィールド説明
| フィールド | 型 | 説明 | 必須 |
|------------|------|------------|------|
| time_range | object | 時間範囲情報 | ✓ |
| summary | string | セグメントの要約 | ✓ |
| importance_score | number | 重要度スコア（0-1） | ✓ |
| metadata | object | メタデータ情報 | ✓ |
| screenshot | object | スクリーンショット情報 | - |
| keywords | array | キーワードリスト | ✓ |
| key_points | array | 重要ポイントリスト | ✓ |

## 6. Notion登録用データ (regist.json)

### 6.1 基本構造
```json
[
  {
    "No": 1,
    "Summary": "セグメントの要約",
    "Timestamp": "0.0秒 - 2.5秒",
    "Thumbnail": "https://gyazo.com/xxx"
  }
]
```

### 6.2 フィールド説明
| フィールド | 型 | 説明 | 必須 |
|------------|------|------------|------|
| No | integer | セグメント番号 | ✓ |
| Summary | string | セグメントの要約 | ✓ |
| Timestamp | string | 時間範囲（表示用） | ✓ |
| Thumbnail | string | サムネイル画像のURL | ✓ |

## 7. データ型の制約

### 7.1 数値
- timestamp: 小数点2桁まで
- confidence: 0.0から1.0の範囲
- frame_number: 0以上の整数

### 7.2 文字列
- path: 相対パス形式
- text: 空文字列を許容
- URL: 有効なURL形式

### 7.3 配列
- 空配列を許容
- 最大要素数は実装依存

## 8. エラー処理

### 8.1 必須フィールド
- 必須フィールドが欠落している場合はエラー
- 型が一致しない場合はエラー

### 8.2 値の検証
- 数値範囲の検証
- URL形式の検証
- パスの存在確認

### 8.3 エラーメッセージ
- フィールド名を含むこと
- エラーの理由を明確に示すこと
- 修正のヒントを提供すること 