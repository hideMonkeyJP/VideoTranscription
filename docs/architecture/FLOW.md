# システムフローチャート

## 1. メインフロー

```mermaid
graph TD
    A[動画入力] --> B[前処理]
    B --> C1[フレーム抽出]
    B --> C2[音声抽出]
    
    C1 --> D1[重要フレーム選定]
    D1 --> E1[OCR処理]
    
    C2 --> D2[音声認識]
    D2 --> E2[テキスト生成]
    
    E1 --> F[テキスト統合]
    E2 --> F
    
    F --> G1[キーワード抽出]
    F --> G2[トピック分析]
    F --> G3[テキスト要約]
    
    G1 --> H[結果統合]
    G2 --> H
    G3 --> H
    
    H --> I[JSON/CSV出力]
```

## 2. フレーム抽出フロー

```mermaid
graph TD
    A[動画読み込み] --> B[フレームレート計算]
    B --> C[総フレーム数取得]
    C --> D[抽出間隔計算]
    D --> E[フレーム抽出]
    E --> F[シーン変化検出]
    F --> G[重要フレーム選定]
    G --> H[画像保存]
```

## 3. 音声処理フロー

```mermaid
graph TD
    A[音声抽出] --> B[音声ファイル生成]
    B --> C[Whisper処理]
    C --> D[セグメント分割]
    D --> E[タイムスタンプ付与]
    E --> F[テキスト生成]
```

## 4. テキスト処理フロー

```mermaid
graph TD
    A[テキスト入力] --> B1[OCRテキスト]
    A --> B2[音声テキスト]
    
    B1 --> C[テキストクリーニング]
    B2 --> C
    
    C --> D1[形態素解析]
    C --> D2[要約処理]
    
    D1 --> E1[キーワード抽出]
    D1 --> E2[トピック分析]
    
    E1 --> F[結果統合]
    E2 --> F
    D2 --> F
```

## 5. データ構造

### 5.1 フレーム情報
```json
{
    "frame_id": "string",
    "timestamp": "float",
    "importance_score": "float",
    "ocr_text": "string",
    "image_path": "string"
}
```

### 5.2 音声セグメント
```json
{
    "segment_id": "string",
    "start_time": "float",
    "end_time": "float",
    "text": "string",
    "confidence": "float"
}
```

### 5.3 最終出力
```json
{
    "video_id": "string",
    "timestamp": "string",
    "frames": [
        {
            "timestamp": "float",
            "ocr_text": "string",
            "image_path": "string"
        }
    ],
    "transcription": [
        {
            "start": "float",
            "end": "float",
            "text": "string"
        }
    ],
    "analysis": {
        "keywords": ["string"],
        "topics": ["string"],
        "summary": "string"
    }
}
```

## 6. エラーハンドリング

```mermaid
graph TD
    A[エラー発生] --> B{エラー種別判定}
    B -->|入力エラー| C1[入力検証]
    B -->|処理エラー| C2[処理再試行]
    B -->|システムエラー| C3[システム復旧]
    
    C1 --> D[エラーログ記録]
    C2 --> D
    C3 --> D
    
    D --> E[エラーレポート生成]