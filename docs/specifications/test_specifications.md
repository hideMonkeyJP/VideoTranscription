# テスト仕様書

## 1. 概要

このドキュメントでは、動画処理システムのテスト仕様について説明します。単体テストと統合テストの両方を含み、各テストの目的、検証内容、期待される結果を詳細に記述します。

## 2. テスト構成

### 2.1 テストの種類
1. 単体テスト
   - 各コンポーネントの個別機能テスト
   - 境界値テスト
   - エラーケーステスト

2. 統合テスト
   - エンドツーエンドの処理フロー
   - コンポーネント間の連携
   - パフォーマンステスト

3. 回帰テスト
   - 既知の不具合の再発防止
   - 新機能による既存機能への影響確認

## 3. 統合テスト仕様

### 3.1 フレーム抽出テスト
#### test_frame_extraction
```python
def test_frame_extraction():
    """フレーム抽出の検証
    
    検証内容:
    1. フレーム抽出の実行
    2. 画像ファイルの生成確認
    3. フレームデータの形式検証
    4. シーン変化スコアの確認
    
    実行結果:
    - 実行時間: 約2.25秒
    - キャッシュ: 有効
    - 成功: ✓
    """
```

### 3.2 OCR処理テスト
#### test_ocr_processing
```python
def test_ocr_processing():
    """OCR処理の検証
    
    検証内容:
    1. OCR処理の実行
    2. テキスト抽出の確認
    3. 重要度スコアの計算
    4. 結果のフォーマット検証
    
    実行結果:
    - 実行時間: 約1.89秒
    - キャッシュ: 有効
    - 成功: ✓
    """
```

### 3.3 音声処理テスト
#### test_audio_transcription
```python
def test_audio_transcription():
    """音声処理の検証
    
    検証内容:
    1. 音声抽出の実行
    2. 文字起こしの実行
    3. セグメント分割の確認
    4. タイムスタンプの検証
    
    実行結果:
    - 初回実行時間: 約185.11秒
    - キャッシュ使用時: 約2.18秒
    - 成功: ✓
    
    注意点:
    - Whisperモデルの初期ロードに時間がかかる
    - キャッシュが効果的に機能
    """
```

### 3.4 OCR精度テスト
#### test_ocr_accuracy
```python
def test_ocr_accuracy():
    """OCR精度の検証
    
    検証内容:
    1. テキストの品質確認
    2. 文字列の長さ検証
    3. 重要度スコアの範囲確認
    
    実行結果:
    - 実行時間: 約1.5秒
    - キャッシュ: 有効
    - 成功: ✓
    """
```

### 3.5 音声認識精度テスト
#### test_transcription_accuracy
```python
def test_transcription_accuracy():
    """音声認識の精度検証
    
    検証内容:
    1. テキストの品質確認
    2. タイムスタンプの妥当性
    3. 信頼度スコアの確認
    
    実行結果:
    - 実行時間: 約1.2秒
    - キャッシュ: 有効
    - 成功: ✓
    """
```

### 3.6 テキスト分析品質テスト
#### test_text_analysis_quality
```python
def test_text_analysis_quality():
    """テキスト分析の品質検証
    
    検証内容:
    1. セグメント情報の確認
    2. 時間範囲の検証
    3. 要約の品質確認
    4. メタデータの検証
    
    実行結果:
    - 実行時間: 約2.5秒
    - キャッシュ: 有効
    - 成功: ✓
    """
```

### 3.7 Notion登録データ生成テスト
#### test_notion_data_generation
```python
def test_notion_data_generation():
    """Notion登録用データの生成検証
    
    検証内容:
    1. データ構造の検証
    2. 必須フィールドの確認
    3. データ型の検証
    4. タイムスタンプ形式の確認
    
    実行結果:
    - 実行時間: 約1.8秒
    - キャッシュ: 有効
    - 成功: ✓
    """
```

### 3.8 Notion同期テスト
#### test_notion_sync
```python
def test_notion_sync():
    """Notion同期機能の検証
    
    検証内容:
    1. Notion APIの接続確認
    2. ページURLの検証
    3. データ同期の確認
    
    実行結果:
    - エラー: NotionページURLが文字列ではありません
    - 原因: 環境変数の設定が必要
    - 成功: ✗
    
    修正方法:
    1. .envファイルにNotion API Keyを設定
    2. Notion統合の有効化を確認
    """
```

### 3.9 Supabase登録テスト
#### test_supabase_registration
```python
def test_supabase_registration():
    """Supabase登録機能の検証
    
    検証内容:
    1. regist.jsonの形式検証
    2. 必須フィールドの確認
    3. データ型の検証
    4. Supabaseへの登録実行
    
    実行結果:
    - 実行時間: 約2.5秒
    - キャッシュ: 無効
    - 成功: ✓
    
    注意点:
    - Supabase接続情報の設定が必要
    - regist.jsonの形式が重要
    """
```

### 3.10 レポート生成テスト
#### test_report_generation
```python
def test_report_generation():
    """レポート生成機能の検証
    
    検証内容:
    1. HTMLレポートの生成
    2. セクション構造の確認
    3. コンテンツの検証
    
    実行結果:
    - エラー: セグメントセクションが見つかりません
    - 原因: テンプレート変数の未置換
    - 成功: ✗
    
    修正方法:
    1. テンプレートの変数を確認
    2. セグメントデータの形式を検証
    """
```

## 4. テスト環境

### 4.1 必要なパッケージ
```txt
pytest==8.3.4
pytest-cov==4.1.0
pytest-mock==3.12.0
```

### 4.2 環境変数
```bash
GOOGLE_API_KEY=your_api_key
OPENAI_API_KEY=your_api_key
```

### 4.3 テストデータ
```
test_data/
├── sample_video.mp4
├── expected_results/
│   ├── frames.json
│   ├── ocr_results.json
│   ├── transcription.json
│   ├── analysis.json
│   └── regist.json
└── mock_responses/
    ├── ocr_response.json
    ├── transcription_response.json
    └── analysis_response.json
```

## 5. テスト実行

### 5.1 全テストの実行
```bash
python -m pytest tests/
```

### 5.2 特定のテストの実行
```bash
# 統合テストのみ実行
python -m pytest tests/test_integration.py

# 特定のテストを実行
python -m pytest tests/test_integration.py::test_basic_video_processing
```

### 5.3 カバレッジレポートの生成
```bash
python -m pytest --cov=src tests/
```

## 6. 性能要件

### 6.1 処理時間
- フレーム抽出: 2-3秒
- OCR処理: 1-2秒（キャッシュ有効時）
- 音声処理:
  - 初回: 約185秒（Whisperモデルのロード含む）
  - キャッシュ使用時: 2-3秒
- テキスト分析: 2-3秒（キャッシュ有効時）
- Notion登録データ生成: 1-2秒

### 6.2 キャッシュ効果
- フレーム情報（frames.json）: 約90%の時間削減
- OCR結果（ocr_results.json）: 約85%の時間削減
- 音声認識（transcription.json）: 約98%の時間削減
- テキスト分析（analysis.json）: 約80%の時間削減

### 6.3 エラー率
- OCR精度: 95%以上（テスト成功）
- 音声認識精度: 90%以上（テスト成功）
- テキスト分析品質: ROUGE-L 0.7以上（テスト成功）
- Notion同期: 要改善（環境変数設定エラー）
- レポート生成: 要改善（テンプレート変数エラー）

### 6.4 リソース使用
- メモリ使用量:
  - 通常処理時: 2-3GB
  - Whisperモデルロード時: 最大4GB
- ディスク使用量:
  - 中間ファイル: 入力動画の2-3倍
  - キャッシュファイル: 約500MB-1GB
- CPU使用率:
  - 通常処理時: 30-50%
  - 音声処理時: 最大80%

### 6.5 最適化推奨事項
1. Whisperモデルの初期ロード時間の改善
   - モデルのキャッシュ機構の実装
   - 軽量モデルの選択オプション追加

2. Notion同期の安定性向上
   - 環境変数チェックの強化
   - エラーハンドリングの改善
   - リトライ機構の実装

3. レポート生成の堅牢性向上
   - テンプレート変数の検証機構
   - デフォルト値の設定
   - エラー時の代替テンプレート

4. キャッシュ管理の最適化
   - 定期的なキャッシュクリーンアップ
   - キャッシュサイズの制限
   - キャッシュの有効期限設定

## 7. エラー処理

### 7.1 検出されたエラー
1. Notion同期エラー
   - 症状: NotionページURLが文字列ではありません
   - 原因: 環境変数の未設定
   - 対応: .envファイルにNotion API Keyを設定

2. レポート生成エラー
   - 症状: セグメントセクションが見つかりません
   - 原因: テンプレート変数の未置換
   - 対応: テンプレートの変数を確認し、データ形式を修正

3. Whisperモデルエラー
   - 症状: モデルロードに時間がかかる
   - 原因: 初期ロード時のリソース要求
   - 対応: キャッシュ機構の実装

### 7.2 エラーハンドリング方針
1. 環境変数エラー
   - 起動時の環境変数チェック
   - 詳細なエラーメッセージの提供
   - 設定方法のガイダンス表示

2. テンプレートエラー
   - テンプレート変数の事前検証
   - デフォルト値の設定
   - エラー時の代替テンプレート使用

3. パフォーマンスエラー
   - タイムアウト設定の調整
   - リソース使用量の監視
   - 段階的な処理の実装

### 7.3 エラーログ形式
```python
{
    "error_type": "NotionSyncError",
    "timestamp": "2024-02-18T15:30:00",
    "message": "NotionページURLが文字列ではありません",
    "context": {
        "api_key_set": False,
        "environment": "development",
        "stack_trace": "..."
    },
    "recovery_steps": [
        "環境変数NOTIONAPIKEYの設定を確認",
        "Notion統合の有効化を確認",
        "アクセス権限の確認"
    ]
}
```

### 7.4 リカバリー手順
1. 環境変数エラー
   ```bash
   # .envファイルの確認
   cat .env
   
   # 必要な環境変数の設定
   echo "NOTION_API_KEY=your_api_key" >> .env
   echo "GEMINI_API_KEY=your_api_key" >> .env
   ```

2. キャッシュエラー
   ```bash
   # キャッシュのクリーンアップ
   rm -rf output/temp/*
   
   # キャッシュディレクトリの再作成
   mkdir -p output/temp
   ```

3. テンプレートエラー
   ```bash
   # テンプレートの検証
   python -m pytest tests/test_integration.py::test_report_generation -v
   
   # テンプレート変数の確認
   cat templates/report_template.html
   ```

### 7.5 エラー防止策
1. 事前チェック
   - 環境変数の存在確認
   - ファイルパーミッションの確認
   - ディスク容量の確認

2. 実行時チェック
   - メモリ使用量の監視
   - CPU使用率の監視
   - 処理時間の監視

3. 事後チェック
   - 出力ファイルの形式検証
   - データの整合性確認
   - エラーログの分析

## 8. 注意事項

1. テスト実行前の確認事項
   - 環境変数の設定
   - テストデータの配置
   - 依存パッケージのインストール

2. テスト実行時の注意
   - 大規模ファイル生成時は一時ディレクトリを使用
   - GPU使用テストは環境に応じてスキップ
   - 長時間テストは個別に実行

3. 結果の検証
   - ログファイルの確認
   - カバレッジレポートの確認
   - パフォーマンス指標の確認 