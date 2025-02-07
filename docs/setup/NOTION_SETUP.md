# Notion連携のセットアップ

このドキュメントでは、VideoTranscriptionシステムとNotionデータベースの連携方法について説明します。

## 前提条件

1. Notionアカウントを持っていること
2. Notion APIの利用権限があること
3. 連携用のデータベースがNotionに作成されていること

## セットアップ手順

### 1. Notion APIの設定

1. [Notion Developers](https://developers.notion.com/) にアクセス
2. "My Integrations" から新しいインテグレーションを作成
3. 以下の情報を設定:
   - Name: "VideoTranscription"
   - Associated workspace: 使用するワークスペース
   - Capabilities: 必要な権限を選択
     - Content Capabilities
       - [x] Read content
       - [x] Update content
       - [x] Insert content
     - Database Capabilities
       - [x] Read database content
       - [x] Insert database content

4. インテグレーションが作成されたら、"Internal Integration Token" をコピー

### 2. データベースの準備

1. Notionで新しいデータベースを作成
2. 以下のプロパティを設定:
   - Title (タイトル型)
   - Summary (リッチテキスト型)
   - Keywords (マルチセレクト型)
   - Duration (数値型)
   - ProcessedDate (日付型)
   - Thumbnail (ファイル型)

3. データベースの共有設定から、作成したインテグレーションを追加
4. データベースIDをコピー (URLの末尾部分)

### 3. 環境変数の設定

`.env`ファイルに以下の環境変数を追加:

```bash
NOTION_AUTH_TOKEN=your_integration_token
NOTION_DATABASE_ID=your_database_id
```

### 4. 設定ファイルの更新

`config/config.yaml`の`notion`セクションを以下のように更新:

```yaml
notion:
  enabled: true
  database_id: "${NOTION_DATABASE_ID}"  # 環境変数から読み込み
  sync_interval: 60
  retry_count: 3
  properties:
    title: "Title"
    summary: "Summary"
    keywords: "Keywords"
    duration: "Duration"
    processed_date: "ProcessedDate"
    thumbnail: "Thumbnail"
```

## 動作確認

1. テストの実行:
```bash
pytest tests/test_notion_integration.py -v
```

2. 実際の動画処理での確認:
```bash
python src/main.py path/to/video.mp4
```

## トラブルシューティング

### よくある問題と解決方法

1. 認証エラー
   - インテグレーショントークンが正しく設定されているか確認
   - データベースへの権限が正しく設定されているか確認

2. データベースエラー
   - データベースIDが正しいか確認
   - プロパティの名前が設定ファイルと一致しているか確認

3. 同期エラー
   - ネットワーク接続を確認
   - Notion APIの制限に達していないか確認

### ログの確認

エラーが発生した場合は、以下のログファイルを確認:

```
logs/video_processor.log
```

## 制限事項

1. Notion APIの制限
   - リクエスト制限: 3回/秒
   - ブロック制限: 1000ブロック/ページ

2. サポートされる機能
   - テキストデータの同期
   - キーワードの同期
   - サムネイル画像の同期

3. 非サポート機能
   - リアルタイム同期
   - 双方向同期