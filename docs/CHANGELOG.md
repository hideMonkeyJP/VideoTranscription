# 変更履歴

## [1.1.0] - 2023-03-01

### 追加

- Whisperモデルのデバイス最適化機能
  - モデルサイズに応じた最適なデバイス（CPU/GPU）選択機能
  - 設定ファイルでのデバイス指定オプション
  - Apple Silicon（M1/M2/M3）のMPSサポート強化

### 変更

- `TranscriptionProcessor`クラスの`_get_device`メソッドを改善
  - 設定ファイルからデバイス設定を取得する機能を追加
  - デバイス使用状況のログ出力を追加
- 設定ファイル（`config/config.yaml`）を更新
  - Whisperモデルを`medium`から`base`に変更
  - デバイス設定を`cpu`に設定（小さいモデル向け最適化）
- テスト設定を更新
  - テスト環境でのデバイス設定を明示的に指定

### ドキュメント

- `docs/troubleshooting/whisper_device_optimization.md` - デバイス最適化の解説
- `docs/troubleshooting/whisper_benchmark.md` - ベンチマーク結果
- `docs/specifications/whisper_config.md` - Whisper設定ガイド
- READMEに最近の更新情報を追加

### パフォーマンス

- baseモデルの処理時間: CPU約17秒 vs MPS約34秒（2倍の高速化）
- mediumモデルの処理時間: MPS約143秒 vs CPU約300秒（2倍の高速化）

## [1.0.0] - 2023-02-10

### 追加

- 初期リリース
- 動画からの音声抽出と文字起こし
- フレーム抽出とOCR処理
- テキスト分析（要約、キーポイント抽出）
- HTMLレポート生成
- Notion連携 