# 開発環境セットアップガイド

## 1. 前提条件

### 1.1 必要なソフトウェア
- Python 3.9
- FFmpeg
- MeCab
- Tesseract-OCR
- Git

### 1.2 推奨スペック
- CPU: 4コア以上
- メモリ: 16GB以上
- ストレージ: 50GB以上の空き容量
- GPU: NVIDIA GPU (CUDA対応) 推奨

## 2. インストール手順

### 2.1 基本環境のセットアップ

#### macOS
```bash
# Homebrewのインストール(未インストールの場合)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 必要なパッケージのインストール
brew install python@3.9
brew install ffmpeg
brew install mecab
brew install mecab-ipadic
brew install tesseract
brew install tesseract-lang
```

#### Ubuntu/Debian
```bash
# システムの更新
sudo apt update
sudo apt upgrade

# 必要なパッケージのインストール
sudo apt install python3.9 python3.9-venv
sudo apt install ffmpeg
sudo apt install mecab libmecab-dev mecab-ipadic-utf8
sudo apt install tesseract-ocr tesseract-ocr-jpn
```

### 2.2 Pythonの仮想環境設定
```bash
# プロジェクトディレクトリの作成
mkdir VideoTranscription
cd VideoTranscription

# 仮想環境の作成
python3.9 -m venv venv

# 仮想環境の有効化
# macOS/Linux:
source venv/bin/activate
# Windows:
# .\venv\Scripts\activate
```

### 2.3 依存ライブラリのインストール
```bash
# 依存関係のインストール
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. 設定ファイルの準備

### 3.1 設定ファイルの配置
```bash
# 設定ディレクトリの作成
mkdir -p config

# 設定ファイルのコピー
cp config.yaml.example config/config.yaml
```

### 3.2 環境変数の設定
`.env`ファイルを作成し、必要な環境変数を設定:
```bash
# .envファイルの作成
touch .env

# 以下の内容を.envに追加
GEMINI_API_KEY=your_api_key_here  # Google Cloud ConsoleでGemini APIキーを取得
```

### 3.3 設定ファイルの編集
config/config.yamlを環境に合わせて編集:
- Tesseractのパスを設定
- 出力ディレクトリを設定
- ログレベルを設定

### 3.4 Gemini APIキーの取得
1. Google Cloud Consoleにアクセス (https://console.cloud.google.com)
2. プロジェクトを作成または選択
3. APIとサービス > 認証情報に移動
4. 認証情報を作成 > APIキー を選択
5. 作成されたAPIキーをコピーし、`.env`ファイルに設定

## 4. 動作確認

### 4.1 単体テストの実行
```bash
# テストの実行
python -m pytest tests/
```

### 4.2 サンプル動画での動作確認
```bash
# サンプル動画での処理実行
python src/main.py videos/Sample.mp4 --output output
```

## 5. トラブルシューティング

### 5.1 一般的な問題
1. MeCabの辞書が見つからない
   ```bash
   # macOS:
   brew install mecab-ipadic
   # Ubuntu:
   sudo apt install mecab-ipadic-utf8
   ```

2. Tesseractの言語ファイルが見つからない
   ```bash
   # macOS:
   brew install tesseract-lang
   # Ubuntu:
   sudo apt install tesseract-ocr-jpn
   ```

3. FFmpegのコーデックエラー
   ```bash
   # macOS:
   brew install ffmpeg --with-fdk-aac
   # Ubuntu:
   sudo apt install ubuntu-restricted-extras
   ```

### 5.2 環境固有の問題
- CUDA関連のエラー
  - NVIDIAドライバーの更新
  - CUDA Toolkitのインストール
- メモリ不足
  - スワップ領域の拡大
  - バッチサイズの調整

## 6. 開発環境の更新

### 6.1 依存関係の更新
```bash
# 依存関係の更新
pip install --upgrade -r requirements.txt
```

### 6.2 設定ファイルの更新
```bash
# 設定ファイルの更新確認
diff config/config.yaml config.yaml.example
```

## 7. バックアップと復元

### 7.1 設定のバックアップ
```bash
# 設定ファイルのバックアップ
cp config/config.yaml config/config.yaml.backup
```

### 7.2 環境の復元
```bash
# 仮想環境の再作成
rm -rf venv
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt