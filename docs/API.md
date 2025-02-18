# VideoTranscription API ドキュメント

## VideoProcessor

動画処理の主要クラス。動画の分析と結果の生成を行います。

### 初期化

```python
from src.video_processor import VideoProcessor

processor = VideoProcessor(config: Dict[str, Any] = None)
```

#### パラメータ
- `config` (Dict[str, Any], optional): 設定辞書
  - `output_dir` (str): 出力ディレクトリ
  - `frame_extractor_config` (Dict): フレーム抽出の設定
  - `audio_extractor_config` (Dict): 音声抽出の設定
  - `ocr_config` (Dict): OCR処理の設定
  - `text_analyzer_config` (Dict): テキスト分析の設定

### メソッド

#### process_video
```python
def process_video(self, video_path: str) -> Dict[str, Any]
```

動画を処理し、分析結果を生成します。

##### パラメータ
- `video_path` (str): 処理する動画のパス

##### 戻り値
- Dict[str, Any]: 処理結果
  - `status`: 処理状態
  - `report_path`: 生成されたレポートのパス
  - `analysis_results`: 分析結果

## FrameExtractor

動画からフレームを抽出するクラス。

### 初期化

```python
from src.video_processing.frame_extractor import FrameExtractor

extractor = FrameExtractor(config: Dict[str, Any] = None)
```

#### パラメータ
- `config` (Dict[str, Any], optional): 設定辞書
  - `output_dir` (str): 出力ディレクトリ
  - `frame_interval` (int): フレーム抽出間隔（秒）
  - `min_scene_change` (float): シーン変更の閾値

### メソッド

#### extract_frames
```python
def extract_frames(self, video_path: str) -> List[Dict[str, Any]]
```

動画からフレームを抽出します。

##### パラメータ
- `video_path` (str): 動画ファイルのパス

##### 戻り値
- List[Dict[str, Any]]: 抽出されたフレーム情報のリスト
  - `path`: フレーム画像のパス
  - `timestamp`: タイムスタンプ
  - `frame_number`: フレーム番号

## AudioExtractor

動画から音声を抽出し、文字起こしを行うクラス。

### 初期化

```python
from src.video_processing.audio_extractor import AudioExtractor

extractor = AudioExtractor(config: Dict[str, Any] = None)
```

#### パラメータ
- `config` (Dict[str, Any], optional): 設定辞書
  - `model_name` (str): Whisperモデル名
  - `output_dir` (str): 出力ディレクトリ
  - `language` (str): 文字起こし言語

### メソッド

#### extract_audio
```python
def extract_audio(self, video_path: str) -> str
```

動画から音声を抽出します。

##### パラメータ
- `video_path` (str): 動画ファイルのパス

##### 戻り値
- str: 抽出された音声ファイルのパス

#### transcribe_audio
```python
def transcribe_audio(self, audio_path: str) -> Dict[str, Any]
```

音声ファイルを文字起こしします。

##### パラメータ
- `audio_path` (str): 音声ファイルのパス

##### 戻り値
- Dict[str, Any]: 文字起こし結果
  - `text`: 文字起こしテキスト
  - `segments`: セグメント情報
  - `language`: 検出された言語

## TextAnalyzer

テキストの分析を行うクラス。

### 初期化

```python
from src.analysis.text_analyzer import TextAnalyzer

analyzer = TextAnalyzer(config: Dict[str, Any] = None)
```

#### パラメータ
- `config` (Dict[str, Any], optional): 設定辞書
  - `model_name` (str): spacyモデル名
  - `min_segment_length` (int): 最小セグメント長
  - `similarity_threshold` (float): 類似度閾値

### メソッド

#### analyze_content
```python
def analyze_content(self, transcription: Dict[str, Any], ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]
```

コンテンツを分析します。

##### パラメータ
- `transcription` (Dict[str, Any]): 文字起こし結果
- `ocr_results` (List[Dict[str, Any]]): OCR結果のリスト

##### 戻り値
- Dict[str, Any]: 分析結果
  - `segments`: 処理されたセグメント
  - `keywords`: 抽出されたキーワード
  - `metadata`: メタデータ

## ReportGenerator

HTMLレポートを生成するクラス。

### 初期化

```python
from src.output.report_generator import ReportGenerator

generator = ReportGenerator(template_dir: str = 'templates')
```

#### パラメータ
- `template_dir` (str): テンプレートファイルのディレクトリパス

### メソッド

#### generate_report
```python
def generate_report(self, data: Dict[str, Any], output_path: str) -> str
```

HTMLレポートを生成します。

##### パラメータ
- `data` (Dict[str, Any]): レポートに含めるデータ
- `output_path` (str): 出力先のパス

##### 戻り値
- str: 生成されたHTMLレポートのパス 