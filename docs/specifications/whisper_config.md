# Whisperモデルの設定ガイド

## 概要

このドキュメントでは、VideoTranscriptionシステムで使用されるWhisperモデルの設定方法について説明します。Whisperは高精度な音声認識モデルで、様々なサイズとパラメータを調整することで、精度とパフォーマンスのバランスを取ることができます。

## 設定ファイル

Whisperモデルの設定は `config/config.yaml` ファイルの `models.whisper` セクションで定義されています：

```yaml
models:
  whisper:
    model:
      name: "base"  # tiny, base, small, medium, large
    device: "cpu"   # cuda, mps, cpu
    optimize_model: true  # TorchScriptによる最適化を有効にする
    use_optimized_model: true  # 最適化済みモデルを使用する
    cache_dir: "~/.cache/whisper"  # モデルキャッシュディレクトリ
    fp16: true  # 半精度浮動小数点を使用（高速化）
    split_long_audio: true  # 長い音声を分割して処理
    remove_silence: false  # 無音区間を除去
    transcribe_params:
      beam_size: 5  # ビームサイズ（小さいほど高速）
      best_of: 3  # 最良の結果を選ぶ候補数（小さいほど高速）
      patience: 1.0  # 探索の忍耐度（小さいほど高速）
    audio_segmentation:
      min_segment_length: 30  # 分割処理を行う最小音声長（秒）
      chunk_length: 30  # 分割するチャンクの長さ（秒）
      overlap: 2  # チャンク間のオーバーラップ（秒）
      overlap_threshold: 0.5  # セグメントをマージする重複率の閾値
```

## モデルサイズ

Whisperモデルには以下のサイズがあります：

| モデル名 | パラメータ数 | 必要メモリ | 精度 | 処理速度 | 推奨デバイス |
|---------|------------|-----------|------|---------|------------|
| tiny    | 39M        | <1GB      | 低   | 非常に高速 | CPU        |
| base    | 74M        | ~1GB      | 中低 | 高速     | CPU        |
| small   | 244M       | ~2GB      | 中   | 中速     | MPS/GPU    |
| medium  | 769M       | ~4GB      | 高   | やや遅い  | MPS/GPU    |
| large   | 1550M      | ~8GB      | 最高 | 遅い     | GPU        |

## デバイス設定

`device` パラメータで使用するハードウェアを指定できます：

- `"cpu"`: CPU処理（小さいモデルに推奨）
- `"mps"`: Apple Silicon GPU処理（M1/M2/M3 Macのみ）
- `"cuda"`: NVIDIA GPU処理（NVIDIA GPUを搭載したシステムのみ）

モデルサイズに応じた推奨デバイス：

- tiny/base: CPU
- small/medium: MPS (Apple Silicon) または CUDA (NVIDIA GPU)
- large: CUDA (NVIDIA GPU)

## 最適化設定

### モデル最適化

- `optimize_model`: TorchScriptによるモデル最適化を有効にします
- `use_optimized_model`: 最適化済みモデルを使用します（初回実行時に最適化）

### 精度設定

- `fp16`: 半精度浮動小数点を使用して処理を高速化します（GPUのみ）

### 音声分割処理

- `split_long_audio`: 長い音声を分割して処理します
- `audio_segmentation`: 音声分割のパラメータを設定します
  - `min_segment_length`: 分割処理を行う最小音声長（秒）
  - `chunk_length`: 分割するチャンクの長さ（秒）
  - `overlap`: チャンク間のオーバーラップ（秒）
  - `overlap_threshold`: セグメントをマージする重複率の閾値

### 文字起こしパラメータ

- `transcribe_params`: 文字起こし処理のパラメータを設定します
  - `beam_size`: ビームサーチの幅（大きいほど精度が上がるが遅くなる）
  - `best_of`: 生成する候補の数
  - `patience`: ビーム探索の打ち切り係数

## パフォーマンスチューニング

### 高速化のためのヒント

1. 小さいモデル（tiny, base）を使用する
2. `beam_size` と `best_of` の値を小さくする
3. 適切なデバイスを選択する（小さいモデルはCPU、大きいモデルはGPU）
4. `fp16` を有効にする（GPUのみ）

### 精度向上のためのヒント

1. 大きいモデル（medium, large）を使用する
2. `beam_size` と `best_of` の値を大きくする
3. `patience` の値を大きくする
4. `temperature` を0に設定する（決定的な出力）

## 使用例

### 高速処理優先の設定

```yaml
models:
  whisper:
    model:
      name: "base"
    device: "cpu"
    transcribe_params:
      beam_size: 3
      best_of: 1
      patience: 0.8
```

### 精度優先の設定

```yaml
models:
  whisper:
    model:
      name: "medium"
    device: "mps"  # または "cuda"
    transcribe_params:
      beam_size: 8
      best_of: 5
      patience: 1.5
```

## トラブルシューティング

- **メモリ不足エラー**: モデルサイズを小さくするか、`chunk_length` を小さくしてください
- **処理が遅い**: 小さいモデルを使用するか、`beam_size` と `best_of` の値を小さくしてください
- **精度が低い**: 大きいモデルを使用するか、`beam_size` と `best_of` の値を大きくしてください 