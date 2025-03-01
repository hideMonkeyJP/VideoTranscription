# Whisperモデルのデバイス最適化

## 概要

このドキュメントでは、Whisperモデルのデバイス最適化に関する変更内容と設定方法について説明します。Apple Silicon（M1/M2/M3）搭載のMacでの音声文字起こし処理の最適化を中心に解説します。

## 変更内容

### デバイス選択ロジックの改善

`TranscriptionProcessor`クラスの`_get_device`メソッドを修正し、設定ファイルからデバイス設定を取得できるようにしました。これにより、モデルサイズに応じて最適なデバイスを選択できるようになりました。

```python
def _get_device(self):
    """最適なデバイスを選択します"""
    # 設定からデバイスを取得
    device = self.config.get('models', {}).get('whisper', {}).get('device')
    if not device:
        # トップレベルのデバイス設定を確認
        device = self.config.get('device')
    
    # デバイスが指定されていない場合は自動検出
    if not device:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    self.logger.info(f"使用デバイス: {device}")
    return device
```

### モデルサイズとデバイスの最適な組み合わせ

ベンチマークテストの結果、以下の最適な組み合わせが判明しました：

- **小さいモデル（tiny, base）**: CPUの方が高速
- **大きいモデル（small, medium, large）**: MPSデバイス（Apple Silicon GPU）の方が高速

### 設定ファイルの変更

`config/config.yaml`ファイルにデバイス設定を追加しました：

```yaml
models:
  whisper:
    model:
      name: "base"  # tiny, base, small, medium, large
    device: "cpu"   # cuda, mps, cpu
```

## パフォーマンス比較

### baseモデルの処理時間

- **CPU**: 約17秒
- **MPS**: 約34秒

### mediumモデルの処理時間

- **CPU**: 約300秒
- **MPS**: 約143秒

## 推奨設定

モデルサイズに応じて以下の設定を推奨します：

1. **tiny/baseモデル**:
   ```yaml
   models:
     whisper:
       model:
         name: "base"
       device: "cpu"
   ```

2. **small/medium/largeモデル**:
   ```yaml
   models:
     whisper:
       model:
         name: "medium"
       device: "mps"
   ```

## 注意事項

- デバイス設定は、使用するハードウェアによって最適な値が異なります
- 大きなモデルをCPUで実行すると、処理時間が大幅に増加する可能性があります
- MPSデバイスは、Apple Silicon搭載のMacでのみ使用可能です

## トラブルシューティング

### MPSデバイスが認識されない場合

1. PyTorchのバージョンが2.0以上であることを確認してください
2. `torch.backends.mps.is_available()`が`True`を返すことを確認してください
3. 最新のmacOSにアップデートすることで解決する場合があります

### CPUモードでのパフォーマンス低下

1. 他のCPU負荷の高いプロセスを終了してください
2. 小さいモデル（tiny, base）を使用してください
3. 音声ファイルを短いセグメントに分割して処理することを検討してください 