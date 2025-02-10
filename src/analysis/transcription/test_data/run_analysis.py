import sys
import os
from pathlib import Path
import json

# TranscriptionProcessorをインポートできるようにパスを追加
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from analysis.transcription import TranscriptionProcessor

def main():
    """音声解析を実行して結果を表示"""
    # 音声ファイルのパス
    audio_path = "/Users/takayanagihidenori/Cursor/VideoTranscription/output/audio/Sample.wav"
    
    if not os.path.exists(audio_path):
        print(f"エラー: 音声ファイルが見つかりません: {audio_path}")
        return
    
    # TranscriptionProcessorの初期化（mediumモデルを使用）
    processor = TranscriptionProcessor(model_name="medium")
    
    try:
        # 音声認識の実行
        print("音声認識を実行中...")
        result = processor.transcribe_audio(audio_path)
        
        # 結果の表示
        print("\n=== 音声認識の完全な結果 ===")
        print("1. 完全なテキスト:")
        print("-" * 80)
        print(result['text'])
        print("-" * 80)
        
        print("\n2. セグメント詳細:")
        print("-" * 80)
        for i, segment in enumerate(result.get("segments", []), 1):
            print(f"\nセグメント {i}:")
            print(f"テキスト: {segment.get('text', '')}")
            print(f"開始時間: {segment.get('start', 0):.2f}秒")
            print(f"終了時間: {segment.get('end', 0):.2f}秒")
            print(f"ID: {segment.get('id', 'N/A')}")
            if 'seek' in segment:
                print(f"シーク位置: {segment.get('seek', 0)}")
            if 'temperature' in segment:
                print(f"温度: {segment.get('temperature', 0)}")
            if 'avg_logprob' in segment:
                print(f"平均対数確率: {segment.get('avg_logprob', 0):.4f}")
            if 'compression_ratio' in segment:
                print(f"圧縮率: {segment.get('compression_ratio', 0):.4f}")
            if 'no_speech_prob' in segment:
                print(f"無音確率: {segment.get('no_speech_prob', 0):.4f}")
            
            print("\n単語詳細:")
            for word in segment.get("words", []):
                print(f"単語: {word.get('word', '')}")
                print(f"  開始時間: {word.get('start', 0):.2f}秒")
                print(f"  終了時間: {word.get('end', 0):.2f}秒")
                print(f"  確信度: {word.get('probability', 0):.4f}")
        
        print("\n3. 言語検出結果:")
        print("-" * 80)
        print(f"検出された言語: {result.get('language', 'N/A')}")
        
        print("\n4. 認識パラメータ:")
        print("-" * 80)
        print(f"タスク: {result.get('task', 'N/A')}")
        if 'duration' in result:
            print(f"音声の長さ: {result.get('duration', 0):.2f}秒")
        
        # 結果をJSONファイルとしても保存
        output_dir = Path(audio_path).parent / "analysis_results"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "transcription_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n詳細な結果をJSONファイルとして保存しました: {output_file}")
            
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()