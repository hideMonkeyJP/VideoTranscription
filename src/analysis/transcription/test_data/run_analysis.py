import sys
import os
from pathlib import Path
import json

# TranscriptionProcessorをインポートできるようにパスを追加
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from analysis.transcription.transcription_processor import TranscriptionProcessor

def main():
    """音声解析を実行して結果を表示"""
    # 音声ファイルのパス
    audio_path = "output/audio/Sample.wav"
    
    if not os.path.exists(audio_path):
        print(f"エラー: 音声ファイルが見つかりません: {audio_path}")
        return
    
    # TranscriptionProcessorの初期化（mediumモデルを使用）
    processor = TranscriptionProcessor(model_name="medium")
    
    try:
        # 音声認識の実行
        print("音声認識を実行中...")
        result = processor.transcribe_audio(audio_path)
        
        # 結果の保存
        output_dir = Path("output/transcription")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "transcription.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n文字起こし結果を保存しました: {output_file}")
            
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()