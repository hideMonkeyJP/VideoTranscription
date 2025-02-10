from audio_extraction.audio_extractor import AudioExtractor
import os

def main():
    # 入力動画ファイルのパス
    video_path = "videos/Sample.mp4"
    
    # 出力ディレクトリの作成
    output_dir = "output/audio"
    os.makedirs(output_dir, exist_ok=True)
    
    # 出力ファイルのパス
    output_path = os.path.join(output_dir, "Sample.wav")
    
    # AudioExtractorの設定
    config = {
        "format": "wav",
        "sample_rate": 16000
    }
    
    # 音声抽出の実行
    extractor = AudioExtractor(config)
    try:
        extractor.extract_audio(video_path, output_path)
        print(f"音声抽出が完了しました: {output_path}")
        
        # 音声ファイルの情報を表示
        audio_info = extractor.get_audio_info(output_path)
        print("\n音声ファイル情報:")
        print(f"チャンネル数: {audio_info['channels']}")
        print(f"サンプル幅: {audio_info['sample_width']} bits")
        print(f"フレームレート: {audio_info['frame_rate']} Hz")
        print(f"長さ: {audio_info['duration']:.2f} 秒")
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main() 