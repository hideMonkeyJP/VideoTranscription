import sys
import os
from pathlib import Path
import json
from PIL import Image
import cv2

# OCRProcessorをインポートできるようにパスを追加
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from analysis.ocr import OCRProcessor
from video_processing.frame_extraction import FrameExtractor

def main():
    """動画からフレームを抽出してOCR処理を実行"""
    # 動画ファイルのパス
    video_path = "/Users/takayanagihidenori/Cursor/VideoTranscription/videos/Sample.mp4"
    
    if not os.path.exists(video_path):
        print(f"エラー: 動画ファイルが見つかりません: {video_path}")
        return
        
    # 出力ディレクトリの作成
    output_dir = Path(__file__).parent.parent.parent.parent / "output" / "ocr_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("フレーム抽出を開始...")
        # フレーム抽出の設定
        frame_config = {
            'interval': 1.0,  # 1秒間隔
            'quality': 95
        }
        frame_extractor = FrameExtractor(frame_config)
        
        # フレームの抽出
        frames = frame_extractor.extract_frames(video_path)
        print(f"抽出されたフレーム数: {len(frames)}")
        
        # OCRプロセッサーの設定
        ocr_config = {
            'lang': 'jpn+eng',  # 日本語と英語
            'psm': 3,
            'oem': 3,
            'min_confidence': 60.0  # 信頼度閾値
        }
        processor = OCRProcessor(ocr_config)
        
        print("\nOCR処理を開始...")
        # OCR処理の実行
        results = processor.process_frames(frames)
        
        # 結果の表示と保存
        print("\n=== OCR処理結果 ===")
        for result in results:
            frame_number = result.get('frame_number', 'unknown')
            timestamp = result.get('timestamp', 0.0)
            texts = result.get('texts', [])
            
            if texts:
                print(f"\nフレーム {frame_number} (タイムスタンプ: {timestamp:.1f}秒):")
                for text_info in texts:
                    confidence = text_info.get('confidence', 0.0)
                    text = text_info.get('text', '')
                    print(f"- {text} (信頼度: {confidence:.1f}%)")
        
        # 結果をJSONファイルとして保存
        output_file = output_dir / "ocr_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n詳細な結果をJSONファイルとして保存しました: {output_file}")
        
        # フレーム画像も保存
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        print("\nフレーム画像を保存中...")
        for i, frame in enumerate(frames):
            image_path = frames_dir / f"frame_{i:03d}.jpg"
            frame['image'].save(image_path, 'JPEG', quality=95)
            
        print(f"フレーム画像を保存しました: {frames_dir}")
            
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main() 