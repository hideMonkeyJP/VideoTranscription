import sys
import os
from pathlib import Path
import json
from PIL import Image
import cv2

# OCRProcessorをインポートできるようにパスを追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.analysis.ocr import OCRProcessor
from src.video_processing.frame_extraction import FrameExtractor

def main():
    """動画からフレームを抽出してOCR処理を実行"""
    # 入力ディレクトリのパス
    input_dir = "/Users/takayanagihidenori/Cursor/VideoTranscription/output_test/notion_test/screenshots_20250209_160200"
    
    if not os.path.exists(input_dir):
        print(f"エラー: 入力ディレクトリが見つかりません: {input_dir}")
        return
        
    # 出力ディレクトリの作成
    output_dir = Path(__file__).parent.parent.parent.parent / "output" / "ocr_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("画像の読み込みを開始...")
        # 画像ファイルの読み込み
        frames = []
        for file in sorted(os.listdir(input_dir)):
            if file.endswith('.png'):
                image_path = os.path.join(input_dir, file)
                image = Image.open(image_path)
                frame_number = int(file.split('_')[1].split('.')[0])
                frames.append({
                    'image': image,
                    'frame_number': frame_number,
                    'timestamp': frame_number * 0.52,  # 0.52秒間隔
                    'importance_score': 1.0  # デフォルトスコア
                })
        print(f"読み込まれた画像数: {len(frames)}")
        
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
        for screenshot in results['screenshots']:
            frame_number = screenshot.get('frame_number', 'unknown')
            timestamp = screenshot.get('timestamp', 0.0)
            texts = screenshot.get('texts', [])
            
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
            
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main() 