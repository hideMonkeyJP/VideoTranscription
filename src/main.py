import sys
import os
from video_processor import VideoProcessor
import json

def main():
    try:
        if len(sys.argv) != 2:
            print("使用方法: python src/main.py <動画ファイルのパス>")
            sys.exit(1)

        video_path = sys.argv[1]
        if not os.path.exists(video_path):
            print(f"エラー: 指定された動画ファイル '{video_path}' が見つかりません。")
            sys.exit(1)

        processor = VideoProcessor()
        print("ビデオの処理を開始します...")
        result = processor.process_video(video_path)
        print("処理が完了しました。")
        
        # 結果の保存
        output_path = os.path.join('output', 'result.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"結果を {output_path} に保存しました。")

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        import traceback
        print("詳細なエラー情報:")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 