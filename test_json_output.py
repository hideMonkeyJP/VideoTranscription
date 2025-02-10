import json
import os
from datetime import datetime
from src.video_processor import VideoProcessor

def test_json_output():
    """新しいJSON出力形式をテスト"""
    # VideoProcessorのインスタンスを作成
    processor = VideoProcessor()
    
    # テスト用の出力ディレクトリ
    output_dir = "output_test/json_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # テスト用の動画を処理
    video_path = "videos/Sample.mp4"
    result = processor.process_video(video_path, output_dir)
    
    # 生成されたJSONファイルを読み込んで内容を確認
    with open(result['json_path'], 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 結果の表示
    print("=== JSON出力テスト ===")
    print(f"文脈数: {len(data['contexts'])}")
    
    for i, context in enumerate(data['contexts'], 1):
        print(f"\n文脈 {i}:")
        print(f"時間範囲: {context['time_range']['start']:.1f}秒 - {context['time_range']['end']:.1f}秒")
        print(f"要約: {context['summary']}")
        print(f"シーン数: {len(context['scenes'])}")
        print(f"セグメント数: {len(context['segments'])}")
        print(f"スクリーンショット数: {len(context['screenshots'])}")
    
    print("\nメタデータ:")
    for key, value in data['metadata'].items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    test_json_output()