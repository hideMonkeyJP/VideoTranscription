import os
import sys
import time
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = str(Path(__file__).parent.absolute())
sys.path.insert(0, project_root)

from src.video_processor import VideoProcessor

def test_intermediate_files():
    # 既存のoutputディレクトリを使用
    output_dir = "output"
    video_path = "videos/Sample.mp4"
    
    # 中間ファイルのパスを確認
    intermediate_files = {
        'transcription': os.path.join(output_dir, 'transcription.json'),
        'ocr': os.path.join(output_dir, 'ocr_results.json'),
        'summaries': os.path.join(output_dir, 'summaries.json')
    }
    
    # 中間ファイルが存在するか確認
    has_intermediate = all(os.path.exists(f) for f in intermediate_files.values())
    
    # VideoProcessorのインスタンス化
    processor = VideoProcessor()
    
    print(f"\n=== 1回目の実行({'中間ファイルあり' if has_intermediate else '中間ファイルなし'})===")
    start_time = time.time()
    result1 = processor.process_video(video_path, output_dir, reuse_intermediate=True)
    processing_time1 = time.time() - start_time
    print(f"処理時間: {processing_time1:.2f}秒")
    
    # 中間ファイルの確認
    print("\n生成された中間ファイル:")
    for file_type, file_path in intermediate_files.items():
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024  # KB単位
            print(f"- {file_type}: {os.path.basename(file_path)} ({file_size:.2f}KB)")
    
    # 2回目の実行(必ず中間ファイルを再利用)
    print("\n=== 2回目の実行(中間ファイル再利用)===")
    start_time = time.time()
    result2 = processor.process_video(video_path, output_dir, reuse_intermediate=True)
    processing_time2 = time.time() - start_time
    print(f"処理時間: {processing_time2:.2f}秒")
    
    # 中間ファイルの最終状態を確認
    print("\n最終的な中間ファイル:")
    for file_type, file_path in intermediate_files.items():
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024  # KB単位
            print(f"- {file_type}: {os.path.basename(file_path)} ({file_size:.2f}KB)")
    
    # 処理時間の比較
    if processing_time1 > 0:  # 0除算を防ぐ
        time_reduction = ((processing_time1 - processing_time2) / processing_time1) * 100
        print(f"\n処理時間の削減: {time_reduction:.1f}%")
    
    return {
        'first_run_time': processing_time1,
        'second_run_time': processing_time2,
        'time_reduction_percent': time_reduction if processing_time1 > 0 else 0,
        'intermediate_files': intermediate_files,
        'had_intermediate_files': has_intermediate
    }

if __name__ == "__main__":
    test_intermediate_files()