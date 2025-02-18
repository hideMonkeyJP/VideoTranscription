import json
import os
from datetime import datetime

def load_intermediate_file(filepath):
    """中間ファイルを読み込む"""
    if not os.path.exists(filepath):
        print(f"ファイルが見つかりません: {filepath}")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_timestamps():
    """タイムスタンプベースの分析を実行"""
    # 中間ファイルの読み込み
    transcription = load_intermediate_file('output_test/main_test10/transcription.json')
    ocr_results = load_intermediate_file('output_test/main_test10/ocr_results.json')
    
    if not transcription or not ocr_results:
        return
    
    # シーンの構築
    scenes = []
    for screenshot in ocr_results.get('screenshots', []):
        scene_time = screenshot['timestamp']
        scene_text = screenshot.get('text', '')
        importance_score = screenshot.get('importance_score', 0)
        
        # このシーンに対応する音声セグメントを収集
        related_segments = []
        for segment in transcription:
            if (segment['start'] <= scene_time <= segment['end'] or
                (len(related_segments) == 0 and segment['end'] > scene_time)):
                related_segments.append(segment)
        
        if importance_score >= 6.0:  # 重要度閾値
            scenes.append({
                'timestamp': scene_time,
                'importance_score': importance_score,
                'screenshot_text': scene_text,
                'segments': related_segments
            })
    
    # 文脈の定義
    context_ranges = [
        (0.0, 7.9, "タスク管理が、稼げる人と稼げない人、できる人とできない人の大きな違いであることが最初に強調される。"),
        (7.9, 19.42, "多くの人を観察した結果、タスク管理のできているかどうかでその人の実力が見えるという点が述べられ、講演者自身も実践していることが分かる。"),
        (19.42, 33.28, "タスクを考える時間や、タスクの切り替えにかかる時間は無駄であり、これらの時間を削減することが重要であると説明される。"),
        (33.28, 43.42, "具体例として、SNS投稿後に次の行動を迷う時間を省くため、ノーションを活用したタスク管理の方法が提案される。"),
        (43.42, 53.72, "最後に、今後の詳細な資料の配布予定について触れ、次のステップへの案内が示される。")
    ]

    print("\n=== 文脈ごとの要約 ===")
    print("以下は、各文脈ごとにタイムスタンプを付けた要約です。\n")
    
    for i, (start, end, summary) in enumerate(context_ranges, 1):
        # この時間範囲に含まれるシーンを収集
        context_scenes = [
            scene for scene in scenes
            if start <= scene['timestamp'] <= end
        ]
        
        # この区間の音声セグメントを収集
        context_segments = [
            segment for segment in transcription
            if start <= segment['start'] <= end
        ]
        
        print(f"{i}. **[{start:.1f}秒 - {end:.1f}秒]**")
        print(f"   {summary}")
        print(f"   重要シーン数: {len(context_scenes)}")
        print(f"   音声セグメント数: {len(context_segments)}\n")

if __name__ == "__main__":
    analyze_timestamps()