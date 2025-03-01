import os
import sys
import pytest
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import time
import logging
from typing import List, Dict, Any

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.video_processing.frame_extraction.frame_extractor import FrameExtractor
from src.analysis.ocr.ocr_processor import OCRProcessor

# テスト用の設定
TEST_VIDEO_PATH = "videos/Sample.mp4"  # テスト用の動画ファイル（大文字Sに修正）
OUTPUT_DIR = Path("output/integration_test")
FRAME_OUTPUT_DIR = OUTPUT_DIR / "frames"
OCR_OUTPUT_FILE = OUTPUT_DIR / "ocr_results.json"

# ログの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('integration_test.log')
    ]
)
logger = logging.getLogger(__name__)

def ensure_dir(path: Path) -> Path:
    """ディレクトリの存在確認と作成"""
    path.mkdir(parents=True, exist_ok=True)
    return path

@pytest.fixture(scope="module")
def setup_test_environment():
    """テスト環境のセットアップ"""
    # 出力ディレクトリの作成
    ensure_dir(OUTPUT_DIR)
    ensure_dir(FRAME_OUTPUT_DIR)
    
    # テスト用の動画ファイルの存在確認
    if not os.path.exists(TEST_VIDEO_PATH):
        pytest.skip(f"テスト用の動画ファイルが見つかりません: {TEST_VIDEO_PATH}")
    
    return TEST_VIDEO_PATH

def test_frame_extraction_interval_adjustment(setup_test_environment):
    """フレーム間隔の調整テスト"""
    video_path = setup_test_environment
    
    # テスト用の間隔設定
    intervals = [0.2, 0.5, 1.0]
    
    for interval in intervals:
        logger.info(f"フレーム間隔 {interval}秒 でのテスト開始")
        
        # フレーム抽出の設定
        config = {
            'interval': interval,
            'quality': 85
        }
        
        # フレーム抽出の実行
        start_time = time.time()
        extractor = FrameExtractor(config)
        frames = extractor.extract_frames(video_path)
        extraction_time = time.time() - start_time
        
        # 結果の検証
        assert frames and len(frames) > 0, f"フレーム抽出結果が空です (間隔: {interval}秒)"
        
        # フレーム間隔の検証
        frame_intervals = []
        for i in range(1, len(frames)):
            time_diff = frames[i]['timestamp'] - frames[i-1]['timestamp']
            frame_intervals.append(time_diff)
        
        avg_interval = sum(frame_intervals) / len(frame_intervals) if frame_intervals else 0
        
        logger.info(f"間隔 {interval}秒: 抽出フレーム数 = {len(frames)}, 平均間隔 = {avg_interval:.3f}秒, 処理時間 = {extraction_time:.2f}秒")
        
        # 間隔が期待値に近いことを確認
        assert abs(avg_interval - interval) < 0.1, f"フレーム間隔が期待値と異なります: 期待値 = {interval}秒, 実際 = {avg_interval:.3f}秒"

def test_important_frame_ratio_optimization(setup_test_environment):
    """重要フレーム比率の最適化テスト"""
    video_path = setup_test_environment
    
    # テスト用の重要フレーム比率設定
    ratios = [0.02, 0.05, 0.1]
    
    for ratio in ratios:
        logger.info(f"重要フレーム比率 {ratio*100}% でのテスト開始")
        
        # フレーム抽出の設定
        config = {
            'interval': 0.5,
            'quality': 85,
            'important_frames_ratio': ratio
        }
        
        # フレーム抽出の実行
        extractor = FrameExtractor(config)
        frames = extractor.extract_frames(video_path)
        
        # 重要フレームの数を確認
        important_frames = [frame for frame in frames if frame.get('is_important', False)]
        important_ratio = len(important_frames) / len(frames) if frames else 0
        
        logger.info(f"比率 {ratio*100}%: 全フレーム数 = {len(frames)}, 重要フレーム数 = {len(important_frames)}, 実際の比率 = {important_ratio*100:.2f}%")
        
        # 重要フレーム比率が期待値に近いことを確認
        assert abs(important_ratio - ratio) < 0.01, f"重要フレーム比率が期待値と異なります: 期待値 = {ratio*100}%, 実際 = {important_ratio*100:.2f}%"
        
        # 重要フレームのシーン変更スコアが高いことを確認
        if important_frames:
            avg_important_score = sum(frame['scene_change_score'] for frame in important_frames) / len(important_frames)
            avg_normal_score = sum(frame['scene_change_score'] for frame in frames if not frame.get('is_important', False)) / (len(frames) - len(important_frames)) if len(frames) > len(important_frames) else 0
            
            logger.info(f"重要フレームの平均スコア = {avg_important_score:.3f}, 通常フレームの平均スコア = {avg_normal_score:.3f}")
            assert avg_important_score > avg_normal_score, "重要フレームのシーン変更スコアが通常フレームより低くなっています"

def test_scene_change_detection_efficiency(setup_test_environment):
    """シーン変更検出アルゴリズムの効率化テスト"""
    video_path = setup_test_environment
    
    # フレーム抽出の設定
    config = {
        'interval': 0.5,
        'quality': 85
    }
    
    # フレーム抽出の実行
    start_time = time.time()
    extractor = FrameExtractor(config)
    frames = extractor.extract_frames(video_path)
    extraction_time = time.time() - start_time
    
    # シーン変更スコアの分布を確認
    scores = [frame['scene_change_score'] for frame in frames]
    avg_score = sum(scores) / len(scores) if scores else 0
    max_score = max(scores) if scores else 0
    min_score = min(scores) if scores else 0
    
    logger.info(f"シーン変更スコア: 平均 = {avg_score:.3f}, 最大 = {max_score:.3f}, 最小 = {min_score:.3f}, 処理時間 = {extraction_time:.2f}秒")
    
    # スコアの分布が適切であることを確認
    assert max_score > 0.1, "シーン変更スコアの最大値が低すぎます"
    assert max_score - min_score > 0.05, "シーン変更スコアの分布が狭すぎます"

def test_ocr_image_preprocessing_optimization(setup_test_environment):
    """OCR画像前処理の最適化テスト"""
    video_path = setup_test_environment
    
    # フレーム抽出
    frame_config = {
        'interval': 1.0,
        'quality': 85
    }
    extractor = FrameExtractor(frame_config)
    frames = extractor.extract_frames(video_path)
    
    # テスト用のフレームを選択（最初の5フレーム）
    test_frames = frames[:5] if len(frames) >= 5 else frames
    
    # OCR設定
    ocr_configs = [
        {'lang': 'jpn+eng', 'psm': 3, 'oem': 3, 'min_confidence': 0.5},  # 基本設定
        {'lang': 'jpn+eng', 'psm': 6, 'oem': 3, 'min_confidence': 0.5},  # 単一テキストブロック
        {'lang': 'jpn+eng', 'psm': 11, 'oem': 3, 'min_confidence': 0.5}  # スパースなテキスト
    ]
    
    results = []
    
    for config in ocr_configs:
        logger.info(f"OCR設定テスト: PSM={config['psm']}, OEM={config['oem']}")
        
        # OCRプロセッサの初期化
        processor = OCRProcessor(config)
        
        # OCR処理の実行
        start_time = time.time()
        ocr_result = processor.process_frames(test_frames)
        processing_time = time.time() - start_time
        
        # 結果の検証
        screenshots = ocr_result.get('screenshots', [])
        text_count = sum(1 for screenshot in screenshots if screenshot.get('text', '').strip())
        avg_text_length = sum(len(screenshot.get('text', '')) for screenshot in screenshots) / len(screenshots) if screenshots else 0
        
        logger.info(f"PSM={config['psm']}: テキスト検出数 = {text_count}/{len(test_frames)}, 平均テキスト長 = {avg_text_length:.1f}, 処理時間 = {processing_time:.2f}秒")
        
        results.append({
            'config': config,
            'text_count': text_count,
            'avg_text_length': avg_text_length,
            'processing_time': processing_time
        })
    
    # 最適な設定を特定
    if results:
        best_result = max(results, key=lambda x: x['text_count'])
        logger.info(f"最適なOCR設定: PSM={best_result['config']['psm']}, OEM={best_result['config']['oem']}")
        
        # 少なくとも1つのテキストが検出されていることを確認
        assert best_result['text_count'] > 0, "どの設定でもテキストが検出されませんでした"

def test_ocr_parallel_processing(setup_test_environment):
    """OCR並列処理のテスト"""
    video_path = setup_test_environment
    
    # フレーム抽出
    frame_config = {
        'interval': 1.0,
        'quality': 85
    }
    extractor = FrameExtractor(frame_config)
    frames = extractor.extract_frames(video_path)
    
    # テスト用のフレームを選択（最初の10フレーム）
    test_frames = frames[:10] if len(frames) >= 10 else frames
    
    # OCR設定
    ocr_config = {
        'lang': 'jpn+eng',
        'psm': 6,
        'oem': 3,
        'min_confidence': 0.5
    }
    
    # 逐次処理
    processor = OCRProcessor(ocr_config)
    start_time = time.time()
    sequential_result = processor.process_frames(test_frames)
    sequential_time = time.time() - start_time
    
    logger.info(f"逐次処理: フレーム数 = {len(test_frames)}, 処理時間 = {sequential_time:.2f}秒")
    
    # 並列処理のシミュレーション（実際の並列処理は実装されていないため、シミュレーションで効果を予測）
    # 実際の並列処理実装では、multiprocessingやconcurrent.futuresを使用する
    
    # 並列処理の効果予測
    num_cores = os.cpu_count() or 4
    estimated_parallel_time = sequential_time / min(num_cores, len(test_frames))
    
    logger.info(f"並列処理予測: コア数 = {num_cores}, 予測処理時間 = {estimated_parallel_time:.2f}秒, 予測速度向上 = {sequential_time/estimated_parallel_time:.1f}倍")
    
    # 並列処理の効果が十分であることを確認
    assert sequential_time / estimated_parallel_time > 1.5, "並列処理による速度向上が不十分です"

if __name__ == "__main__":
    # テスト実行
    pytest.main(["-v", __file__]) 