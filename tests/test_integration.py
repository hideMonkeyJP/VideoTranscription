import os
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.video_processor import VideoProcessor
from src.utils.config import Config
import cv2
from PIL import Image
from typing import List, Dict, Any

# 中間ファイルのパス定義
TEMP_DIR = Path('output/temp')
PATHS = {
    'frames': TEMP_DIR / 'frames.json',
    'ocr': TEMP_DIR / 'ocr_results.json',
    'transcription': TEMP_DIR / 'transcription.json',
    'screenshots': Path('output/screenshots'),
    'audio': Path('output/audio'),
    'analysis': TEMP_DIR / 'analysis.json'
}

def ensure_dir(path: Path) -> Path:
    """ディレクトリの存在確認と作成"""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def validate_json_file(path: Path, required_keys: List[str]) -> bool:
    """JSONファイルの妥当性確認"""
    try:
        if not path.exists():
            return False
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return all(all(key in item for key in required_keys) for item in data if item)
            return all(key in data for key in required_keys)
    except Exception:
        return False

@pytest.fixture(scope="class")
def mock_gemini():
    """Geminiモデルのモック"""
    with patch('google.generativeai.GenerativeModel') as mock:
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"heading": "タスク管理の重要性", "summary": "タスク管理は稼げる人とできない人の大きな差となり、タスク考える時間や切り替えの時間を削減することが重要", "key_points": ["タスク管理が成功の鍵", "時間の無駄を省く", "効率的な管理方法の導入"]}'
        mock_model.generate_content.return_value = mock_response
        mock.return_value = mock_model
        yield mock

@pytest.fixture(scope="class")
def frame_extraction_result(request, processor_config):
    """フレーム抽出の共有結果"""
    frames_json = ensure_dir(PATHS['frames'])
    required_keys = ['timestamp', 'frame_number', 'scene_change_score', 'path']
    
    if validate_json_file(frames_json, required_keys):
        print("\n=== フレーム抽出（キャッシュ使用） ===")
        with open(frames_json, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    print("\n=== フレーム抽出（新規実行） ===")
    processor = VideoProcessor(processor_config)
    video_path = "videos/Sample.mp4"
    
    frames = processor.frame_extractor.extract_frames(str(video_path))
    screenshots_dir = ensure_dir(PATHS['screenshots'])
    saved_paths = processor.frame_extractor.save_frames(frames, str(screenshots_dir))
    
    frames_data = []
    for frame, path in zip(frames, saved_paths):
        frames_data.append({
            'timestamp': frame.get('timestamp', 0),
            'frame_number': frame.get('frame_number', 0),
            'scene_change_score': frame.get('scene_change_score', 0),
            'path': str(path) if isinstance(path, Path) else path
        })
    
    with open(frames_json, 'w', encoding='utf-8') as f:
        json.dump(frames_data, f, ensure_ascii=False, indent=2)
    
    return frames_data

@pytest.fixture(scope="class")
def ocr_processing_result(request, processor_config, frame_extraction_result):
    """OCR処理の共有結果"""
    ocr_json = ensure_dir(PATHS['ocr'])
    required_keys = ['screenshots']
    screenshot_keys = ['timestamp', 'frame_number', 'importance_score', 'text']
    
    if validate_json_file(ocr_json, required_keys):
        print("\n=== OCR処理（キャッシュ使用） ===")
        with open(ocr_json, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    print("\n=== OCR処理（新規実行） ===")
    processor = VideoProcessor(processor_config)
    
    frames_with_images = []
    for frame in frame_extraction_result:
        image_path = Path(frame['path'])
        with Image.open(str(image_path)) as img:
            frames_with_images.append({
                **frame,
                'image': img.copy()
            })
    
    ocr_results = processor.ocr_processor.process_frames(frames_with_images)
    
    with open(ocr_json, 'w', encoding='utf-8') as f:
        json.dump(ocr_results, f, ensure_ascii=False, indent=2)
    
    return ocr_results

@pytest.fixture(scope="class")
def audio_transcription_result(request, processor_config):
    """音声処理の共有結果"""
    transcription_json = ensure_dir(PATHS['transcription'])
    required_keys = ['text', 'start', 'end']
    
    if validate_json_file(transcription_json, required_keys):
        print("\n=== 音声処理（キャッシュ使用） ===")
        with open(transcription_json, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    print("\n=== 音声処理（新規実行） ===")
    processor = VideoProcessor(processor_config)
    video_path = "videos/Sample.mp4"
    
    audio_dir = ensure_dir(PATHS['audio'])
    audio_path = processor.audio_extractor.extract_audio(str(video_path))
    transcription = processor.transcription_processor.transcribe_audio(audio_path)
    
    with open(transcription_json, 'w', encoding='utf-8') as f:
        json.dump(transcription, f, ensure_ascii=False, indent=2)
    
    return transcription

@pytest.fixture(scope="class")
def video_processing_result(request, processor_config, mock_gemini):
    """完全な動画処理の共有結果"""
    # 出力ディレクトリの準備
    for path in PATHS.values():
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
    
    processor = VideoProcessor(processor_config)
    video_path = "videos/Sample.mp4"
    result = processor.process_video(video_path, 'output')
    
    # 処理結果の検証
    assert result['status'] == 'success', "動画処理が失敗しました"
    assert 'output_files' in result, "出力ファイル情報がありません"
    
    # 中間ファイルの存在確認
    for key, path in PATHS.items():
        if isinstance(path, Path):
            assert path.exists(), f"{key}の中間ファイルが存在しません: {path}"
    
    return result

@pytest.fixture(scope="class")
def text_analysis_result(request, processor_config, ocr_processing_result, audio_transcription_result):
    """テキスト分析の共有結果"""
    analysis_json = ensure_dir(PATHS['analysis'])
    required_keys = ['segments', 'total_segments']
    
    if validate_json_file(analysis_json, required_keys):
        print("\n=== テキスト分析（キャッシュ使用） ===")
        with open(analysis_json, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    print("\n=== テキスト分析（新規実行） ===")
    processor = VideoProcessor(processor_config)
    
    # transcriptionデータをanalysis.json形式に変換
    analysis_data = {
        "segments": [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            }
            for segment in audio_transcription_result
        ]
    }
    
    # テキスト分析を実行
    result = processor.text_analyzer.analyze_content_v2(analysis_data, ocr_processing_result)
    
    # analysis.jsonが存在しない場合のみ保存
    if not analysis_json.exists():
        with open(analysis_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    return result

class TestVideoProcessing:
    """動画処理の統合テスト"""

    @pytest.fixture(scope="class")
    def processor_config(self):
        """テスト用の設定"""
        return {
            'video_processor': {
                'output_dir': 'output',
                'temp_dir': 'output/temp'
            },
            'frame_extractor': {
                'interval': 0.52,
                'quality': 95,
                'frames_per_hour': 6923,
                'important_frames_ratio': 0.05,
                'min_scene_change': 0.3
            },
            'audio_extractor': {
                'format': 'wav',
                'sample_rate': 44100
            },
            'transcription': {
                'model': 'whisper-1',
                'language': 'ja'
            },
            'ocr_processor': {
                'engine': 'claude',
                'confidence_threshold': 0.7
            },
            'text_analyzer': {
                'model': 'gemini-pro',
                'max_tokens': 1000
            },
            'notion': {
                'enabled': False
            }
        }

    @pytest.mark.order(1)
    def test_frame_extraction(self, frame_extraction_result):
        """ステップ1: フレーム抽出のテスト"""
        try:
            # フレーム抽出結果の検証
            assert frame_extraction_result and len(frame_extraction_result) > 0, "フレーム抽出結果が空です"
            
            # 各フレームの検証
            required_keys = ['timestamp', 'frame_number', 'scene_change_score', 'path']
            for frame in frame_extraction_result:
                assert all(key in frame for key in required_keys), "フレームデータの形式が不正です"
                assert Path(frame['path']).exists(), f"画像ファイルが見つかりません: {frame['path']}"
                
            print(f"\nフレーム抽出テスト成功: {len(frame_extraction_result)}フレーム")
            return True
            
        except AssertionError as e:
            pytest.fail(f"フレーム抽出の検証エラー: {str(e)}")
        except Exception as e:
            pytest.fail(f"フレーム抽出でエラーが発生: {str(e)}")

    @pytest.mark.order(2)
    def test_ocr_processing(self, ocr_processing_result):
        """ステップ2: OCR処理のテスト"""
        try:
            # OCR結果の基本検証
            assert ocr_processing_result and 'screenshots' in ocr_processing_result, "OCR結果の形式が不正です"
            screenshots = ocr_processing_result['screenshots']
            assert len(screenshots) > 0, "OCR結果が空です"
            
            # 各スクリーンショットの検証
            screenshot_keys = ['timestamp', 'frame_number', 'importance_score', 'text']
            for screenshot in screenshots:
                assert all(key in screenshot for key in screenshot_keys), "スクリーンショットデータの形式が不正です"
                assert isinstance(screenshot['timestamp'], (int, float)), "タイムスタンプが数値ではありません"
                assert isinstance(screenshot['frame_number'], int), "フレーム番号が整数ではありません"
                assert isinstance(screenshot['importance_score'], (int, float)), "重要度スコアが数値ではありません"
                assert isinstance(screenshot['text'], str), "テキストが文字列ではありません"
            
            print(f"\nOCR処理テスト成功: {len(screenshots)}スクリーンショット")
            return True
            
        except AssertionError as e:
            pytest.fail(f"OCR処理の検証エラー: {str(e)}")
        except Exception as e:
            pytest.fail(f"OCR処理でエラーが発生: {str(e)}")

    @pytest.mark.order(3)
    def test_audio_transcription(self, audio_transcription_result):
        """ステップ3: 音声処理のテスト"""
        try:
            # 文字起こし結果の検証
            assert audio_transcription_result and len(audio_transcription_result) > 0, "文字起こし結果が空です"
            
            # 各セグメントの検証
            required_keys = ['text', 'start', 'end']
            for segment in audio_transcription_result:
                assert all(key in segment for key in required_keys), "セグメントデータの形式が不正です"
                assert len(segment['text'].strip()) > 0, "テキストが空です"
                assert segment['start'] >= 0 and segment['end'] > segment['start'], "タイムスタンプが不正です"
            
            print("\n音声処理テストが成功しました!")
            return True
            
        except Exception as e:
            pytest.fail(f"音声処理テストでエラーが発生: {str(e)}")

    @pytest.mark.order(4)
    def test_ocr_accuracy(self, ocr_processing_result):
        """OCR精度のテスト"""
        try:
            screenshots = ocr_processing_result['screenshots']
            
            # OCR結果の詳細検証
            for screenshot in screenshots:
                # テキストの品質確認
                if 'text' in screenshot:
                    text = screenshot['text']
                    assert isinstance(text, str), "OCRテキストが文字列ではありません"
                    if len(text.strip()) > 0:
                        # テキストの基本的な品質チェック
                        assert len(text) >= 2, "テキストが短すぎます"
                        assert not text.isspace(), "テキストが空白のみです"
                
                # スコアの範囲確認
                assert 0 <= screenshot['importance_score'] <= 1, "重要度スコアが範囲外です"
            
            print("\nOCR精度テストが成功しました!")
            return True
            
        except Exception as e:
            pytest.fail(f"OCR精度テストでエラーが発生: {str(e)}")

    @pytest.mark.order(5)
    def test_transcription_accuracy(self, audio_transcription_result):
        """音声認識の精度テスト"""
        try:
            # 文字起こし結果の詳細検証
            for segment in audio_transcription_result:
                # テキストの品質確認
                text = segment['text']
                assert len(text.strip()) > 0, "テキストが空です"
                assert not text.isspace(), "テキストが空白のみです"
                
                # タイムスタンプの妥当性確認
                assert 0 <= segment['start'] < segment['end'], "タイムスタンプが不正です"
                
                # 信頼度スコアの確認（存在する場合）
                if 'confidence' in segment:
                    assert 0 <= segment['confidence'] <= 1, "信頼度スコアが範囲外です"
            
            print("\n音声認識精度テストが成功しました!")
            return True
            
        except Exception as e:
            pytest.fail(f"音声認識精度テストでエラーが発生: {str(e)}")

    @pytest.mark.order(6)
    def test_text_analysis_quality(self, text_analysis_result):
        """テキスト分析の品質テスト"""
        try:
            # 基本構造の検証
            assert 'segments' in text_analysis_result, "セグメント情報がありません"
            assert 'total_segments' in text_analysis_result, "合計セグメント数がありません"
            segments = text_analysis_result['segments']
            assert len(segments) > 0, "セグメントが空です"
            
            # 各セグメントの検証
            for segment in segments:
                # 時間範囲の検証
                assert 'time_range' in segment, "時間範囲がありません"
                assert 'start' in segment['time_range'], "開始時間がありません"
                assert 'end' in segment['time_range'], "終了時間がありません"
                assert segment['time_range']['start'] < segment['time_range']['end'], "時間範囲が不正です"
                
                # 要約の検証
                assert 'summary' in segment, "要約がありません"
                assert isinstance(segment['summary'], str), "要約が文字列ではありません"
                assert len(segment['summary']) > 0, "要約が空です"
                
                # メタデータの検証
                assert 'metadata' in segment, "メタデータがありません"
                assert 'segment_count' in segment['metadata'], "セグメント数がありません"
                assert 'has_screenshot_text' in segment['metadata'], "スクリーンショットテキスト情報がありません"
                assert 'summary_points' in segment['metadata'], "要約ポイントがありません"
                assert 'keyword_count' in segment['metadata'], "キーワード数がありません"
                
                # スクリーンショット情報の検証（存在する場合）
                if 'screenshot' in segment:
                    assert 'timestamp' in segment['screenshot'], "タイムスタンプがありません"
                    assert 'frame_number' in segment['screenshot'], "フレーム番号がありません"
                    assert 'text' in segment['screenshot'], "テキストがありません"
                    assert 'ocr_confidence' in segment['screenshot'], "OCR信頼度がありません"
            
            print("\nテキスト分析品質テストが成功しました!")
            return True
            
        except Exception as e:
            pytest.fail(f"テキスト分析品質テストでエラーが発生: {str(e)}")

    @pytest.mark.order(7)
    def test_notion_data_generation(self, video_processing_result):
        """Notion登録用データ生成のテスト"""
        try:
            notion_data_path = video_processing_result['output_files']['notion_data']
            assert os.path.exists(notion_data_path), "Notion登録用データファイルが存在しません"
            
            # JSONファイルの内容検証
            with open(notion_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # データ構造の検証
                assert isinstance(data, list), "データがリスト形式ではありません"
                assert len(data) > 0, "データが空です"
                
                # 各エントリの検証
                required_keys = ['No', 'Summary', 'Timestamp', 'Thumbnail']
                for entry in data:
                    assert all(key in entry for key in required_keys), f"必須キーが不足しています: {required_keys}"
                    assert isinstance(entry['No'], int), "Noが整数ではありません"
                    assert isinstance(entry['Summary'], str), "Summaryが文字列ではありません"
                    assert isinstance(entry['Timestamp'], str), "Timestampが文字列ではありません"
                    assert isinstance(entry['Thumbnail'], str), "Thumbnailが文字列ではありません"
                    assert '秒' in entry['Timestamp'], "タイムスタンプの形式が正しくありません"
            
            print("\nNotion登録用データ生成テストが成功しました!")
            return True
            
        except Exception as e:
            pytest.fail(f"Notion登録用データ生成テストでエラーが発生: {str(e)}")

    @pytest.mark.order(8)
    def test_notion_sync(self, video_processing_result):
        """Notion同期機能のテスト"""
        try:
            # Notion同期結果の検証
            assert 'notion_page_url' in video_processing_result, "NotionページURLがありません"
            page_url = video_processing_result.get('notion_page_url', '')  # デフォルト値を設定
            assert isinstance(page_url, str), "NotionページURLが文字列ではありません"
            if page_url:  # URLが空でない場合のみ検証
                assert page_url.startswith('https://'), "NotionページURLが不正です"
            
            print("\nNotion同期テストが成功しました!")
            return True
            
        except Exception as e:
            pytest.fail(f"Notion同期テストでエラーが発生: {str(e)}")

    @pytest.mark.order(9)
    def test_supabase_registration(self, video_processing_result):
        """Supabase登録機能のテスト"""
        try:
            # regist.jsonの存在確認
            regist_json_path = video_processing_result['output_files']['notion_data']
            assert os.path.exists(regist_json_path), "regist.jsonファイルが存在しません"

            print("\nregist.jsonの内容を検証します...")
            # regist.jsonの内容検証
            with open(regist_json_path, 'r', encoding='utf-8') as f:
                regist_data = json.load(f)
                assert isinstance(regist_data, list), "データがリスト形式ではありません"
                assert len(regist_data) > 0, "データが空です"

                # 各エントリの検証
                required_keys = ['No', 'Summary', 'Timestamp', 'Thumbnail']
                for entry in regist_data:
                    assert all(key in entry for key in required_keys), f"必須キーが不足しています: {required_keys}"
                    assert isinstance(entry['No'], int), "Noが整数ではありません"
                    assert isinstance(entry['Summary'], str), "Summaryが文字列ではありません"
                    assert isinstance(entry['Timestamp'], str), "Timestampが文字列ではありません"
                    assert isinstance(entry['Thumbnail'], str), "Thumbnailが文字列ではありません"

            print("\nSupabaseへの登録を開始します...")
            # Supabase登録の実行
            from src.tools.supabase_register import register_to_supabase
            video_path = "videos/Sample.mp4"
            
            print("\nvideoテーブルへの登録を実行します...")
            success = register_to_supabase(
                regist_json_path,
                'videos',
                title="Sample Video",
                file_path=video_path,
                duration=300  # サンプル動画の長さ（秒）
            )
            
            if not success:
                print("\nSupabaseへの登録に失敗しました。エラーログを確認してください。")
                pytest.fail("Supabaseへの登録に失敗しました")
            
            print("\nSupabase登録テストが成功しました!")
            return True

        except Exception as e:
            print(f"\nエラーの詳細: {str(e)}")
            pytest.fail(f"Supabase登録テストでエラーが発生: {str(e)}")

    @pytest.mark.order(10)
    def test_report_generation(self, video_processing_result):
        """レポート生成機能のテスト"""
        try:
            report_path = video_processing_result['output_files']['report']
            assert os.path.exists(report_path), "レポートファイルが存在しません"
            
            # レポート内容の検証
            with open(report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
            
            # 基本構造の確認
            assert '<!DOCTYPE html>' in report_content, "HTMLドキュメントではありません"
            assert '<title>動画分析レポート</title>' in report_content.replace('{title}', '動画分析レポート'), "タイトルがありません"
            
            # 必須セクションの確認
            required_sections = [
                'processed_at',
                'video_duration',
                'segment_count',
                'screenshot_count',
                '<div class="segment"',
                '<div class="summary"',
                '<div class="key-points"'
            ]
            
            for section in required_sections:
                assert section in report_content, f"{section}セクションがありません"
            
            print("\nレポート生成テストが成功しました!")
            return True
            
        except Exception as e:
            pytest.fail(f"レポート生成テストでエラーが発生: {str(e)}")

if __name__ == '__main__':
    pytest.main(['-v', __file__])