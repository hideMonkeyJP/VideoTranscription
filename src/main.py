import os
import sys
import argparse
import logging
import shutil
import json
from pathlib import Path
import cv2
from PIL import Image

# プロジェクトルートをPYTHONPATHに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.video_processor import VideoProcessor
from src.utils.config import Config

def setup_logging(output_dir: Path) -> logging.Logger:
    """ロギングの設定を行います"""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'video_processing.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s - (%(filename)s:%(lineno)d)',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

def check_output_directory(output_dir: Path, force: bool, logger: logging.Logger) -> None:
    """出力ディレクトリの状態をチェックします"""
    if output_dir.exists() and force:
        logger.info(f"出力ディレクトリをクリーンアップします: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

def get_video_duration(video_path: str) -> int:
    """動画の長さ（秒）を取得します"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("動画ファイルを開けません")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = int(frame_count / fps)
        
        cap.release()
        return duration
    except Exception as e:
        raise Exception(f"動画の長さの取得に失敗: {e}")

def save_regist_data(analysis_result: dict, output_dir: Path, logger: logging.Logger) -> None:
    """Notion/Supabase登録用のデータを生成して保存します"""
    try:
        # analysis.jsonを一時的に保存
        analysis_json = output_dir / 'analysis.json'
        with open(analysis_json, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)

        # VideoProcessorを使用してNotion登録用データを生成
        config = {
            'video_processor': {
                'output_dir': str(output_dir),
                'temp_dir': str(output_dir / 'temp')
            }
        }
        processor = VideoProcessor(config=config)
        
        regist_json = output_dir / 'regist.json'
        success = processor.generate_notion_data(str(analysis_json), str(regist_json))
        
        if success:
            logger.info(f"Notion/Supabase登録用データを保存しました: {regist_json}")
        else:
            logger.error("Notion/Supabase登録用データの生成に失敗しました")
            raise Exception("データ生成に失敗")
            
    except Exception as e:
        logger.error(f"Notion/Supabase登録用データの保存に失敗: {e}")
        raise

def register_to_supabase(video_path: Path, output_dir: Path, logger: logging.Logger) -> bool:
    """Supabaseにデータを登録します"""
    try:
        from src.tools.supabase_register import register_to_supabase as supabase_register
        
        regist_json = output_dir / 'regist.json'
        if not regist_json.exists():
            logger.error("regist.jsonが見つかりません")
            return False

        # 動画の長さを取得
        try:
            duration = get_video_duration(str(video_path))
            logger.info(f"動画の長さ: {duration}秒")
        except Exception as e:
            logger.error(f"動画の長さの取得に失敗: {e}")
            return False

        # Supabaseに登録
        success = supabase_register(
            str(regist_json),
            'videos',
            title=video_path.stem,
            file_path=str(video_path),
            duration=duration
        )
        
        if success:
            logger.info("Supabaseへの登録が完了しました")
        else:
            logger.error("Supabaseへの登録に失敗しました")
        
        return success
    except Exception as e:
        logger.error(f"Supabase登録中にエラーが発生: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='動画から文字起こしと要約を生成します')
    parser.add_argument('video_path', help='処理する動画ファイルのパス')
    parser.add_argument('--output', '-o', default='output_test', help='出力ディレクトリ')
    parser.add_argument('--force', '-f', action='store_true', help='既存のファイルを上書きする')
    parser.add_argument('--language', '-l', help='言語コード (ja/en)', default='ja')
    parser.add_argument('--notion', '-n', help='Notionに同期する', action='store_true')
    parser.add_argument('--supabase', '-s', help='Supabaseに登録する', action='store_true')
    parser.add_argument('--use-v2', help='テキスト分析にV2アルゴリズムを使用する（デフォルトはV3の一括要約処理）', action='store_true')
    args = parser.parse_args()

    try:
        output_dir = Path(args.output)
        logger = setup_logging(output_dir)
        logger.info(f"処理を開始します - 出力ディレクトリ: {output_dir}")

        # 入力ファイルの検証
        video_path = Path(args.video_path)
        if not video_path.exists():
            logger.error(f"エラー: 指定された動画ファイル '{video_path}' が見つかりません。")
            sys.exit(1)

        # 出力ディレクトリの準備
        check_output_directory(output_dir, args.force, logger)

        # 設定ファイルから設定を読み込む
        config_obj = Config('config/config.yaml')
        config = config_obj.get_all()
        
        # 出力ディレクトリの設定を上書き
        if 'video_processor' not in config:
            config['video_processor'] = {}
        config['video_processor']['output_dir'] = str(output_dir)
        config['video_processor']['temp_dir'] = str(output_dir / 'temp')

        # VideoProcessorの初期化
        processor = VideoProcessor(config=config)

        # 必要なディレクトリの作成
        temp_dir = output_dir / 'temp'
        temp_dir.mkdir(parents=True, exist_ok=True)
        screenshots_dir = output_dir / 'screenshots'
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        audio_dir = output_dir / 'audio'
        audio_dir.mkdir(parents=True, exist_ok=True)

        # ステップ1: フレーム抽出
        logger.info("フレーム抽出を開始します...")
        frames = processor.frame_extractor.extract_frames(str(video_path))
        saved_paths = processor.frame_extractor.save_frames(frames, str(screenshots_dir))
        
        # フレームデータをJSONファイルに保存
        frames_data = []
        for frame, path in zip(frames, saved_paths):
            frames_data.append({
                'timestamp': frame.get('timestamp', 0),
                'frame_number': frame.get('frame_number', 0),
                'scene_change_score': frame.get('scene_change_score', 0),
                'path': str(path) if isinstance(path, Path) else path
            })
        
        frames_json = temp_dir / 'frames.json'
        with open(frames_json, 'w', encoding='utf-8') as f:
            json.dump(frames_data, f, ensure_ascii=False, indent=2)
        logger.info(f"フレーム抽出が完了しました: {len(frames_data)}フレーム")

        # ステップ2: OCR処理
        logger.info("OCR処理を開始します...")
        frames_with_images = []
        for frame in frames_data:
            image_path = Path(frame['path'])
            with Image.open(str(image_path)) as img:
                frames_with_images.append({
                    **frame,
                    'image': img.copy()
                })
        
        ocr_results = processor.ocr_processor.process_frames(frames_with_images)
        
        ocr_json = temp_dir / 'ocr_results.json'
        with open(ocr_json, 'w', encoding='utf-8') as f:
            json.dump(ocr_results, f, ensure_ascii=False, indent=2)
        logger.info("OCR処理が完了しました")

        # ステップ3: 音声処理
        logger.info("音声処理を開始します...")
        audio_path = processor.audio_extractor.extract_audio(str(video_path))
        transcription = processor.transcription_processor.transcribe_audio(audio_path)
        
        transcription_json = temp_dir / 'transcription.json'
        with open(transcription_json, 'w', encoding='utf-8') as f:
            json.dump(transcription, f, ensure_ascii=False, indent=2)
        logger.info("音声処理が完了しました")

        # ステップ4: テキスト分析
        logger.info("テキスト分析を開始します...")
        analysis_data = {
            "segments": [
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"]
                }
                for segment in transcription
            ]
        }
        
        analysis_result = processor.text_analyzer.analyze_content_v2(analysis_data, ocr_results)
        
        analysis_json = temp_dir / 'analysis.json'
        with open(analysis_json, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        logger.info("テキスト分析が完了しました")

        # ステップ5: Notion登録用データの生成とSupabase登録
        logger.info("Notion登録用データの生成を開始します...")
        notion_data = save_regist_data(analysis_result, output_dir, logger)
        regist_json = output_dir / 'regist.json'

        # Supabaseへの登録
        logger.info("Supabaseへの登録を開始します...")
        duration = get_video_duration(str(video_path))
        supabase_success = register_to_supabase(video_path, output_dir, logger)
        
        if supabase_success:
            logger.info("Supabaseへの登録が完了しました")
        else:
            logger.warning("Supabaseへの登録に失敗しました")

        return {
            'status': 'success',
            'output_files': {
                'frames': str(frames_json),
                'ocr': str(ocr_json),
                'transcription': str(transcription_json),
                'analysis': str(analysis_json),
                'notion_data': str(regist_json)
            }
        }

    except Exception as e:
        logger.error(f"エラーが発生しました: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 