"""
動画処理の統合モジュール
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json
import cv2
from PIL import Image
import numpy as np
from .utils.performance_monitor import PerformanceMonitor
from jinja2 import Template
import torch

from .utils.config import Config
from .utils.logger import Logger
from .video_processing.frame_extraction import FrameExtractor
from .video_processing.audio_extraction import AudioExtractor
from .analysis.transcription import TranscriptionProcessor
from .analysis.ocr import OCRProcessor
from .analysis.text_analyzer import TextAnalyzer
from .output.result_formatter import ResultFormatter
from .output.report_generator import ReportGenerator
from .output.notion_synchronizer import NotionSynchronizer
from .exceptions import VideoProcessingError
from .utils.gyazo_client import GyazoClient

class VideoProcessingError(Exception):
    """動画処理中のエラーを表す例外クラス"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        
    def __str__(self):
        error_msg = super().__str__()
        if self.context:
            error_msg += f"\nコンテキスト: {self.context}"
        return error_msg

class AudioExtractionError(VideoProcessingError):
    """音声抽出に関するエラー"""
    pass

class VideoProcessor:
    """動画処理を行うクラス"""

    def __init__(self, config: Optional[Union[str, Dict[str, Any], Config]] = None):
        """
        VideoProcessorを初期化します。

        Args:
            config: 設定情報。以下のいずれかを指定できます:
                - 設定ファイルのパス(str)
                - 設定辞書(Dict[str, Any])
                - Configオブジェクト
        """
        # 設定の初期化
        if isinstance(config, Config):
            self.config_obj = config
        else:
            self.config_obj = Config(config)
            
        # 出力ディレクトリの設定
        video_processor_config = self.config_obj.get('video_processor', {})
        self.output_dir = Path(video_processor_config.get('output_dir', 'output'))
        self.temp_dir = Path(video_processor_config.get('temp_dir', self.output_dir / 'temp'))
            
        # ロガーの初期化
        self.logger = Logger.get_logger("video_processor", self.config_obj)

        # フレーム抽出の設定を更新
        frame_extractor_config = self.config_obj.get('frame_extractor', {})
        
        # 1時間あたりのフレーム数から間隔を計算
        frames_per_hour = frame_extractor_config.get('target_frames_per_hour', 1000)
        interval = 3600 / frames_per_hour  # 1時間（3600秒）をフレーム数で割って間隔を算出
        
        frame_extractor_config.update({
            'interval': interval,  # 設定から計算した間隔
            'quality': frame_extractor_config.get('quality', 95),
            'important_frames_ratio': frame_extractor_config.get('important_frames_ratio', 0.05),
            'min_scene_change': frame_extractor_config.get('min_scene_change', 0.3)
        })
        
        self.logger.info(f"フレーム抽出設定: 1時間あたり{frames_per_hour}フレーム（間隔: {interval:.2f}秒）")

        # 各コンポーネントの初期化
        self.frame_extractor = FrameExtractor(frame_extractor_config)
        self.audio_extractor = AudioExtractor(self.config_obj.get('audio_extractor', {}))
        
        # TranscriptionProcessorの設定を統合
        transcription_config = self.config_obj.get_all()
        
        # テスト用の設定をオーバーライドしないように、元の設定を保持
        whisper_config = transcription_config.get('models', {}).get('whisper', {})
        if not whisper_config.get('model', {}).get('name'):
            # 設定が不足している場合のみデフォルト値を設定
            if 'models' not in transcription_config:
                transcription_config['models'] = {}
            if 'whisper' not in transcription_config['models']:
                transcription_config['models']['whisper'] = {}
            if 'model' not in transcription_config['models']['whisper']:
                transcription_config['models']['whisper']['model'] = {}
            if 'name' not in transcription_config['models']['whisper']['model']:
                transcription_config['models']['whisper']['model']['name'] = 'base'
        
        # デバイス設定が不足している場合のみデフォルト値を設定
        if not whisper_config.get('device'):
            # 小さいモデル（tiny, base）はCPUの方が高速
            if transcription_config['models']['whisper']['model']['name'] in ['tiny', 'base']:
                transcription_config['models']['whisper']['device'] = 'cpu'
            # 大きいモデル（small, medium, large）はGPUの方が高速
            else:
                # デフォルトのデバイス選択ロジックを使用
                if torch.cuda.is_available():
                    transcription_config['models']['whisper']['device'] = 'cuda'
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    transcription_config['models']['whisper']['device'] = 'mps'
                else:
                    transcription_config['models']['whisper']['device'] = 'cpu'
                
        self.logger.info(f"Whisperモデル設定: {transcription_config['models']['whisper']['model']['name']}, デバイス: {transcription_config['models']['whisper'].get('device', '自動検出')}")
        self.transcription_processor = TranscriptionProcessor(transcription_config)
        
        self.ocr_processor = OCRProcessor(self.config_obj.get('ocr_processor', {}))
        
        # TextAnalyzerに完全な設定を渡す
        text_analyzer_config = self.config_obj.get_all()
        self.text_analyzer = TextAnalyzer(text_analyzer_config)
        self.logger.info(f"TextAnalyzer初期化: Geminiモデル設定を適用")
        
        self.report_generator = ReportGenerator(self.config_obj)

        # Notion同期の初期化(設定で有効な場合のみ)
        notion_config = self.config_obj.get('notion', {})
        if notion_config.get('enabled', False):
            self.notion_sync = NotionSynchronizer(self.config_obj)
        else:
            self.notion_sync = None
            self.logger.info("Notion同期は無効化されています")

        # パフォーマンスモニタリング
        self.performance_monitor = PerformanceMonitor()
        self.logger.info("VideoProcessor initialized successfully")
        
    def process_video(self, video_path: str, output_dir: str = 'output') -> Dict[str, Any]:
        """動画を処理し、結果を返します
        
        Args:
            video_path (str): 処理する動画のパス
            output_dir (str): 出力ディレクトリ
            
        Returns:
            Dict[str, Any]: 処理結果
        """
        try:
            video_path = Path(video_path)
            if not video_path.exists():
                raise VideoProcessingError(f"動画ファイルが見つかりません: {video_path}")

            # 出力ディレクトリの更新(設定値を上書き)
            if output_dir:
                self.output_dir = Path(output_dir)
                self.temp_dir = self.output_dir / "temp"
            
            # ディレクトリの作成
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.temp_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"動画処理を開始: {video_path}")
            self.performance_monitor.start_monitoring("video_processing")

            # 中間ファイルのパスを設定
            frames_json = self.temp_dir / "frames.json"
            ocr_json = self.temp_dir / "ocr_results.json"
            transcription_json = self.temp_dir / "transcription.json"
            screenshots_dir = self.output_dir / "screenshots"
            
            # フレーム抽出（中間ファイルがある場合は再利用）
            if frames_json.exists():
                self.logger.info("既存のフレーム情報を使用")
                with open(frames_json, 'r', encoding='utf-8') as f:
                    frames = json.load(f)
            else:
                self.logger.info("フレーム抽出を開始")
                frames = self.frame_extractor.extract_frames(str(video_path))
                screenshots_dir.mkdir(parents=True, exist_ok=True)
                saved_paths = self.frame_extractor.save_frames(frames, str(screenshots_dir))
                frames = [{k: str(v) if isinstance(v, Path) else v
                          for k, v in frame.items() if k != 'image'}
                         for frame in frames]
                with open(frames_json, 'w', encoding='utf-8') as f:
                    json.dump(frames, f, ensure_ascii=False, indent=2)

            # OCR処理（中間ファイルがある場合は再利用）
            if ocr_json.exists():
                self.logger.info("既存のOCR結果を使用")
                with open(ocr_json, 'r', encoding='utf-8') as f:
                    ocr_results = json.load(f)
            else:
                self.logger.info("OCR処理を開始")
                ocr_results = self.ocr_processor.process_frames(frames)
                with open(ocr_json, 'w', encoding='utf-8') as f:
                    json.dump(ocr_results, f, ensure_ascii=False, indent=2)

            # 音声処理（中間ファイルがある場合は再利用）
            if transcription_json.exists():
                self.logger.info("既存の文字起こし結果を使用")
                with open(transcription_json, 'r', encoding='utf-8') as f:
                    transcription = json.load(f)
            else:
                self.logger.info("音声処理を開始")
                audio_path = self.extract_audio(str(video_path))
                transcription = self.transcription_processor.transcribe_audio(audio_path)
                with open(transcription_json, 'w', encoding='utf-8') as f:
                    json.dump(transcription, f, ensure_ascii=False, indent=2)

            # テキスト分析
            self.logger.info("テキスト分析を開始")
            
            # 一時ディレクトリの作成（存在しない場合のみ）
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            
            # analysis.jsonの存在確認
            analysis_json = self.temp_dir / "analysis.json"
            if analysis_json.exists():
                self.logger.info("既存のテキスト分析結果を使用")
                with open(analysis_json, 'r', encoding='utf-8') as f:
                    analysis_result = json.load(f)
            else:
                self.logger.info("新しいテキスト分析を実行")
                # 新しい分析を実行
                analysis_result = self.text_analyzer.analyze_content(transcription, ocr_results)
                
                # 分析結果を保存
                with open(analysis_json, 'w', encoding='utf-8') as f:
                    json.dump(analysis_result, f, ensure_ascii=False, indent=2)

            # 最終結果の生成
            final_result = self._merge_results(frames, ocr_results, transcription, analysis_result)
            final_json = self.output_dir / "final_result.json"
            with open(final_json, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)

            # HTMLレポートの生成
            self.logger.info("レポート生成を開始")
            report_path = self.output_dir / "report.html"
            self.report_generator.generate_report(final_result, str(report_path))

            # Notion登録用データの生成
            self.logger.info("Notion登録用データの生成を開始")
            notion_data_path = self.output_dir / "regist.json"
            self.generate_notion_data(str(analysis_json), str(notion_data_path))

            # Notion同期(有効な場合のみ)
            notion_result = {}
            if self.notion_sync:
                self.logger.info("Notion同期を開始")
                notion_result = self.notion_sync.sync_results(final_result)

            # パフォーマンス情報の記録
            performance_metrics = self.performance_monitor.stop_monitoring("video_processing")

            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "video_path": str(video_path),
                "output_files": {
                    "frames": str(frames_json),
                    "screenshots_dir": str(screenshots_dir),
                    "ocr": str(ocr_json),
                    "transcription": str(transcription_json),
                    "analysis": str(analysis_json),
                    "final": str(final_json),
                    "report": str(report_path),
                    "notion_data": str(notion_data_path)
                },
                "notion_page_url": notion_result.get("url"),
                "performance": performance_metrics
            }
            
        except Exception as e:
            self.logger.error(f"動画処理中にエラーが発生: {str(e)}")
            raise VideoProcessingError(f"動画処理に失敗: {str(e)}")
    
    def _merge_results(self, frames: List[Dict[str, Any]], ocr_results: Dict[str, Any], 
                      transcription: Dict[str, Any], analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """各処理の結果を統合して最終結果を生成します
        
        Args:
            frames (List[Dict[str, Any]]): フレーム情報のリスト
            ocr_results (Dict[str, Any]): OCR処理結果
            transcription (Dict[str, Any]): 文字起こし結果
            analysis_result (Dict[str, Any]): 分析・要約結果
            
        Returns:
            Dict[str, Any]: 統合された最終結果
        """
        try:
            self.logger.info("結果の統合を開始します")
            
            # パフォーマンス情報を取得
            performance_summary = self.performance_monitor.get_task_summary("video_processing")
            processing_time = performance_summary['duration'] if performance_summary else 0
            
            # セグメントの形式を修正
            segments = []
            for segment in analysis_result.get('segments', []):
                segments.append({
                    'time_range': {
                        'start': segment.get('start', 0),
                        'end': segment.get('end', 0)
                    },
                    'text': segment.get('text', ''),
                    'ocr': segment.get('ocr', []),
                    'analysis': segment.get('analysis', {})
                })

            # 結果の統合
            final_result = {
                'metadata': {
                    'processed_at': datetime.now().isoformat(),
                    'version': '1.0.0',
                    'screenshot_count': len(frames),
                    'processing_time': processing_time,
                    'intermediate_files': {
                        'transcription': str(self.temp_dir / "transcription.json"),
                        'ocr': str(self.temp_dir / "ocr_results.json"),
                        'summaries': str(self.temp_dir / "analysis.json")
                    }
                },
                'transcription': transcription,
                'analysis': {
                    'segments': segments,
                    'total_segments': len(segments)
                },
                'keywords': [],
                'topics': []
            }
            
            self.logger.info("結果の統合が完了しました")
            return final_result
            
        except Exception as e:
            self.logger.error(f"結果の統合中にエラーが発生: {str(e)}")
            raise VideoProcessingError("結果の統合に失敗しました", {
                "error": str(e)
            })

    def extract_audio(self, video_path: str) -> str:
        """動画から音声を抽出する

        Args:
            video_path (str): 動画ファイルのパス

        Returns:
            str: 抽出された音声ファイルのパス

        Raises:
            AudioExtractionError: 音声抽出に失敗した場合
        """
        try:
            self.logger.info(f"音声抽出開始: {video_path}")

            if not os.path.exists(video_path):
                raise AudioExtractionError(f"動画ファイルが見つかりません: {video_path}")

            # 音声抽出の設定を取得
            audio_config = self.config_obj.get('audio_extractor', {})
            output_format = audio_config.get('format', 'wav')
            sample_rate = audio_config.get('sample_rate', 44100)

            # 出力パスの設定
            audio_dir = self.output_dir / 'audio'
            audio_dir.mkdir(parents=True, exist_ok=True)
            output_path = audio_dir / f"{Path(video_path).stem}_audio.{output_format}"

            # AudioExtractorを使用して音声を抽出
            audio_extractor = AudioExtractor(self.config_obj.get('audio_extractor', {}))
            output_path = audio_extractor.extract_audio(video_path, str(output_path))

            self.logger.info(f"音声抽出完了: {output_path}")
            return output_path
            
        except Exception as e:
            error_context = {
                'video_path': video_path,
                'error_type': type(e).__name__,
                'error_message': str(e)
            }
            raise AudioExtractionError("音声抽出に失敗しました", error_context) from e

    def _clean_llm_response(self, response: Any) -> str:
        """LLMのレスポンスをクリーンアップします
        
        Args:
            response (Any): LLMからのレスポンス
            
        Returns:
            str: クリーンアップされたテキスト
        """
        if isinstance(response, dict):
            return response.get('generated_text', '').replace('見出し:', '').strip()
        elif isinstance(response, list) and response:
            return response[0].get('generated_text', '').replace('要約:', '').strip()
        elif isinstance(response, str):
            return response.replace('ポイント:', '').strip()
        return ''
        
    def _extract_key_points(self, response: str) -> List[str]:
        """レスポンスから重要なポイントを抽出します
        
        Args:
            response (str): レスポンステキスト
            
        Returns:
            List[str]: 重要なポイントのリスト
        """
        points = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('・'):
                points.append(line[1:].strip())
        return points[:3]  # 最大3つまで
        
    def generate_html_report(self, data: Dict[str, Any], output_path: str) -> bool:
        """HTMLレポートを生成します
        
        Args:
            data (Dict[str, Any]): レポートデータ
            output_path (str): 出力パス
            
        Returns:
            bool: 生成が成功したかどうか
        """
        try:
            # テンプレートを読み込み
            template_path = Path('templates') / 'report_template.html'
            if not template_path.exists():
                self.logger.error(f"テンプレートファイルが見つかりません: {template_path}")
                return False
                
            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()
                
            # データを埋め込み
            template = Template(template)
            html = template.render(
                title="動画分析レポート",
                processed_at=data['metadata']['processed_at'],
                video_duration=data['metadata'].get('video_duration', 'N/A'),
                segment_count=data['metadata'].get('segment_count', 0),
                screenshot_count=data['metadata'].get('screenshot_count', 0),
                segments=data.get('segments', [])
            )
            
            # 出力
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
                
            self.logger.info(f"HTMLレポートを生成しました: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"HTMLレポートの生成に失敗: {str(e)}")
            return False
            
    def generate_notion_data(self, data: str, output_path: str) -> bool:
        """Notion登録用のデータを生成します
        
        Args:
            data (str): analysis.jsonのパス
            output_path (str): 出力パス（JSON形式）
            
        Returns:
            bool: 生成が成功したかどうか
        """
        try:
            # analysis.jsonからデータを読み込む
            with open(data, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)

            # GyazoClientの初期化
            gyazo_access_token = os.getenv('GYAZO_ACCESS_TOKEN', "8bAQUk5x4GrEqT6-1xmIMClIRt2F6QGpWds_LY3kDGs")
            self.logger.info(f"Gyazo APIキーの長さ: {len(gyazo_access_token)}")
            gyazo_client = GyazoClient(gyazo_access_token)

            result = []
            # セグメントごとにデータを生成
            for i, segment in enumerate(analysis_data['segments'], 1):
                # 時間範囲の取得
                time_range = segment['time_range']
                start = time_range['start']
                end = time_range['end']
                timestamp = f"{start}秒 - {end}秒"

                # スクリーンショット情報の取得（存在する場合）
                screenshot_info = segment.get('screenshot', {})
                image_path = screenshot_info.get('image_path', '')
                
                # パスの検証とログ出力
                if image_path:
                    self.logger.info(f"スクリーンショットパス: {image_path}")
                    if not os.path.exists(image_path):
                        self.logger.warning(f"スクリーンショットファイルが存在しません: {image_path}")
                        image_path = ''
                else:
                    self.logger.warning(f"セグメント{i}にスクリーンショットパスがありません")
                
                # Gyazoにアップロードしてサムネイルを取得
                thumbnail_url = ""
                if image_path and os.path.exists(image_path):
                    thumbnail_url = gyazo_client.upload_image(
                        image_path,
                        description=f"セグメント{i}のスクリーンショット ({timestamp})"
                    ) or ""
                
                # セグメントデータの作成
                segment_data = {
                    "No": i,  # 1から始まる連番
                    "Summary": segment['summary'],
                    "Timestamp": timestamp,
                    "Thumbnail": thumbnail_url  # Gyazoの画像URLを設定
                }
                result.append(segment_data)
            
            # 結果が空の場合は少なくとも1つのダミーデータを追加
            if not result:
                self.logger.warning("有効なセグメントが見つかりませんでした。ダミーデータを追加します。")
                result.append({
                    "No": 1,
                    "Summary": "動画の要約データがありません",
                    "Timestamp": "0秒 - 0秒",
                    "Thumbnail": ""
                })
            
            # JSON形式で保存
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"Notion登録用データを生成しました: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Notion登録用データの生成に失敗: {str(e)}")
            return False

