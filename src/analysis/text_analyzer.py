import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
import google.generativeai as genai
from dotenv import load_dotenv
from src.exceptions import TextAnalysisError

class TextAnalyzer:
    _instance = None
    _model = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.logger = logging.getLogger(__name__)
            
            # 設定の初期化
            if config:
                self.min_segment_length = config.get('min_segment_length', 50)
                self.similarity_threshold = config.get('similarity_threshold', 0.7)
            else:
                self.min_segment_length = 50
                self.similarity_threshold = 0.7
            
            # Geminiモデルの初期化
            load_dotenv()
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise TextAnalysisError("Gemini API Keyが設定されていません")
            
            # Geminiの設定
            genai.configure(api_key=api_key)
            
            # モデルの設定
            generation_config = {
                'temperature': 0.3,
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 8192,
            }
            
            if not TextAnalyzer._model:
                TextAnalyzer._model = genai.GenerativeModel(
                    model_name='gemini-1.5-pro',
                    generation_config=generation_config
                )
                self.logger.info("Geminiモデルを初期化しました")

    def analyze_content(self, transcription: Union[List[Dict[str, Any]], Dict[str, Any]], ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        音声書き起こしとOCR結果を分析して、コンテンツの要約を生成します。

        Args:
            transcription: リストまたは辞書形式の文字起こし結果
            ocr_results: OCR結果のリスト

        Returns:
            Dict[str, Any]: 分析結果
        """
        try:
            # transcriptionがリストの場合は辞書形式に変換
            if isinstance(transcription, list):
                segments = transcription
            else:
                segments = transcription.get('segments', [])

            if not segments:
                raise TextAnalysisError("書き起こしセグメントが空です")

            analyzed_segments = []
            for segment in segments:
                segment_text = segment.get('text', '')
                if not segment_text:
                    continue

                # OCR結果の取得
                try:
                    segment_ocr = self._get_segment_ocr(segment, ocr_results)
                except Exception as e:
                    self.logger.warning(f"OCR結果の取得に失敗しましたが、処理を継続します: {str(e)}")
                    segment_ocr = []

                # OCRテキストの抽出
                ocr_texts = []
                for ocr_result in segment_ocr:
                    if isinstance(ocr_result, dict):
                        ocr_texts.append(ocr_result.get('text', ''))
                    elif isinstance(ocr_result, str):
                        ocr_texts.append(ocr_result)

                # Geminiを使用してセグメントを分析
                prompt = f"""
                以下のセグメントを分析し、JSONで返してください:

                音声テキスト:
                {segment_text}

                画面のテキスト:
                {', '.join(ocr_texts)}

                形式:
                {{
                    "heading": "30文字以内の見出し（です・ます調）",
                    "summary": "100文字以内の要約（です・ます調）",
                    "key_points": ["重要なポイント（3つまで、です・ます調）"]
                }}
                """
                
                try:
                    response = TextAnalyzer._model.generate_content(prompt)
                    result = json.loads(response.text)
                    
                    analyzed_segments.append({
                        'time_range': {
                            'start': segment.get('start', 0),
                            'end': segment.get('end', 0)
                        },
                        'text': segment_text,
                        'ocr': segment_ocr,
                        'analysis': result
                    })

                except Exception as e:
                    self.logger.error(f"テキスト分析エラー: {str(e)}")
                    raise TextAnalysisError(f"テキスト分析に失敗: {str(e)}")

            return {
                'segments': analyzed_segments,
                'total_segments': len(analyzed_segments)
            }

        except Exception as e:
            raise TextAnalysisError(f"コンテンツ分析に失敗: {str(e)}")

    def _get_segment_ocr(self, segment: Dict[str, Any], ocr_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        指定されたセグメントの時間範囲内のOCR結果を取得します。
        """
        try:
            segment_start = float(segment.get('start', 0))
            segment_end = float(segment.get('end', 0))
            
            matched_results = []
            for result in ocr_results:
                if isinstance(result, dict):
                    timestamp = float(result.get('timestamp', 0))
                    if segment_start <= timestamp <= segment_end:
                        matched_results.append(result)
                elif isinstance(result, str):
                    matched_results.append({'text': result})
            
            return matched_results

        except Exception as e:
            self.logger.error(f"OCR結果の取得に失敗: {str(e)}")
            return []

    def analyze_content_v2(self, analysis_json: dict, ocr_results: dict) -> dict:
        """
        新しい処理フローに基づくコンテンツ分析メソッド

        Args:
            analysis_json (dict): 音声書き起こしデータ
            ocr_results (dict): OCR結果データ

        Returns:
            dict: 分析結果（analysis.json形式）

        Raises:
            TextAnalysisError: 分析処理中にエラーが発生した場合
        """
        try:
            # 1. データ入力とパース
            if analysis_json is None or ocr_results is None:
                raise TextAnalysisError("入力データがNoneです")

            if not isinstance(analysis_json, dict) or not isinstance(ocr_results, dict):
                raise TextAnalysisError("入力データの形式が不正です")

            segments = analysis_json.get("segments")
            screenshots = ocr_results.get("screenshots")

            if segments is None or screenshots is None:
                raise TextAnalysisError("必須フィールドが存在しません")

            if not isinstance(segments, list) or not isinstance(screenshots, list):
                raise TextAnalysisError("segmentsとscreenshotsはリスト形式である必要があります")

            if not segments or not screenshots:
                raise TextAnalysisError("入力データが不足しています")

            # 2. 前処理
            processed_segments = []
            for i, segment in enumerate(segments):
                if not isinstance(segment, dict):
                    raise TextAnalysisError(f"セグメント{i}の形式が不正です")
                try:
                    start = float(segment.get('start', 0))
                    end = float(segment.get('end', 0))
                    if end <= start:
                        raise TextAnalysisError(f"セグメント{i}の時間範囲が不正です: start={start}, end={end}")
                    
                    processed_segments.append({
                        'start': start,
                        'end': end,
                        'text': str(segment.get('text', '')),
                        'analysis': segment.get('analysis', {}),
                        'duration': end - start
                    })
                except (ValueError, TypeError) as e:
                    raise TextAnalysisError(f"セグメント{i}のデータ変換に失敗しました: {str(e)}")

            processed_screenshots = []
            for i, shot in enumerate(screenshots):
                if not isinstance(shot, dict):
                    raise TextAnalysisError(f"スクリーンショット{i}の形式が不正です")
                try:
                    processed_screenshots.append({
                        'timestamp': float(shot.get('timestamp', 0)),
                        'frame_number': int(shot.get('frame_number', 0)),
                        'importance_score': float(shot.get('importance_score', 0)),
                        'ocr_confidence': float(shot.get('ocr_confidence', 0)),
                        'text': str(shot.get('text', '')),
                        'text_quality_score': self.calculate_text_quality(str(shot.get('text', ''))),
                        'image_path': str(shot.get('image_path', ''))
                    })
                except (ValueError, TypeError) as e:
                    raise TextAnalysisError(f"スクリーンショット{i}のデータ変換に失敗しました: {str(e)}")

            if not processed_segments:
                raise TextAnalysisError("有効なセグメントデータがありません")

            # 3. 動的な要約行数の決定
            video_length = max(seg['end'] for seg in processed_segments)
            total_segments = len(processed_segments)
            
            # 動画の長さと元のセグメント数に基づいて要約行数を決定
            base_summary_lines = max(3, round(total_segments * 0.3))  # 最低3行
            length_factor = max(1, video_length / 60)  # 1分あたりの係数
            target_summary_lines = min(int(base_summary_lines * length_factor), 15)  # 最大15行
            
            # 入力サンプル数に基づいてクラスタ数を調整
            n_clusters = min(target_summary_lines, total_segments)
            if n_clusters < 1:
                n_clusters = 1

            # セグメントのクラスタリングと要約
            X = np.array([[s['start'], s['end']] for s in processed_segments])
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X)

            # クラスタごとにセグメントを統合
            summarized_segments = []
            for i in range(n_clusters):
                cluster_segments = [s for j, s in enumerate(processed_segments) if clusters[j] == i]
                if not cluster_segments:
                    continue

                # 時間範囲の計算
                start_time = min(s['start'] for s in cluster_segments)
                end_time = max(s['end'] for s in cluster_segments)
                
                # クラスタ内のテキストを結合
                combined_text = ' '.join(s['text'] for s in cluster_segments)
                if not combined_text.strip():
                    continue

                # Geminiを使用して要約を生成
                prompt = f"""
                以下のテキストを要約してください。
                テキストの長さや内容に応じて、重要なポイントと具体的な情報を適切に含めた要約を生成してください。
                短いテキストは簡潔に、長いテキストはより詳細に要約してください。

                要約のガイドライン:
                1. テキストの主要なポイントを漏らさず含めてください
                2. 具体的な数値や重要な用語は保持してください
                3. 文脈や因果関係が明確になるように表現してください
                4. 冗長な表現は避け、簡潔で分かりやすい日本語を使用してください
                5. 必ず「です・ます調」で要約を生成してください
                6. 文末は「～です。」「～ます。」で終わるようにしてください

                テキスト:
                {combined_text}
                """
                try:
                    response = self._model.generate_content(prompt)
                    summary = response.text.strip()
                    if not summary:
                        raise TextAnalysisError("要約の生成に失敗しました")
                except Exception as e:
                    raise TextAnalysisError(f"要約の生成中にエラーが発生しました: {str(e)}")

                # スクショの選定
                relevant_shots = [
                    shot for shot in processed_screenshots
                    if start_time <= shot['timestamp'] <= end_time
                ]

                best_shot = None
                if relevant_shots:
                    best_shot = max(relevant_shots, key=lambda x: (
                        x['importance_score'] * 0.4 +
                        x['ocr_confidence'] * 0.3 +
                        x['text_quality_score'] * 0.3
                    ))

                # データ統合（analysis.json形式に合わせる）
                segment_data = {
                    "time_range": {
                        "start": start_time,
                        "end": end_time
                    },
                    "summary": summary,
                    "importance_score": best_shot['importance_score'] if best_shot else 0.0,
                    "metadata": {
                        "segment_count": len(cluster_segments),
                        "has_screenshot_text": bool(best_shot),
                        "summary_points": len(summary.split('。')),
                        "keyword_count": len(set(summary.split()))
                    }
                }

                if best_shot:
                    segment_data["screenshot"] = {
                        "timestamp": best_shot['timestamp'],
                        "frame_number": best_shot['frame_number'],
                        "text": best_shot['text'],
                        "ocr_confidence": best_shot['ocr_confidence'],
                        "image_path": best_shot.get('image_path', '')
                    }

                summarized_segments.append(segment_data)

            if not summarized_segments:
                raise TextAnalysisError("有効な要約セグメントを生成できませんでした")

            # 時間順にソート
            summarized_segments.sort(key=lambda x: x['time_range']['start'])

            return {
                "segments": summarized_segments,
                "total_segments": len(summarized_segments),
                "metadata": {
                    "video_length": video_length,
                    "total_segments": total_segments,
                    "summary_segments": len(summarized_segments)
                }
            }

        except TextAnalysisError:
            raise
        except Exception as e:
            self.logger.error(f"分析処理中にエラーが発生しました: {str(e)}")
            raise TextAnalysisError(f"分析処理に失敗しました: {str(e)}")

    def analyze_content_v3(self, analysis_json, ocr_results):
        """
        テキスト分析の実行（バージョン3）- 一括要約処理による最適化版
        
        Args:
            analysis_json (dict): 分析用のJSONデータ
            ocr_results (dict): OCR結果のJSONデータ
            
        Returns:
            dict: 分析結果
            
        Raises:
            TextAnalysisError: テキスト分析中にエラーが発生した場合
        """
        try:
            # 1. 入力データの検証
            if not isinstance(analysis_json, dict):
                raise TextAnalysisError("analysis_jsonはdictである必要があります")
            if not isinstance(ocr_results, dict):
                raise TextAnalysisError("ocr_resultsはdictである必要があります")
                
            # セグメントの検証
            if 'segments' not in analysis_json:
                raise TextAnalysisError("analysis_jsonにsegmentsフィールドがありません")
            if not isinstance(analysis_json['segments'], list):
                raise TextAnalysisError("segmentsはlistである必要があります")
                
            # スクリーンショットの検証
            if 'screenshots' not in ocr_results:
                raise TextAnalysisError("ocr_resultsにscreenshotsフィールドがありません")
            if not isinstance(ocr_results['screenshots'], list):
                raise TextAnalysisError("screenshotsはlistである必要があります")
                
            # 2. データの前処理
            processed_segments = []
            for segment in analysis_json['segments']:
                try:
                    processed_segment = {
                        'text': segment['text'],
                        'start': float(segment['start']),
                        'end': float(segment['end']),
                        'confidence': float(segment.get('confidence', 0.0))
                    }
                    processed_segments.append(processed_segment)
                except (KeyError, ValueError) as e:
                    self.logger.error(f"セグメントの変換エラー: {str(e)}")
                    
            processed_screenshots = []
            for screenshot in ocr_results['screenshots']:
                try:
                    text_quality_score = self.calculate_text_quality(screenshot.get('text', ''))
                    processed_screenshot = {
                        'timestamp': float(screenshot['timestamp']),
                        'frame_number': int(screenshot['frame_number']),
                        'text': screenshot.get('text', ''),
                        'ocr_confidence': float(screenshot.get('confidence', 0.0)),
                        'importance_score': float(screenshot.get('importance_score', 0.0)),
                        'text_quality_score': text_quality_score,
                        'image_path': screenshot.get('image_path', '')
                    }
                    processed_screenshots.append(processed_screenshot)
                except (KeyError, ValueError) as e:
                    self.logger.error(f"スクリーンショットの変換エラー: {str(e)}")
                    
            if not processed_segments:
                raise TextAnalysisError("有効なセグメントデータがありません")

            # 3. 動的な要約行数の決定
            video_length = max(seg['end'] for seg in processed_segments)
            total_segments = len(processed_segments)
            
            # 動画の長さと元のセグメント数に基づいて要約行数を決定
            base_summary_lines = max(3, round(total_segments * 0.3))  # 最低3行
            length_factor = max(1, video_length / 60)  # 1分あたりの係数
            target_summary_lines = min(int(base_summary_lines * length_factor), 15)  # 最大15行
            
            # 入力サンプル数に基づいてクラスタ数を調整
            n_clusters = min(target_summary_lines, total_segments)
            if n_clusters < 1:
                n_clusters = 1

            # セグメントのクラスタリングと要約
            X = np.array([[s['start'], s['end']] for s in processed_segments])
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X)

            # クラスタごとにセグメントを整理
            cluster_data = []
            for i in range(n_clusters):
                cluster_segments = [s for j, s in enumerate(processed_segments) if clusters[j] == i]
                if not cluster_segments:
                    continue

                # 時間範囲の計算
                start_time = min(s['start'] for s in cluster_segments)
                end_time = max(s['end'] for s in cluster_segments)
                
                # クラスタ内のテキストを結合
                combined_text = ' '.join(s['text'] for s in cluster_segments)
                if not combined_text.strip():
                    continue
                
                cluster_data.append({
                    'id': i,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': combined_text,
                    'segments': cluster_segments
                })
            
            # 4. 一括要約処理
            # トークン数を考慮して分割処理
            MAX_TOKENS = 30000
            batches = []
            current_batch = []
            current_token_count = 0
            
            # 簡易的なトークン数推定（日本語の場合、文字数の約1.5倍がトークン数の目安）
            for cluster in cluster_data:
                estimated_tokens = len(cluster['text']) * 1.5
                
                if current_token_count + estimated_tokens > MAX_TOKENS and current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_token_count = 0
                
                current_batch.append(cluster)
                current_token_count += estimated_tokens
            
            if current_batch:
                batches.append(current_batch)
            
            # バッチごとに処理
            all_summaries = {}
            for batch_idx, batch in enumerate(batches):
                # バッチ内の全クラスタを一度に要約
                prompt = """
                以下の複数のテキストセクションを個別に要約してください。
                各セクションは<section id="数字">タグで区切られています。
                
                要約のガイドライン:
                1. 各セクションの主要なポイントを漏らさず含めてください
                2. 具体的な数値や重要な用語は保持してください
                3. 文脈や因果関係が明確になるように表現してください
                4. 冗長な表現は避け、簡潔で分かりやすい日本語を使用してください
                5. 必ず「です・ます調」で要約を生成してください
                6. 文末は「～です。」「～ます。」で終わるようにしてください
                
                出力形式:
                <summary id="0">
                セクション0の要約文
                </summary>
                <summary id="1">
                セクション1の要約文
                </summary>
                （以下同様）
                
                セクション:
                """
                
                for cluster in batch:
                    prompt += f"""
                    <section id="{cluster['id']}">
                    {cluster['text']}
                    </section>
                    """
                
                try:
                    self.logger.info(f"バッチ {batch_idx+1}/{len(batches)} の要約を生成中 ({len(batch)}クラスタ)")
                    response = self._model.generate_content(prompt)
                    result_text = response.text.strip()
                    
                    # 応答から各セクションの要約を抽出
                    import re
                    summary_pattern = r'<summary id="(\d+)">(.*?)</summary>'
                    summaries = re.findall(summary_pattern, result_text, re.DOTALL)
                    
                    for summary_id, summary_text in summaries:
                        all_summaries[int(summary_id)] = summary_text.strip()
                    
                except Exception as e:
                    raise TextAnalysisError(f"要約の生成中にエラーが発生しました: {str(e)}")
            
            # 5. 結果の統合
            summarized_segments = []
            for cluster in cluster_data:
                cluster_id = cluster['id']
                if cluster_id not in all_summaries:
                    self.logger.warning(f"クラスタID {cluster_id} の要約が見つかりません")
                    continue
                
                summary = all_summaries[cluster_id]
                start_time = cluster['start_time']
                end_time = cluster['end_time']
                
                # スクショの選定
                relevant_shots = [
                    shot for shot in processed_screenshots
                    if start_time <= shot['timestamp'] <= end_time
                ]

                best_shot = None
                if relevant_shots:
                    best_shot = max(relevant_shots, key=lambda x: (
                        x['importance_score'] * 0.4 +
                        x['ocr_confidence'] * 0.3 +
                        x['text_quality_score'] * 0.3
                    ))

                # データ統合（analysis.json形式に合わせる）
                segment_data = {
                    "time_range": {
                        "start": start_time,
                        "end": end_time
                    },
                    "summary": summary,
                    "importance_score": best_shot['importance_score'] if best_shot else 0.0,
                    "metadata": {
                        "segment_count": len(cluster['segments']),
                        "has_screenshot_text": bool(best_shot),
                        "summary_points": len(summary.split('。')),
                        "keyword_count": len(set(summary.split()))
                    }
                }

                if best_shot:
                    segment_data["screenshot"] = {
                        "timestamp": best_shot['timestamp'],
                        "frame_number": best_shot['frame_number'],
                        "text": best_shot['text'],
                        "ocr_confidence": best_shot['ocr_confidence'],
                        "image_path": best_shot.get('image_path', '')
                    }

                summarized_segments.append(segment_data)

            if not summarized_segments:
                raise TextAnalysisError("有効な要約セグメントを生成できませんでした")

            # 時間順にソート
            summarized_segments.sort(key=lambda x: x['time_range']['start'])

            return {
                "segments": summarized_segments,
                "metadata": {
                    "video_length": video_length,
                    "total_segments": total_segments,
                    "summary_segments": len(summarized_segments),
                    "api_requests": len(batches)  # APIリクエスト回数を記録
                }
            }
            
        except Exception as e:
            if isinstance(e, TextAnalysisError):
                raise
            raise TextAnalysisError(f"テキスト分析中にエラーが発生しました: {str(e)}")

    def calculate_text_quality(self, text: str) -> float:
        """テキストの品質スコアを計算します（0.0 ~ 1.0）"""
        if not text or len(text.strip()) < 3:
            return 0.0

        try:
            score = 1.0
            
            # 文字の多様性評価
            unique_ratio = len(set(text)) / len(text)
            score *= min(1.0, unique_ratio * 2)
            
            # 意味のある文字の割合評価
            meaningful_chars = sum(1 for c in text if c.isalnum() or 0x3000 <= ord(c) <= 0x9FFF)
            score *= meaningful_chars / len(text)
            
            # 記号の割合評価
            symbols = sum(1 for c in text if not c.isalnum() and not 0x3000 <= ord(c) <= 0x9FFF)
            score *= (1.0 - min(1.0, symbols / len(text) * 2))
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.error(f"テキスト品質計算中にエラー: {str(e)}")
            return 0.0