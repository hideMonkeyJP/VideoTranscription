import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
from collections import Counter
import itertools
import google.generativeai as genai
from dotenv import load_dotenv

class TextAnalysisError(Exception):
    """テキスト分析に関するエラー"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}

class TextAnalyzer:
    def __init__(self, config: Dict[str, Any] = None):
        """テキスト分析プロセッサを初期化します
        
        Args:
            config (Dict[str, Any], optional): 設定辞書
                - min_confidence (float): 最小信頼度スコア (default: 0.6)
        """
        self.config = config or {}
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.logger = logging.getLogger(__name__)
        
        try:
            self.logger.info("Geminiモデルを初期化しています...")
            # 環境変数の読み込み
            load_dotenv()
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEYが設定されていません")
            
            # Geminiの設定
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.logger.info("Geminiモデルの初期化が完了しました")
            
        except Exception as e:
            self.logger.error(f"モデルの初期化中にエラー: {str(e)}")
            raise TextAnalysisError("モデルの初期化に失敗しました", {"error": str(e)})

    def generate_heading(self, text: str) -> str:
        """テキストから見出しを生成します
        
        Args:
            text (str): 入力テキスト
            
        Returns:
            str: 生成された見出し
        """
        try:
            prompt = f"""
            以下のテキストに対して、適切な見出しを生成してください。
            見出しは30文字以内で、テキストの主要なトピックを簡潔に表現してください。

            テキスト:
            {text}

            注意事項:
            - 簡潔で分かりやすい表現を使用
            - 重要なキーワードを含める
            - 文末の句点は不要
            """
            return self._generate_text(prompt)
        except Exception as e:
            self.logger.error(f"見出し生成中にエラー: {str(e)}")
            raise TextAnalysisError("見出しの生成に失敗しました", {"error": str(e)})

    def generate_summary(self, text: str) -> str:
        """テキストの要約を生成します
        
        Args:
            text (str): 入力テキスト
            
        Returns:
            str: 生成された要約
        """
        try:
            prompt = f"""
            以下のテキストを要約してください。
            要約は100文字以内で、テキストの主要なポイントを簡潔にまとめてください。

            テキスト:
            {text}

            注意事項:
            - 重要な情報を優先
            - 具体的な数値やキーワードを含める
            - 簡潔な日本語で表現
            """
            return self._generate_text(prompt)
        except Exception as e:
            self.logger.error(f"要約生成中にエラー: {str(e)}")
            raise TextAnalysisError("要約の生成に失敗しました", {"error": str(e)})

    def generate_key_points(self, text: str) -> List[str]:
        """テキストから重要なポイントを抽出します
        
        Args:
            text (str): 入力テキスト
            
        Returns:
            List[str]: 抽出された重要ポイントのリスト
        """
        try:
            prompt = f"""
            以下のテキストから重要なポイントを抽出してください。
            箇条書きで3点以内にまとめ、各要点は50文字以内で記述してください。

            テキスト:
            {text}

            出力形式:
            • 要点1
            • 要点2
            • 要点3

            注意事項:
            - 重要な情報を優先
            - 具体的な数値やキーワードを含める
            - 簡潔な日本語で表現
            """
            response = self._generate_text(prompt)
            points = []
            for line in response.split('\n'):
                line = line.strip()
                if line and line != "内容なし":
                    # 箇条書き記号と番号を削除
                    line = re.sub(r'^[-・\*\d\.\)、] *', '', line)
                    # 不要な記号を削除
                    line = re.sub(r'[「」『』【】\(\)（）\[\]］\[\{\}:：。、．，!！?？\^_\+\-\*/=]', '', line)
                    if line.strip():
                        points.append(line.strip())
            return points[:3] if points else ["内容なし"]
        except Exception as e:
            self.logger.error(f"キーポイント抽出中にエラー: {str(e)}")
            raise TextAnalysisError("キーポイントの抽出に失敗しました", {"error": str(e)})

    def calculate_text_quality(self, text: str) -> float:
        """テキストの品質スコアを計算します（0.0 ~ 1.0）
        
        Args:
            text (str): 入力テキスト
            
        Returns:
            float: 品質スコア
        """
        if not text or len(text.strip()) < 3:
            return 0.0

        try:
            # 基本スコアの初期化
            score = 1.0

            # 1. 文字種類の評価
            chars = Counter(text)
            unique_ratio = len(chars) / len(text)
            score *= min(1.0, unique_ratio * 2)  # 文字の多様性を評価

            # 2. 意味のある文字の割合
            meaningful_chars = sum(1 for c in text if c.isalnum() or 0x3000 <= ord(c) <= 0x9FFF)
            meaningful_ratio = meaningful_chars / len(text)
            score *= meaningful_ratio

            # 3. 記号の割合評価
            symbol_ratio = sum(1 for c in text if not c.isalnum() and not 0x3000 <= ord(c) <= 0x9FFF) / len(text)
            score *= (1.0 - min(1.0, symbol_ratio * 2))

            # 4. パターン検出
            # 連続する同じ文字
            max_repeat = max(len(list(g)) for _, g in itertools.groupby(text))
            if max_repeat > 3:
                score *= 0.5

            # 5. 日本語文字の評価
            jp_ratio = sum(1 for c in text if 0x3000 <= ord(c) <= 0x9FFF) / len(text)
            if jp_ratio > 0:
                score *= (1.0 + jp_ratio)  # 日本語文字が含まれる場合はスコアを上げる

            # 6. アルファベットの評価
            if text.isascii():
                # 母音の存在確認
                vowel_ratio = sum(1 for c in text.lower() if c in 'aeiou') / len(text)
                if vowel_ratio < 0.1:  # 母音が少なすぎる場合
                    score *= 0.5

            return min(1.0, score)

        except Exception as e:
            self.logger.error(f"テキスト品質計算中にエラー: {str(e)}")
            return 0.0

    def detect_topic_change(self, prev_text: str, current_text: str) -> bool:
        """2つのテキスト間でトピックが変更されたかどうかを判定します
        
        Args:
            prev_text (str): 前のテキスト
            current_text (str): 現在のテキスト
            
        Returns:
            bool: トピックが変更された場合はTrue
        """
        try:
            # 類似度が低い場合はトピックが変更されたと判断
            similarity = self._calculate_similarity(prev_text, current_text)
            return similarity < 0.3  # 類似度が30%未満の場合はトピック変更とみなす
        except Exception as e:
            self.logger.error(f"トピック変更検出中にエラー: {str(e)}")
            return True  # エラーの場合は安全のためトピック変更とみなす

    def _generate_text(self, prompt: str) -> str:
        """Geminiモデルを使用してテキストを生成します"""
        try:
            response = self.model.generate_content(prompt)
            if not response or not hasattr(response, 'text') or not response.text:
                raise ValueError("Geminiモデルからの応答が不正です")
            return self._clean_response(response.text)
        except Exception as e:
            self.logger.error(f"テキスト生成中にエラー: {str(e)}")
            return "内容なし"

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """2つのテキスト間の類似度を計算します"""
        # 文字レベルのn-gramを使用して類似度を計算
        def get_ngrams(text, n=3):
            return set(''.join(gram) for gram in zip(*[text[i:] for i in range(n)]))
        
        # 両方のテキストのn-gramを取得
        ngrams1 = get_ngrams(text1)
        ngrams2 = get_ngrams(text2)
        
        # Jaccard類似度を計算
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        return intersection / union if union > 0 else 0

    def format_segment(self, segment: Dict[str, Any], cache_dir: str = 'output/temp/text_analysis') -> Dict[str, Any]:
        """セグメントを正しい形式にフォーマットします
        
        Args:
            segment (Dict[str, Any]): 入力セグメント
            cache_dir (str): 中間ファイルの保存ディレクトリ
            
        Returns:
            Dict[str, Any]: フォーマットされたセグメント
        """
        try:
            # 時間範囲の整形
            start_time = segment.get('start', 0.0)
            end_time = segment.get('end', 0.0)
            time_range = {
                "start": start_time,
                "end": end_time
            }
            
            # OCRテキストの取得と結合
            ocr_texts = [item.get('text', '') for item in segment.get('ocr', [])]
            ocr_text = ' '.join(ocr_texts)
            
            # メタデータの生成
            metadata = {
                "segment_count": 1,
                "has_screenshot_text": bool(ocr_text),
                "summary_points": len(segment.get('analysis', {}).get('key_points', [])),
                "keyword_count": len(set(' '.join(segment.get('analysis', {}).get('key_points', [])).split()))
            }
            
            # 結果の生成
            result = {
                "time_range": time_range,
                "summary": segment.get('analysis', {}).get('summary', ''),
                "key_points": segment.get('analysis', {}).get('key_points', []),
                "metadata": metadata
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"セグメントのフォーマット中にエラー: {str(e)}")
            raise TextAnalysisError("セグメントのフォーマットに失敗しました", {"error": str(e)})

    def analyze_content(self, transcription: List[Dict[str, Any]], ocr_results: Dict[str, Any]) -> Dict[str, Any]:
        """文字起こしとOCR結果を分析します
        
        Args:
            transcription (List[Dict[str, Any]]): 文字起こし結果のリスト
            ocr_results (Dict[str, Any]): OCR結果
            
        Returns:
            Dict[str, Any]: 分析結果
        """
        try:
            segments = []
            for segment in transcription:
                # セグメントの基本情報を取得
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                text = segment.get('text', '')

                # このセグメントの時間範囲に含まれるOCRテキストを取得
                ocr_texts = []
                for screenshot in ocr_results.get('screenshots', []):
                    if start <= screenshot.get('timestamp', 0) <= end:
                        ocr_texts.append(screenshot.get('text', ''))

                # セグメントごとの分析を実行
                prompt = f"""
                以下のセグメントを分析してください。
                このセグメントは動画全体の一部で、タイムスタンプは{start}秒から{end}秒の部分です。
                
                セグメントのテキスト:
                {text}
                
                OCRテキスト:
                {' '.join(ocr_texts)}
                
                以下の形式でJSONを返してください:
                {{
                    "heading": "このセグメントの内容を端的に表す見出し（30文字以内）",
                    "summary": "このセグメントの内容を要約（100文字以内）",
                    "key_points": [
                        "このセグメントで語られている重要なポイント（3つまで）"
                    ]
                }}
                
                重要な注意事項:
                1. このセグメントの独自性を重視:
                   - このセグメントでのみ語られている具体的な内容や発見に焦点を当てる
                   - 他のセグメントと同じような一般的な表現は完全に避ける
                   - このセグメントならではの新しい視点や気づきを強調する
                
                2. 具体的な表現の使用:
                   - 「重要」「大切」などの抽象的な表現は使用禁止
                   - 実際の発言から具体的な行動、結果、数値を抽出
                   - 話者の感情や反応（「衝撃を受けた」「驚いた」など）も含める
                
                3. セグメント固有の文脈を反映:
                   - このセグメントで初めて登場する概念や主張を特定
                   - 前後のセグメントの内容は一切参照しない
                   - このセグメントだけを読んでも理解できる表現にする
                
                4. 分析の具体例:
                   悪い例:
                   - 見出し: 「タスク管理の重要性」（一般的すぎる）
                   - 要約: 「タスク管理は大切」（抽象的すぎる）
                   - キーポイント: 「効率的な管理が必要」（具体性が不足）
                
                   良い例:
                   - 見出し: 「タスク管理で人材の実力が一目瞭然に」（このセグメントの具体的な発見）
                   - 要約: 「タスク管理の巧拙を見るだけで個人の実力レベルが判断でき、その事実に衝撃を受けた」（具体的な内容と反応）
                   - キーポイント: 「タスク管理能力が人材評価の新しい指標になる可能性」（セグメント固有の気づき）
                
                このセグメントでしか語られていない独自の内容を最大限に引き出し、
                一般的な表現や他のセグメントと重複する内容は完全に避けてください。
                """
                
                response = self.model.generate_content(prompt)
                analysis = json.loads(response.text)
                
                segments.append({
                    'start': start,
                    'end': end,
                    'text': text,
                    'ocr': ocr_texts,
                    'analysis': analysis
                })

            return {
                'segments': segments,
                'total_segments': len(segments)
            }
            
        except Exception as e:
            self.logger.error(f"コンテンツ分析中にエラー: {str(e)}")
            raise TextAnalysisError("コンテンツの分析に失敗しました", {"error": str(e)})

    def _clean_response(self, response: str) -> str:
        """生成されたテキストをクリーニングします"""
        if not response:
            return "内容なし"
        
        # 不要な記号を削除
        response = re.sub(r'[「」『』【】\(\)（）\[\]］\[\{\}]', '', response)
        response = re.sub(r'[:：]', '', response)
        response = re.sub(r'[。、．，]', '。', response)
        response = re.sub(r'[!！?？\^_\+\-\*/=]', '', response)
        
        # 不要な表現を削除
        unnecessary = [
            'です', 'ます', 'ください', '次の', '文章', '見出し',
            '要約', 'ポイント', '抽出', '生成', '入力', '出力',
            'から', 'まで', 'とか', 'みたいな', '感じ', '笑',
            'ね', 'よ', 'な', 'っと', 'この', 'もの', 'こと'
        ]
        for word in unnecessary:
            response = response.replace(word, '')
        
        # 連続する句点を1つに
        response = re.sub(r'。+', '。', response)
        
        # 前後の空白と句点を削除
        response = response.strip('。 　')
        
        # 数字のみの応答を削除
        if re.match(r'^\d+$', response):
            return "内容なし"
        
        # 空の応答の場合はデフォルト値を返す
        if not response or len(response) < 2:
            return "内容なし"
        
        return response 

    def analyze_content_v2(self, analysis_json: Dict[str, Any], ocr_results: Dict[str, Any]) -> Dict[str, Any]:
        """新しい分析処理フローを実装したメソッド
        
        Args:
            analysis_json (Dict[str, Any]): analysis.jsonの内容
            ocr_results (Dict[str, Any]): ocr_results.jsonの内容
            
        Returns:
            Dict[str, Any]: 新しい分析結果（analysis.json形式）
            
        Raises:
            TextAnalysisError: 分析処理中にエラーが発生した場合
        """
        try:
            # 入力データの検証
            if not isinstance(analysis_json, dict) or not isinstance(ocr_results, dict):
                raise TextAnalysisError("入力データが正しい形式ではありません")
            
            # セグメントとスクリーンショットの取得と検証
            segments = analysis_json.get('segments')
            screenshots = ocr_results.get('screenshots')
            
            if not isinstance(segments, list) or not isinstance(screenshots, list):
                raise TextAnalysisError("segments または screenshots が正しい形式ではありません")
            
            if not segments or not screenshots:
                raise TextAnalysisError("セグメントまたはスクリーンショットデータが空です")
            
            # データ構造の検証
            self._validate_data_structure(segments, screenshots)
            
            # 動画の長さと要約行数の計算
            video_length = self._calculate_video_length(segments)
            num_summary_lines = self._calculate_summary_lines(len(segments), video_length)
            
            # セグメントの統合と要約
            merged_segments = self._merge_segments_v2(segments, num_summary_lines)
            
            # スクリーンショットの選定と結果の統合
            final_segments = self._integrate_screenshots_v2(merged_segments, screenshots)
            
            # 最終結果の検証
            if not final_segments:
                raise TextAnalysisError("分析結果の生成に失敗しました")
            
            return {
                "segments": final_segments,
                "total_segments": len(final_segments)
            }
            
        except TextAnalysisError:
            raise
        except Exception as e:
            self.logger.error(f"予期せぬエラーが発生: {str(e)}")
            raise TextAnalysisError(f"予期せぬエラーが発生しました: {str(e)}")

    def _validate_data_structure(self, segments: List[Dict[str, Any]], screenshots: List[Dict[str, Any]]) -> None:
        """データ構造を検証します
        
        Args:
            segments (List[Dict[str, Any]]): セグメントデータ
            screenshots (List[Dict[str, Any]]): スクリーンショットデータ
            
        Raises:
            TextAnalysisError: データ構造が不正な場合
        """
        # セグメントの検証
        required_segment_fields = ['start', 'end', 'text']
        for segment in segments:
            if not all(field in segment for field in required_segment_fields):
                raise TextAnalysisError(f"セグメントに必要なフィールドが不足しています: {required_segment_fields}")
            
            try:
                float(segment['start'])
                float(segment['end'])
            except (ValueError, TypeError):
                raise TextAnalysisError("セグメントの時間情報が不正です")
        
        # スクリーンショットの検証
        required_screenshot_fields = ['timestamp', 'frame_number', 'importance_score', 'ocr_confidence', 'text']
        for shot in screenshots:
            if not all(field in shot for field in required_screenshot_fields):
                raise TextAnalysisError(f"スクリーンショットに必要なフィールドが不足しています: {required_screenshot_fields}")
            
            try:
                float(shot['timestamp'])
                float(shot['importance_score'])
                float(shot['ocr_confidence'])
            except (ValueError, TypeError):
                raise TextAnalysisError("スクリーンショットの数値情報が不正です")

    def _calculate_video_length(self, segments: List[Dict[str, Any]]) -> float:
        """動画の長さを計算します
        
        Args:
            segments (List[Dict[str, Any]]): セグメントデータ
            
        Returns:
            float: 動画の長さ（秒）
            
        Raises:
            TextAnalysisError: 計算に失敗した場合
        """
        try:
            return max(float(seg['end']) for seg in segments)
        except (ValueError, TypeError) as e:
            raise TextAnalysisError(f"動画長の計算に失敗しました: {str(e)}")

    def _calculate_summary_lines(self, total_segments: int, video_length: float, min_lines: int = 3) -> int:
        """要約行数を動的に計算します
        
        Args:
            total_segments (int): セグメントの総数
            video_length (float): 動画の長さ（秒）
            min_lines (int): 最小行数
            
        Returns:
            int: 要約行数
            
        Raises:
            TextAnalysisError: 計算に失敗した場合
        """
        try:
            if total_segments <= 0 or video_length <= 0:
                raise ValueError("不正な入力値です")
            
            base_count = max(min_lines, round(total_segments * 0.1))
            length_factor = max(1, video_length / 60)
            return min(int(base_count * length_factor), 15)
            
        except Exception as e:
            raise TextAnalysisError(f"要約行数の計算に失敗しました: {str(e)}")

    def _merge_segments_v2(self, segments: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
        """セグメントを統合して要約を生成します（新バージョン）
        
        Args:
            segments (List[Dict[str, Any]]): 統合するセグメント
            target_count (int): 目標セグメント数
            
        Returns:
            List[Dict[str, Any]]: 統合されたセグメント
            
        Raises:
            TextAnalysisError: 統合に失敗した場合
        """
        try:
            if len(segments) <= target_count:
                return segments
            
            # セグメントを重要度でソート
            sorted_segments = sorted(
                segments,
                key=lambda x: len(x.get('analysis', {}).get('key_points', [])),
                reverse=True
            )
            
            # 上位N件を選択
            selected_segments = sorted_segments[:target_count]
            
            # 時間順にソート
            return sorted(selected_segments, key=lambda x: float(x['start']))
            
        except Exception as e:
            raise TextAnalysisError(f"セグメントの統合処理に失敗しました: {str(e)}")

    def _integrate_screenshots_v2(self, segments: List[Dict[str, Any]], screenshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """各セグメントに最適なスクリーンショットを統合します（新バージョン）
        
        Args:
            segments (List[Dict[str, Any]]): 統合するセグメント
            screenshots (List[Dict[str, Any]]): スクリーンショットデータ
            
        Returns:
            List[Dict[str, Any]]: スクリーンショットが統合されたセグメント
            
        Raises:
            TextAnalysisError: 統合に失敗した場合
        """
        try:
            result_segments = []
            
            for segment in segments:
                start_time = float(segment['start'])
                end_time = float(segment['end'])
                
                # セグメントの時間範囲内のスクリーンショットを抽出
                relevant_shots = [
                    shot for shot in screenshots
                    if float(shot['timestamp']) >= start_time and float(shot['timestamp']) <= end_time
                ]
                
                # 最適なスクリーンショットを選択
                best_shot = None
                if relevant_shots:
                    best_shot = max(
                        relevant_shots,
                        key=lambda x: (
                            float(x['importance_score']) * 0.4 +
                            float(x['ocr_confidence']) * 0.3 +
                            self.calculate_text_quality(x['text']) * 0.3
                        )
                    )
                
                # 結果を統合
                result_segment = {
                    "time_range": {
                        "start": segment['start'],
                        "end": segment['end']
                    },
                    "text": segment['text'],
                    "ocr": [
                        {
                            "text": best_shot['text'],
                            "timestamp": best_shot['timestamp'],
                            "frame_number": best_shot['frame_number'],
                            "confidence": best_shot['ocr_confidence'],
                            "image_path": best_shot.get('image_path', '')
                        }
                    ] if best_shot else [],
                    "analysis": segment.get('analysis', {})
                }
                if best_shot:
                    result_segment["screenshot"] = {
                        "timestamp": best_shot['timestamp'],
                        "frame_number": best_shot['frame_number'],
                        "text": best_shot['text'],
                        "ocr_confidence": best_shot['ocr_confidence'],
                        "image_path": best_shot.get('image_path', '')
                    }
                result_segments.append(result_segment)
            
            return result_segments
            
        except TextAnalysisError:
            raise
        except Exception as e:
            raise TextAnalysisError(f"スクリーンショットの統合処理に失敗しました: {str(e)}") 