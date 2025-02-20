from typing import Dict, List, Optional, Any
import whisper
import librosa
import numpy as np
import logging
from pathlib import Path
import re
import torch

class TranscriptionError(Exception):
    """音声認識処理に関するエラー"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}

class TranscriptionProcessor:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        TranscriptionProcessorを初期化します。

        Args:
            config (Optional[Dict[str, Any]], optional): 設定辞書
                - model_name (str): Whisperモデル名（デフォルト: "base"）
                - language (str): 言語（デフォルト: "ja"）
                - auto_optimize (bool): 自動最適化を有効にするかどうか
                - device (str): 使用するデバイス（"cuda"または"cpu"または"mps"）
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 基本設定
        self.model_name = self.config.get('model_name', 'medium')
        self.language = self.config.get('language', 'ja')
        self.auto_optimize = self.config.get('auto_optimize', True)
        
        # デバイスの自動選択
        self.device = self._select_optimal_device()
        self.logger.info(f"選択されたデバイス: {self.device}")
        
        try:
            self.model = whisper.load_model(self.model_name, device=self.device)
            self.logger.info(f"Whisperモデルを読み込みました: {self.model_name} (device: {self.device})")
        except Exception as e:
            raise TranscriptionError(f"モデルの読み込みに失敗: {str(e)}")
        
    def _select_optimal_device(self) -> str:
        """
        利用可能な最適なデバイスを選択します。
        
        Returns:
            str: 選択されたデバイス ("cuda", "mps", "cpu")
        """
        # 設定で指定されている場合はそれを使用
        if 'device' in self.config:
            return self.config['device']
            
        try:
            # NVIDIA GPU (CUDA)の確認
            if torch.cuda.is_available():
                self.logger.info("CUDA対応GPUが利用可能です")
                return "cuda"
                
            # Apple Silicon (MPS)の確認
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.logger.info("Apple Silicon GPUが利用可能です")
                return "mps"
                
        except Exception as e:
            self.logger.warning(f"GPUの確認中にエラーが発生: {str(e)}")
            
        # GPUが利用できない場合はCPUを使用
        self.logger.info("CPUを使用します")
        return "cpu"
        
    def transcribe_audio(self, audio_path: str) -> List[Dict[str, Any]]:
        """音声ファイルを文字起こしします"""
        try:
            # 音声の前処理
            audio = self._preprocess_audio(audio_path)
            
            # Whisperによる文字起こし
            params = {
                'language': self.language,
                'task': 'transcribe',
                'beam_size': 5,
                'best_of': 3,
                'patience': 1.0,
                'length_penalty': 1.0,
                'condition_on_previous_text': True,
                'initial_prompt': """
                以下は日本語の会話の文字起こしです。
                自然な話し言葉で、句読点を適切に含みます。
                """,
                'suppress_tokens': [-1],
                'without_timestamps': False,
                'temperature': 0.0
            }
            
            result = self.model.transcribe(audio_path, **params)
            
            # Whisperの出力を直接使用
            segments = []
            for segment in result.get("segments", []):
                text = segment.get("text", "").strip()
                if not text:
                    continue
                
                segments.append({
                    "text": text,
                    "start": round(segment.get("start", 0.0), 2),
                    "end": round(segment.get("end", 0.0), 2),
                    "confidence": segment.get("confidence", 0.0)
                })
            
            return segments
            
        except Exception as e:
            self.logger.error(f"音声認識中にエラーが発生: {str(e)}")
            raise TranscriptionError(f"音声認識に失敗しました: {str(e)}")
            
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """認識結果の信頼度を計算"""
        segments = result.get("segments", [])
        if not segments:
            return 0.0
        
        # 各セグメントの信頼度の平均を計算
        confidences = [seg.get("confidence", 0.0) for seg in segments]
        return sum(confidences) / len(confidences)
        
    def _get_transcribe_params(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        音声認識パラメータを取得します。

        Args:
            audio (np.ndarray): 音声データ

        Returns:
            Dict[str, Any]: 音声認識パラメータ
        """
        params = {
            'language': self.language,
            'task': 'transcribe',
            'beam_size': 10,  # 5から10に増加
            'best_of': 5,     # 3から5に増加
            'patience': 2.0,  # 1.0から2.0に増加
            'length_penalty': 0.8,  # 1.0から0.8に変更
            'condition_on_previous_text': True,
            'initial_prompt': """
            以下は日本語の会話の文字起こしです。
            自然な話し言葉で、句読点を適切に含みます。
            タスク管理や業務効率化に関する内容を含みます。
            例：
            - タスク管理について説明します。
            - 効率的な管理方法として重要なのは、
            - 日々の作業を確実に記録することです。
            - それによって進捗が明確になります。
            """,
            'suppress_tokens': [-1],
            'without_timestamps': False,
            'temperature': 0.0
        }

        # 音声の長さに応じて温度パラメータを調整
        duration = len(audio) / 16000
        if duration > 300:  # 5分以上の場合
            params['temperature'] = [0.0, 0.2, 0.4]  # より広い温度範囲で試行
            params['best_of'] = 3  # 候補数を減らして処理を高速化
        
        return params
        
    def _process_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """セグメントを処理して最適な形式に変換"""
        processed_segments = []
        
        for segment in segments:
            # テキストの正規化と誤認識の修正
            text = self._normalize_text(segment.get("text", "").strip())
            
            # 長いセグメントを分割（句点や読点で区切る）
            sub_texts = self._split_long_segment(text)
            
            if len(sub_texts) > 1:
                # セグメントの時間を文の長さに応じて分割
                processed_segments.extend(
                    self._create_length_based_segments(sub_texts, segment)
                )
            else:
                processed_segments.append(
                    self._create_segment(text, segment)
                )
                
        return processed_segments
        
    def _normalize_text(self, text: str) -> str:
        """テキストを正規化"""
        # 一般的な誤認識の修正
        replacements = {
            # 基本的な誤認識
            "化石で": "稼げる",
            "化石でない": "稼げない",
            "SNス": "SNS",
            "くす": "なくす",
            "しうか": "しようか",
            "くしましょう": "くしましょう",
            "管理しましょう": "管理しましょう",
            
            # 語尾の正規化
            "です。": "ですね",
            "ます。": "ますね",
            "した。": "したね",
            "思います。": "思いますね",
            
            # 口語表現
            "えーと": "ええと",
            "あのー": "あの",
            "えっと": "ええと",
            "まぁ": "まあ",
            "じゃ": "では",
            "ちょっと": "少し",
            "すごく": "とても",
            "めっちゃ": "とても",
            "全然": "まったく",
            "結構": "かなり",
            
            # 縮約形
            "わかんない": "わからない",
            "できんない": "できない",
            "してんの": "しているの",
            "やってんの": "やっているの",
            "みたいな": "のような",
            "っていう": "という",
            
            # フィラー除去
            "なんか": "",
            "みたいな": "",
            "的な": "",
            "感じ": "",
            "とか": "",
            "って": "",
            "まあ": "",
            "ね": "",
            "よ": "",
            "な": "",
            "さ": ""
        }
        
        # 長い表現から順に置換するためにソート
        replacements = dict(sorted(replacements.items(), key=lambda x: len(x[0]), reverse=True))
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        # 余分な空白の削除
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 記号の正規化
        text = text.replace('…', '...')
        text = text.replace('―', 'ー')
        text = text.replace('～', 'ー')
        
        # 不自然な句点の除去
        text = re.sub(r'。\s*$', '', text)
        
        return text
        
    def _split_long_segment(self, text: str, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """長い文を時間で強制的に分割"""
        duration = end_time - start_time
        # 5秒ごとに分割
        time_step = 5.0
        num_splits = max(2, int(duration / time_step))
        
        # テキストを文字数で均等に分割
        chars = list(text)
        chars_per_split = len(chars) // num_splits
        splits = []
        
        for i in range(num_splits):
            start_idx = i * chars_per_split
            end_idx = start_idx + chars_per_split if i < num_splits - 1 else len(chars)
            split_text = "".join(chars[start_idx:end_idx]).strip()
            
            if split_text:
                split_start = start_time + (duration * (i / num_splits))
                split_end = start_time + (duration * ((i + 1) / num_splits))
                
                splits.append({
                    "text": split_text,
                    "start": round(split_start, 2),
                    "end": round(split_end, 2),
                    "confidence": 0.0
                })
        
        return splits
        
    def _create_length_based_segments(
        self, sub_texts: List[str], segment: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """文の長さに基づいてセグメントを作成"""
        total_length = sum(len(text) for text in sub_texts)
        start_time = segment.get("start", 0.0)
        end_time = segment.get("end", 0.0)
        duration = end_time - start_time
        
        segments = []
        current_position = start_time
        
        for text in sub_texts:
            # 文の長さに応じて時間を配分
            segment_duration = duration * (len(text) / total_length)
            segment_end = round(current_position + segment_duration, 2)
            
            segments.append({
                "text": text,
                "start": round(current_position, 2),
                "end": segment_end,
                "confidence": segment.get("confidence", 0.0)
            })
            
            current_position = segment_end
            
        return segments
        
    def _create_segment(self, text: str, segment: Dict[str, Any]) -> Dict[str, Any]:
        """単一のセグメントを作成"""
        return {
            "text": text,
            "start": round(segment.get("start", 0.0), 2),
            "end": round(segment.get("end", 0.0), 2),
            "confidence": segment.get("confidence", 0.0)
        }
            
    def _preprocess_audio(self, audio_path: str) -> np.ndarray:
        """音声ファイルを前処理します"""
        try:
            # 音声の読み込みとリサンプリング
            audio, sr = librosa.load(audio_path, sr=16000, dtype=np.float32)
            
            if self.auto_optimize:
                # SNRの計算
                noise_floor = np.mean(np.abs(audio[:int(sr/10)]))
                signal = np.abs(audio)
                snr = 20 * np.log10(np.mean(signal) / noise_floor)
                
                # 無音区間の検出
                non_silent = librosa.effects.split(audio, top_db=20)
                silence_ratio = 1 - (np.sum([end-start for start, end in non_silent]) / len(audio))
                
                # パラメータの最適化
                top_db = self._optimize_top_db(snr)
                frame_length = self._optimize_frame_length(sr, silence_ratio)
                hop_length = self._optimize_hop_length(sr)
            else:
                top_db = 20
                frame_length = 1024
                hop_length = 256
            
            # ノイズ除去（プリエンファシスフィルタ）
            audio = librosa.effects.preemphasis(audio, coef=0.95).astype(np.float32)
            
            # 無音区間の検出と除去
            non_silent = librosa.effects.split(
                audio,
                top_db=top_db,
                frame_length=frame_length,
                hop_length=hop_length
            )
            
            if len(non_silent) > 0:
                # 無音区間を除去した音声を結合
                audio_cleaned = np.concatenate([
                    audio[start:end] for start, end in non_silent
                ]).astype(np.float32)
                
                # 音量の正規化（RMSベース）
                audio_cleaned = librosa.util.normalize(audio_cleaned, norm=2).astype(np.float32)
                
                return audio_cleaned
            
            return audio.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"音声の前処理中にエラーが発生: {str(e)}")
            raise TranscriptionError(f"音声の前処理に失敗しました: {str(e)}")
            
    def _optimize_top_db(self, snr: float) -> float:
        """top_dbパラメータを最適化"""
        if snr > 30:  # SNRが高い場合
            return 30
        elif snr > 20:
            return 25
        else:  # ノイズが多い場合
            return 20

    def _optimize_frame_length(self, sr: int, silence_ratio: float) -> int:
        """frame_lengthパラメータを最適化"""
        if silence_ratio > 0.3:  # 無音区間が多い場合
            return int(sr * 0.05)  # 50ms
        else:
            return int(sr * 0.03)  # 30ms

    def _optimize_hop_length(self, sr: int) -> int:
        """hop_lengthパラメータを最適化"""
        return int(sr * 0.01)  # 10ms
            
    def get_word_timestamps(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """文字起こし結果から単語単位のタイムスタンプを取得します
        
        Args:
            result (Dict[str, Any]): transcribe_audio()の結果
            
        Returns:
            List[Dict[str, Any]]: 単語ごとのタイムスタンプ情報
            
        Raises:
            TranscriptionError: タイムスタンプの取得に失敗した場合
        """
        try:
            words = []
            for segment in result.get("segments", []):
                for word_info in segment.get("words", []):
                    word = {
                        "text": word_info.get("word", "").strip(),
                        "start": round(word_info.get("start", 0), 2),
                        "end": round(word_info.get("end", 0), 2),
                        "probability": round(word_info.get("probability", 0), 2)
                    }
                    words.append(word)
            return words
            
        except Exception as e:
            self.logger.error(f"単語タイムスタンプの取得中にエラーが発生: {str(e)}")
            raise TranscriptionError(f"単語タイムスタンプの取得に失敗しました: {str(e)}")
            
    def _split_segment(self, text: str, segment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """セグメントを適切な長さに分割します"""
        # 文の区切りパターン
        sentence_patterns = [
            r'(?<=[。！？\.\!\?])',  # 句点での分割
            r'(?<=。)(?![\)）])',    # 閉じ括弧を考慮した句点での分割
            r'(?<=[\.\!\?])(?![\.!\?])',  # 英文の句点での分割
            r'(?<=。)(?!\s*[\)）])'  # 句点と閉じ括弧の間のスペースを考慮
        ]
        
        # パターンに基づいて分割
        parts = [text]
        for pattern in sentence_patterns:
            new_parts = []
            for part in parts:
                split_result = re.split(pattern, part)
                new_parts.extend([p.strip() for p in split_result if p.strip()])
            parts = new_parts
        
        # 長い部分（40文字以上）を読点で分割
        final_parts = []
        for part in parts:
            if len(part) > 40:
                # 読点による分割
                sub_parts = re.split(r'(?<=、)', part)
                sub_parts = [p.strip() for p in sub_parts if p.strip()]
                if sub_parts:
                    final_parts.extend(sub_parts)
                else:
                    final_parts.append(part)
            else:
                final_parts.append(part)
        
        # 短すぎる部分（5文字未満）を前後のセグメントと結合
        merged_parts = []
        current = ""
        
        for part in final_parts:
            if len(part) < 5:
                current += part
            else:
                if current:
                    if len(current) < 5:
                        current += part
                    else:
                        merged_parts.append(current)
                        current = part
                else:
                    current = part
        
        if current:
            merged_parts.append(current)
        
        if not merged_parts:
            return []
        
        # 時間を文字数に応じて配分
        start_time = segment.get("start", 0.0)
        end_time = segment.get("end", 0.0)
        duration = end_time - start_time
        
        total_length = sum(len(p) for p in merged_parts)
        segments = []
        current_time = start_time
        
        for part in merged_parts:
            part_duration = duration * (len(part) / total_length)
            part_end = round(current_time + part_duration, 2)
            
            segments.append({
                "text": part,
                "start": round(current_time, 2),
                "end": part_end,
                "confidence": 0.0
            })
            
            current_time = part_end
        
        return segments 