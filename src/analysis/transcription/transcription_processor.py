from typing import Dict, List, Optional, Any
import whisper
import librosa
import numpy as np
import logging
from pathlib import Path
import re
import torch
import time
import os

class TranscriptionError(Exception):
    """音声認識処理に関するエラー"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}

class TranscriptionProcessor:
    def __init__(self, config=None):
        """初期化"""
        self.logger = logging.getLogger(__name__)
        
        # 設定の読み込み
        if config is None:
            config_loader = Config()
            self.config = config_loader.get_all()
        else:
            self.config = config
            
        # Whisper設定の取得
        whisper_config = self.config.get('models', {}).get('whisper', {})
        model_config = whisper_config.get('model', {})
        self.model_name = model_config.get('name', 'tiny')
        
        # 最適化設定
        self.optimize_model = whisper_config.get('optimize_model', False)
        self.use_optimized_model = whisper_config.get('use_optimized_model', False)
        self.optimized_model_path = os.path.join(
            os.path.expanduser(whisper_config.get('cache_dir', '~/.cache/whisper')),
            f"whisper_{self.model_name}_optimized.pt"
        )
        
        # 音声処理設定
        self.fp16 = whisper_config.get('fp16', True)
        self.split_long_audio = whisper_config.get('split_long_audio', True)
        self.remove_silence = whisper_config.get('remove_silence', False)
        
        # 音声分割設定
        segmentation_config = whisper_config.get('audio_segmentation', {})
        self.min_segment_length = segmentation_config.get('min_segment_length', 30)
        self.chunk_length = segmentation_config.get('chunk_length', 30)
        self.overlap = segmentation_config.get('overlap', 2)
        self.overlap_threshold = segmentation_config.get('overlap_threshold', 0.5)
        
        # 日本語設定の取得
        japanese_config = self.config.get('japanese', {})
        self.initial_prompt = japanese_config.get('initial_prompt', '')
        
        # モデルの読み込み
        self.logger.info(f"Whisperモデルを読み込み中: {self.model_name}")
        
        # デバイスの選択
        self.device = self._get_device()
        self.logger.info(f"使用デバイス: {self.device}")
        
        # モデルの読み込み
        self.model = self._load_model(self.model_name)
        
    def _optimize_model(self):
        """モデルをTorchScriptに変換して最適化"""
        try:
            self.logger.info("モデルをTorchScriptに変換して最適化します...")
            # TorchScriptモデルに変換
            torch_model = torch.jit.script(self.model)
            # 保存
            torch.jit.save(torch_model, self.optimized_model_path)
            self.logger.info(f"モデルを最適化して保存しました: {self.optimized_model_path}")
            # 最適化されたモデルを読み込み
            self.model = torch_model
            self.is_optimized = True
        except Exception as e:
            self.logger.warning(f"モデルの最適化に失敗しました: {str(e)}")
            
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
        """音声を文字起こしします"""
        try:
            start_total = time.time()
            
            self.logger.info(f"音声認識を開始: {audio_path}")
            
            # 音声の前処理
            preprocess_start = time.time()
            audio = self._preprocess_audio(audio_path)
            preprocess_time = time.time() - preprocess_start
            self.logger.info(f"音声前処理完了: {preprocess_time:.2f}秒")
            
            # 音声の長さを取得
            audio_duration = len(audio) / 16000  # 16kHzサンプリングレートを想定
            self.logger.info(f"音声の長さ: {audio_duration:.2f}秒")
            
            # 音声の長さが設定値を超える場合、または分割処理が有効な場合は分割処理
            if audio_duration > self.min_segment_length or self.split_long_audio:
                self.logger.info(f"音声を分割して処理します（長さ: {audio_duration:.2f}秒）")
                segments = self._process_long_audio(audio)
            else:
                # Whisperモデルによる文字起こし
                transcribe_start = time.time()
                params = self._get_transcribe_params(audio)
                self.logger.info(f"文字起こしパラメータ: {params}")
                
                # 30秒以下の場合はpad_or_trimを使用
                if audio_duration <= 30:
                    audio = whisper.pad_or_trim(audio)
                
                result = self.model.transcribe(audio, **params)
                transcribe_time = time.time() - transcribe_start
                self.logger.info(f"Whisper文字起こし完了: {transcribe_time:.2f}秒")
                
                # セグメントの処理
                segment_start = time.time()
                segments = self._process_segments(result.get('segments', []))
                segment_time = time.time() - segment_start
                self.logger.info(f"セグメント処理完了: {segment_time:.2f}秒")
            
            total_time = time.time() - start_total
            self.logger.info(f"音声認識全体の処理時間: {total_time:.2f}秒")
            
            return segments
            
        except Exception as e:
            error_msg = f"音声認識中にエラーが発生: {str(e)}"
            self.logger.error(error_msg)
            raise TranscriptionError(error_msg)
    
    def _process_long_audio(self, audio: np.ndarray) -> List[Dict[str, Any]]:
        """長い音声を分割して処理します"""
        # 設定から分割パラメータを取得
        chunk_length = int(self.chunk_length * 16000)  # 秒 * 16kHz
        overlap = int(self.overlap * 16000)    # 秒 * 16kHz
        
        all_segments = []
        offset = 0
        
        # 音声の総長を取得
        total_duration = len(audio) / 16000
        self.logger.info(f"音声の総長: {total_duration:.2f}秒、分割処理を開始します")
        
        chunk_count = 0
        while offset < len(audio):
            chunk_count += 1
            # チャンクの終了位置を計算
            end = min(offset + chunk_length, len(audio))
            
            # チャンクを抽出
            chunk = audio[offset:end]
            
            # チャンクが短すぎる場合はスキップ
            if len(chunk) < 3 * 16000:  # 3秒未満
                self.logger.info(f"チャンクが短すぎるためスキップ: {len(chunk)/16000:.2f}秒")
                break
                
            self.logger.info(f"チャンク {chunk_count} 処理中: {offset/16000:.2f}秒 - {end/16000:.2f}秒 (長さ: {len(chunk)/16000:.2f}秒)")
            
            # チャンクを処理
            params = self._get_transcribe_params(chunk)
            
            # 30秒に合わせてパディングまたはトリミング
            chunk_processed = whisper.pad_or_trim(chunk)
            
            # チャンクを処理
            chunk_start_time = time.time()
            result = self.model.transcribe(chunk_processed, **params)
            chunk_process_time = time.time() - chunk_start_time
            self.logger.info(f"チャンク {chunk_count} の処理時間: {chunk_process_time:.2f}秒")
            
            # セグメントを処理して時間オフセットを追加
            chunk_segments = self._process_segments(result.get('segments', []))
            
            # 空のセグメントをスキップ
            if not chunk_segments:
                self.logger.info(f"チャンク {chunk_count} にセグメントがないためスキップします")
                offset = end - overlap
                continue
                
            # 時間オフセットを追加
            for segment in chunk_segments:
                # 30秒以内の相対時間から実際の時間に変換
                rel_start = segment['start']
                rel_end = segment['end']
                
                # 相対時間が30秒を超える場合は30秒に制限（Whisperの制限による）
                if rel_start > 30:
                    rel_start = 30
                if rel_end > 30:
                    rel_end = 30
                
                # 開始時間と終了時間が同じ場合は修正
                if rel_start == rel_end:
                    rel_end += 0.1
                    self.logger.warning(f"相対タイムスタンプが同じセグメントを修正: {rel_start} → {rel_end}")
                
                # 実際の時間に変換
                segment['start'] = offset / 16000 + rel_start
                segment['end'] = offset / 16000 + rel_end
                
            all_segments.extend(chunk_segments)
            
            # オーバーラップを考慮して次のオフセットを計算
            offset = end - overlap
            
        # セグメントを時間順にソート
        all_segments.sort(key=lambda x: x['start'])
        
        # 重複するセグメントをマージ
        merged_segments = self._merge_overlapping_segments(all_segments)
        
        self.logger.info(f"分割処理完了: {len(merged_segments)}個のセグメントを生成")
        
        return merged_segments
    
    def _merge_overlapping_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """重複するセグメントをマージします"""
        if not segments:
            return []
            
        # セグメントを時間順にソート
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        
        merged = []
        current = sorted_segments[0]
        
        for next_seg in sorted_segments[1:]:
            # 重複判定（設定された閾値以上の重複がある場合）
            current_duration = current['end'] - current['start']
            next_duration = next_seg['end'] - next_seg['start']
            
            # 重複部分の計算
            overlap_start = max(current['start'], next_seg['start'])
            overlap_end = min(current['end'], next_seg['end'])
            overlap = max(0, overlap_end - overlap_start)
            
            # 重複率の計算
            current_overlap_ratio = overlap / current_duration if current_duration > 0 else 0
            next_overlap_ratio = overlap / next_duration if next_duration > 0 else 0
            
            # 重複が閾値を超える場合はマージ
            if current_overlap_ratio > self.overlap_threshold or next_overlap_ratio > self.overlap_threshold:
                # テキストの長さに基づいて選択（長い方を優先）
                if len(next_seg['text']) > len(current['text']):
                    # 次のセグメントのテキストが長い場合は、それを採用
                    current['text'] = next_seg['text']
                
                # 時間範囲を更新（より広い範囲を採用）
                current['start'] = min(current['start'], next_seg['start'])
                current['end'] = max(current['end'], next_seg['end'])
            else:
                # 重複が少ない場合は新しいセグメントとして追加
                merged.append(current)
                current = next_seg
        
        # 最後のセグメントを追加
        merged.append(current)
        
        # 開始時間と終了時間が同じセグメントを修正
        for segment in merged:
            if segment['start'] == segment['end']:
                # 終了時間を少し延長する（0.1秒）
                segment['end'] += 0.1
                self.logger.warning(f"タイムスタンプが同じセグメントを修正: {segment['start']} → {segment['end']}")
        
        self.logger.info(f"セグメントマージ: {len(segments)}個 → {len(merged)}個")
        return merged
            
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """認識結果の信頼度を計算"""
        segments = result.get("segments", [])
        if not segments:
            return 0.0
        
        # 各セグメントの信頼度の平均を計算
        confidences = [seg.get("confidence", 0.0) for seg in segments]
        return sum(confidences) / len(confidences)
        
    def _get_device(self):
        """最適なデバイスを選択します"""
        # 設定からデバイスを取得
        device = self.config.get('models', {}).get('whisper', {}).get('device')
        if not device:
            # トップレベルのデバイス設定を確認
            device = self.config.get('device')
        
        # デバイスが指定されていない場合は自動検出
        if not device:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.logger.info(f"使用デバイス: {device}")
        return device
            
    def _load_model(self, model_name):
        """Whisperモデルを読み込みます"""
        try:
            # キャッシュディレクトリの設定
            whisper_config = self.config.get('models', {}).get('whisper', {})
            cache_dir = whisper_config.get('cache_dir', '~/.cache/whisper')
            if cache_dir.startswith('~'):
                cache_dir = os.path.expanduser(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            
            # 最適化済みモデルのパス
            optimized_model_path = os.path.join(cache_dir, f"whisper_{model_name}_optimized.pt")
            
            # 最適化済みモデルが存在し、使用が有効な場合はそれを読み込む
            if os.path.exists(optimized_model_path) and self.use_optimized_model:
                self.logger.info(f"最適化済みモデルを読み込みます: {optimized_model_path}")
                model = torch.jit.load(optimized_model_path)
                self.is_optimized = True
            else:
                # 通常のモデル読み込み
                self.logger.info(f"通常のモデルを読み込みます: {model_name}")
                model = whisper.load_model(model_name, device=self.device)
                self.is_optimized = False
                
                # モデルの最適化（設定で有効な場合）
                if self.optimize_model:
                    try:
                        self.logger.info("モデルを最適化しています...")
                        # 入力サンプルの作成
                        sample_input = torch.zeros((1, 80, 3000), device=self.device)
                        
                        # TorchScriptモデルに変換
                        with torch.no_grad():
                            script_model = torch.jit.trace(model.encoder, sample_input)
                            
                        # 最適化済みモデルを保存
                        script_model.save(optimized_model_path)
                        self.logger.info(f"最適化済みモデルを保存しました: {optimized_model_path}")
                        self.is_optimized = True
                    except Exception as e:
                        self.logger.warning(f"モデルの最適化に失敗しました: {str(e)}")
            
            self.logger.info(f"Whisperモデルを読み込みました: {model_name} (device: {self.device})")
            return model
        except Exception as e:
            error_msg = f"モデルの読み込みに失敗: {str(e)}"
            self.logger.error(error_msg)
            raise TranscriptionError(error_msg)
            
    def _get_transcribe_params(self, audio):
        """文字起こしパラメータを取得します"""
        # 設定からパラメータを取得
        whisper_config = self.config.get('models', {}).get('whisper', {})
        transcribe_params = whisper_config.get('transcribe_params', {})
        
        # 言語設定
        language = self.config.get('speech_recognition', {}).get('language', 'ja')
        
        # 日本語設定
        japanese_config = self.config.get('japanese', {})
        initial_prompt = japanese_config.get('initial_prompt', '')
        
        # パラメータの設定
        params = {
            'language': language,
            'task': 'transcribe',
            'fp16': self.fp16,
            'beam_size': transcribe_params.get('beam_size', 5),
            'best_of': transcribe_params.get('best_of', 5),
            'patience': transcribe_params.get('patience', 1.0),
            'length_penalty': transcribe_params.get('length_penalty', 1.0),
            'condition_on_previous_text': True,
            'initial_prompt': initial_prompt,
            'suppress_tokens': [-1],
            'without_timestamps': False,
            'temperature': transcribe_params.get('temperature', 0.0),
        }
        
        return params
        
    def _process_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Whisperの出力セグメントを処理します"""
        processed_segments = []
        
        for segment in segments:
            text = segment.get("text", "").strip()
            if not text:
                continue
                
            # テキストの正規化
            text = self._normalize_text(text)
            
            # 長いセグメントの分割
            if len(text) > 100:  # 100文字以上の場合は分割
                start_time = segment.get("start", 0.0)
                end_time = segment.get("end", 0.0)
                sub_segments = self._split_long_segment(text, start_time, end_time)
                processed_segments.extend(sub_segments)
            else:
                # 単一のセグメントとして追加
                processed_segments.append(self._create_segment(text, segment))
                
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
            start_time = time.time()
            self.logger.info(f"音声前処理を開始: {audio_path}")
            
            # 音声の読み込み（最適化：直接whisperの関数を使用）
            load_start = time.time()
            audio = whisper.load_audio(audio_path)
            load_time = time.time() - load_start
            self.logger.info(f"音声ファイル読み込み完了: {load_time:.2f}秒")
            
            # 音声の長さを記録（切り詰める前）
            original_duration = len(audio) / 16000
            self.logger.info(f"元の音声の長さ: {original_duration:.2f}秒")
            
            # 無音区間の検出と除去（処理時間短縮のため条件付きで実行）
            if self.remove_silence:
                vad_start = time.time()
                intervals = librosa.effects.split(
                    audio,
                    top_db=20,
                    frame_length=2048,
                    hop_length=512
                )
                vad_time = time.time() - vad_start
                self.logger.info(f"無音区間検出完了: {vad_time:.2f}秒")
            else:
                self.logger.info("無音区間除去をスキップします")
            
            # 音声の正規化 - pad_or_trimを使用しない
            norm_start = time.time()
            # 30秒に切り詰めるのではなく、元の長さを保持
            # audio = whisper.pad_or_trim(audio)  # Whisperの標準前処理を使用
            norm_time = time.time() - norm_start
            self.logger.info(f"音声正規化完了: {norm_time:.2f}秒")
            
            total_time = time.time() - start_time
            self.logger.info(f"音声前処理全体の時間: {total_time:.2f}秒")
            
            return audio
            
        except Exception as e:
            error_msg = f"音声前処理中にエラーが発生: {str(e)}"
            self.logger.error(error_msg)
            raise TranscriptionError(error_msg)
            
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