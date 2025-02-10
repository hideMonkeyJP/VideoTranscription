from typing import Dict, List, Optional, Any
import whisper
import librosa
import numpy as np
import logging
from pathlib import Path

class TranscriptionError(Exception):
    """音声認識処理に関するエラー"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}

class TranscriptionProcessor:
    def __init__(self, model_name: str = "medium", device: str = "cpu"):
        """音声認識プロセッサを初期化します
        
        Args:
            model_name (str): 使用するWhisperモデルの名前 (tiny, base, small, medium, large)
            device (str): 使用するデバイス (cpu, cuda)
        """
        self.model = whisper.load_model(model_name, device=device)
        self.logger = logging.getLogger(__name__)
        
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """音声ファイルを文字起こしします
        
        Args:
            audio_path (str): 音声ファイルのパス
            
        Returns:
            Dict[str, Any]: 文字起こし結果
            
        Raises:
            TranscriptionError: 音声認識に失敗した場合
        """
        try:
            # 音声の前処理
            audio = self._preprocess_audio(audio_path)
            
            # Whisperによる文字起こし
            result = self.model.transcribe(
                audio,
                language="ja",
                task="transcribe",
                condition_on_previous_text=True,
                word_timestamps=True
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"音声認識中にエラーが発生: {str(e)}")
            raise TranscriptionError(f"音声認識に失敗しました: {str(e)}")
            
    def _preprocess_audio(self, audio_path: str) -> np.ndarray:
        """音声ファイルを前処理します
        
        Args:
            audio_path (str): 音声ファイルのパス
            
        Returns:
            np.ndarray: 前処理済みの音声データ
            
        Raises:
            TranscriptionError: 前処理に失敗した場合
        """
        try:
            # 音声の読み込みとリサンプリング
            audio, sr = librosa.load(audio_path, sr=16000, dtype=np.float32)
            
            # ノイズ除去（プリエンファシスフィルタ）
            audio = librosa.effects.preemphasis(audio).astype(np.float32)
            
            # 無音区間の検出と除去
            non_silent = librosa.effects.split(
                audio,
                top_db=30,
                frame_length=1024,
                hop_length=256
            )
            
            if len(non_silent) > 0:
                # 無音区間を除去した音声を結合
                audio_cleaned = np.concatenate([
                    audio[start:end] for start, end in non_silent
                ]).astype(np.float32)
                
                # 音量の正規化
                audio_cleaned = librosa.util.normalize(audio_cleaned).astype(np.float32)
                
                return audio_cleaned
            
            return audio.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"音声の前処理中にエラーが発生: {str(e)}")
            raise TranscriptionError(f"音声の前処理に失敗しました: {str(e)}")
            
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