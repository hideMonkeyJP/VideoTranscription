import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import subprocess
import whisper
from pydub import AudioSegment
import json
import numpy as np
import librosa

class AudioProcessingError(Exception):
    """音声処理に関するエラー"""
    pass

class AudioProcessor:
    """動画から音声を抽出し処理するクラス"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初期化
        
        Args:
            config (Dict[str, Any], optional): 設定辞書
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 音声処理の設定
        self.output_dir = Path(self.config.get('output_dir', 'output/audio'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Whisperモデルの設定
        self.model_name = self.config.get('model_name', 'base')
        self.device = self.config.get('device', 'cpu')
        self.model = whisper.load_model(self.model_name, device=self.device)
        
        # 音声設定
        self._format = self.config.get('format', 'wav')
        self._sample_rate = self.config.get('sample_rate', 16000)
    
    @property
    def format(self) -> str:
        """音声フォーマットを取得"""
        return self._format
    
    @property
    def sample_rate(self) -> int:
        """サンプリングレートを取得"""
        return self._sample_rate
    
    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """動画から音声を抽出
        
        Args:
            video_path (str): 動画ファイルのパス
            output_path (Optional[str], optional): 出力ファイルのパス
            
        Returns:
            str: 抽出された音声ファイルのパス
            
        Raises:
            FileNotFoundError: 動画ファイルが存在しない場合
            AudioProcessingError: 音声抽出に失敗した場合
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")
        
        try:
            video_path = Path(video_path)
            if output_path is None:
                output_path = self.output_dir / f"{video_path.stem}_audio.{self.format}"
            else:
                output_path = Path(output_path)
                os.makedirs(output_path.parent, exist_ok=True)
            
            # FFmpegを使用して音声を抽出
            command = [
                'ffmpeg', '-i', str(video_path),
                '-vn',  # 映像を無効化
                '-acodec', 'pcm_s16le',  # WAV形式で出力
                '-ar', str(self.sample_rate),  # サンプリングレート
                '-ac', '1',  # モノラル
                '-y',  # 既存ファイルを上書き
                str(output_path)
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise AudioProcessingError(f"FFmpegによる音声抽出に失敗: {result.stderr}")
            
            self.logger.info(f"音声を抽出しました: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"音声抽出中にエラーが発生: {str(e)}")
            raise AudioProcessingError(f"音声抽出に失敗しました: {str(e)}")
    
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """音声を文字起こし
        
        Args:
            audio_path (str): 音声ファイルのパス
            
        Returns:
            Dict[str, Any]: 文字起こし結果
            
        Raises:
            AudioProcessingError: 文字起こしに失敗した場合
        """
        try:
            self.logger.info("文字起こしを開始します")
            
            # Whisperで文字起こし
            result = self.model.transcribe(
                audio_path,
                language='ja',
                task='transcribe',
                verbose=False
            )
            
            # 結果を整形
            transcription = {
                'text': result['text'],
                'segments': result['segments'],
                'language': result['language']
            }
            
            # 結果をJSONファイルとして保存
            output_path = self.output_dir / "transcription.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(transcription, f, ensure_ascii=False, indent=2)
            
            self.logger.info("文字起こしが完了しました")
            return transcription
            
        except Exception as e:
            self.logger.error(f"文字起こし中にエラーが発生: {str(e)}")
            raise AudioProcessingError(f"文字起こしに失敗しました: {str(e)}")
    
    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """音声の特徴を分析
        
        Args:
            audio_path (str): 音声ファイルのパス
            
        Returns:
            Dict[str, Any]: 分析結果
            
        Raises:
            AudioProcessingError: 分析に失敗した場合
        """
        try:
            self.logger.info("音声分析を開始します")
            
            # 音声ファイルを読み込み
            audio = AudioSegment.from_wav(audio_path)
            
            # 基本的な音声特徴を抽出
            analysis = {
                'duration': len(audio) / 1000.0,  # 秒単位
                'channels': audio.channels,
                'sample_width': audio.sample_width,
                'frame_rate': audio.frame_rate,
                'frame_width': audio.frame_width,
                'rms': audio.rms,
                'max_possible_amplitude': audio.max_possible_amplitude,
                'max_dBFS': audio.max_dBFS
            }
            
            # 結果をJSONファイルとして保存
            output_path = self.output_dir / "audio_analysis.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            
            self.logger.info("音声分析が完了しました")
            return analysis
            
        except Exception as e:
            self.logger.error(f"音声分析中にエラーが発生: {str(e)}")
            raise AudioProcessingError(f"音声分析に失敗しました: {str(e)}")
    
    def get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """音声ファイルの情報を取得
        
        Args:
            audio_path (str): 音声ファイルのパス
            
        Returns:
            Dict[str, Any]: 音声ファイルの情報
            
        Raises:
            FileNotFoundError: 音声ファイルが存在しない場合
            AudioProcessingError: 情報取得に失敗した場合
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")
        
        try:
            # FFprobeを使用して音声情報を取得
            command = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                audio_path
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise AudioProcessingError(f"FFprobeによる情報取得に失敗: {result.stderr}")
            
            probe_data = json.loads(result.stdout)
            audio_stream = next(s for s in probe_data['streams'] if s['codec_type'] == 'audio')
            
            return {
                'channels': int(audio_stream.get('channels', 1)),
                'sample_width': int(audio_stream.get('bits_per_sample', 16)) // 8,
                'frame_rate': int(float(audio_stream.get('sample_rate', 0))),
                'frames': int(float(audio_stream.get('duration', 0)) * float(audio_stream.get('sample_rate', 0))),
                'duration': float(audio_stream.get('duration', 0)),
                'format': os.path.splitext(audio_path)[1][1:]
            }
            
        except Exception as e:
            self.logger.error(f"音声情報の取得中にエラーが発生: {str(e)}")
            raise AudioProcessingError(f"音声情報の取得に失敗しました: {str(e)}")
    
    def optimize_parameters(self, audio_path: str) -> Dict[str, Any]:
        """音声処理パラメータを自動最適化します。

        Args:
            audio_path (str): 音声ファイルのパス

        Returns:
            Dict[str, Any]: 最適化されたパラメータ
        """
        try:
            # 音声データの読み込み
            y, sr = librosa.load(audio_path, sr=None)
            
            # SNR（信号対雑音比）の計算
            noise_floor = np.mean(np.abs(y[:int(sr/10)]))  # 最初の0.1秒を使用
            signal = np.abs(y)
            snr = 20 * np.log10(np.mean(signal) / noise_floor)
            
            # 無音区間の検出
            non_silent = librosa.effects.split(y, top_db=20)
            silence_ratio = 1 - (np.sum([end-start for start, end in non_silent]) / len(y))
            
            # パラメータの最適化
            optimized_params = {
                'top_db': self._optimize_top_db(snr, silence_ratio),
                'frame_length': self._optimize_frame_length(sr, silence_ratio),
                'hop_length': self._optimize_hop_length(sr),
                'min_silence_duration': self._optimize_silence_duration(silence_ratio)
            }
            
            return optimized_params
            
        except Exception as e:
            raise AudioProcessingError(f"パラメータ最適化に失敗: {str(e)}")

    def _optimize_top_db(self, snr: float, silence_ratio: float) -> float:
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

    def _optimize_silence_duration(self, silence_ratio: float) -> float:
        """最小無音区間長を最適化"""
        if silence_ratio > 0.4:
            return 0.3  # 300ms
        else:
            return 0.2  # 200ms 