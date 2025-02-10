from moviepy.editor import VideoFileClip
import os
from typing import Dict, Any, Optional
import wave
import numpy as np

class AudioExtractor:
    def __init__(self, config: Dict[str, Any] = None):
        """音声抽出クラスの初期化"""
        self.config = config or {}
        self.format = self.config.get('format', 'wav')
        self.sample_rate = self.config.get('sample_rate', 16000)

    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """動画から音声を抽出する"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")

        # 出力パスが指定されていない場合、動画と同じディレクトリに作成
        if output_path is None:
            base_path = os.path.splitext(video_path)[0]
            output_path = f"{base_path}.{self.format}"

        # 出力ディレクトリの作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            # 動画を読み込み
            video = VideoFileClip(video_path)
            
            # 音声を抽出
            audio = video.audio
            
            if audio is None:
                raise ValueError(f"動画に音声トラックが含まれていません: {video_path}")

            # 音声を保存
            if self.format.lower() == 'wav':
                audio.write_audiofile(
                    output_path,
                    fps=self.sample_rate,
                    nbytes=2,
                    codec='pcm_s16le'
                )
            else:
                audio.write_audiofile(
                    output_path,
                    fps=self.sample_rate
                )

            # メタデータを取得
            metadata = {
                'duration': audio.duration,
                'fps': audio.fps,
                'format': self.format,
                'sample_rate': self.sample_rate
            }

            return output_path

        except Exception as e:
            raise RuntimeError(f"音声抽出中にエラーが発生しました: {str(e)}")

        finally:
            if 'video' in locals():
                video.close()

    def get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """音声ファイルの情報を取得する"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

        if audio_path.lower().endswith('.wav'):
            with wave.open(audio_path, 'rb') as wav:
                return {
                    'channels': wav.getnchannels(),
                    'sample_width': wav.getsampwidth(),
                    'frame_rate': wav.getframerate(),
                    'frames': wav.getnframes(),
                    'duration': wav.getnframes() / wav.getframerate(),
                    'format': 'wav'
                }
        else:
            # WAV以外のフォーマットの場合はmovieoyを使用
            audio = VideoFileClip(audio_path).audio
            return {
                'duration': audio.duration,
                'fps': audio.fps,
                'format': os.path.splitext(audio_path)[1][1:],
                'sample_rate': self.sample_rate
            }
