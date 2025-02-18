import os
from pathlib import Path
import wave
from typing import Dict, Any, Optional
import ffmpeg
import logging

class AudioExtractionError(Exception):
    """音声抽出時のエラーを表すカスタム例外クラス"""
    pass

class AudioExtractor:
    def __init__(self, config: Dict[str, Any] = None):
        """
        音声抽出クラスの初期化

        Args:
            config (Dict[str, Any], optional): 設定情報。デフォルトはNone。
        """
        self.config = config or {}
        self._format = self.config.get('format', 'wav')
        self._sample_rate = self.config.get('sample_rate', 16000)
        self.output_dir = Path(self.config.get('output_dir', 'output/audio'))
        self.logger = logging.getLogger(__name__)
        
        # 出力ディレクトリの作成
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def format(self) -> str:
        """音声フォーマットを取得します。"""
        return self._format

    @property
    def sample_rate(self) -> int:
        """サンプリングレートを取得します。"""
        return self._sample_rate

    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        動画から音声を抽出します。

        Args:
            video_path (str): 動画ファイルのパス
            output_path (Optional[str], optional): 出力ファイルのパス。デフォルトはNone。

        Returns:
            str: 抽出された音声ファイルのパス

        Raises:
            FileNotFoundError: 動画ファイルが存在しない場合
            AudioExtractionError: 音声抽出中にエラーが発生した場合
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

            # ffmpegを使用して音声を抽出
            stream = ffmpeg.input(str(video_path))
            stream = ffmpeg.output(stream, str(output_path),
                                 acodec='pcm_s16le',
                                 ac=1,
                                 ar=str(self.sample_rate))

            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, overwrite_output=True)
            self.logger.info(f"音声を抽出しました: {output_path}")

            return str(output_path)

        except ffmpeg.Error as e:
            if hasattr(e, 'stderr') and e.stderr is not None:
                error_message = e.stderr.decode()
            else:
                error_message = str(e)
            raise AudioExtractionError(f"音声抽出中にエラーが発生しました: {error_message}")
        except Exception as e:
            raise AudioExtractionError(f"音声抽出中にエラーが発生しました: {str(e)}")

    def get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """
        音声ファイルの情報を取得します。

        Args:
            audio_path (str): 音声ファイルのパス

        Returns:
            Dict[str, Any]: 音声ファイルの情報

        Raises:
            FileNotFoundError: 音声ファイルが存在しない場合
            AudioExtractionError: 情報取得中にエラーが発生した場合
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

        try:
            probe = ffmpeg.probe(audio_path)
            audio_info = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
            
            return {
                'channels': int(audio_info.get('channels', 0)),
                'sample_width': int(audio_info.get('bits_per_sample', 0)) // 8 or 2,  # デフォルト値を2に設定
                'frame_rate': int(float(audio_info.get('sample_rate', 0))),
                'frames': int(float(audio_info.get('duration', 0)) * float(audio_info.get('sample_rate', 0))),
                'duration': float(audio_info.get('duration', 0)),
                'format': os.path.splitext(audio_path)[1][1:]
            }

        except ffmpeg.Error as e:
            if hasattr(e, 'stderr') and e.stderr is not None:
                error_message = e.stderr.decode()
            else:
                error_message = str(e)
            raise AudioExtractionError(f"音声情報の取得中にエラーが発生しました: {error_message}")
        except Exception as e:
            raise AudioExtractionError(f"音声情報の取得中にエラーが発生しました: {str(e)}")
