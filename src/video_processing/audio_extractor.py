import os
from pathlib import Path
import logging
from typing import Dict, Any, Optional

import ffmpeg
import whisper

class AudioExtractionError(Exception):
    """音声抽出時のエラーを表すカスタム例外クラス"""
    pass

class AudioExtractor:
    """動画から音声を抽出し、文字起こしを行うクラス"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        AudioExtractorを初期化します。
        
        Args:
            config (Dict[str, Any], optional): 設定辞書
                - model_name (str): Whisperモデル名
                - output_dir (str): 出力ディレクトリ
                - language (str): 文字起こし言語
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 設定の初期化
        self.model_name = self.config.get('model_name', 'medium')
        self.output_dir = Path(self.config.get('output_dir', 'output/audio'))
        self.language = self.config.get('language', 'ja')
        
        # 出力ディレクトリの作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Whisperモデルの読み込み
            self.model = whisper.load_model(self.model_name)
            self.logger.info(f"Whisperモデルを読み込みました: {self.model_name}")
        except Exception as e:
            raise AudioExtractionError(f"Whisperモデルの読み込みに失敗しました: {str(e)}")
    
    def extract_audio(self, video_path: str) -> str:
        """
        動画から音声を抽出します。
        
        Args:
            video_path (str): 動画ファイルのパス
            
        Returns:
            str: 抽出された音声ファイルのパス
        """
        try:
            video_path = Path(video_path)
            output_path = self.output_dir / f"{video_path.stem}_audio.wav"
            
            # ffmpegを使用して音声を抽出
            stream = ffmpeg.input(str(video_path))
            stream = ffmpeg.output(stream, str(output_path),
                                 acodec='pcm_s16le',
                                 ac=1,
                                 ar='16k')
            
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            self.logger.info(f"音声を抽出しました: {output_path}")
            
            return str(output_path)
            
        except ffmpeg.Error as e:
            raise AudioExtractionError(f"音声抽出中にエラーが発生しました: {str(e.stderr.decode())}")
        except Exception as e:
            raise AudioExtractionError(f"音声抽出中にエラーが発生しました: {str(e)}")
    
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        音声ファイルを文字起こしします。
        
        Args:
            audio_path (str): 音声ファイルのパス
            
        Returns:
            Dict[str, Any]: 文字起こし結果
        """
        try:
            # 文字起こしの実行
            result = self.model.transcribe(
                audio_path,
                language=self.language,
                task='transcribe',
                verbose=False
            )
            
            self.logger.info(f"文字起こしが完了しました: {audio_path}")
            
            # 結果の整形
            transcription = {
                'text': result['text'],
                'segments': result['segments'],
                'language': result['language']
            }
            
            return transcription
            
        except Exception as e:
            raise AudioExtractionError(f"文字起こし中にエラーが発生しました: {str(e)}")
    
    def cleanup(self):
        """一時ファイルをクリーンアップします"""
        try:
            # 音声ファイルの削除
            for file in self.output_dir.glob('*_audio.wav'):
                file.unlink()
            self.logger.info("一時ファイルをクリーンアップしました")
        except Exception as e:
            self.logger.warning(f"クリーンアップ中にエラー: {str(e)}") 