import numpy as np
from scipy.io import wavfile

def generate_test_audio():
    """テスト用の音声ファイルを生成します"""
    # サンプリングレート
    sample_rate = 16000
    
    # 2秒間の無音
    duration = 2
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 440Hzの正弦波を生成（A4音）
    frequency = 440
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # 音量を調整
    audio = np.int16(audio * 32767)
    
    # WAVファイルとして保存
    wavfile.write('test_audio.wav', sample_rate, audio)

if __name__ == "__main__":
    generate_test_audio() 