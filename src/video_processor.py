import os
import argparse
import datetime
import json
import json
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from PIL import Image
import pytesseract
import yaml

class VideoTranscriber:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.recognizer = sr.Recognizer()
        self.output_dir = self.config['output']['directory']
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_audio(self, video_path):
        video = VideoFileClip(video_path)
        audio_path = os.path.join(self.output_dir, "temp_audio.wav")
        video.audio.write_audiofile(audio_path)
        return audio_path

    def transcribe_audio(self, audio_path):
        import whisper
        from whisper.utils import get_writer
        
        # Whisperモデルのロード
        model = whisper.load_model(self.config['speech_recognition'].get('whisper_model', 'medium'))
        
        # 音声認識の実行
        # 認識精度向上のためパラメータ調整
        # 認識精度向上のためのパラメータ最適化
        result = model.transcribe(
            audio_path,
            language=self.config['speech_recognition']['language'],
            temperature=0.2,                # 確定的な出力
            beam_size=3,                    # 精度と速度のバランス
            best_of=3,                      # 候補数最適化
            verbose=False,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,# 圧縮率フィルタリング
            no_speech_threshold=0.6,        # 無音検出閾値
            suppress_tokens=[-1]            # 特殊トークン抑制
        )
        
        # 単語レベルのタイムスタンプ情報を抽出（信頼度0.7以上のみ）
        word_entries = []
        # 信頼度フィルタリングとテキスト正規化
        for segment in result.get('segments', []):
            for word in segment.get('words', []):
                try:
                    if word.get('probability', 0) >= 0.7:
                        # 厳格な文字制限と正規化
                        clean_text = re.sub(
                            r'[^ぁ-んァ-ン一-龯a-zA-Z0-9・ー]',
                            '',
                            word['word']
                        ).strip()
                        if len(clean_text) > 0:
                            word_entries.append({
                                "text": clean_text,
                                "start": round(word['start'], 2),
                                "end": round(word['end'], 2),
                                "confidence": round(word['probability'], 2)
                            })
                except KeyError as e:
                    print(f"警告: 単語データに欠損フィールドがあります - {e}")
                    continue
        
        # デバッグ用に生データを保存
        debug_path = os.path.join(self.output_dir, "whisper_raw_debug.json")
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        return word_entries
            
        results = []
        for entry in raw_text['alternative']:
            if 'words' in entry:
                for word_info in entry['words']:
                    start_time = word_info['startTime'][:-1]  # 's'を除去
                    end_time = word_info['endTime'][:-1]
                    results.append({
                        "text": word_info['word'],
                        "start": float(start_time),
                        "end": float(end_time)
                    })
        return results

    def capture_screenshots(self, video_path):
        video = VideoFileClip(video_path)
        interval = self.config['screenshot']['interval']
        timestamps = []
        
        for t in range(0, int(video.duration), interval):
            frame = video.get_frame(t)
            img = Image.fromarray(frame)
            img_path = os.path.join(self.output_dir, f"screenshot_{t}.png")
            img.save(img_path)
            timestamps.append({
                "time": t,
                "image_path": img_path,
                "ocr_text": pytesseract.image_to_string(img, lang='jpn+eng')
            })
        return timestamps

    def process_video(self, video_path):
        audio_path = self.extract_audio(video_path)
        word_entries = self.transcribe_audio(audio_path)
        screenshots = self.capture_screenshots(video_path)
        
        # タイムスタンプ付き文字起こしデータの整形
        # 音声認識デバッグ用に生データを保存
        debug_path = os.path.join(self.output_dir, "word_entries_debug.json")
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(word_entries, f, ensure_ascii=False, indent=2)
            
        transcription = []
        current_sentence = []
        sentence_start = 0
        
        for i, word in enumerate(word_entries):
            # 単語データのバリデーション
            if 'start' not in word or 'end' not in word or 'text' not in word:
                print(f"無効な単語データ: index {i} - {word}")
                continue
                
            if not current_sentence:
                sentence_start = word['start']
            current_sentence.append(word['text'])
            
            # 1秒以上の間隔がある場合や最後の単語でセンテンスを確定
            if i == len(word_entries)-1 or (word_entries[i+1]['start'] - word['end'] > 1.0):
                transcription.append({
                    "text": "".join(current_sentence),
                    "start": sentence_start,
                    "end": word['end'],
                    "screenshots": [
                        ss for ss in screenshots
                        if ss['time'] >= sentence_start and ss['time'] <= word['end']
                    ]
                })
                current_sentence = []

        # JSON出力用データ整形
        output_data = {
            "full_transcription": transcription,
            "screenshots": [{
                "time": ss["time"],
                "image_file": os.path.basename(ss["image_path"]),
                "ocr_text": ss["ocr_text"]
            } for ss in screenshots],
            "metadata": {
                "video_duration": VideoFileClip(video_path).duration,
                "processed_at": datetime.datetime.now().isoformat(),
                "screenshot_interval": self.config['screenshot']['interval'],
                "audio_engine": self.config['speech_recognition']['engine']
            }
        }
        
        # JSONファイル保存
        output_path = os.path.join(self.output_dir, "result.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # 一時ファイル削除
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return output_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to input video file")
    args = parser.parse_args()
    
    transcriber = VideoTranscriber()
    result = transcriber.process_video(args.video_path)
    print("Processing completed. Results saved in:", transcriber.output_dir)