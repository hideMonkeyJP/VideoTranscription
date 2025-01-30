import os
import argparse
import datetime
import json
import re
import itertools
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import yaml
import cv2
import html
import unicodedata
from collections import Counter
from transformers import pipeline

class VideoProcessor:
    def __init__(self):
        # デフォルトの設定
        self.config = {
            'speech_recognition': {
                'whisper_model': 'medium',  # モデルサイズ
                'language': 'ja',           # 日本語
                'min_confidence': 0.5       # 最小信頼度
            },
            'ocr': {
                'languages': 'jpn+eng'      # 日本語と英語のOCR
            }
        }
        
        # 出力ディレクトリの設定
        self.output_dir = 'output'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def process_video(self, video_path):
        """動画を処理してテキストと画像を抽出"""
        try:
            # 1. 音声の抽出
            audio_path = self.extract_audio(video_path)
            if not audio_path:
                raise Exception("音声の抽出に失敗しました")

            # 2. 音声の文字起こし
            transcription = self.transcribe_audio(audio_path)
            if not transcription:
                raise Exception("音声認識に失敗しました")

            # 3. スクリーンショットの生成と処理
            screenshots = self.capture_screenshots(video_path)
            if not screenshots:
                raise Exception("スクリーンショットの生成に失敗しました")

            # 4. スクリーンショットのOCR処理
            screenshots = self.process_screenshots(screenshots)

            # 5. 結果の整形
            result = {
                "video_file": os.path.basename(video_path),
                "transcription": transcription,
                "screenshots": screenshots
            }

            # 6. 結果の保存
            self.save_results(result)

            return result

        except Exception as e:
            print(f"ビデオ処理中にエラー: {e}")
            return None

    def extract_audio(self, video_path):
        """動画から音声を抽出"""
        try:
            video = VideoFileClip(video_path)
            audio_path = os.path.join(self.output_dir, "temp_audio.wav")
            video.audio.write_audiofile(audio_path)
            return audio_path
        except Exception as e:
            print(f"音声抽出中にエラー: {e}")
            return None

    def transcribe_audio(self, audio_path):
        import whisper
        from whisper.utils import get_writer
        
        # Whisperモデルのロード
        model = whisper.load_model(self.config['speech_recognition'].get('whisper_model', 'medium'))
        
        # 音声認識の実行
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
        
        word_entries = []
        if not result.get('segments'):
            print("警告: 音声認識結果が空です。音声ファイルを確認してください")
            return word_entries
            
        for segment_idx, segment in enumerate(result.get('segments', [])):
            segment_text = segment.get('text', '')
            
            clean_segment_text = re.sub(
                r'[^\wぁ-んァ-ン一-龯ａ-ｚＡ-Ｚ０-９・ー、。]',
                '',
                segment_text
            ).strip()
            
            if not clean_segment_text:
                print(f"警告: セグメント{segment_idx}のテキストが空です")
                continue
            
            if not segment.get('words'):
                print(f"情報: セグメント{segment_idx}の文全体を使用します")
                word_entries.append({
                    "text": clean_segment_text,
                    "start": round(segment['start'], 2),
                    "end": round(segment['end'], 2),
                    "confidence": round(segment.get('avg_logprob', 0), 2)
                })
                continue
            
            for word_idx, word in enumerate(segment.get('words', [])):
                try:
                    confidence = word.get('probability', 0)
                    if confidence >= self.config['speech_recognition'].get('min_confidence', 0.5):
                        clean_text = re.sub(
                            r'[^\wぁ-んァ-ン一-龯ａ-ｚＡ-Ｚ０-９・ー、。]',
                            '',
                            word['word']
                        ).strip()
                        
                        if len(clean_text) > 0:
                            word_entries.append({
                                "text": clean_text,
                                "start": round(word['start'], 2),
                                "end": round(word['end'], 2),
                                "confidence": round(confidence, 2)
                            })
                except Exception as e:
                    print(f"エラー: {segment_idx}-{word_idx} - {str(e)}")
        
        return word_entries

    def capture_screenshots(self, video_path):
        """ビデオからスクリーンショットを生成"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("ビデオファイルを開けませんでした")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            interval = 10
            screenshots = []
            
            for time in range(0, int(duration), interval):
                frame_number = int(time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret:
                    screenshot_path = f"screenshot_{time}.jpg"
                    output_path = os.path.join(self.output_dir, screenshot_path)
                    if cv2.imwrite(output_path, frame):
                        print(f"スクリーンショットを保存: {output_path}")
                        screenshots.append({
                            "timestamp": time,
                            "image_path": output_path,  # フルパスを保存
                            "text": ""
                        })
                    else:
                        print(f"スクリーンショットの保存に失敗: {output_path}")
            
            cap.release()
            return screenshots
            
        except Exception as e:
            print(f"スクリーンショット生成中にエラー: {e}")
            return []

    def _calculate_text_quality(self, text):
        """テキストの品質スコアを計算（0.0 ~ 1.0）"""
        if not text or len(text.strip()) < 3:
            return 0.0

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

    def process_screenshots(self, screenshots):
        """スクリーンショットの処理とOCR"""
        for ss in screenshots:
            try:
                # 画像の前処理を強化（フルパスを使用）
                image = Image.open(ss["image_path"])
                
                # 前処理パイプライン
                # 1. グレースケール変換
                image = image.convert('L')
                
                # 2. バイナリ化のための閾値を自動計算
                threshold = int(sum(image.histogram()[i] * i for i in range(256)) / sum(image.histogram()))
                
                # 3. コントラスト強調
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(2.5)
                
                # 4. シャープネス強調
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(2.5)
                
                # 5. ノイズ除去（メディアンフィルタ）
                image = image.filter(ImageFilter.MedianFilter(size=3))
                
                # 6. 画像のスケーリング（必要に応じて）
                if image.size[0] < 1000:  # 小さすぎる画像は拡大
                    scale = 1000 / image.size[0]
                    new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)

                # OCR実行（設定を最適化）
                text = pytesseract.image_to_string(
                    image,
                    lang=self.config['ocr']['languages'],
                    config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン、。，．・ー'
                )

                # テキストのクリーニングと文字コード処理
                try:
                    # 文字コードの正規化とクリーニング
                    text = text.encode('utf-8', errors='ignore').decode('utf-8')
                    # Unicodeの正規化（全角・半角の統一）
                    text = unicodedata.normalize('NFKC', text)
                    
                    lines = [line.strip() for line in text.split('\n')]
                    cleaned_lines = []
                    for line in lines:
                        if len(line) <= 1:  # 空行や1文字の行を除外
                            continue
                            
                        # 制御文字と特殊文字を除去
                        line = ''.join(c for c in line if ord(c) >= 32 or c == '\n')
                        
                        # 1. 行の前処理
                        line = line.strip()
                        if len(line) <= 3:  # 短すぎる行は除外
                            continue

                        # 2. 特殊文字と記号の処理
                        # 特殊文字や記号が多すぎる行を除外
                        symbol_count = sum(1 for c in line if not c.isalnum() and not 0x3000 <= ord(c) <= 0x9FFF)
                        if symbol_count > len(line) * 0.2:  # 20%以上が記号の場合は除外
                            continue

                        # 連続する記号や特殊文字のパターンをチェック
                        if re.search(r'[-_@#$%^&*(){}\[\]|;:]{2,}|[O\-—]{2,}|[A-Z0-9]{4,}', line):
                            continue

                        # URL、ファイルパス、特殊な識別子のパターンを検出
                        if re.search(r'(https?:\/\/|www\.|\/|\[|\]|\(\)|#\d+|[A-Z]+\d+|\d+[A-Z]+)', line):
                            continue

                        # 特定のパターンで始まる行を除外
                        if any(line.startswith(prefix) for prefix in ['©', '®', '™', '[]', '【', '》', '-O', '@', '#', '*', '=']):
                            continue

                        # 3. テキスト品質の詳細評価
                        # 日本語文字の検出
                        jp_chars = sum(1 for c in line if 0x3000 <= ord(c) <= 0x9FFF)
                        
                        # 意味のある文字列かチェック
                        meaningful_chars = sum(1 for c in line if c.isalnum() or 0x3000 <= ord(c) <= 0x9FFF)
                        if meaningful_chars < len(line) * 0.7:  # 70%以上が意味のある文字であること
                            continue
                        
                        # 文字列の最小長チェック
                        if len(line) < 5:
                            continue
                            
                        # 記号や特殊文字の連続をチェック
                        if re.search(r'[-_@#$%^&*(){}\[\]|;:]{2,}|[O\-—]{2,}', line):
                            continue
                            
                        # 文字の多様性チェック
                        char_freq = Counter(line)
                        unique_ratio = len(char_freq) / len(line)
                        if unique_ratio < 0.6:  # 文字の種類が60%未満は除外
                            continue
                            
                        # アルファベットのみの文字列の場合の追加チェック
                        if line.isascii() and line.replace(' ', '').isalpha():
                            # 母音の割合チェック
                            vowels = sum(1 for c in line.lower() if c in 'aeiou')
                            if vowels / len(line) < 0.15:  # 母音が少なすぎる場合は除外
                                continue

                        # 4. テキスト品質の総合評価
                        quality_score = self._calculate_text_quality(line)
                        if quality_score <= 0.6:  # より厳しい品質閾値
                            continue

                        # 5. 追加のフィルタリング
                        # 連続する特殊文字のパターンを検出
                        if re.search(r'[-_@#$%^&*(){}\[\]|;:]+', line):
                            continue
                            
                        # 無意味な大文字の連続を検出
                        if re.search(r'[A-Z]{4,}', line) and not re.search(r'[あ-んア-ン一-龯]', line):
                            continue

                        # 数字とアルファベットの混在パターンを検出
                        if re.search(r'\d+[a-zA-Z]+\d+|[a-zA-Z]+\d+[a-zA-Z]+', line):
                            continue
                            
                        # 3. 意味のある文字列の判定
                        has_japanese = any(0x3000 <= ord(c) <= 0x9FFF for c in line)
                        has_meaningful_ascii = (
                            any(c.isalpha() for c in line) and  # アルファベットを含む
                            sum(1 for c in line if c.isalnum()) > len(line) * 0.4 and  # より厳しい英数字の比率
                            len(line) >= 5 and  # 最小長を増加
                            not re.search(r'[A-Z0-9]{4,}', line)  # 大文字と数字の連続を制限
                        )
                        
                        # URLやファイルパスのようなパターンを除外
                        if any(pattern in line for pattern in ['http', '://', '.com', '.jp', '#', '@']):
                            continue
                            
                        if has_japanese or has_meaningful_ascii:
                            # 4. テキストのクリーニング
                            # 連続する空白を1つに
                            line = ' '.join(line.split())
                            # 前後の記号を除去
                            line = line.strip('_-=@#$%^&*()[]{}|;:,.<>?/\\')
                            if len(line) > 3:  # 再度長さチェック
                                cleaned_lines.append(line)

                    # 最低文字数チェック
                    if len(''.join(cleaned_lines)) < 5:  # 合計5文字未満は除外
                        cleaned_lines = []

                    # クリーニングされたテキストを保存
                    cleaned_text = '\n'.join(cleaned_lines)
                    if cleaned_text.strip():
                        ss["text"] = cleaned_text
                    else:
                        ss["text"] = ""
                    
                except UnicodeError as e:
                    print(f"文字コードエラー {ss['image_path']}: {str(e)}")
                    ss["text"] = ""

            except Exception as e:
                print(f"OCRエラー {ss['image_path']}: {str(e)}")
                ss["text"] = ""

        return screenshots

    def analyze_content(self, transcription, screenshots):
        """音声認識結果とスクリーンショットを組み合わせてコンテンツを分析"""
        segments = []
        current_segment = []
        current_screenshots = []
        start_time = 0
        segment_duration = 60  # 1分ごとにセグメント分割

        sorted_transcription = sorted(transcription, key=lambda x: x['start'])

        for entry in sorted_transcription:
            while screenshots and screenshots[0]["timestamp"] <= entry["end"]:
                current_screenshots.append(screenshots.pop(0))

            current_segment.append(entry)
            
            if (entry['end'] - start_time > segment_duration or
                (len(current_segment) > 1 and
                 self._topic_changed(current_segment[-2]['text'], entry['text']))):
                
                segment_text = ' '.join(item['text'] for item in current_segment)
                
                    segment_info = {
                        'start': start_time,
                        'end': entry['end'],
                        'text': segment_text,
                        'heading': self.generate_heading(segment_text),
                        'summary': self.generate_summary(segment_text),
                        'key_points': self.extract_key_points(segment_text),
                        'screenshots': [{
                            'path': ss['image_path'],
                            'timestamp': ss['timestamp'],
                            'text': ss.get('text', '')
                        } for ss in current_screenshots]
                    }
                    segments.append(segment_info)
                
                start_time = entry['end']
                current_segment = []
                current_screenshots = []

        if current_segment:
            segment_text = ' '.join(item['text'] for item in current_segment)
                segments.append({
                    'start': start_time,
                    'end': current_segment[-1]['end'],
                    'text': segment_text,
                    'heading': self.generate_heading(segment_text),
                    'summary': self.generate_summary(segment_text),
                    'key_points': self.extract_key_points(segment_text),
                    'screenshots': [{
                        'path': ss['image_path'],
                        'timestamp': ss['timestamp'],
                        'text': ss.get('text', '')
                } for ss in current_screenshots + screenshots]
                })

        return segments

    def generate_heading(self, text):
        """見出し生成の実装"""
        try:
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese")
            model = T5ForConditionalGeneration.from_pretrained("sonoisa/t5-base-japanese")
            
            input_text = f"タイトル: {text[:500]}"
            input_ids = tokenizer.encode(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            outputs = model.generate(
                input_ids,
                max_length=30,
                min_length=10,
                num_beams=4,
                temperature=0.7,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                early_stopping=True
            )
            
            heading = tokenizer.decode(outputs[0], skip_special_tokens=True)
            heading = heading.replace("タイトル:", "").strip()
            return heading
            
        except Exception as e:
            print(f"見出し生成エラー: {str(e)}")
            words = text[:100].split()
            return ' '.join(words[:5]) + "..."

    def generate_summary(self, text):
        """要約生成の実装"""
        try:
            summarizer = pipeline("summarization",
                                model="ku-nlp/bart-base-japanese",
                                tokenizer="ku-nlp/bart-base-japanese")
            
            text_length = len(text)
            if text_length < 100:
                max_length = 30
                min_length = 10
            else:
                max_length = min(100, text_length // 3)
                min_length = max(30, text_length // 6)
            
            summary = summarizer(
                text[:1000],
                max_length=max_length,
                min_length=min_length,
                no_repeat_ngram_size=3,
                num_beams=4,
                temperature=0.7,
                do_sample=True
            )[0]['summary_text']
            
            return summary.strip()
            
        except Exception as e:
            print(f"要約生成エラー: {str(e)}")
            sentences = text.split('。')[:2]
            return '。'.join(sentences) + '。'

    def _calculate_similarity(self, text1, text2):
        """2つのテキスト間の類似度を計算"""
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

    def extract_key_points(self, text):
        """キーポイント抽出の改善実装"""
        try:
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            
            # テキストの前処理
            cleaned_text = re.sub(r'[\(\)\[\]「」『』]', '', text)  # 括弧類を削除
            cleaned_text = re.sub(r'[:：]', '', cleaned_text)  # コロンを削除
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # 空白の正規化
            
            if not cleaned_text:
                return []
            
            tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese")
            model = T5ForConditionalGeneration.from_pretrained("sonoisa/t5-base-japanese")
            
            input_text = f"以下の文章から重要なポイントを抽出してください: {cleaned_text[:500]}"
            input_ids = tokenizer.encode(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            outputs = model.generate(
                input_ids,
                max_length=150,
                min_length=30,
                num_beams=5,
                temperature=0.6,  # より確実な生成のため温度を下げる
                no_repeat_ngram_size=3,
                top_k=30,
                top_p=0.92,
                early_stopping=True,
                repetition_penalty=1.2  # 繰り返しを抑制
            )
            
            key_points_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # キーポイントの詳細な後処理
            key_points = []
            seen_points = set()
            
            for point in key_points_text.split('。'):
                # 基本的なクリーニング
                point = point.strip()
                point = re.sub(r'(重要|大事)(\s*な)?\s*(ポイント|点)\s*[:：]?', '', point)  # 接頭辞の除去
                point = re.sub(r'[「」『』（）\(\)\[\]\{\}]', '', point)  # 括弧類の除去
                point = re.sub(r'[:：。、]$', '', point)  # 末尾の区切り文字を除去
                point = point.strip()
                
                # 意味のある内容かチェック
                if not point or len(point) < 8:  # 最小長さを増加
                    continue
                
                # 日本語文字を含むかチェック
                if not re.search(r'[ぁ-んァ-ン一-龯]', point):
                    continue
                    
                # 記号のみの行を除外
                if re.match(r'^[\s\W]+$', point):
                    continue
                    
                # 省略記号で終わる不完全な文を除外
                if point.endswith(('...', '…', '→')):
                    continue
                    
                # 重複や類似のチェック
                is_duplicate = any(
                    self._calculate_similarity(point, existing) > 0.6  # 類似度の閾値を調整
                    for existing in seen_points
                )
                
                if not is_duplicate:
                    key_points.append(point)
                    seen_points.add(point)
            
            return key_points if key_points else [cleaned_text[:100] + "..."]
            
        except Exception as e:
            print(f"キーポイント抽出エラー: {str(e)}")
            return [text[:100] + "..."]

    def generate_html_report(self, results):
        """HTMLレポートを生成"""
        try:
            # CSS定義
            css = '''
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans JP", sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 0;
                    background: #ffffff;
                }
                .container {
                    max-width: 900px;
                    margin: 0 auto;
                    padding: 2rem;
                }
                h1 { font-size: 2.5em; font-weight: 700; margin-bottom: 1em; color: #333; }
                h2 { font-size: 1.8em; font-weight: 600; margin: 1em 0 0.5em; color: #333; }
                h3 { font-size: 1.2em; font-weight: 600; color: #333; }
                .segment {
                    margin-bottom: 2.5em;
                    padding: 1.5em;
                    border: 1px solid #e1e4e8;
                    border-radius: 6px;
                    background: #fff;
                }
                .segment-header {
                    margin-bottom: 1em;
                    padding-bottom: 0.5em;
                    border-bottom: 1px solid #e1e4e8;
                }
                .segment-time { color: #666; font-size: 0.9em; }
                .summary-section {
                    background: #f6f8fa;
                    padding: 1em;
                    border-radius: 4px;
                    margin: 1em 0;
                }
                .key-points {
                    list-style-type: none;
                    padding-left: 0;
                }
                .key-points li {
                    position: relative;
                    padding-left: 1.5em;
                    margin-bottom: 0.5em;
                }
                .key-points li:before {
                    content: "•";
                    position: absolute;
                    left: 0.5em;
                    color: #0366d6;
                }
                .screenshot-container {
                    margin: 1.5em 0;
                    border: 1px solid #e1e4e8;
                    border-radius: 4px;
                    overflow: hidden;
                }
                .screenshot {
                    max-width: 100%;
                    display: block;
                }
                .timestamp {
                    color: #666;
                    font-size: 0.9em;
                    margin: 0.5em 1em;
                }
                .ocr-text {
                    font-family: "Noto Sans JP", "Noto Sans CJK JP", monospace;
                    font-size: 0.95em;
                    line-height: 1.8;
                    background: #f6f8fa;
                    color: #24292e;
                    padding: 1em;
                    margin: 1em;
                    border-radius: 4px;
                    white-space: pre-line;
                    border-left: 3px solid #0366d6;
                }
                .metadata {
                    margin-top: 3em;
                    padding-top: 1em;
                    border-top: 1px solid #e1e4e8;
                    color: #666;
                    font-size: 0.9em;
                }
            '''

            # HTMLの構築
            html_parts = [
                "<!DOCTYPE html>",
                '<html lang="ja">',
                "<head>",
                '<meta charset="UTF-8">',
                '<meta http-equiv="Content-Type" content="text/html; charset=utf-8">',
                '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
                "<title>動画コンテンツ要約</title>",
                f"<style>{css}</style>",
                "</head>",
                "<body>",
                "<div class='container'>",
                "<h1>📝 動画コンテンツ要約</h1>",
                f"<p>ファイル名: {results.get('video_file', '不明')}</p>",
                "<div class='content'>"
            ]

            # セグメントの処理
            for segment in results.get("segments", []):
                # 時間の文字列化
                start_time = str(datetime.timedelta(seconds=int(segment["start"])))
                end_time = str(datetime.timedelta(seconds=int(segment["end"])))

                # セグメントの内容を追加
                html_parts.extend([
                    "<div class='segment'>",
                    "<div class='segment-header'>",
                    f"<div class='segment-time'>{start_time} - {end_time}</div>",
                    f"<h2>{html.escape(segment['heading'])}</h2>",
                    "</div>",
                    "<div class='summary-section'>",
                    "<h3>📝 要約</h3>",
                    f"<p>{html.escape(segment['summary'])}</p>",
                    "</div>",
                    "<h3>🎯 キーポイント</h3>",
                    "<ul class='key-points'>"
                ])

                # キーポイントの追加
                for point in segment['key_points']:
                    html_parts.append(f"<li>{html.escape(point)}</li>")

                html_parts.append("</ul>")

                # スクリーンショットとOCRテキストの処理
                for ss in segment.get('screenshots', []):
                    timestamp = str(datetime.timedelta(seconds=int(ss['timestamp'])))
                    # スクリーンショットのパスを相対パスに変換
                    relative_path = os.path.relpath(ss['path'], self.output_dir)
                    html_parts.extend([
                        "<div class='screenshot-container'>",
                        f"<img src='{relative_path}' alt='Screenshot' class='screenshot'>",
                        f"<div class='timestamp'>タイムスタンプ: {timestamp}</div>"
                    ])

                    # OCRテキストの処理（文字化け対策強化版）
                    ocr_text = ss.get('text', '').strip()
                    if ocr_text:
                        try:
                            # 文字コードの正規化
                            ocr_text = unicodedata.normalize('NFKC', ocr_text)
                            # 制御文字の除去と空白の正規化
                            ocr_text = ''.join(char for char in ocr_text if ord(char) >= 32 or char == '\n')
                            ocr_text = re.sub(r'\s+', ' ', ocr_text)
                            # HTMLエスケープと改行処理
                            ocr_text = html.escape(ocr_text).replace('\n', '<br>')
                            
                            if ocr_text.strip():
                                html_parts.append(
                                    '<div class="ocr-text">'
                                    '<span style="color: #666;">📄 検出されたテキスト:</span><br>'
                                    f'{ocr_text}'
                                    '</div>'
                                )
                        except Exception as e:
                            print(f"OCRテキスト処理エラー: {str(e)}")

                    html_parts.append("</div>")  # screenshot-container終了

                html_parts.append("</div>")  # segment終了

            # メタデータの追加
            metadata = results.get("metadata", {})
            duration = str(datetime.timedelta(seconds=int(metadata.get("video_duration", 0))))
            html_parts.extend([
                "<div class='metadata'>",
                f"<p>処理日時: {metadata.get('processed_at', '不明')}</p>",
                f"<p>動画時間: {duration}</p>",
                f"<p>セグメント数: {metadata.get('segment_count', 0)}</p>",
                "</div>",
                "</div>",  # content終了
                "</div>",  # container終了
                "</body>",
                "</html>"
            ])

            # HTMLファイルの保存（self.output_dirを使用）
            os.makedirs(self.output_dir, exist_ok=True)
            html_path = os.path.join(self.output_dir, "report.html")
            
            # HTMLファイルをBOM付きUTF-8で保存
            with open(html_path, "wb") as f:
                content = "\n".join(html_parts)
                # BOMを追加してUTF-8でエンコード
                f.write(b'\xef\xbb\xbf')
                f.write(content.encode('utf-8', errors='replace'))

            print(f"HTMLレポートを生成しました: {html_path}")
            return html_path

        except Exception as e:
            import traceback
            print(f"HTMLレポート生成中にエラー: {e}")
            print(f"エラーの詳細:\n{traceback.format_exc()}")
            return None

    def save_results(self, result):
        """結果をファイルに保存"""
        try:
            # 出力ディレクトリの作成
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 結果をJSONファイルとして保存
            output_json = os.path.join(self.output_dir, 'transcription_result.json')
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # HTMLレポートの生成
            html_path = self.generate_html_report(result)
            
            print(f"結果を保存しました:")
            print(f"- JSON: {output_json}")
                print(f"- HTML: {html_path}")
            
        except Exception as e:
            import traceback
            print(f"結果の保存中にエラー: {e}")
            print(f"エラーの詳細:\n{traceback.format_exc()}")

    def _generate_html_report(self, result):
        """HTMLレポートを生成"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>動画書き起こしレポート</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                    margin: 20px;
                    line-height: 1.6;
                }
                .section {
                    margin-bottom: 30px;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 8px;
                }
                .timestamp {
                    color: #666;
                    font-size: 0.9em;
                }
                .screenshot {
                    max-width: 100%;
                    height: auto;
                    margin: 10px 0;
                    border-radius: 4px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .transcription {
                    margin: 10px 0;
                    padding: 10px;
                    background: white;
                    border-radius: 4px;
                }
                .screenshot-section {
                    margin-bottom: 20px;
                    padding: 15px;
                    background: white;
                    border-radius: 4px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }
                h1, h2, h3 {
                    color: #333;
                }
                .text {
                    margin: 10px 0;
                    line-height: 1.6;
                }
            </style>
        </head>
        <body>
            <h1>動画書き起こしレポート</h1>
            <h2>ファイル: {video_file}</h2>
            
            <div class="section">
                <h3>スクリーンショット</h3>
                {screenshots_html}
            </div>
            
            <div class="section">
                <h3>書き起こし</h3>
                {transcription_html}
            </div>
        </body>
        </html>
        """
        
        # スクリーンショットのHTML生成
        screenshots_html = ""
        for ss in result.get('screenshots', []):
            screenshots_html += f"""
            <div class="screenshot-section">
                <p class="timestamp">時間: {ss['timestamp']}秒</p>
                <img class="screenshot" src="{os.path.basename(ss['image_path'])}" alt="Screenshot">
                <p class="text">{html.escape(ss.get('text', ''))}</p>
            </div>
            """
        
        # 書き起こしのHTML生成
        transcription_html = ""
        for entry in result.get('transcription', []):
            transcription_html += f"""
            <div class="transcription">
                <span class="timestamp">[{entry['start']}s - {entry['end']}s]</span>
                <span class="text">{html.escape(entry['text'])}</span>
            </div>
            """
        
        # テンプレートに値を挿入
        html_content = html_template.format(
            video_file=html.escape(result.get('video_file', '不明')),
            screenshots_html=screenshots_html,
            transcription_html=transcription_html
        )
        
        return html_content

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python video_processor.py <動画ファイルのパス>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    processor = VideoProcessor()
    result = processor.process_video(video_path)
    
    if result:
        processor.save_results(result)
        print("処理が完了しました")
    else:
        print("処理中にエラーが発生しました")
