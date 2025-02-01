import os
import argparse
from datetime import datetime
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
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
import sys
import glob
import traceback

class VideoProcessor:
    def __init__(self, output_dir='output'):
        print("VideoProcessorの初期化を開始します...")
        try:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            print("T5モデルを読み込んでいます...")
            self.tokenizer = T5Tokenizer.from_pretrained('sonoisa/t5-base-japanese')
            self.model = T5ForConditionalGeneration.from_pretrained('sonoisa/t5-base-japanese')
            self.model.eval()
            print("T5モデルの読み込みが完了しました")
            
            # 音声認識の設定
            self.config = {
                'whisper_model': 'medium',
                'language': 'ja',
                'min_confidence': 0.5,
                'languages': 'jpn+eng'
            }
            
            # Tesseractの設定
            if sys.platform == 'darwin':  # macOS
                pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
            elif sys.platform == 'win32':  # Windows
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            else:  # Linux
                pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
            
            # 音声認識の設定
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 4000
            self.recognizer.dynamic_energy_threshold = True
            
            # モデルの読み込み
            print("モデルを読み込んでいます...")
            self.llm = pipeline(
                "text2text-generation",
                model="sonoisa/t5-base-japanese",
                tokenizer="sonoisa/t5-base-japanese"
            )
            print("モデルの読み込みが完了しました")

            print("初期化が完了しました")
        except Exception as e:
            print(f"初期化中にエラーが発生しました: {str(e)}")
            print("詳細なエラー情報:")
            print(traceback.format_exc())
            raise

    def process_video(self, video_path):
        """ビデオを処理し、文字起こし、要約、キーポイントを生成します"""
        print("ビデオの処理を開始します...")
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"ビデオファイルが見つかりません: {video_path}")

            # 音声認識を実行
            segments = self.transcribe_audio(video_path)
            if not segments:
                print("警告: 音声認識結果が空です")
                segments = []
            
            print(f"音声認識完了: {len(segments)}個のセグメントを検出")
            
            # スクリーンショットを取得
            screenshots = self.capture_screenshots(video_path)
            if not screenshots:
                print("警告: スクリーンショットの取得に失敗しました")
                screenshots = []
            
            # 各セグメントを処理
            processed_segments = []
            total_segments = len(segments)
            
            for i, segment in enumerate(segments):
                try:
                    if segment.get('text'):  # テキストが存在する場合のみ処理
                        processed_segment = self.process_segment(segment, i, total_segments)
                        if processed_segment:
                            processed_segments.append(processed_segment)
                    else:
                        print(f"警告: セグメント{i}のテキストが空です")
                except Exception as e:
                    print(f"セグメント{i}の処理中にエラー: {str(e)}")
                    traceback.print_exc()
                    continue
            
            # 最低1つのセグメントが必要
            if not processed_segments:
                print("警告: 処理されたセグメントがありません")
                # 空のセグメントを作成
                processed_segments.append({
                    "start_time": 0,
                    "end_time": 0,
                    "text": "処理可能なテキストが見つかりませんでした。",
                    "heading": "処理失敗",
                    "summary": "処理可能なテキストが見つかりませんでした。",
                    "key_points": ["処理可能なテキストが見つかりませんでした。"],
                    "screenshot": ""
                })
            
            # 結果を保存
            result = {
                "metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "video_duration": getattr(self, 'video_duration', 0),
                    "segment_count": len(processed_segments),
                    "screenshot_count": len(screenshots),
                    "success": len(processed_segments) > 0
                },
                "segments": processed_segments
            }
            
            print("結果の保存を開始します...")
            output_path = os.path.join(self.output_dir, "result.json")
            os.makedirs(self.output_dir, exist_ok=True)
            
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"結果をJSONファイルに保存しました: {output_path}")
            except Exception as e:
                print(f"結果の保存中にエラー: {str(e)}")
                traceback.print_exc()
            
            # HTMLレポートを生成
            try:
                html_output_path = os.path.join(self.output_dir, "report.html")
                if self.generate_html_report(result, html_output_path):
                    print(f"HTMLレポートを生成しました: {html_output_path}")
                else:
                    print("警告: HTMLレポートの生成に失敗しました")
            except Exception as e:
                print(f"HTMLレポート生成中にエラー: {str(e)}")
                traceback.print_exc()
            
            return result
            
        except Exception as e:
            print(f"ビデオ処理中にエラーが発生しました: {str(e)}")
            traceback.print_exc()
            # 最低限の結果を返す
            return {
                "metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "error": str(e),
                    "success": False
                },
                "segments": []
            }

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
        model = whisper.load_model(self.config['whisper_model'])
        
        # 音声認識の実行
        result = model.transcribe(
            audio_path,
            language=self.config['language'],
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
                    if confidence >= self.config['min_confidence']:
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
                    lang=self.config['languages'],
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
        """文字起こしとスクリーンショットを分析してセグメントに分割"""
        segments = []
        current_segment = None
        current_screenshots = []

        # スクリーンショットを時間でソート
        sorted_screenshots = sorted(screenshots, key=lambda x: x['timestamp'])

        for i, trans in enumerate(transcription):
            # 新しいセグメントの開始条件をチェック
            start_new_segment = (
                current_segment is None or  # 最初のセグメント
                len(current_segment['text']) > 150 or  # テキストが一定の長さを超えた
                trans['start'] - current_segment['end'] > 2  # 2秒以上の間隔
            )

            if start_new_segment:
                # 現在のセグメントを保存
                if current_segment is not None:
                    # スクリーンショットの割り当て
                    current_segment['screenshots'] = current_screenshots
                    segments.append(current_segment)
                    current_screenshots = []

                # 新しいセグメントの作成
                current_segment = {
                    'start': trans['start'],
                    'end': trans['end'],
                    'text': trans['text'],
                    'heading': '',
                    'summary': '',
                    'key_points': []
                }

                # 見出しの生成
                try:
                    current_segment['heading'] = self.llm(f"次の文章の内容を30文字以内の見出しにまとめてください：{trans['text']}", max_length=50)
                    current_segment['heading'] = current_segment['heading'][:30]
                except Exception as e:
                    print(f"見出し生成エラー: {e}")
                    current_segment['heading'] = trans['text'][:30] + "..."

                # 要約の生成
                try:
                    current_segment['summary'] = self.llm(f"次の文章を100文字以内で要約してください：{trans['text']}", max_length=150)
                    current_segment['summary'] = current_segment['summary'][:100]
                except Exception as e:
                    print(f"要約生成エラー: {e}")
                    current_segment['summary'] = trans['text'][:100] + "..."

                # キーポイントの抽出
                try:
                    current_segment['key_points'] = [point.strip() for point in self.llm(f"次の文章から重要なポイントを3つ抽出してください：{trans['text']}", max_length=200).split('\n') if point.strip()][:3]
                except Exception as e:
                    print(f"キーポイント抽出エラー: {e}")
                    current_segment['key_points'] = [trans['text'][:40] + "..."]

            else:
                # セグメントの更新
                current_segment['end'] = trans['end']
                current_segment['text'] += f" {trans['text']}"

            # スクリーンショットの割り当て
            while sorted_screenshots and sorted_screenshots[0]['timestamp'] <= trans['end']:
                current_screenshots.append(sorted_screenshots.pop(0))

        # 最後のセグメントの保存
        if current_segment is not None:
            current_segment['screenshots'] = current_screenshots
            segments.append(current_segment)

        return segments

    def llm(self, prompt, max_length=100):
        """LLMを使用してテキスト生成を行います"""
        try:
            # プロンプトのエンコード
            inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
            
            # テキスト生成
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # 出力のデコード
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # プロンプトの部分を削除
            if "文章：" in response:
                response = response.split("文章：")[0]
            if "見出し：" in response:
                response = response.split("見出し：")[-1]
            if "要約：" in response:
                response = response.split("要約：")[-1]
            if "ポイント：" in response:
                response = response.split("ポイント：")[-1]
            
            # 箇条書きの処理
            if "・" in response:
                points = []
                for line in response.split('\n'):
                    line = line.strip()
                    if line.startswith('・'):
                        points.append(line[1:].strip())
                    elif line:
                        points.append(line)
                response = '\n'.join(points)
            
            return response.strip()
            
        except Exception as e:
            print(f"LLM処理中にエラーが発生しました: {str(e)}")
            return ""

    def process_segment(self, segment, segment_index, total_segments):
        """
        セグメントを処理し、メタデータを生成します。
        """
        try:
            print(f"セグメント {segment_index + 1}/{total_segments} を処理中...")
            
            # セグメントのテキストを取得
            text = segment.get('text', '')
            if not text:
                print("セグメントのテキストが空です")
                # 空のテキストの場合でもメタデータを返す
                return {
                    "start_time": segment.get('start', 0),
                    "end_time": segment.get('end', 0),
                    "text": "テキストなし",
                    "heading": "テキストなし",
                    "summary": "テキストなし",
                    "key_points": ["テキストが検出されませんでした"],
                    "screenshot": f"screenshot_{segment_index * 10}.jpg"
                }
            
            # 見出しを生成
            heading_prompt = f"""この文章の内容を簡潔な見出し（30文字以内）にしてください。
            装飾的な表現は避け、内容を端的に表現してください。
            文章：「{text}」
            見出し："""
            heading = self.llm(heading_prompt, max_length=50)
            
            # 要約を生成
            summary_prompt = f"""この文章を100文字程度で要約してください。
            文章：「{text}」
            要約："""
            summary = self.llm(summary_prompt, max_length=150)
            
            # キーポイントを生成
            key_points_prompt = f"""この文章から重要なポイントを3つ箇条書きで抽出してください。
            文章：「{text}」
            ポイント："""
            key_points = self.llm(key_points_prompt, max_length=200)
            
            # キーポイントが文字列の場合はそのまま使用し、リストの場合は結合
            if isinstance(key_points, list):
                # リストの各要素が辞書型の場合は、テキスト部分を抽出
                key_points = [point.get('text', str(point)) if isinstance(point, dict) else str(point) for point in key_points]
                key_points_text = "\n".join(key_points)
            else:
                key_points_text = str(key_points)
            
            # スクリーンショットのファイル名を生成
            screenshot_filename = f"screenshot_{segment_index * 10}.jpg"
            
            # メタデータを生成
            metadata = {
                "start_time": segment.get('start', 0),
                "end_time": segment.get('end', 0),
                "text": text,
                "heading": heading[:30],  # 30文字に制限
                "summary": summary[:100],  # 100文字に制限
                "key_points": key_points_text.split('\n')[:3],  # 3つのポイントに制限
                "screenshot": screenshot_filename
            }
            
            print(f"セグメント {segment_index + 1} の処理が完了しました")
            return metadata
            
        except Exception as e:
            print(f"セグメント処理中にエラーが発生しました: {str(e)}")
            traceback.print_exc()
            # エラーが発生した場合でも最低限のメタデータを返す
            return {
                "start_time": segment.get('start', 0),
                "end_time": segment.get('end', 0),
                "text": "エラーが発生しました",
                "heading": "エラー",
                "summary": f"処理中にエラーが発生しました: {str(e)}",
                "key_points": ["エラーが発生しました"],
                "screenshot": f"screenshot_{segment_index * 10}.jpg"
            }

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

    def _topic_changed(self, prev_text, current_text):
        """トピックが変更されたかどうかを判定"""
        # 類似度が低い場合はトピックが変更されたと判断
        similarity = self._calculate_similarity(prev_text, current_text)
        return similarity < 0.3  # 類似度が30%未満の場合はトピック変更とみなす

    def generate_html_report(self, result_data, output_path):
        """HTMLレポートを生成します"""
        print(f"HTMLレポート生成を開始します: {output_path}")
        
        try:
            metadata = result_data.get("metadata", {})
            segments = result_data.get("segments", [])
            
            print(f"結果データのメタデータ: {metadata}")
            print(f"セグメント数: {len(segments)}")
            
            # セグメントのHTMLを生成
            print("セグメントHTMLの生成を開始します")
            segments_html = []
            for i, segment in enumerate(segments, 1):
                if segment:
                    segment_html = f"""
                    <div class="segment">
                        <div class="segment-header">
                            <h2>{html.escape(str(segment['heading']))}</h2>
                            <span class="timestamp">{int(segment['start_time'])}秒 - {int(segment['end_time'])}秒</span>
                        </div>
                        <div class="segment-content">
                            <div class="screenshot">
                                <img src="{html.escape(str(segment['screenshot']))}" alt="スクリーンショット">
                            </div>
                            <div class="text-content">
                                <div class="summary">
                                    <h3>要約</h3>
                                    <p>{html.escape(str(segment['summary']))}</p>
                                </div>
                                <div class="key-points">
                                    <h3>キーポイント</h3>
                                    <ul>
                                        {"".join(f"<li>{html.escape(str(point))}</li>" for point in segment['key_points'])}
                                    </ul>
                                </div>
                                <div class="transcript">
                                    <h3>文字起こし</h3>
                                    <p>{html.escape(str(segment['text']))}</p>
                                </div>
                            </div>
                        </div>
                    </div>"""
                    segments_html.append(segment_html)
                print(f"セグメント {i} のHTML生成が完了しました")
            
            # HTMLの全体構造
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>動画文字起こしレポート</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .segment {{
            background-color: white;
            margin-bottom: 20px;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .segment-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .segment-header h2 {{
            margin: 0;
            color: #333;
        }}
        .timestamp {{
            color: #666;
        }}
        .segment-content {{
            display: flex;
            gap: 20px;
        }}
        .screenshot {{
            flex: 0 0 300px;
        }}
        .screenshot img {{
            width: 100%;
            border-radius: 4px;
        }}
        .text-content {{
            flex: 1;
        }}
        h3 {{
            color: #444;
            margin: 15px 0 10px;
        }}
        .summary p {{
            color: #333;
            line-height: 1.6;
        }}
        .key-points ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .key-points li {{
            color: #333;
            margin-bottom: 5px;
        }}
        .transcript p {{
            color: #666;
            line-height: 1.6;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>動画文字起こしレポート</h1>
        {"".join(segments_html)}
    </div>
</body>
</html>
"""
            
            # HTMLファイルに保存
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"HTMLレポートを生成しました: {output_path}")
            return True
            
        except Exception as e:
            print(f"HTMLレポート生成中にエラーが発生しました: {str(e)}")
            traceback.print_exc()
            return False

    def save_results(self, result):
        """結果をファイルに保存"""
        try:
            # 出力ディレクトリの作成
            os.makedirs(self.output_dir, exist_ok=True)
            
            # メタデータの追加
            result['metadata'] = {
                'processed_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'video_duration': sum([segment['end'] - segment['start'] for segment in result.get('segments', [])]),
                'segment_count': len(result.get('segments', [])),
                'screenshot_count': len(result.get('screenshots', []))
            }
            
            # 結果をJSONファイルとして保存
            output_json = os.path.join(self.output_dir, 'result.json')
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # HTMLレポートの生成
            html_path = self.generate_html_report(result)
            
            print(f"結果を保存しました:")
            print(f"- JSON: {output_json}")
            if html_path:
                print(f"- HTML: {html_path}")
            
        except Exception as e:
            print(f"結果の保存中にエラー: {e}")
            print(f"エラーの詳細:\n{traceback.format_exc()}")

    def _is_topic_change(self, prev_text, current_text):
        """2つのテキスト間でトピックが変更されたかどうかを判定"""
        try:
            # 前後のテキストをLLMに渡して、トピックの変更があったかどうかを判定
            prompt = f"""
            以下の2つのテキストを比較して、トピックが変更されたかどうかを判定してください。
            「はい」または「いいえ」で答えてください。

            テキスト1: {prev_text}
            テキスト2: {current_text}
            """
            
            response = self.llm(prompt, max_length=5, temperature=0.3)[0]['generated_text'].strip().lower()
            return 'はい' in response or 'yes' in response
        except Exception as e:
            print(f"トピック変更判定エラー: {e}")
            return False  # エラーの場合は変更なしとみなす

    def extract_frames(self, video_path):
        """ビデオからフレームを抽出します"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            return frames
            
        except Exception as e:
            print(f"フレーム抽出中にエラーが発生しました: {str(e)}")
            return []

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
