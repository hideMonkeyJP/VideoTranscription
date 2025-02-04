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
from typing import Dict, Any, List, Optional
import torch

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
            
            # 音声認識の設定を調整
            self.config = {
                'whisper_model': 'medium',  # モデルサイズを大きくして精度を向上
                'language': 'ja',
                'min_confidence': 0.3,  # 信頼度の閾値を下げて検出を増やす
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
            self.recognizer.energy_threshold = 3000  # エネルギー閾値を下げて検出を増やす
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
            else:
                print(f"スクリーンショット取得完了: {len(screenshots)}枚")
                # スクリーンショットを保存
                for i, screenshot in enumerate(screenshots):
                    screenshot_path = os.path.join(self.output_dir, f'screenshot_{i}.jpg')
                    cv2.imwrite(screenshot_path, screenshot)
                    print(f"スクリーンショットを保存: {screenshot_path}")
            
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
        """ビデオからスクリーンショットを取得します"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("エラー: ビデオファイルを開けませんでした")
                return []

            # ビデオの情報を取得
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            self.video_duration = duration
            
            # スクリーンショットを取得する間隔（秒）
            interval = 1.0  # 1秒ごと
            screenshots = []
            
            print(f"ビデオ情報: {total_frames}フレーム, {fps}fps, {duration:.2f}秒")
            print(f"スクリーンショット取得間隔: {interval}秒")

            for frame_idx in range(0, total_frames, int(fps * interval)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    screenshots.append(frame)
                    print(f"スクリーンショット取得: {len(screenshots)}枚目 ({frame_idx/fps:.1f}秒)")
            
            cap.release()
            return screenshots
            
        except Exception as e:
            print(f"スクリーンショット取得中にエラーが発生しました: {str(e)}")
            traceback.print_exc()
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
                    current_segment['heading'] = self._generate_heading(trans['text'])
                except Exception as e:
                    print(f"見出し生成エラー: {e}")
                    current_segment['heading'] = trans['text'][:30] + "..."

                # 要約の生成
                try:
                    current_segment['summary'] = self._generate_summary(trans['text'])
                except Exception as e:
                    print(f"要約生成エラー: {e}")
                    current_segment['summary'] = trans['text'][:100] + "..."

                # キーポイントの抽出
                try:
                    current_segment['key_points'] = self._generate_key_points(trans['text'])
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

    def _generate_text(self, prompt, max_length=100):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_beams=5,
                length_penalty=0.8,
                no_repeat_ngram_size=3,
                temperature=0.3,
                top_k=20,
                top_p=0.85,
                min_length=5,
                repetition_penalty=3.0,
                early_stopping=True
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._clean_response(response)
        except Exception as e:
            print(f"テキスト生成中にエラーが発生しました: {str(e)}")
            return "内容なし"

    def _generate_heading(self, text):
        prompt = f"次の文章を30文字以内の簡潔な見出しにまとめてください。数字や記号は含めないでください：\n{text}"
        return self._generate_text(prompt, max_length=30)

    def _generate_summary(self, text):
        prompt = f"次の文章を100文字以内で要約してください。数字や記号は含めないでください：\n{text}"
        return self._generate_text(prompt, max_length=100)

    def _generate_key_points(self, text):
        prompt = f"次の文章から重要なポイントを3つ抽出してください。各ポイントは簡潔に記述し、数字や記号は含めないでください：\n{text}"
        response = self._generate_text(prompt)
        points = []
        for line in response.split('\n'):
            line = line.strip()
            if line and line != "内容なし":
                # 箇条書き記号と番号を削除
                line = re.sub(r'^[-・\*\d\.\)、] *', '', line)
                # 不要な記号を削除
                line = re.sub(r'[「」『』【】\(\)（）\[\]］\[\{\}:：。、．，!！?？\^_\+\-\*/=]', '', line)
                if line.strip():
                    points.append(line.strip())
        return points[:3] if points else ["内容なし"]

    def _clean_response(self, response):
        if not response:
            return "内容なし"
        
        # 不要な記号を削除
        response = re.sub(r'[「」『』【】\(\)（）\[\]］\[\{\}]', '', response)
        response = re.sub(r'[:：]', '', response)
        response = re.sub(r'[。、．，]', '。', response)
        response = re.sub(r'[!！?？\^_\+\-\*/=]', '', response)
        
        # 不要な表現を削除
        unnecessary = [
            'です', 'ます', 'ください', '次の', '文章', '見出し',
            '要約', 'ポイント', '抽出', '生成', '入力', '出力',
            'から', 'まで', 'とか', 'みたいな', '感じ', '笑',
            'ね', 'よ', 'な', 'っと', 'この', 'もの', 'こと',
            '上記', '内容', '以内', '文字', '箇条書き', 'お願い',
            'して', 'する', 'した', 'しょう', 'する', 'され',
            '思い', '思う', '考え', '考える', '言う', '言った',
            'という', 'といった', 'という風', 'というよう',
            '的', '的な', '的に', '的で', '的な', '的です',
            '以下', '略', '本文', '中', '記事', '項目', '例',
            'また', '下さい', '可', '半角', '英数', '⇒',
            'で', 'に', 'を', 'の', 'が', 'と', 'へ', 'や',
            'ok', '結構', 'し', '簡潔', 'タイトル', 'タグ',
            '字', '以上', '全文', 'キーワード', '重要', '注意',
            'すべき', '押さえておき', 'たい', 'ポイント', '点',
            'それ', 'これ', 'あれ', 'どれ', 'そう', 'こう',
            'あの', 'その', 'どの', 'いう', 'って', 'けど',
            'だけ', 'だった', 'だろう', 'かも', 'かな', 'わけ',
            'はず', 'まま', 'ほど', 'くらい', 'ぐらい', 'なり',
            'たり', 'だり', 'れる', 'られる', 'せる', 'させる',
            'いく', 'くる', 'いる', 'ある', 'なる', 'いい',
            'わかる', 'みる', 'くれる', 'もらう', 'あげる',
            'いただく', 'おく', 'しまう', 'ちゃう', 'じゃう',
            'てる', 'でる', 'とく', 'どく', 'いく', 'てく',
            'でく', 'とる', 'どる', 'いる', 'てる', 'でる',
            'のに', 'のは', 'のが', 'のを', 'には', 'には',
            'にも', 'でも', 'から', 'まで', 'より', 'ほど',
            'など', 'なんて', 'なんか', 'なんて', 'なんか'
        ]
        for word in unnecessary:
            response = response.replace(word, '')
        
        # 連続する句点を1つに
        response = re.sub(r'。+', '。', response)
        
        # 前後の空白と句点を削除
        response = response.strip('。 　')
        
        # 数字のみの応答を削除
        if re.match(r'^\d+$', response):
            return "内容なし"
        
        # 空の応答の場合はデフォルト値を返す
        if not response or len(response) < 2:
            return "内容なし"
        
        return response

    def _clean_key_points(self, points):
        if not points or len(points) == 0:
            return ["内容なし"]
        
        cleaned = []
        for point in points:
            point = self._clean_response(point)
            if point and point != "内容なし":
                # 箇条書き記号を削除
                point = re.sub(r'^[-・\*] *', '', point)
                # 番号を削除
                point = re.sub(r'^\d+[\.\)、] *', '', point)
                cleaned.append(point)
        
        # 重複を削除
        cleaned = list(dict.fromkeys(cleaned))
        
        # 3つまでに制限
        return cleaned[:3] if cleaned else ["内容なし"]

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
            
            # セクション一覧の生成
            sections_html = []
            for i, segment in enumerate(segments, 1):
                start_time = int(segment.get('start_time', 0))
                end_time = int(segment.get('end_time', 0))
                sections_html.append(
                    f'<li><a href="#segment-{i}">'
                    f'{start_time//60:02d}:{start_time%60:02d} - '
                    f'{end_time//60:02d}:{end_time%60:02d} '
                    f'{html.escape(str(segment.get("heading", "")))}</a></li>'
                )
            
            sections_list = "\n".join(sections_html)
            
            # 要約部分の生成
            summary_html = f"""
            <div class="summary-section">
                <h2>動画の要約</h2>
                <div class="metadata">
                    <div class="metadata-item">
                        <span class="metadata-label">処理日時:</span>
                        <span class="metadata-value">{metadata.get('processed_at', '')}</span>
                    </div>
                    <div class="metadata-item">
                        <span class="metadata-label">動画の長さ:</span>
                        <span class="metadata-value">{metadata.get('video_duration', 0)}秒</span>
                    </div>
                    <div class="metadata-item">
                        <span class="metadata-label">セグメント数:</span>
                        <span class="metadata-value">{metadata.get('segment_count', 0)}</span>
                    </div>
                    <div class="metadata-item">
                        <span class="metadata-label">スクリーンショット数:</span>
                        <span class="metadata-value">{metadata.get('screenshot_count', 0)}</span>
                    </div>
                </div>
            </div>"""
            
            # セグメントのHTMLを生成
            segments_html = []
            for i, segment in enumerate(segments, 1):
                if segment:
                    start_time = int(segment.get('start_time', 0))
                    end_time = int(segment.get('end_time', 0))
                    
                    # スクリーンショットのギャラリーを生成
                    screenshots_html = ""
                    for j in range(5):  # 最大5枚のスクリーンショットを表示
                        screenshot_path = f"screenshot_{j}.jpg"
                        if os.path.exists(os.path.join(self.output_dir, screenshot_path)):
                            screenshots_html += f"""
                            <div class="screenshot">
                                <img src="{screenshot_path}" 
                                     alt="スクリーンショット {j+1}"
                                     loading="lazy">
                                <div class="screenshot-time">{j}秒</div>
                            </div>
                            """
                    
                    segment_html = f"""
                    <div class="segment" id="segment-{i}">
                        <div class="segment-header">
                            <div class="segment-title">
                                <span class="segment-number">#{i}</span>
                                <h2>{html.escape(str(segment.get("heading", "")))}</h2>
                            </div>
                            <div class="timestamp">
                                <span class="time-start">{start_time//60:02d}:{start_time%60:02d}</span>
                                <span class="time-separator">-</span>
                                <span class="time-end">{end_time//60:02d}:{end_time%60:02d}</span>
                            </div>
                        </div>
                        <div class="segment-content">
                            <div class="screenshots-gallery">
                                {screenshots_html}
                            </div>
                            <div class="text-content">
                                <div class="summary">
                                    <h3>要約</h3>
                                    <p>{html.escape(str(segment.get("summary", "")))}</p>
                                </div>
                                <div class="key-points">
                                    <h3>キーポイント</h3>
                                    <ul>
                                        {chr(10).join([f'<li>{html.escape(str(point))}</li>' for point in segment.get("key_points", [])])}
                                    </ul>
                                </div>
                                <div class="transcript">
                                    <h3>文字起こし</h3>
                                    <p>{html.escape(str(segment.get("text", "")))}</p>
                                </div>
                            </div>
                        </div>
                        <div class="segment-footer">
                            <a href="#" class="back-to-top">▲ トップへ戻る</a>
                        </div>
                    </div>
                    """
                    segments_html.append(segment_html)
                print(f"セグメント {i} のHTML生成が完了しました")
            
            # HTMLの全体構造
            html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>動画文字起こしレポート</title>
    <style>
        :root {{
            --primary-color: #2c3e50;
            --secondary-color: #34495e;
            --accent-color: #3498db;
            --background-color: #f5f7fa;
            --text-color: #2c3e50;
            --border-radius: 8px;
            --shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--background-color);
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        h1, h2, h3 {{
            color: var(--primary-color);
            margin-bottom: 1rem;
        }}
        
        h1 {{
            font-size: 2rem;
            text-align: center;
            margin-bottom: 2rem;
        }}
        
        .toc {{
            background-color: white;
                    padding: 2rem;
            border-radius: var(--border-radius);
            box-shadow: var(--shadow);
            margin-bottom: 2rem;
        }}
        
        .toc h2 {{
            margin-bottom: 1rem;
        }}
        
        .toc ul {{
            list-style: none;
                    padding-left: 0;
        }}
        
        .toc li {{
            margin-bottom: 0.5rem;
        }}
        
        .toc a {{
            color: var(--accent-color);
            text-decoration: none;
            transition: color 0.3s;
        }}
        
        .toc a:hover {{
            color: var(--primary-color);
        }}
        
        .summary-section {{
            background-color: white;
            padding: 2rem;
            border-radius: var(--border-radius);
            box-shadow: var(--shadow);
            margin-bottom: 2rem;
        }}
        
        .metadata {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }}
        
        .metadata-item {{
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}
        
        .metadata-label {{
            font-weight: bold;
            color: var(--secondary-color);
        }}
        
        .segment {{
            background-color: white;
            padding: 2rem;
            border-radius: var(--border-radius);
            box-shadow: var(--shadow);
            margin-bottom: 2rem;
        }}
        
        .segment-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid var(--background-color);
        }}
        
        .segment-title {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        
        .segment-number {{
            font-size: 1.2rem;
            font-weight: bold;
            color: var(--accent-color);
        }}
        
        .timestamp {{
            background-color: var(--background-color);
            padding: 0.5rem 1rem;
            border-radius: var(--border-radius);
            font-family: monospace;
        }}
        
        .segment-content {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
        }}
        
        .screenshots-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        
        .screenshot {{
                    position: relative;
            border-radius: var(--border-radius);
                    overflow: hidden;
            box-shadow: var(--shadow);
        }}
        
        .screenshot img {{
            width: 100%;
            height: auto;
                    display: block;
        }}
        
        .screenshot-time {{
            position: absolute;
            bottom: 0;
            right: 0;
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 0.3rem 0.6rem;
            font-size: 0.9rem;
            border-top-left-radius: var(--border-radius);
        }}
        
        @media (max-width: 768px) {{
            .screenshots-gallery {{
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            }}
        }}
        
        .text-content > div {{
            margin-bottom: 1.5rem;
        }}
        
        .text-content h3 {{
            font-size: 1.1rem;
            color: var(--secondary-color);
            margin-bottom: 0.5rem;
        }}
        
        .key-points ul {{
            list-style: none;
            padding-left: 0;
        }}
        
        .key-points li {{
            position: relative;
            padding-left: 1.5rem;
            margin-bottom: 0.5rem;
        }}
        
        .key-points li::before {{
            content: "•";
            position: absolute;
            left: 0;
            color: var(--accent-color);
            font-weight: bold;
        }}
        
        .transcript {{
            background-color: var(--background-color);
            padding: 1rem;
            border-radius: var(--border-radius);
        }}
        
        .segment-footer {{
            margin-top: 1.5rem;
            text-align: right;
        }}
        
        .back-to-top {{
            color: var(--accent-color);
            text-decoration: none;
            font-size: 0.9rem;
        }}
        
        .back-to-top:hover {{
            color: var(--primary-color);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>動画文字起こしレポート</h1>
        
        <div class="toc">
            <h2>目次</h2>
            <ul>
                {sections_list}
            </ul>
        </div>
        
        {summary_html}
        
        <div class="segments">
            {"".join(segments_html)}
        </div>
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
                'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'video_duration': sum([segment.get('end_time', 0) - segment.get('start_time', 0) for segment in result.get('segments', [])]),
                'segment_count': len(result.get('segments', [])),
                'screenshot_count': len(result.get('screenshots', []))
            }
            
            # 結果をJSONファイルとして保存
            output_json = os.path.join(self.output_dir, 'result.json')
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # HTMLレポートの生成
            html_path = os.path.join(self.output_dir, 'report.html')
            if self.generate_html_report(result, html_path):
                print(f"結果を保存しました:")
                print(f"- JSON: {output_json}")
                print(f"- HTML: {html_path}")
            else:
                print("警告: HTMLレポートの生成に失敗しました")
            
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

    def _generate_system_prompt(self) -> str:
        return """
あなたは簡潔で正確な応答を生成するAIアシスタントです。
以下のガイドラインに従ってください：
1. 指示語（「〜してください」など）は使用しない
2. 文字数制限を厳守する
3. 余分な説明や前置きを省く
4. 箇条書きは「・」を使用する
5. 応答は結果のみを含める
"""

    def process_segment(self, segment, segment_index, total_segments):
        """セグメントを処理し、メタデータを生成します。"""
        try:
            print(f"セグメント {segment_index + 1}/{total_segments} を処理中...")
            
            # セグメントのテキストを取得
            text = segment.get('text', '')
            if not text or len(text.strip()) < 2:  # 短すぎるテキストは処理しない
                print("セグメントのテキストが空または短すぎます")
                return None
            
            # ヘッディングの生成
            heading = self._generate_heading(text)
            
            # 要約の生成
            summary = self._generate_summary(text)
            
            # キーポイントの生成
            key_points = self._generate_key_points(text)
            
            # メタデータを生成
            metadata = {
                "start_time": segment.get('start', 0),
                "end_time": segment.get('end', 0),
                "text": text,
                "heading": heading,
                "summary": summary,
                "key_points": key_points,
                "screenshot": f"screenshot_{segment_index}.jpg"
            }
            
            print(f"セグメント {segment_index + 1} の処理が完了しました")
            return metadata
            
        except Exception as e:
            print(f"セグメント処理中にエラーが発生しました: {str(e)}")
            traceback.print_exc()
            return None

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
