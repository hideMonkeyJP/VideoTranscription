import unittest
from unittest.mock import patch, MagicMock
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from src.analysis.text_analyzer import TextAnalyzer
from src.exceptions import TextAnalysisError

class TestTextAnalyzer(unittest.TestCase):
    def setUp(self):
        # 環境変数を読み込みます
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            self.skipTest("GEMINI_API_KEYが設定されていません")
        
        # 環境変数をパッチします
        self.env_patcher = patch.dict(os.environ, {
            'GEMINI_API_KEY': api_key
        })
        self.env_patcher.start()
        
        self.analyzer = TextAnalyzer()
        
        # テストデータディレクトリを設定します
        self.test_data_dir = Path('output/temp')
        self.test_output_dir = Path('output_test/json_test')
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.env_patcher.stop()

    def test_analyze_content_v2_with_real_data(self):
        """実際のデータを使用してanalyze_content_v2をテストします"""
        try:
            # 実データを読み込みます
            with open(self.test_data_dir / 'transcription.json', 'r', encoding='utf-8') as f:
                transcription = json.load(f)
            
            with open(self.test_data_dir / 'ocr_results.json', 'r', encoding='utf-8') as f:
                ocr_results = json.load(f)
            
            # transcriptionデータをanalysis.json形式に変換します
            analysis_json = {
                "segments": [
                    {
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"]
                    }
                    for segment in transcription
                ]
            }
            
            # 実際のAPIを使用してメソッドを実行します
            result = self.analyzer.analyze_content_v2(analysis_json, ocr_results)
            
            # 基本構造を検証します
            self.assertIn('segments', result)
            self.assertIn('total_segments', result)
            self.assertIsInstance(result['segments'], list)
            self.assertGreater(len(result['segments']), 0)
            
            # 各セグメントを検証します
            for segment in result['segments']:
                # 時間範囲を検証します
                self.assertIn('time_range', segment)
                self.assertIn('start', segment['time_range'])
                self.assertIn('end', segment['time_range'])
                self.assertLess(segment['time_range']['start'], segment['time_range']['end'])
                
                # 要約を検証します
                self.assertIn('summary', segment)
                self.assertIsInstance(segment['summary'], str)
                self.assertGreater(len(segment['summary']), 0)  # 空の要約でないことを確認します
                
                # メタデータを検証します
                self.assertIn('metadata', segment)
                self.assertIn('segment_count', segment['metadata'])
                self.assertIn('has_screenshot_text', segment['metadata'])
                self.assertIn('summary_points', segment['metadata'])
                self.assertIn('keyword_count', segment['metadata'])
                
                # スクリーンショット情報を検証します（存在する場合）
                if 'screenshot' in segment:
                    self.assertIn('timestamp', segment['screenshot'])
                    self.assertIn('frame_number', segment['screenshot'])
                    self.assertIn('text', segment['screenshot'])
                    self.assertIn('ocr_confidence', segment['screenshot'])
            
            # 結果を保存します（デバッグ用）
            output_file = self.test_output_dir / 'analysis_test_result.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            self.assertTrue(output_file.exists())
            
            # 要約の内容を表示します（デバッグ用）
            print("\n=== 生成された要約の例 ===")
            for i, segment in enumerate(result['segments'][:3], 1):
                print(f"\nセグメント {i}:")
                print(f"時間範囲: {segment['time_range']['start']} - {segment['time_range']['end']}")
                print(f"要約: {segment['summary']}")
            
        except Exception as e:
            self.fail(f"テスト実行中にエラーが発生しました: {str(e)}")

    def test_dynamic_summary_lines(self):
        """動的な要約行数の決定ロジックをテストします"""
        # テストデータを準備します
        short_video = {
            "segments": [
                {"start": 0, "end": 10, "text": "テスト1"},
                {"start": 10, "end": 20, "text": "テスト2"}
            ]
        }
        
        long_video = {
            "segments": [
                {"start": i, "end": i+1, "text": f"テスト{i}"}
                for i in range(60)
            ]
        }
        
        ocr_data = {
            "screenshots": [
                {
                    "timestamp": 5,
                    "frame_number": 1,
                    "importance_score": 0.8,
                    "ocr_confidence": 0.9,
                    "text": "テストテキスト"
                }
            ]
        }

        # モックを設定します
        with patch.object(TextAnalyzer._model, 'generate_content') as mock_generate:
            mock_response = MagicMock()
            mock_response.text = "テスト要約文"
            mock_generate.return_value = mock_response
            
            # 短い動画をテストします
            short_result = self.analyzer.analyze_content_v2(short_video, ocr_data)
            self.assertLessEqual(len(short_result['segments']), 5)  # 短い動画は少ないセグメント数になります
            
            # 長い動画をテストします
            long_result = self.analyzer.analyze_content_v2(long_video, ocr_data)
            self.assertGreater(len(long_result['segments']), 5)  # 長い動画は多いセグメント数になります
            self.assertLessEqual(len(long_result['segments']), 15)  # 最大15行の制限を確認します

    def test_screenshot_selection(self):
        """スクリーンショット選定ロジックをテストします"""
        test_data = {
            "segments": [
                {
                    "start": 0,
                    "end": 10,
                    "text": "テストセグメント"
                }
            ]
        }
        
        test_screenshots = {
            "screenshots": [
                {
                    "timestamp": 2,
                    "frame_number": 1,
                    "importance_score": 0.9,
                    "ocr_confidence": 0.8,
                    "text": "高品質テキスト"
                },
                {
                    "timestamp": 5,
                    "frame_number": 2,
                    "importance_score": 0.5,
                    "ocr_confidence": 0.5,
                    "text": "低品質テキスト"
                }
            ]
        }
        
        # モックを設定します
        with patch.object(TextAnalyzer._model, 'generate_content') as mock_generate:
            mock_response = MagicMock()
            mock_response.text = "テスト要約文"
            mock_generate.return_value = mock_response
            
            result = self.analyzer.analyze_content_v2(test_data, test_screenshots)
            
            # スクリーンショットが選択されていることを確認します
            self.assertIn('screenshot', result['segments'][0])
            # より高品質なスクリーンショットが選択されていることを確認します
            self.assertEqual(result['segments'][0]['screenshot']['frame_number'], 1)

    def test_error_handling(self):
        """エラーハンドリングをテストします"""
        # 無効なデータでテストします
        with self.assertRaises(TextAnalysisError):
            self.analyzer.analyze_content_v2({}, {})
        
        with self.assertRaises(TextAnalysisError):
            self.analyzer.analyze_content_v2(
                {"segments": []},
                {"screenshots": []}
            )
        
        # 不正なデータ構造でテストします
        with self.assertRaises(TextAnalysisError):
            self.analyzer.analyze_content_v2(
                {"segments": [{"invalid": "data"}]},
                {"screenshots": [{"invalid": "data"}]}
            )
        
        # 不正な型のデータでテストします
        with self.assertRaises(TextAnalysisError):
            self.analyzer.analyze_content_v2(
                {"segments": [{"start": "invalid", "end": "invalid", "text": "test"}]},
                {"screenshots": [{"timestamp": "invalid", "frame_number": "invalid"}]}
            )
        
        # Noneデータでテストします
        with self.assertRaises(TextAnalysisError):
            self.analyzer.analyze_content_v2(None, None)
        
        # リストデータでテストします
        with self.assertRaises(TextAnalysisError):
            self.analyzer.analyze_content_v2([], [])

if __name__ == '__main__':
    unittest.main() 