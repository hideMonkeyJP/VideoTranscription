import unittest
from unittest.mock import Mock, patch
import os
from dotenv import load_dotenv
from .text_analyzer import TextAnalyzer, TextAnalysisError

class TestTextAnalyzer(unittest.TestCase):
    def setUp(self):
        """テストの前準備"""
        # 環境変数の読み込み
        load_dotenv()
        
        # テスト用の設定
        self.config = {
            'min_confidence': 0.6
        }
        
        # テスト用のサンプルテキスト
        self.sample_text = """
        2023年度第3四半期の業績報告について説明します。
        売上高は前年同期比15%増の100億円となりました。
        主力製品のシェアが20%から25%に拡大し、新規顧客も10社獲得しました。
        一方、原材料費の高騰により、営業利益率は0.5ポイント低下しています。
        来期に向けては、コスト削減と新製品の開発を進めていく方針です。
        """
        
        # TextAnalyzerのインスタンス化
        self.analyzer = TextAnalyzer(self.config)

    @patch('google.generativeai.GenerativeModel')
    def test_initialization(self, mock_model):
        """初期化のテスト"""
        analyzer = TextAnalyzer(self.config)
        self.assertEqual(analyzer.min_confidence, 0.6)
        self.assertIsNotNone(analyzer.logger)
        self.assertIsNotNone(analyzer.model)

    @patch('google.generativeai.GenerativeModel')
    def test_generate_heading(self, mock_model):
        """見出し生成のテスト"""
        # モックの設定
        mock_response = Mock()
        mock_response.text = "2023年度第3四半期の業績報告"
        mock_model.return_value.generate_content.return_value = mock_response
        
        analyzer = TextAnalyzer(self.config)
        heading = analyzer.generate_heading(self.sample_text)
        
        self.assertIsInstance(heading, str)
        self.assertLess(len(heading), 31)
        self.assertNotIn("。", heading)

    @patch('google.generativeai.GenerativeModel')
    def test_generate_summary(self, mock_model):
        """要約生成のテスト"""
        # モックの設定
        mock_response = Mock()
        mock_response.text = "売上高15%増の100億円、主力製品シェア25%に拡大。原材料費高騰で営業利益率低下。"
        mock_model.return_value.generate_content.return_value = mock_response
        
        analyzer = TextAnalyzer(self.config)
        summary = analyzer.generate_summary(self.sample_text)
        
        self.assertIsInstance(summary, str)
        self.assertLess(len(summary), 101)

    @patch('google.generativeai.GenerativeModel')
    def test_generate_key_points(self, mock_model):
        """キーポイント抽出のテスト"""
        # モックの設定
        mock_response = Mock()
        mock_response.text = """
        • 売上高が前年同期比15%増の100億円
        • 主力製品のシェアが25%に拡大
        • 原材料費高騰により営業利益率が低下
        """
        mock_model.return_value.generate_content.return_value = mock_response
        
        analyzer = TextAnalyzer(self.config)
        key_points = analyzer.generate_key_points(self.sample_text)
        
        self.assertIsInstance(key_points, list)
        self.assertLessEqual(len(key_points), 3)
        for point in key_points:
            self.assertLess(len(point), 51)

    def test_calculate_text_quality(self):
        """テキスト品質計算のテスト"""
        # 高品質なテキスト
        good_text = "今年度の売上高は前年比20%増加し、過去最高を記録しました。"
        good_score = self.analyzer.calculate_text_quality(good_text)
        self.assertGreater(good_score, 0.7)
        
        # 低品質なテキスト
        bad_text = "あああああああ!!!!!!@@@@@@"
        bad_score = self.analyzer.calculate_text_quality(bad_text)
        self.assertLess(bad_score, 0.3)
        
        # 空のテキスト
        empty_score = self.analyzer.calculate_text_quality("")
        self.assertEqual(empty_score, 0.0)

    def test_detect_topic_change(self):
        """トピック変更検出のテスト"""
        text1 = "今年度の売上高について説明します。"
        text2 = "来年度の採用計画について説明します。"
        text3 = "今年度の売上高は好調に推移しています。"
        
        # 異なるトピック
        self.assertTrue(self.analyzer.detect_topic_change(text1, text2))
        
        # 同じトピック
        self.assertFalse(self.analyzer.detect_topic_change(text1, text3))

    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        with patch('google.generativeai.GenerativeModel') as mock_model:
            # モデルがエラーを発生させる場合
            mock_model.return_value.generate_content.side_effect = Exception("API Error")
            
            analyzer = TextAnalyzer(self.config)
            
            # 見出し生成のエラー
            with self.assertRaises(TextAnalysisError):
                analyzer.generate_heading("テストテキスト")
            
            # 要約生成のエラー
            with self.assertRaises(TextAnalysisError):
                analyzer.generate_summary("テストテキスト")
            
            # キーポイント抽出のエラー
            with self.assertRaises(TextAnalysisError):
                analyzer.generate_key_points("テストテキスト")

    def test_clean_response(self):
        """レスポンスのクリーニングテスト"""
        # 不要な記号を含むテキスト
        dirty_text = "「これは」【テスト】（です）。！？"
        cleaned = self.analyzer._clean_response(dirty_text)
        self.assertNotIn("「」", cleaned)
        self.assertNotIn("【】", cleaned)
        self.assertNotIn("（）", cleaned)
        self.assertNotIn("！？", cleaned)
        
        # 空のテキスト
        self.assertEqual(self.analyzer._clean_response(""), "内容なし")
        
        # 数字のみのテキスト
        self.assertEqual(self.analyzer._clean_response("12345"), "内容なし")

if __name__ == '__main__':
    unittest.main() 