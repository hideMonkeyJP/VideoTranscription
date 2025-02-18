import os
import sys
import unittest
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# プロジェクトルートへのパスを追加
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

class TestGemini(unittest.TestCase):
    def setUp(self):
        """テストの前準備"""
        # 環境変数の読み込み
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            self.fail("GEMINI_API_KEYが設定されていません")
            
        # Geminiの設定
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def test_text_generation(self):
        """テキスト生成の基本機能テスト"""
        test_text = """
        人工知能技術の進化により、画像認識や自然言語処理などの分野で
        革新的な進歩が見られています。
        """
        
        prompt = f"""
        以下のテキストを要約してください：
        {test_text}
        """
        
        try:
            response = self.model.generate_content(prompt)
            self.assertIsNotNone(response)
            self.assertTrue(hasattr(response, 'text'))
            self.assertGreater(len(response.text), 0)
        except Exception as e:
            self.fail(f"テキスト生成に失敗: {str(e)}")

    def test_heading_generation(self):
        """見出し生成のテスト"""
        test_text = """
        2024年第1四半期のAI市場分析によると、
        特に自然言語処理分野での需要が急増しています。
        """
        
        prompt = f"""
        以下のテキストに対して、30文字以内の見出しを生成してください：
        {test_text}
        """
        
        try:
            response = self.model.generate_content(prompt)
            self.assertIsNotNone(response)
            self.assertTrue(hasattr(response, 'text'))
            self.assertLess(len(response.text), 31)
        except Exception as e:
            self.fail(f"見出し生成に失敗: {str(e)}")

    def test_key_points_extraction(self):
        """キーポイント抽出のテスト"""
        test_text = """
        近年のAI技術の発展は目覚ましく、特に以下の分野で大きな進展が見られます：
        1. 自然言語処理による多言語翻訳の精度向上
        2. コンピュータビジョンによる医療診断支援
        3. 強化学習を用いた自動運転技術の進化
        4. 音声認識による新しいインターフェースの開発
        """
        
        prompt = f"""
        以下のテキストから3つの重要なポイントを抽出してください：
        {test_text}
        """
        
        try:
            response = self.model.generate_content(prompt)
            self.assertIsNotNone(response)
            self.assertTrue(hasattr(response, 'text'))
            # 改行で分割して要点の数を確認
            points = [p for p in response.text.split('\n') if p.strip()]
            self.assertLessEqual(len(points), 3)
        except Exception as e:
            self.fail(f"キーポイント抽出に失敗: {str(e)}")

if __name__ == '__main__':
    unittest.main() 