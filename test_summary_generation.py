import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import logging

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_summary_generation():
    try:
        # 環境変数の読み込み
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEYが設定されていません")
        
        logger.info(f"APIキーの長さ: {len(api_key)}")
        logger.info(f"APIキーの最初の10文字: {api_key[:10]}...")

        # Geminiの設定
        genai.configure(api_key=api_key)
        
        try:
            # モデルリストの取得を試みる
            logger.info("モデルリストの取得を開始...")
            models = genai.list_models()
            logger.info(f"利用可能なモデル: {[model.name for model in models]}")
        except Exception as e:
            logger.error(f"モデルリストの取得に失敗: {str(e)}")
            logger.error("モデルリストの取得はスキップしますが、処理は続行します")
        
        logger.info("Gemini-proモデルの初期化を開始...")
        model = genai.GenerativeModel('gemini-pro')
        logger.info("モデルの初期化が完了しました")

        # 中間ファイルから音声認識結果を読み込む
        logger.info("transcription.jsonの読み込みを開始...")
        file_path = 'output_test/main_test10/transcription.json'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            transcription = json.load(f)
        logger.info("transcription.jsonの読み込みが完了しました")
        
        # テキストを結合
        test_text = ' '.join([segment['text'] for segment in transcription])
        logger.info(f"入力テキストの長さ: {len(test_text)} 文字")

        # プロンプトの作成
        prompt = f"""
        以下のテキストを要約してください。
        箇条書きで3点以内にまとめ、各要点は50文字以内で記述してください。

        テキスト:
        {test_text}

        出力形式:
        • 要点1
        • 要点2
        • 要点3

        注意事項:
        - 重要な情報を優先
        - 具体的な数値やキーワードを含める
        - 簡潔な日本語で表現
        """

        logger.info("要約の生成を開始...")
        response = model.generate_content(prompt)
        logger.info("要約の生成が完了しました")
        
        if not response:
            raise ValueError("空の応答を受信しました")
            
        if not hasattr(response, 'text'):
            raise ValueError(f"応答にtextプロパティがありません: {response}")
            
        if not response.text:
            raise ValueError("応答のtextが空です")
            
        logger.info(f"生成された要約: {response.text}")
        return True

    except Exception as e:
        logger.error(f"要約生成中にエラーが発生: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    result = test_summary_generation()
    print(f"テスト結果: {'成功' if result else '失敗'}")