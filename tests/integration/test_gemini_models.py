import os
from dotenv import load_dotenv
import google.generativeai as genai

def list_available_models():
    # 環境変数の読み込み
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("GEMINI_API_KEYが設定されていません")
        return

    # Gemini APIの初期化
    genai.configure(api_key=api_key)
    
    try:
        # 利用可能なモデルの一覧を取得
        models = genai.list_models()
        
        print("利用可能なGeminiモデル:")
        print("-" * 50)
        
        for model in models:
            print(f"\nモデル名: {model.name}")
            print(f"表示名: {model.display_name}")
            print(f"説明: {model.description}")
            print(f"生成タイプ: {model.supported_generation_methods}")
            print(f"入力トークン制限: {model.input_token_limit}")
            print(f"出力トークン制限: {model.output_token_limit}")
            print("-" * 50)
            
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    list_available_models()