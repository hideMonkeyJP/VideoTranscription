import os
from dotenv import load_dotenv
import google.generativeai as genai

def split_text(text, max_length=2000):
    """テキストを適切な長さに分割"""
    if len(text) <= max_length:
        return [text]
    
    sentences = text.split('。')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip() + '。'
        if current_length + len(sentence) > max_length and current_chunk:
            chunks.append(''.join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)
    
    if current_chunk:
        chunks.append(''.join(current_chunk))
    
    return chunks

def test_gemini():
    # 環境変数の読み込み
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("GEMINI_API_KEYが設定されていません")
        return

    # Gemini APIの初期化
    genai.configure(api_key=api_key)
    
    # モデルの設定
    generation_config = {
        'temperature': 0.3,
        'top_p': 0.8,
        'top_k': 40,
        'max_output_tokens': 8192,  # Gemini 1.5 Proの最大出力トークン
    }
    
    # モデルの初期化
    model = genai.GenerativeModel(
        model_name='models/gemini-1.5-pro',
        generation_config=generation_config
    )
    
    # テスト用のテキスト(より長いサンプル)
    test_text = """
    人工知能(AI)技術の進化は、社会に大きな変革をもたらしています。

    機械学習とディープラーニングの発展により、画像認識、自然言語処理、音声認識など
    様々な分野で人間の能力に匹敵する、あるいはそれを上回る性能を示すようになりました。

    特に、大規模言語モデル(LLM)の登場により、テキスト生成、対話、翻訳、要約など、
    言語に関連するタスクで革新的な進歩が見られます。これらのモデルは、膨大なデータから
    学習することで、文脈を理解し、適切な応答を生成する能力を獲得しています。

    また、コンピュータビジョンの分野では、物体検出、顔認識、医療画像診断など、
    高度な視覚的理解を必要とするタスクで実用的なレベルに達しています。
    自動運転技術への応用も進んでおり、安全性と効率性の向上に貢献しています。

    さらに、強化学習の発展により、ゲームやロボット制御などの分野で、
    環境との相互作用を通じて最適な行動を学習するシステムが実現されています。

    これらのAI技術は、医療、教育、製造、金融など、様々な産業分野に
    革新的なソリューションをもたらし、人々の生活をより豊かにする可能性を秘めています。
    """
    
    # テキストを分割(必要な場合)
    chunks = split_text(test_text)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nチャンク {i}/{len(chunks)} の処理:")
        
        # プロンプトの準備(トークン数を考慮)
        prompt = f"""
以下のテキストを要約してください:
- 重要な情報を保持
- 簡潔な日本語で
- 100文字以内

テキスト:
{chunk}

要約:"""

        try:
            # 要約の生成
            response = model.generate_content(prompt)
            print("要約結果:")
            print(response.text)
        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")
            continue
    
    print("\nテスト完了!")

if __name__ == "__main__":
    test_gemini()