import json
import difflib
import re
from typing import Union, Dict, List, Any

def normalize_text(text: str) -> str:
    """テキストを正規化します
    
    Args:
        text (str): 正規化する文字列
        
    Returns:
        str: 正規化された文字列
    """
    # 文末表現の正規化
    endings = [
        'です', 'ます', 'でした', 'ました', 'だ', 'である', 'だった', 'であった',
        'のだ', 'のです', 'のである', 'ください', 'ましょう', 'でしょう'
    ]
    for end in endings:
        text = text.replace(end, '')
    
    # 基本的な文字正規化
    text = text.lower()  # 小文字化
    text = ''.join(char for char in text if not char.isspace())  # 空白除去
    
    # 記号の正規化
    text = text.translate(str.maketrans({
        '、': '',
        '。': '',
        '!': '',
        '?': '',
        '(': '',
        ')': '',
        '「': '',
        '」': '',
        '『': '',
        '』': '',
        '・': '',
        ' ': ''
    }))
    
    return text

def compare_json_files(file1: str, file2: str) -> bool:
    """2つのJSONファイルを比較します
    
    Args:
        file1 (str): 比較対象の1つ目のJSONファイルパス
        file2 (str): 比較対象の2つ目のJSONファイルパス
        
    Returns:
        bool: 比較結果(True: 一致、False: 不一致)
    """
    try:
        with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)
            return compare_generic_json(data1, data2)
    except Exception as e:
        print(f"ファイル比較中にエラーが発生: {str(e)}")
        return False

def compare_summaries(data1: List[Dict], data2: List[Dict]) -> bool:
    """summaries.json用の比較関数
    
    Args:
        data1 (List[Dict]): 比較対象の1つ目のサマリーデータ
        data2 (List[Dict]): 比較対象の2つ目のサマリーデータ
        
    Returns:
        bool: 比較結果(True: 一致、False: 不一致)
    """
    try:
        if len(data1) != len(data2):
            print(f"セグメント数が異なります: {len(data1)} vs {len(data2)}")
            return False

        for item1, item2 in zip(data1, data2):
            # タイムスタンプの比較(誤差を許容)
            if abs(float(item1.get('timestamp', 0)) - float(item2.get('timestamp', 0))) > 0.2:
                print("タイムスタンプが大きく異なります")
                return False

            # スクリーンショットテキストの比較
            text1 = normalize_text(item1.get('screenshot_text', ''))
            text2 = normalize_text(item2.get('screenshot_text', ''))
            if text1 != text2:
                similarity = difflib.SequenceMatcher(None, text1, text2).ratio() * 100
                if similarity < 80.0:  # 類似度閾値を80%に設定
                    print(f"スクリーンショットテキストの類似度が低すぎます: {similarity:.2f}%")
                    return False

            # メタデータの比較
            if item1.get('metadata') != item2.get('metadata'):
                print("メタデータが一致しません")
                return False

        return True
    except Exception as e:
        print(f"サマリー比較中にエラーが発生: {str(e)}")
        return False

def compare_contexts(contexts1: List[Dict], contexts2: List[Dict]) -> bool:
    """contexts用の比較関数
    
    Args:
        contexts1 (List[Dict]): 比較対象の1つ目のコンテキストデータ
        contexts2 (List[Dict]): 比較対象の2つ目のコンテキストデータ
        
    Returns:
        bool: 比較結果(True: 一致、False: 不一致)
    """
    try:
        if len(contexts1) != len(contexts2):
            print(f"コンテキスト数が異なります: {len(contexts1)} vs {len(contexts2)}")
            return False

        for ctx1, ctx2 in zip(contexts1, contexts2):
            # time_rangeの比較
            time_range1 = ctx1.get('time_range', {})
            time_range2 = ctx2.get('time_range', {})
            
            start_diff = abs(float(time_range1.get('start', 0)) - float(time_range2.get('start', 0)))
            end_diff = abs(float(time_range1.get('end', 0)) - float(time_range2.get('end', 0)))
            
            if start_diff > 0.2 or end_diff > 0.2:
                print("time_rangeが大きく異なります")
                return False

            # シーンの比較
            scenes1 = ctx1.get('scenes', [])
            scenes2 = ctx2.get('scenes', [])
            
            if len(scenes1) != len(scenes2):
                print(f"シーン数が異なります: {len(scenes1)} vs {len(scenes2)}")
                return False

            for scene1, scene2 in zip(scenes1, scenes2):
                # シーンテキストの比較
                text1 = normalize_text(scene1.get('screenshot_text', ''))
                text2 = normalize_text(scene2.get('screenshot_text', ''))
                if text1 != text2:
                    similarity = difflib.SequenceMatcher(None, text1, text2).ratio() * 100
                    if similarity < 80.0:
                        print(f"シーンテキストの類似度が低すぎます: {similarity:.2f}%")
                        return False

                # メタデータの比較
                if scene1.get('metadata') != scene2.get('metadata'):
                    print("シーンのメタデータが一致しません")
                    return False

            # サマリーの比較
            summary1 = normalize_text(ctx1.get('summary', ''))
            summary2 = normalize_text(ctx2.get('summary', ''))
            if summary1 != summary2:
                print("サマリーが一致しません")
                return False

        return True
    except Exception as e:
        print(f"コンテキスト比較中にエラーが発生: {str(e)}")
        return False

def compare_scenes(scene1: Dict, scene2: Dict) -> bool:
    """シーン同士の比較関数
    
    Args:
        scene1 (Dict): 比較対象の1つ目のシーンデータ
        scene2 (Dict): 比較対象の2つ目のシーンデータ
        
    Returns:
        bool: 比較結果(True: 一致、False: 不一致)
    """
    try:
        # タイムスタンプの比較
        if abs(float(scene1.get('timestamp', 0)) - float(scene2.get('timestamp', 0))) > 0.2:
            print("タイムスタンプが大きく異なります")
            return False

        # テキストの比較
        text1 = normalize_text(scene1.get('screenshot_text', ''))
        text2 = normalize_text(scene2.get('screenshot_text', ''))
        if text1 != text2:
            similarity = difflib.SequenceMatcher(None, text1, text2).ratio() * 100
            if similarity < 80.0:
                print(f"テキストの類似度が低すぎます: {similarity:.2f}%")
                return False

        # メタデータの比較
        if scene1.get('metadata') != scene2.get('metadata'):
            print("メタデータが一致しません")
            return False

        # キーワードの比較
        if set(scene1.get('keywords', [])) != set(scene2.get('keywords', [])):
            print("キーワードが一致しません")
            return False

        return True
    except Exception as e:
        print(f"シーン比較中にエラーが発生: {str(e)}")
        return False

def compare_generic_json(data1: Any, data2: Any) -> bool:
    """その他のJSONデータの汎用的な比較関数
    
    Args:
        data1 (Any): 比較対象の1つ目のデータ
        data2 (Any): 比較対象の2つ目のデータ
        
    Returns:
        bool: 比較結果(True: 一致、False: 不一致)
    """
    try:
        # None値のチェック
        if data1 is None or data2 is None:
            raise Exception("比較対象にNone値が含まれています")

        if isinstance(data1, dict) and isinstance(data2, dict):
            if set(data1.keys()) != set(data2.keys()):
                print("キーが一致しません")
                return False
            return all(compare_generic_json(data1[key], data2[key]) for key in data1)
        elif isinstance(data1, list) and isinstance(data2, list):
            if len(data1) != len(data2):
                print("配列の長さが一致しません")
                return False
            return all(compare_generic_json(item1, item2) for item1, item2 in zip(data1, data2))
        else:
            return data1 == data2
    except Exception as e:
        print(f"汎用比較中にエラーが発生: {str(e)}")
        raise