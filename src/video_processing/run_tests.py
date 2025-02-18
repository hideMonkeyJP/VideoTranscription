import unittest
import sys
import os
import datetime
from unittest.runner import TextTestRunner
from unittest.loader import TestLoader
from src.video_processing.frame_extraction.frame_extractor_test import TestFrameExtractor
from src.video_processing.audio_extraction.audio_extractor_test import TestAudioExtractor
import xmlrunner
import HtmlTestRunner

def run_tests():
    """すべてのテストを実行し、結果をレポートとして保存"""
    # テスト対象のクラスを収集
    test_classes = [TestFrameExtractor, TestAudioExtractor]
    
    # テストスイートの作成
    loader = TestLoader()
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # 現在の日時を取得（ファイル名用）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # レポート保存ディレクトリ
    report_dir = os.path.join(os.path.dirname(__file__), 'test_reports')
    os.makedirs(report_dir, exist_ok=True)

    # XMLレポートの生成
    xml_report_file = os.path.join(report_dir, f'test_results_{timestamp}.xml')
    with open(xml_report_file, 'wb') as output:
        xml_runner = xmlrunner.XMLTestRunner(output=output)
        xml_runner.run(suite)

    # HTMLレポートの生成
    html_report_dir = os.path.join(report_dir, 'html')
    html_runner = HtmlTestRunner.HTMLTestRunner(
        output=html_report_dir,
        report_name=f"test_results_{timestamp}",
        combine_reports=True,
        add_timestamp=False
    )
    html_runner.run(suite)

    # テキストレポートの生成
    text_report_file = os.path.join(report_dir, f'test_results_{timestamp}.txt')
    with open(text_report_file, 'w') as f:
        runner = TextTestRunner(stream=f, verbosity=2)
        runner.run(suite)

    print(f"\nテストレポートが生成されました:")
    print(f"XML レポート: {xml_report_file}")
    print(f"HTML レポート: {os.path.join(html_report_dir, f'test_results_{timestamp}.html')}")
    print(f"テキスト レポート: {text_report_file}")

if __name__ == '__main__':
    run_tests() 