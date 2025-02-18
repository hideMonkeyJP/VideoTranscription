import unittest
from unittest.mock import Mock, patch, MagicMock
import os
from pathlib import Path
import json
import tempfile
import shutil
from datetime import datetime

from src.output import (
    ResultFormatter, FormatterError,
    ReportGenerator, ReportError,
    NotionSynchronizer, NotionSyncError
)

class TestResultFormatter(unittest.TestCase):
    def setUp(self):
        """テストの前準備"""
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            'output_dir': self.test_dir,
            'default_format': 'json'
        }
        self.formatter = ResultFormatter(self.config)
        
        # テスト用のデータ
        self.test_data = {
            'metadata': {
                'duration': 300,
                'total_segments': 5
            },
            'segments': [
                {
                    'heading': 'テストセグメント1',
                    'start_time': 0,
                    'end_time': 60,
                    'quality_score': 0.8,
                    'summary': 'テストの要約1',
                    'key_points': ['ポイント1', 'ポイント2']
                }
            ],
            'keywords': ['テスト', '分析', 'キーワード']
        }

    def test_format_json(self):
        """JSON形式のフォーマットテスト"""
        result = self.formatter.format_results(self.test_data, 'json')
        self.assertIsInstance(result, str)
        # JSONとして解析可能か確認
        parsed = json.loads(result)
        self.assertEqual(parsed['metadata']['total_segments'], 5)

    def test_format_markdown(self):
        """Markdown形式のフォーマットテスト"""
        result = self.formatter.format_results(self.test_data, 'markdown')
        self.assertIsInstance(result, str)
        self.assertIn('# 動画分析レポート', result)
        self.assertIn('テストセグメント1', result)

    def test_save_results(self):
        """結果の保存テスト"""
        output_path = self.formatter.save_results(
            self.test_data,
            'test_output',
            'json'
        )
        self.assertTrue(output_path.exists())
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        self.assertEqual(saved_data['metadata']['total_segments'], 5)

    def tearDown(self):
        """テスト後のクリーンアップ"""
        shutil.rmtree(self.test_dir)

class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        """テストの前準備"""
        self.test_dir = tempfile.mkdtemp()
        self.template_dir = Path(self.test_dir) / 'templates'
        self.output_dir = Path(self.test_dir) / 'reports'
        
        self.config = {
            'template_dir': str(self.template_dir),
            'output_dir': str(self.output_dir)
        }
        
        # テンプレートディレクトリの作成
        self.template_dir.mkdir(parents=True)
        
        # テスト用のテンプレート作成
        self.template_content = """
        <html>
        <body>
            <h1>{{ metadata.title }}</h1>
            {% for segment in segments %}
            <div class="segment">
                <h2>{{ segment.heading }}</h2>
                <p>{{ segment.summary }}</p>
            </div>
            {% endfor %}
        </body>
        </html>
        """
        
        # テスト用のデータ
        self.test_data = {
            'metadata': {
                'title': 'テストレポート',
                'duration': 300
            },
            'segments': [
                {
                    'heading': 'セグメント1',
                    'summary': 'テストの要約1'
                }
            ]
        }

    @patch('jinja2.Environment')
    def test_generate_report(self, mock_env):
        """レポート生成のテスト"""
        # テンプレートファイルの作成
        template_path = self.template_dir / 'test_template.html'
        template_path.parent.mkdir(parents=True, exist_ok=True)
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(self.template_content)

        # モックの設定
        mock_template = Mock()
        mock_template.render.return_value = "<html>テストレポート</html>"
        mock_env.return_value.get_template.return_value = mock_template
        
        generator = ReportGenerator(self.config)
        
        # レポートの生成
        output_path = generator.generate_report(
            self.test_data,
            'test_template.html'
        )
        
        self.assertTrue(output_path.exists())
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn('テストレポート', content)

    def test_create_template(self):
        """テンプレート作成のテスト"""
        generator = ReportGenerator(self.config)
        template_path = generator.create_template(
            'test_template.html',
            self.template_content
        )
        
        self.assertTrue(template_path.exists())
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertEqual(content, self.template_content)

    def tearDown(self):
        """テスト後のクリーンアップ"""
        shutil.rmtree(self.test_dir)

class TestNotionSynchronizer(unittest.TestCase):
    def setUp(self):
        """テストの前準備"""
        self.config = {
            'auth_token': 'test_token',
            'database_id': 'test_database'
        }
        
        # テスト用のデータ
        self.test_data = {
            'metadata': {
                'title': 'テスト動画',
                'duration': 300
            },
            'segments': [
                {
                    'heading': 'セグメント1',
                    'summary': 'テストの要約1'
                }
            ],
            'keywords': ['テスト', '分析']
        }

    @patch('requests.get')
    def test_get_database_properties(self, mock_get):
        """データベースプロパティ取得のテスト"""
        # モックの設定
        mock_response = Mock()
        mock_response.json.return_value = {
            'properties': {
                'Title': {'type': 'title'},
                'Summary': {'type': 'rich_text'}
            }
        }
        mock_get.return_value = mock_response
        
        synchronizer = NotionSynchronizer(self.config)
        properties = synchronizer.get_database_schema()
        
        self.assertEqual(properties['Title'], 'title')
        self.assertEqual(properties['Summary'], 'rich_text')

    @patch('requests.post')
    @patch('requests.get')
    def test_sync_results(self, mock_get, mock_post):
        """結果の同期テスト"""
        # モックの設定
        mock_get_response = Mock()
        mock_get_response.json.return_value = {'properties': {}}
        mock_get.return_value = mock_get_response
        
        mock_post_response = Mock()
        mock_post_response.json.return_value = {
            'id': 'test_page_id',
            'url': 'https://notion.so/test_page',
            'created_time': '2024-01-01T00:00:00.000Z',
            'last_edited_time': '2024-01-01T00:00:00.000Z'
        }
        mock_post.return_value = mock_post_response
        
        synchronizer = NotionSynchronizer(self.config)
        result = synchronizer.sync_results(self.test_data)
        
        self.assertEqual(result['page_id'], 'test_page_id')
        self.assertEqual(result['url'], 'https://notion.so/test_page')

    @patch('requests.get')
    def test_prepare_notion_data(self, mock_get):
        """Notionデータの準備テスト"""
        # モックの設定
        mock_response = Mock()
        mock_response.json.return_value = {
            'properties': {
                'Title': {'type': 'title'},
                'Summary': {'type': 'rich_text'},
                'Keywords': {'type': 'multi_select'}
            }
        }
        mock_get.return_value = mock_response
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()

        synchronizer = NotionSynchronizer(self.config)
        notion_data = synchronizer._prepare_notion_data(self.test_data)
        
        self.assertIn('parent', notion_data)
        self.assertIn('properties', notion_data)
        self.assertEqual(notion_data['parent']['database_id'], 'test_database')
        self.assertIn('Title', notion_data['properties'])
        self.assertIn('Keywords', notion_data['properties'])

if __name__ == '__main__':
    unittest.main() 