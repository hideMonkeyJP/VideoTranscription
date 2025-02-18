import os
import sys
import pytest

# プロジェクトルートディレクトリをPYTHONPATHに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

@pytest.fixture(scope="session")
def project_root():
    """プロジェクトのルートディレクトリを返す"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@pytest.fixture(scope="session")
def test_video_path():
    """テスト用の動画ファイルパスを返す"""
    return os.path.join('videos', 'Sample.mp4')

@pytest.fixture(scope="session")
def output_dir():
    """テスト用の出力ディレクトリを返す"""
    output_path = os.path.join('output', 'test')
    os.makedirs(output_path, exist_ok=True)
    return output_path

@pytest.fixture(autouse=False)
def cleanup_output(output_dir):
    """テスト実行後に出力ファイルをクリーンアップ"""
    yield
    # テスト終了後のクリーンアップ
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            try:
                os.remove(os.path.join(root, file))
            except Exception as e:
                print(f"クリーンアップ中にエラーが発生しました: {e}")

@pytest.fixture(scope="session")
def temp_dir():
    """テスト用の一時ディレクトリを返す"""
    temp_path = os.path.join('temp', 'test')
    os.makedirs(temp_path, exist_ok=True)
    return temp_path

@pytest.fixture(autouse=False)
def cleanup_temp(temp_dir):
    """テスト実行後に一時ファイルをクリーンアップ"""
    yield
    # テスト終了後のクリーンアップ
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            try:
                os.remove(os.path.join(root, file))
            except Exception as e:
                print(f"クリーンアップ中にエラーが発生しました: {e}")