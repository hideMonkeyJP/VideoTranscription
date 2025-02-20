# Supabase接続トラブルシューティング

## 発生した問題

### 1. Invalid API Keyエラー
初期設定では`anon`キーを使用していたため、以下のエラーが発生：
```
Connection test failed: {'message': 'Invalid API key', 'hint': 'Double check your Supabase `anon` or `service_role` API key.'}
```

### 2. クエリ実行エラー
複雑なクエリを使用していたため、以下のエラーが発生：
```
'code': 'PGRST100', 'details': 'unexpected \'(\' expecting letter, digit, "-", "->>"'
```

## 解決方法

### 1. APIキーの変更
1. `.env`ファイルのAPIキーを`service_role`キーに変更
```env
SUPABASE_URL=https://mlidxmirlzzrvdujgtab.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

2. 環境変数の再読み込み
```bash
source venv/bin/activate
```

### 2. クエリの最適化
`src/database/supabase_client.py`のテスト接続クエリを簡素化：
```python
def test_connection(self) -> bool:
    try:
        # シンプルなクエリに変更
        self.client.table('videos').select("id").limit(1).execute()
        print("Connection test successful")
        return True
    except Exception as e:
        print(f"Connection test failed: {str(e)}")
        return False
```

## 重要なポイント

1. APIキーの選択
   - 開発時：`service_role`キーを使用
   - 本番環境：適切な権限設定を行い、必要に応じて`anon`キーを使用

2. クエリの設計
   - シンプルなクエリを使用
   - 複雑な集計は避ける
   - 適切なリミットを設定

3. 環境変数の管理
   - `.env`ファイルを使用
   - 環境変数の再読み込みを確実に行う
   - システム環境変数との競合に注意

## 参考資料
- [Supabase Python Client Documentation](https://supabase.com/docs/reference/python/introduction)
- [Supabase Authentication](https://supabase.com/docs/guides/auth)
- [Database Queries](https://supabase.com/docs/reference/python/select)

## 使用方法

### 1. コマンドラインから実行
```bash
python -m src.tools.register_to_supabase output/regist.json "タスク管理の基礎" "videos/Sample.mp4" 60
```

### 2. Pythonコードから実行
```python
from src.tools.supabase_register import register_to_supabase

# データ登録の実行
success = register_to_supabase(
    json_path="output/regist.json",
    video_title="タスク管理の基礎",
    video_path="videos/Sample.mp4",
    duration=60
)

if success:
    print("登録成功！")
else:
    print("登録失敗...") 