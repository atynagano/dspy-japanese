# DSPyを用いたドキュメントからの自動コード生成

本チュートリアルでは、DPSyを使用してURLからドキュメントを自動取得し、任意のライブラリに対応した動作可能なコード例を生成する方法を解説します。本システムはドキュメントサイトを解析し、主要な概念を抽出した上で、特定のユースケースに合わせたコード例を生成することが可能です。

## 構築するシステム

ドキュメントを活用したコード生成システムで、以下の機能を備えています：

- 複数のURLからドキュメントを取得・解析する機能
- APIのパターン、メソッド、および使用例を抽出する機能
- 特定の使用ケースに対応した動作可能なコードを生成する機能
- 解説文やベストプラクティスを提供する機能
- あらゆるライブラリのドキュメントに対応可能な汎用性

## セットアップ手順

```bash
pip install dspy requests beautifulsoup4 html2text
```

## ステップ1：ドキュメントの取得と処理

```python
import dspy
import requests
from bs4 import BeautifulSoup
import html2text
from typing import List, Dict, Any
import json
from urllib.parse import urljoin, urlparse
import time

# DSPyの設定
lm = dspy.LM(model='openai/gpt-4o-mini')
dspy.configure(lm=lm)

class DocumentationFetcher:
    """URLからドキュメントを取得・処理するクラス"""
    
    def __init__(self, max_retries=3, delay=1):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.max_retries = max_retries
        self.delay = delay
        self.html_converter = html

## ステップ2：ドキュメントURLからの学習

```python
def learn_library_from_urls(library_name: str, documentation_urls: list[str]) -> Dict:
    """ドキュメントURLから任意のライブラリの詳細情報を取得する"""
    
    try:
        library_info = agent.learn_from_urls(library_name, documentation_urls)
        
        print(f"\n🔍 {library_name} ライブラリ分析結果:")
        print(f"参照元: {len(library_info['source_urls'])} 件の取得成功")
        print(f"主要概念: {library_info['core_concepts']}")
        print(f"一般的な使用パターン: {library_info['patterns']}")
        print(f"主要メソッド: {library_info['methods']}")
        print(f"インストール方法: {library_info['installation']}")
        print(f"コード例 {len(library_info['examples'])} 件を発見")
        
        return library_info
        
    except Exception as e:
        print(f"❌ ライブラリ学習中にエラー発生: {e}")
        raise

# 使用例 1: FastAPIの公式ドキュメントから詳細を取得
fastapi_urls = [
    "https://fastapi.tiangolo.com/",
    "https://fastapi.tiangolo.com/tutorial/first-steps/",
    "https://fastapi.tiangolo.com/tutorial/path-params/",
    "https://fastapi.tiangolo.com/tutorial/query-params/"
]

print("🚀 FastAPIの公式ドキュメントから詳細を取得中...")
fastapi_info = learn_library_from_urls("FastAPI", fastapi_urls)

# 使用例 2: 別のライブラリの詳細を取得（任意のライブラリで置き換え可能）
streamlit_urls = [
    "https://docs.streamlit.io/",
    "https://docs.streamlit.io/get-started",
    "https://docs.streamlit.io/develop/api-reference"
]

print("\n\n📊 Streamlitの公式ドキュメントから詳細を取得中...")
streamlit_info = learn_library_from_urls("Streamlit", streamlit_urls)
```

## ステップ3: コード例の生成

```python
def generate_examples_for_library(library_info: Dict, library_name: str):
    """ライブラリのドキュメントに基づいて、任意のライブラリのコードサンプルを生成します。"""
    
    # ほとんどのライブラリに適用可能な汎用的なユースケースを定義します
    use_cases = [
        {
            "name": "基本設定とHello World",
            "description": f"{library_name}を使用した最小限の動作例を作成します",
            "requirements": "インストール手順、インポート文、および基本的な使用方法を含める"
        },
        {
            "name": "一般的な操作例",
            "description": f"{library_name}で最も頻繁に使用される操作を実演します",
            "requirements": "典型的なワークフローとベストプラクティスを示す"
        },
        {
            "name": "高度な使用例",
            "description": f"{library_name}の機能を最大限に活用したより複雑な例を作成します",
            "requirements": "エラー処理と最適化手法を含める"
        }
    ]
    
    generated_examples = []
    
    print(f"\n🔧 {library_name}のコードサンプルを生成中...")
    
    for use_case in use_cases:
        print(f"\n📝 {use_case['name']}")
        print(f"説明: {use_case['description']}")
        
        example = agent.generate_example(
            library_info=library_info,
            use_case=use_case['description'],
            requirements=use_case['requirements']
        )
        
        print("\n💻 生成されたコード:")
        print("```python")
        print(example['code'])
        print("```")
        
        print("\n📦 必要なインポート文:")
        for imp in example['imports']:
            print(f"  • {imp}")
        
        print("\n📝 解説:")
        print(example['explanation'])
        
        print("\n✅ ベストプラクティス:")
        for practice in example['best_practices']:
            print(f"  • {practice}")
        
        generated_examples.append({
            "use_case": use_case['name'],
            "code": example['code'],
            "imports": example['imports'],
            "explanation": example['explanation'],
            "best_practices": example['best_practices']
        })
        
        print("-" * 80)
    
    return generated_examples

# 両ライブラリのコードサンプルを生成します
print("🎯 FastAPIのコードサンプルを生成中:")
fastapi_examples = generate_examples_for_library(fastapi_info, "FastAPI")

print("\n\n🎯 Streamlitのコードサンプルを生成中:")
streamlit_examples = generate_examples_for_library(streamlit_info, "Streamlit")
```

## ステップ4：インタラクティブライブラリ学習機能

```python
def learn_any_library(library_name: str, documentation_urls: list[str], use_cases: list[str] = None):
    """ドキュメントを参照しながら任意のライブラリを学習し、サンプルコードを生成します。"""
    
    if use_cases is None:
        use_cases = [
            "基本的なセットアップとhello worldのサンプル実装",
            "一般的な操作手順と開発ワークフロー",
            "ベストプラクティスを考慮した高度な使用例"
        ]
    
    print(f"🚀 {

## 出力例

インタラクティブ学習システムを実行すると、以下のような画面が表示されます：

**インタラクティブセッション開始：**
```
🎯 インタラクティブライブラリ学習システムへようこそ！
本システムは、Pythonライブラリの公式ドキュメントから学習を進めるための支援ツールです。

============================================================
🚀 ライブラリ学習セッション開始
============================================================

📚 学習したいライブラリ名を入力してください（終了する場合は「quit」と入力）： FastAPI

🔗 FastAPIのドキュメントURLを1行ずつ入力してください（入力終了には空行を使用）：
  URL: https://fastapi.tiangolo.com/
  URL: https://fastapi.tiangolo.com/tutorial/first-steps/
  URL: https://fastapi.tiangolo.com/tutorial/path-params/
  URL: 

🎯 FastAPIの使用シナリオを定義してください（任意、デフォルト値を使用する場合はEnterキーを押す）：
   デフォルトの使用シナリオは以下の通りです：基本設定、一般的な操作、高度な活用
   カスタム使用シナリオを定義しますか？（y/n）： y
   使用シナリオを入力してください（1シナリオずつ改行して、入力終了には空行を使用）：
     使用シナリオ：認証機能を備えたREST APIの作成
     使用シナリオ：ファイルアップロードエンドポイントの構築
     使用シナリオ：SQLAlchemyを用いたデータベース連携の追加
     使用シナリオ：
```

**文書処理工程:**
```
🚀 FastAPI の学習プロセスを開始します...
🚀 FastAPI の自動学習を開始します...
ドキュメントソース：3 つの URL
📡 取得中：https://fastapi.tiangolo.com/（試行 1回目）
📡 取得中：https://fastapi.tiangolo.com/tutorial/first-steps/（試行 1回目）
📡 取得中：https://fastapi.tiangolo.com/tutorial/path-params/（試行 1回目）
📚 3 つの URL から FastAPI について学習中...

🔍 FastAPI に関するライブラリ解析結果：
取得成功数：3 件
主要概念：['FastAPI アプリケーション', 'パス操作', '依存関係管理', 'リクエスト/レスポンスモデル']
頻出パターン：['app = FastAPI()', 'デコレータベースのルーティング', 'Pydantic モデル']
主要メソッド：['FastAPI()', '@app.get()', '@app.post()', 'uvicorn.run()']
インストール方法：pip install fastapi uvicorn
```

**コード生成:**
```
📝 例1/3生成中：認証機能を備えたREST APIの作成

✅ FastAPIの学習を完了しました！

📊 FastAPI学習のまとめ：
   • 主要概念：4項目を特定
   • 頻出パターン：3パターンを確認
   • 生成した実装例：3件

👀 FastAPIの生成例を表示しますか？（y/n）： y

──────────────────────────────────────────────────
📝 例1：認証機能を備えたREST APIの作成
──────────────────────────────────────────────────

💻 生成されたコード：
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from typing import Dict
import jwt
from datetime import datetime, timedelta

app = FastAPI(title="認証対応API", version="1.0.0")
security = HTTPBearer()

# JWT用シークレットキー（本番環境では環境変数を使用）
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="無効なトークン")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="無効なトークン")

@app.post("/login")
async def login(username: str, password: str) -> dict[str, str]:
    # 本番環境ではデータベースによる認証を実施
    if username == "admin" and password == "secret":
        token_data = {"sub": username, "exp": datetime.utcnow() + timedelta(hours=24)}
        token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="無効な認証情報")

@app.get("/protected")
async def protected_route(current_user: str = Depends(verify_token)) -> dict[str, str]:
    return {"message": f"こんにちは {current_user} 様！こちらは保護されたルートです。"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

📦 必要なインポート：
  • pip install fastapi uvicorn python-jose[cryptography]
  • from fastapi import FastAPI, Depends, HTTPException, status
  • from fastapi.security import HTTPBearer
  • import jwt

📝 解説：
本例では、JWTベースの認証機能を備えたFastAPIアプリケーションを実装しています。ログインエンドポイントではJWTトークンを発行し、保護されたルートでは認証済みユーザーのみがアクセス可能となる仕組みを採用しています...

✅ ベストプラクティス：
  • シークレットキーは環境変数で管理すること
  • 本番環境では適切なパスワードハッシュ処理を実装すること
  • トークンの有効期限設定と更新ロジックを追加すること
  • 適切なエラーハンドリングを実装すること

次の例に進みますか？（y/n）： n

💾 FastAPIの学習結果をファイルに保存しますか？（y/n）： y
   ファイル名を入力してください（デフォルト：fastapi_learning.json）： 
   ✅ fastapi_learning.json に結果を保存しました

📚 これまでに学習したライブラリ： ['FastAPI']

🔄 別のライブラリを学習しますか？（y/n）： n

🎉 セッションまとめ：
1つのライブラリの学習を完了しました：
   • FastAPI：3件の実装例を生成
```


## 次のステップ

- **GitHub連携**: READMEファイルやサンプルリポジトリから学習する
- **動画チュートリアル処理**: 動画形式のドキュメントから情報を抽出する
- **コミュニティ事例の収集**: Stack Overflowやフォーラムから収集した使用例を集約する
- **バージョン比較**: ライブラリの各バージョン間でAPIの変更点を追跡する
- **テストケース生成**: 自動生成されたコードに対してユニットテストを自動的に作成する
- **ページクローリング**: ドキュメントページを自動的に巡回し、実際の使用状況を積極的に把握する

本チュートリアルでは、DSPyを使用することで、未知のライブラリのドキュメントからその学習プロセス全体を自動化できる方法を示しています。これは、迅速な技術習得と探索において非常に有用な手法です。
