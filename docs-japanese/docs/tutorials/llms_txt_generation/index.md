# DSPyを用いたコードドキュメント用`llms.txt`ファイルの自動生成

本チュートリアルでは、DSpyを使用してDSpyリポジトリ自体の`llms.txt`ファイルを自動生成する方法を説明します。`llms.txt`規格は、LLM（大規模言語モデル）向けに最適化されたコードベース理解を支援するドキュメント形式です。

## `llms.txt`とは何か？

`llms.txt`は、プロジェクトに関する構造化されたLLMフレンドリーなドキュメントを提供するための提案規格です。通常、以下の内容を含みます：

- プロジェクトの概要と目的
- 主要な概念と専門用語
- アーキテクチャと構成構造
- 使用例
- 重要なファイルとディレクトリ

## DSPyプログラムによる`llms.txt`生成システムの構築

リポジトリを解析し、包括的な`llms.txt`ドキュメントを生成するDSpyプログラムを作成しましょう。

### ステップ1：署名定義の作成

まず、ドキュメント生成の各要素に対応する署名を定義します：

```python
import dspy
from typing import List

class AnalyzeRepository(dspy.Signature):
    """リポジトリ構造を分析し、主要な構成要素を特定します。"""
    repo_url: str = dspy.InputField(desc="GitHubリポジトリのURL")
    file_tree: str = dspy.InputField(desc="リポジトリのファイル構成")
    readme_content: str = dspy.InputField(desc="README.mdファイルの内容")
    
    project_purpose: str = dspy.OutputField(desc="プロジェクトの主要な目的と目標")
    key_concepts: list[str] = dspy.OutputField(desc="重要な概念および専門用語の一覧")
    architecture_overview: str = dspy.OutputField(desc="高レベルなアーキテクチャの概要説明")

class AnalyzeCodeStructure(dspy.Signature):
    """コード構造を分析し、重要なディレクトリとファイルを特定します。"""
    file_tree: str = dspy.InputField(desc="リポジトリのファイル構成")
    package_files: str = dspy.InputField(desc="主要なパッケージおよび設定ファイル")
    
    important_directories: list[str] = dspy.OutputField(desc="重要なディレクトリとその目的")
    entry_points: list[str] = dspy.OutputField(desc="主要なエントリポイントおよび重要ファイル")
    development_info: str = dspy.OutputField(desc="開発環境のセットアップとワークフローに関する情報")

class GenerateLLMsTxt(dspy.Signature):
    """分析済みリポジトリ情報に基づいて包括的な llms.txt ファイルを生成します。"""
    project_purpose: str = dspy.InputField()
    key_concepts: list[str] = dspy.InputField()
    architecture_overview: str = dspy.InputField()
    important_directories: list[str] = dspy.InputField()
    entry_points: list[str] = dspy.InputField()
    development_info: str = dspy.InputField()
    usage_examples: str = dspy.InputField(desc="一般的な使用パターンと具体例")
    
    llms_txt_content: str = dspy.OutputField(desc="標準フォーマットに準拠した完全な llms.txt ファイルの内容")
```

### ステップ2：リポジトリアナライザーモジュールの作成

```python
class RepositoryAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        # リポジトリ分析用の思考連鎖処理を定義
        self.analyze_repo = dspy.ChainOfThought(AnalyzeRepository)
        # コード構造分析用の思考連鎖処理を定義
        self.analyze_structure = dspy.ChainOfThought(AnalyzeCodeStructure)
        # 使用例生成用の思考連鎖処理を定義
        self.generate_examples = dspy.ChainOfThought("repo_info -> usage_examples")
        # LLMs用テキスト生成用の思考連鎖処理を定義
        self.generate_llms_txt = dspy.ChainOfThought(GenerateLLMsTxt)

    def forward(self, repo_url, file_tree, readme_content, package_files):
        # リポジトリの目的と主要概念を分析
        repo_analysis = self.analyze_repo(
            repo_url=repo_url,
            file_tree=file_tree,
            readme_content=readme_content
        )
        
        # コード構造を分析
        structure_analysis = self.analyze_structure(
            file_tree=file_tree,
            package_files=package_files
        )
        
        # 使用例を生成
        usage_examples = self.generate_examples(
            repo_info=f"目的: {repo_analysis.project_purpose}\n主要概念: {repo_analysis.key_concepts}"
        )
        
        # LLMs用テキストファイルを生成
        llms_txt = self.generate_llms_txt(
            project_purpose=repo_analysis.project_purpose,
            key_concepts=repo_analysis.key_concepts,
            architecture_overview=repo_analysis.architecture_overview,
            important_directories=structure_analysis.important_directories,
            entry_points=structure_analysis.entry_points,
            development_info=structure_analysis.development_info,
            usage_examples=usage_examples.usage_examples
        )
        
        # 最終的な出力として予測結果を返す
        return dspy.Prediction(
            llms_txt_content=llms_txt.llms_txt_content,
            analysis=repo_analysis,
            structure=structure_analysis
        )
```

### ステップ3: リポジトリ情報の取得

リポジトリ情報を抽出するための補助関数を作成します：

```python
import requests
import os
from pathlib import Path

os.environ["GITHUB_ACCESS_TOKEN"] = "<your_access_token>"

def get_github_file_tree(repo_url):
    """GitHub APIからリポジトリのファイル構造を取得する"""
    # URLから所有者名とリポジトリ名を抽出
    parts = repo_url.rstrip('/').split('/')
    owner, repo = parts[-2], parts[-1]
    
    api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
    response = requests.get(api_url, headers={
        "Authorization": f"Bearer {os.environ.get('GITHUB_ACCESS_TOKEN')}"
    })
    
    if response.status_code == 200:
        tree_data = response.json()
        file_paths = [item['path'] for item in tree_data['tree'] if item['type'] == 'blob']
        return '\n'.join(sorted(file_paths))
    else:
        raise Exception(f"リポジトリツリーの取得に失敗しました: {response.status_code}")

def get_github_file_content(repo_url, file_path):
    """GitHubから特定のファイル内容を取得する"""
    parts = repo_url.rstrip('/').split('/')
    owner, repo = parts[-2], parts[-1]
    
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    response = requests.get(api_url, headers={
        "Authorization": f"Bearer {os.environ.get('GITHUB_ACCESS_TOKEN')}"
    })
    
    if response.status_code == 200:
        import base64
        content = base64.b64decode(response.json()['content']).decode('utf-8')
        return content
    else:
        return f"{file_path}の取得に失敗しました"

def gather_repository_info(repo_url):
    """リポジトリに関する必要な情報をすべて収集する"""
    file_tree = get_github_file_tree(repo_url)
    readme_content = get_github_file_content(repo_url, "README.md")
    
    # 主要パッケージファイルの取得
    package_files = []
    for file_path in ["pyproject.toml", "setup.py", "requirements.txt", "package.json"]:
        try:
            content = get_github_file_content(repo_url, file_path)
            if "取得に失敗しました" not in content:
                package_files.append(f"=== {file_path} ===\n{content}")
        except:
            continue
    
    package_files_content = "\n\n".join(package_files)
    
    return file_tree, readme_content, package_files_content
```

### ステップ4: DSPyの設定と llms.txt ファイルの生成

```python
def generate_llms_txt_for_dspy():
    # DSPy 環境の設定（使用する言語モデルを指定）
    lm = dspy.LM(model="gpt-4o-mini")
    dspy.configure(lm=lm)
    os.environ["OPENAI_API_KEY"] = "<YOUR_OPENAI_API_KEY>"
    
    # 分析器の初期化
    analyzer = RepositoryAnalyzer()
    
    # DSPy リポジトリ情報の取得
    repo_url = "https://github.com/stanfordnlp/dspy"
    file_tree, readme_content, package_files = gather_repository_info(repo_url)
    
    # llms.txt ファイルの生成
    result = analyzer(
        repo_url=repo_url,
        file_tree=file_tree,
        readme_content=readme_content,
        package_files=package_files
    )
    
    return result

# 生成処理の実行
if __name__ == "__main__":
    result = generate_llms_txt_for_dspy()
    
    # 生成された llms.txt ファイルの保存
    with open("llms.txt", "w") as f:
        f.write(result.llms_txt_content)
    
    print("llms.txt ファイルの生成が完了しました！")
    print("\nプレビュー表示:")
    print(result.llms_txt_content[:500] + "...")
```

## 期待される出力形式

DSPy向けに生成される`llms.txt`の形式は以下の通りです：

```
# DSPy：言語モデル向けプログラミングフレームワーク

## プロジェクト概要
DSPyは、言語モデルに対するプロンプティングではなく、プログラミングを行うためのフレームワークである...

## 主要概念
- **モジュール**：言語モデルプログラムの構成要素
- **シグネチャ**：入出力仕様の定義
- **テレプロンプター**：最適化アルゴリズム
- **予測器**：中核的な推論コンポーネント

## アーキテクチャ
- `/dspy/`：メインパッケージディレクトリ
  - `/adapters/`：入出力フォーマット処理モジュール
  - `/clients/`：言語モデルクライアントインターフェース
  - `/predict/`：中核的な予測処理モジュール
  - `/teleprompt/`：最適化アルゴリズム群

## 使用事例
1. **分類器の構築**：DSPyを用いることで、ユーザーはテキストデータを入力として受け取り、事前に定義されたクラスに分類するモジュール式分類器を定義できる。ユーザーは分類ロジックを宣言的に指定可能であり、これにより容易な調整や最適化が可能となる。
2. **RAGパイプラインの実装**：開発者は、クエリに基づいて関連文書を検索した後、それらの文書を用いて一貫性のある応答を生成する検索拡張生成パイプラインを実装できる。DSPyは検索処理と生成処理のコンポーネントをシームレスに統合する機能を提供する。
3. **プロンプト最適化**：ユーザーはDSPyを活用することで、言語モデルのパフォーマンス指標に基づいてプロンプトを自動最適化するシステムを構築できる。これにより、手動介入なしに応答品質を継続的に向上させることが可能となる。
4. **エージェントループの実装**：ユーザーは、ユーザーとの継続的な対話を通じて学習し、フィードバックに基づいて応答を洗練させていくエージェントループを設計できる。これは、DSPyフレームワークの自己改善機能を実証する事例である。
5. **構成的コード**：開発者は、AIシステムの異なるモジュール間で相互作用を可能にする構成的コードを記述できる。これにより、複雑なワークフローを容易に変更・拡張可能な形で実装できる。
```

生成された`llms.txt`ファイルは、DSPyリポジトリの包括的でLLM（大規模言語モデル）に適した概要を提供し、他のAIシステムがコードベースをより効果的に理解・活用する手助けとなります。

## 今後の展開

- 複数リポジトリを分析可能なプログラムへの拡張
- 各種ドキュメントフォーマットへの対応追加
- ドキュメント品質評価のための指標体系の構築
- インタラクティブなリポジトリ分析を実現するWebインターフェースの開発
