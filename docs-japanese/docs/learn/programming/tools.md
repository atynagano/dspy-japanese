---
sidebar_position: 2
---

# ツール機能

DSPyは、外部関数・API・サービスと連携可能な**ツール利用エージェント**に対して強力なサポートを提供します。ツール機能を活用することで、言語モデルは単なるテキスト生成を超え、動的なアクション実行、情報取得、データ処理といった高度なタスクを実行できるようになります。

DSPyでツールを利用する主な方法は以下の2つです：

1. **`dspy.ReAct`** 方式 - 推論とツール呼び出しを完全自動管理するツールエージェント
2. **手動ツール操作** - `dspy.Tool`、`dspy.ToolCalls`、およびカスタム署名を用いたツール呼び出しの直接制御

## アプローチ1：`dspy.ReAct`の利用（完全自動管理型）

`dspy.ReAct`モジュールは、Reasoning and Acting（ReAct）パターンを実装しており、言語モデルが現在の状況について反復的に推論を行い、どのツールを呼び出すかを自動的に決定します。

### 基本例

```python
import dspy

# ツールを関数として定義する
def get_weather(city: str) -> str:
    """指定した都市の現在の気象情報を取得する"""
    # 実際の実装では気象APIを呼び出す処理となる
    return f"{city}の天気は晴れ、気温75°Fです"

def search_web(query: str) -> str:
    """ウェブ検索を実行して情報を取得する"""
    # 実際の実装では検索APIを呼び出す処理となる
    return f"「{query}」に関する検索結果: [関連性の高い情報...]"

# ReActエージェントを作成する
react_agent = dspy.ReAct(
    signature="question -> answer",
    tools=[get_weather, search_web],
    max_iters=5
)

# エージェントを使用する
result = react_agent(question="東京の天気はどうなっていますか?")
print(result.answer)
print("ツールの呼び出し履歴:", result.trajectory)
```

### ReActの主要機能

- **自動推論機能**：問題を段階的に論理的に解決する
- **ツール選択機能**：状況に応じて最適なツールを自動的に選択する
- **反復実行機能**：複数のツール呼び出しを可能とし、必要な情報を段階的に収集できる
- **エラー処理機能**：ツール呼び出しが失敗した場合の自動リカバリー機構を備えている
- **推論履歴追跡機能**：推論過程とツール呼び出しの完全な履歴を保持する

### ReActのパラメータ設定

```python
react_agent = dspy.ReAct(
    signature="question -> answer",  # 入力/出力仕様
    tools=[tool1, tool2, tool3],     # 利用可能なツールのリスト
    max_iters=10                     # ツール呼び出しの最大反復回数
)
```

## アプローチ2：手動ツール操作

ツール呼び出しプロセスをより詳細に制御する必要がある場合、DSPyのツール型機能を使用して手動でツールを操作することが可能です。

!!! note "バージョン要件"
    以下の例で使用している`ToolCall.execute()`メソッドは、**dspy 3.0.4b2**以降のバージョンで利用可能です。バージョン3.0.3以前をご利用の場合は、この機能を使用するためにアップグレードが必要です。

### 基本的な設定

```python
import dspy

class ToolSignature(dspy.Signature):
    """手動ツール操作用のシグネチャ定義"""
    question: str = dspy.InputField()  # ユーザー入力用フィールド
    tools: list[dspy.Tool] = dspy.InputField()  # 使用するツールのリスト
    outputs: dspy.ToolCalls = dspy.OutputField()  # ツール実行結果の出力フィールド

def weather(city: str) -> str:
    """指定された都市の天気予報を取得する"""
    return f"{city}の天気は晴れです"

def calculator(expression: str) -> str:
    """数学式を評価する"""
    try:
        result = eval(expression)  # ※本番環境では安全な評価方法を採用してください
        return f"結果は {result} です"
    except:
        return "無効な式です"

# ツールインスタンスの作成
tools = {
    "weather": dspy.Tool(weather),
    "calculator": dspy.Tool(calculator)
}

# 予測器の初期化
predictor = dspy.Predict(ToolSignature)

# 予測処理の実行
response = predictor(
    question="ニューヨークの天気はどうなっていますか？",
    tools=list(tools.values())
)

# ツール呼び出しの実行
for call in response.outputs.tool_calls:
    # ツール呼び出しを実行
    result = call.execute()
    # ※バージョン 3.0.4b2 以前の場合は、以下の方法を使用してください：result = tools[call.name](**call.args)
    print(f"ツール名: {call.name}")
    print(f"引数: {call.args}")
    print(f"結果: {result}")
```

### `dspy.Tool` クラスの理解

`dspy.Tool` クラスは、通常の Python 関数を DSPy のツールシステムと互換性を持たせるためにラップする機能を提供します：

```python
def my_function(param1: str, param2: int = 5) -> str:
    """パラメータを持つサンプル関数"""
    return f"{param1} を値 {param2} で処理しました"

# ツールの作成
tool = dspy.Tool(my_function)

# ツールのプロパティ
print(tool.name)        # "my_function"
print(tool.desc)        # 関数のドキュメント文字列
print(tool.args)        # パラメータスキーマ
print(str(tool))        # ツールの詳細情報
```

### `dspy.ToolCalls` の理解

!!! note "バージョン要件"
    `ToolCall.execute()` メソッドは **dspy 3.0.4b2** 以降のバージョンで利用可能です。これより前のバージョンを使用している場合、この機能を利用するにはアップグレードが必要です。

`dspy.ToolCalls` 型は、ツール呼び出し機能を備えたモデルからの出力結果を表します。個々のツール呼び出しは、`execute` メソッドを使用して実行可能です：

```python
# ツール呼び出しを含むレスポンスを取得した後
for call in response.outputs.tool_calls:
    print(f"ツール名: {call.name}")
    print(f"引数: {call.args}")
    
    # 各ツール呼び出しを異なるオプションで実行:
    
    # オプション1: 自動発見機能を使用（ローカル/グローバルスコープ内の関数を検索）
    result = call.execute()  # 関数名に基づいて自動的に関数を検索

    # オプション2: 辞書形式でツールを指定（最も明示的な方法）
    result = call.execute(functions={"weather": weather, "calculator": calculator})
    
    # オプション3: Toolオブジェクトのリストとしてツールを指定
    result = call.execute(functions=[dspy.Tool(weather), dspy.Tool(calculator)])
    
    # オプション4: バージョン3.0.4b2以前の場合（手動でツールを検索）
    # tools_dict = {"weather": weather, "calculator": calculator}
    # result = tools_dict[call.name](**call.args)
    
    print(f"結果: {result}")
```

## ネイティブ関数呼び出し機能の活用

DSPyアダプタでは、**ネイティブ関数呼び出し**機能をサポートしています。これは、テキストベースの構文解析に依存するのではなく、基盤となる言語モデルが本来備えている関数呼び出し機能を活用するものです。このアプローチにより、より信頼性の高いツール実行が可能となり、ネイティブ関数呼び出しをサポートするモデルとの統合性も向上します。

!!! warning "ネイティブ関数呼び出しが必ずしも品質向上を保証するものではない"

    ネイティブ関数呼び出しでは、カスタム関数呼び出しに比べて品質が低下する可能性がある点にご注意ください。

### アダプタの動作仕様

DSPyの各アダプタでは、ネイティブ関数呼び出しに関するデフォルト設定が異なります：

- **`ChatAdapter`** - デフォルトでは `use_native_function_calling=False` が設定されています（テキスト解析方式を採用）
- **`JSONAdapter`** - デフォルトでは `use_native_function_calling=True` が設定されています（ネイティブ関数呼び出しを使用）

アダプタを作成する際に `use_native_function_calling` パラメータを明示的に設定することで、これらのデフォルト値を上書きすることが可能です。

### 設定オプション

```python
import dspy

# ネイティブ関数呼び出しを有効にした ChatAdapter
chat_adapter_native = dspy.ChatAdapter(use_native_function_calling=True)

# ネイティブ関数呼び出しを無効にした JSONAdapter
json_adapter_manual = dspy.JSONAdapter(use_native_function_calling=False)

# DSPy の設定で指定のアダプタを使用
dspy.configure(lm=dspy.LM(model="openai/gpt-4o"), adapter=chat_adapter_native)
```

[MLflow tracing](https://dspy.ai/tutorials/observability/)を有効にすることで、ネイティブツールの呼び出し方法を確認できます。[上記セクション](tools.md#basic-setup)で提供されているコードスニペットで`JSONAdapter`または`ChatAdapter`を使用し、ネイティブ関数呼び出しが有効になっている場合、ネイティブ関数呼び出し用引数`tools`が以下のように設定されているはずです：

![ネイティブツール呼び出しの例](../figures/native_tool_call.png)


### モデル互換性について

ネイティブ関数呼び出し機能は、`litellm.supports_function_calling()`を使用してモデルの対応状況を自動的に検出します。もしモデルがネイティブ関数呼び出しをサポートしていない場合、DSPyでは`use_native_function_calling=True`が設定されている場合でも、従来のテキストベースの解析方式にフォールバックします。

## 非同期ツールの使用

DSPyのツールは、同期処理と非同期処理の両方をサポートしています。非同期ツールを使用する場合、以下の2つの選択肢があります：

### 非同期ツールで`acall`を使用する方法

非同期ツールを扱う際には、`acall`を使用することが推奨される方法です：

```python
import asyncio
import dspy

async def async_weather(city: str) -> str:
    """非同期的に気象情報を取得する関数"""
    await asyncio.sleep(0.1)  # 非同期API呼び出しを模擬
    return f"{city} の天気は晴れです"

tool = dspy.Tool(async_weather)

# 非同期ツールの場合はacallメソッドを使用する
result = await tool.acall(city="ニューヨーク")
print(result)
```

### 非同期ツールを同期モードで実行する場合

同期処理コードから非同期ツールを呼び出す必要がある場合、`allow_tool_async_sync_conversion` 設定を有効にすることで自動変換機能を利用できます：

```python
import asyncio
import dspy

async def async_weather(city: str) -> str:
    """非同期で気象情報を取得する関数"""
    await asyncio.sleep(0.1)
    return f"{city} の天気は晴れです"

tool = dspy.Tool(async_weather)

# 非同期-同期変換機能を有効にする
with dspy.context(allow_tool_async_sync_conversion=True):
    # これで非同期ツールに対しても __call__ メソッドを使用可能になる
    result = tool(city="ニューヨーク")
    print(result)
```

## ベストプラクティス

### 1. ツール機能の設計方針

- **明確なドキュメントストリング**: ツールは詳細なドキュメントが付与されている場合に効果的に機能する
- **型ヒントの活用**: パラメータと戻り値の型を明確に指定する
- **シンプルなパラメータ設計**: 基本的な型（str、int、bool、dict、list）またはPydanticモデルを使用する

```python
def good_tool(city: str, units: str = "celsius") -> str:
    """
    特定の都市の気象情報を取得する関数です。

    引数:
        city: 気象情報を取得する都市名
        units: 温度単位 ('celsius' または 'fahrenheit')

    戻り値:
        現在の気象状況を説明する文字列
    """
    # 適切なエラー処理を含む実装
    if not city.strip():
        return "エラー: 都市名は空にできません"

    # 気象情報の処理ロジックをここに記述...
    return f"{city} の気象状況: {25}{units[0].upper()}度、晴天"
```

### 2. ReActと手動ツール処理の選択基準

**`dspy.ReAct`を使用するべき場合：**

- 自動的な推論とツール選択が必要な場合
- 複数のツール呼び出しが必要なタスクを実行する場合
- 組み込みのエラー回復機能が必要な場合
- ツールのオーケストレーションよりもツール実装自体に注力したい場合

**手動ツール処理を使用するべき場合：**

- ツール実行プロセスに対して精密な制御が必要な場合
- 独自のエラー処理ロジックを実装したい場合
- 遅延を最小限に抑えたい場合
- ツールが何も返さない（void関数）場合


DSPyのツール機能は、言語モデルの能力をテキスト生成の枠を超えて拡張する強力な手段を提供します。完全に自動化されたReActアプローチを採用する場合でも、手動でツールを処理する場合でも、コードを通じて世界と相互作用する高度なエージェントを構築することが可能です。
