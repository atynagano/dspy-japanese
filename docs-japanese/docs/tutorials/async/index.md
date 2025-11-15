# 非同期DSPyプログラミング

DSPyはネイティブに非同期プログラミングをサポートしており、より効率的でスケーラブルなアプリケーションの構築を可能にします。本ガイドでは、DSPyにおける非同期機能の活用方法について、組み込みモジュールとカスタム実装の両方を網羅的に解説します。

## DSPyで非同期処理を利用する理由

DSPyにおける非同期プログラミングには以下のような利点があります：
- 並行処理によるパフォーマンス向上
- リソース利用効率の改善
- I/Oバウンド操作における待機時間の短縮
- 複数リクエストを処理する際のスケーラビリティ向上

## 同期処理と非同期処理の使い分け

DSPyにおいて同期処理と非同期処理のどちらを選択するかは、具体的なユースケースによって異なります。適切な選択を行うためのガイドラインを以下に示します：

同期処理を使用するべき場合：

- 新しいアイデアの探索やプロトタイピング段階にある場合
- 研究や実験を行っている場合
- 小規模から中規模のアプリケーションを開発している場合
- より簡潔で分かりやすいコードが求められる場合
- デバッグやエラー追跡を容易に行いたい場合

非同期処理を使用するべき場合：

- 高スループットサービス（高QPS）を構築する場合
- 非同期操作のみをサポートするツールを使用する場合
- 複数の同時リクエストを効率的に処理する必要がある場合
- 高いスケーラビリティが求められる本番サービスを構築する場合

### 重要な考慮事項

非同期プログラミングはパフォーマンス面での利点がある一方で、いくつかのトレードオフも存在します：

- より複雑なエラー処理とデバッグが必要となる
- 微妙なバグが発生しやすく、追跡が困難になる可能性がある
- コード構造がより複雑化する
- ipython環境（Colab、Jupyter lab、Databricksノートブックなど）と通常のPython実行環境では、コード記述方法が異なる

ほとんどの開発シナリオでは、まずは同期処理から始め、非同期処理の利点が明確に必要な場合にのみ移行することを推奨します。このアプローチにより、非同期プログラミングの追加的な複雑さに対処する前に、アプリケーションの中核ロジックに集中することができます。

## 組み込みモジュールの非同期利用

DSPyのほとんどの組み込みモジュールは、`acall()`メソッドを通じて非同期操作をサポートしています。このメソッドは同期版の`__call__`メソッドと同じインターフェースを維持しつつ、非同期的に動作します。

以下に`dspy.Predict`モジュールを使用した基本的な例を示します：

```python
import dspy
import asyncio
import os

os.environ["OPENAI_API_KEY"] = "your_api_key"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
predict = dspy.Predict("question->answer")

async def main():
    # 非同期処理にはacall()メソッドを使用する
    output = await predict.acall(question="なぜ鶏はキッチンを渡ったのか？")
    print(output)


asyncio.run(main())
```

### 非同期ツールとの連携方法

DSPyの`Tool`クラスは非同期関数とシームレスに連携します。`dspy.Tool`に非同期関数を
渡す場合、`acall()`メソッドを使用して実行することが可能です。これは、I/Oバウンドな処理や
外部サービスとの連携が必要な場合に特に有用です。

```python
import asyncio
import dspy
import os

os.environ["OPENAI_API_KEY"] = "your_api_key"

async def foo(x):
    # 非同期処理を模擬
    await asyncio.sleep(0.1)
    print(f"取得した値: {x}")

# 非同期関数からツールを作成
tool = dspy.Tool(foo)

async def main():
    # ツールを非同期実行
    await tool.acall(x=2)

asyncio.run(main())
```

#### 同期処理環境での非同期ツールの使用

同期処理コードから非同期ツールを呼び出す必要がある場合、非同期処理を同期処理に自動変換する機能を有効にすることができます：

```python
import dspy

async def async_tool(x: int) -> int:
    """数値を2倍にする非同期処理ツール"""
    await asyncio.sleep(0.1)
    return x * 2

tool = dspy.Tool(async_tool)

# オプション1: コンテキストマネージャを使用した一時的な変換
with dspy.context(allow_tool_async_sync_conversion=True):
    result = tool(x=5)  # 同期処理環境でも使用可能
    print(result)  # 10

# オプション2: グローバル設定による恒久的な有効化
dspy.configure(allow_tool_async_sync_conversion=True)
result = tool(x=5)  # 以降すべての環境で使用可能
print(result)  # 10
```

非同期ツールに関する詳細情報については、[ツールのドキュメント](../../learn/programming/tools.md#async-tools)を参照してください。

注意：`dspy.ReAct`をツールと組み合わせて使用する場合、ReActインスタンス上で`acall()`を呼び出すと、
自動的にすべてのツールがそれぞれの`acall()`メソッドを使用して非同期的に実行されます。

## カスタム非同期 DSPy モジュールの作成方法

独自の非同期 DSPy モジュールを作成するには、`forward()`メソッドではなく`aforward()`メソッドを実装してください。
このメソッドには、モジュールの非同期処理ロジックを記述します。以下に、2つの非同期処理を連鎖させるカスタムモジュールの
具体例を示します：

```python
import dspy
import asyncio
import os

os.environ["OPENAI_API_KEY"] = "your_api_key"  # 実際のAPIキーに置き換えてください
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class MyModule(dspy.Module):
    def __init__(self):
        self.predict1 = dspy.ChainOfThought("question->answer")  # 質問→回答の思考連鎖
        self.predict2 = dspy.ChainOfThought("answer->simplified_answer")  # 回答→簡略化回答の思考連鎖

    async def aforward(self, question, **kwargs):
        # 予測処理を逐次的に実行するが非同期で行う
        answer = await self.predict1.acall(question=question)
        return await self.predict2.acall(answer=answer)


async def main():
    mod = MyModule()
    result = await mod.acall(question="Why did a chicken cross the kitchen?")
    print(result)


asyncio.run(main())
```
