# チュートリアル：DSPyにおけるデバッグと可観測性

本ガイドでは、DSPy環境で発生する問題のデバッグ方法と、システムの可観測性を向上させる手法について解説します。現代のAIアプリケーションでは、言語モデル、情報検索システム、ツール群など複数のコンポーネントが連携して動作することが一般的です。DSPyは、これらの複雑なAIシステムをクリーンでモジュール化された方法で構築・最適化することを可能にします。

しかしながら、システムが複雑化するにつれ、**システムの動作内容を理解可能であること**が極めて重要になります。透明性が欠如している場合、予測プロセスはブラックボックス化しやすく、不具合や品質問題の原因特定が困難になるだけでなく、本番環境での保守作業も著しく困難になります。

本チュートリアルを修了する頃には、[MLflow Tracing](#tracing)を用いた問題のデバッグ方法と可観測性向上の手法を習得できます。さらに、コールバック機能を活用したカスタムロギングソリューションの構築方法についても理解を深めることができます。



## プログラムの定義

まずは簡単なReActエージェントを作成し、ColBERTv2のWikipediaデータセットを検索ソースとして使用する例から始めます。この部分をより高度なプログラムに置き換えることも可能です。

```python
import dspy
import os

os.environ["OPENAI_API_KEY"] = "{your_openai_api_key}"

# OpenAI APIキーを環境変数に設定
lm = dspy.LM("openai/gpt-4o-mini")
colbert = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
dspy.configure(lm=lm)


def retrieve(query: str):
    """ColBERTから関連性の高い上位3件の情報を取得"""
    results = colbert(query, k=3)
    return [x["text"] for x in results]


agent = dspy.ReAct("question -> answer", tools=[retrieve], max_iters=3)
```

それでは、エージェントに対して簡単な質問をしてみましょう：

```python
prediction = agent(question="2025年6月時点で、大谷翔平はどの野球チームに所属していますか？")
print(prediction.answer)
```

```
入手可能な情報によれば、大谷翔平選手は2025年6月に北海道日本ハムファイターズでプレーする予定である。
```

これは誤りです。彼は現在北海道日本ハムファイターズでプレーしておらず、ドジャースに移籍して2024年にはワールドシリーズで優勝しています！プログラムのデバッグを行い、可能な修正方法を検討しましょう。

## ``inspect_history`` の使用

DSPyには`inspect_history()`というユーティリティ関数が用意されており、これまでに実行されたすべてのLLM呼び出しを出力します：

```python
# 5回分のLLM呼び出し履歴を表示
dspy.inspect_history(n=5)
```

```
[2024年12月1日 10:23:29.144257]

システムメッセージ:

入力フィールドは以下の通りです:
1. `question` (文字列)

...

応答:

応答:

[[ ## 推論 ## ]]
2025年6月時点における大谷翔平選手の所属チームに関する情報を検索しましたが、具体的な結果は得られませんでした。取得されたデータはいずれも彼が北海道日本ハムファイターズに所属していることを示していましたが、指定された日付時点でのチーム変更や更新に関する情報は一切確認できませんでした。現時点で入手可能な情報に不足があることから、今後新たな展開が生じない限り、彼は依然として北海道日本ハムファイターズに所属していると推測するのが妥当です。

[[ ## 回答 ## ]]
入手可能な情報に基づくと、大谷翔平選手は2025年6月時点においても北海道日本ハムファイターズでプレーを継続しているものと判断されます。

[[ ## 処理完了 ## ]]

```
ログ記録によると、エージェントは検索ツールから有用な情報を取得できなかった。では、実際にリトリーバーはどのような結果を返したのか？ `inspect_history`は有用な機能ではあるものの、以下のような制約が存在する：

* 実際のシステム環境では、リトリーバーやツール、カスタムモジュールなど他のコンポーネントも重要な役割を果たすが、`inspect_history`はLLMへの呼び出し記録のみを対象としている
* DSPyプログラムでは、単一の予測処理中に複数のLLM呼び出しが行われることが多い。モノリシックなログ履歴では、特に複数の質問を扱う場合にログの整理が困難になる
* パラメータ値、レイテンシ、モジュール間の関係性といったメタデータは取得されない

**トレーシング**はこれらの制約を克服し、より包括的なソリューションを提供する。

## トレーシング機能

[MLflow](https://mlflow.org/docs/latest/llms/tracing/index.html)は、エンドツーエンドの機械学習プラットフォームであり、DSPyとシームレスに統合されることでLLMOpsのベストプラクティスをサポートする。DSPy環境でMLflowの自動トレーシング機能を利用するのは簡単で、**サービスへの新規登録やAPIキーの取得は一切不要**である。必要なのはMLflowをインストールし、ノートブックまたはスクリプト内で`mlflow.dspy.autolog()`を呼び出すだけだ。

```bash
pip install -U mlflow>=2.18.0
```

インストール完了後は、以下のコマンドを使用してサーバーを起動してください。

```
# MLflow のトレース機能を使用する場合、SQL ストアの利用を強く推奨します
mlflow server --backend-store-uri sqlite:///mydb.sqlite
```

`--port` フラグで別のポートを指定しない場合、MLflowサーバーはデフォルトでポート5000で起動します。

次に、MLflowのトレーシング機能を有効にするためにコードを修正します。具体的には以下の2点を行います：

- MLflowに対して、サーバーのホスト場所を指定する
- `mlflow.autolog()` を適用し、DSPyのトレーシング情報を自動的に記録させる

完全なコードは以下の通りです。それでは、再度実行してみましょう！

```python
import dspy
import os
import mlflow

os.environ["OPENAI_API_KEY"] = "{your_openai_api_key}"

# MLflowにサーバーURIを通知します。
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# 実験用の一意の名前を設定します。
mlflow.set_experiment("DSPy")

lm = dspy.LM("openai/gpt-4o-mini")
colbert = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
dspy.configure(lm=lm)


def retrieve(query: str):
    """ColBertから関連性の高い上位3件の情報を取得します"""
    results = colbert(query, k=3)
    return [x["text"] for x in results]


agent = dspy.ReAct("question -> answer", tools=[retrieve], max_iters=3)
print(agent(question="大谷翔平が所属する野球チームはどこですか？"))
```


MLflowは、各予測に対して自動的に**トレース**を生成し、実験記録として保存します。これらのトレースを視覚的に確認するには、ブラウザで`http://127.0.0.1:5000/`にアクセスしてください。その後、作成した実験を選択し、[トレース]タブに移動します：

![MLflowトレースUI](./mlflow_trace_ui.png)

最新のトレースをクリックすると、その詳細な内訳を表示できます：

![MLflowトレース表示](./mlflow_trace_view.png)

ここでは、ワークフローの各ステップにおける入力と出力を確認できます。例えば、上記のスクリーンショットでは`retrieve`関数の入力と出力が表示されています。リトリービング機能の出力を確認したところ、情報が古くなっていることが判明しました。これは、2025年6月時点で大谷翔平選手が所属するチームを特定するには不十分な情報です。また、言語モデルの入力・出力・設定パラメータなど、他のステップも同様に確認可能です。

情報の古さという問題に対処するため、`retrieve`関数を[Tavily検索](https://www.tavily.com/)を利用したウェブ検索ツールに置き換えることができます。

```python
from tavily import TavilyClient
import dspy
import mlflow

# MLflowに対してサーバーのURIを通知します。
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# 実験用の一意の名前を設定します。
mlflow.set_experiment("DSPy")

search_client = TavilyClient(api_key="<YOUR_TAVILY_API_KEY>")

def web_search(query: str) -> list[str]:
    """ウェブ検索を実行し、上位5件の検索結果からコンテンツを取得して返します"""
    response = search_client.search(query)
    return [r["content"] for r in response["results"]]

agent = dspy.ReAct("question -> answer", tools=[web_search])

prediction = agent(question="大谷翔平はどの野球チームに所属していますか？")
print(agent.answer)
```

```
ロサンゼルス・ドジャース
```

以下に、MLflow UIの操作手順をGIFアニメーションで示します：

![MLflow Trace UI操作手順](./mlflow_trace_ui_navigation.gif)


MLflowのトレーシング機能の詳細な使用方法については、
[MLflowトレーシングガイド](https://mlflow.org/docs/3.0.0rc0/tracing)
をご参照ください。



!!! info MLflowについてさらに詳しく

    MLflowはエンドツーエンドのLLMOpsプラットフォームであり、実験追跡、評価、デプロイメントなど多岐にわたる機能を提供しています。DSPyとMLflowの統合について詳しく知りたい方は、[こちらのチュートリアル](../deployment/index.md#deploying-with-mlflow)をご覧ください。


## カスタムロギングソリューションの構築

場合によっては、独自のロギングソリューションを実装したいケースもあるでしょう。例えば、特定のモジュールによってトリガーされる特定のイベントをログに記録する必要がある場合などです。DSPyの**コールバック**機構はこのようなユースケースをサポートしています。``BaseCallback``クラスでは、ロギング動作をカスタマイズするための複数のハンドラが用意されています：

|ハンドラ|説明|
|:--|:--|
|`on_module_start` / `on_module_end` | `dspy.Module`サブクラスが呼び出された際にトリガーされます。|
|`on_lm_start` / `on_lm_end` | `dspy.LM`サブクラスが呼び出された際にトリガーされます。|
|`on_adapter_format_start` / `on_adapter_format_end`| `dspy.Adapter`サブクラスが入力プロンプトをフォーマットする際にトリガーされます。|
|`on_adapter_parse_start` / `on_adapter_parse_end`| `dspy.Adapter`サブクラスがLMからの出力テキストを後処理する際にトリガーされます。|
|`on_tool_start` / `on_tool_end` | `dspy.Tool`サブクラスが呼び出された際にトリガーされます。|
|`on_evaluate_start` / `on_evaluate_end` | `dspy.Evaluate`インスタンスが呼び出された際にトリガーされます。|

以下に、ReActエージェントの中間処理ステップをログに記録するカスタムコールバックの具体例を示します：

```python
import dspy
from dspy.utils.callback import BaseCallback

# 1. BaseCallback クラスを継承したカスタムコールバッククラスを定義する
class AgentLoggingCallback(BaseCallback):

    # 2. on_module_end ハンドラを実装し、カスタムログ出力処理を実行する
    def on_module_end(self, call_id, outputs, exception):
        step = "推論" if self._is_reasoning_output(outputs) else "実行"
        print(f"== {step} ステップ ===")
        for k, v in outputs.items():
            print(f"  {k}: {v}")
        print("\n")

    def _is_reasoning_output(self, outputs):
        # 出力辞書のキーに「Thought」で始まる要素が存在する場合に True を返す
        return any(k.startswith("Thought") for k in outputs.keys())

# 3. このコールバックを DSPy の設定に登録し、プログラム実行時に適用されるようにする
dspy.configure(callbacks=[AgentLoggingCallback()])
```


```
== 推論ステップ ===
  Thought_1: 大谷翔平が現在所属しているメジャーリーグベースボールのチームを特定する必要がある。
  Action_1: 検索[大谷翔平 所属チーム 2023年]

== 実行ステップ ===
  passages: ["大谷翔平は..."]

...
```

!!! info コールバック関数における入力/出力データの取り扱いについて

    コールバック関数内で入力データや出力データを扱う際には注意が必要です。これらのデータを直接変更（インプレイス変更）すると、プログラムに渡された元データを意図せず変更してしまう可能性があり、予期しない動作を引き起こす原因となります。このような事態を回避するため、データを変更する可能性のある操作を行う前には、必ずデータのコピーを作成することを強く推奨します。
