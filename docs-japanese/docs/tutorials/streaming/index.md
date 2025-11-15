# ストリーミング機能の実装

本ガイドでは、DSPyプログラムにストリーミング機能を実装する方法について解説します。DSPyのストリーミング機能は大きく分けて以下の2つの要素で構成されています：

- **トークン単位のストリーミング出力**：完全なレスポンスを待つことなく、生成されたトークンを逐次出力する機能
- **中間状態のストリーミング表示**：プログラムの実行状況をリアルタイムで表示する機能（例：「ウェブ検索を実行中...」「検索結果を処理中...」）

## トークン単位のストリーミング出力

DSPyのトークンストリーミング機能は、パイプラインの最終出力モジュールに限らず、あらゆるモジュールで利用可能です。唯一の要件は、ストリーミング対象のフィールドが`str`型であることです。トークンストリーミングを有効にするには、以下の手順を実施してください：

1. `dspy.streamify`デコレータでプログラムをラップする
2. ストリーミング対象のフィールドを指定するため、`dspy.streaming.StreamListener`オブジェクトを1つ以上作成する

以下に基本的な実装例を示します：

```python
import os

import dspy

os.environ["OPENAI_API_KEY"] = "your_api_key"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

predict = dspy.Predict("question->answer")

# 'answer'フィールドのストリーミング機能を有効にする
stream_predict = dspy.streamify(
    predict,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
)
```

ストリーミング出力を消費するには：

```python
import asyncio

async def read_output_stream():
    output_stream = stream_predict(question="なぜニワトリはキッチンを渡ったのか？")

    async for chunk in output_stream:
        print(chunk)

asyncio.run(read_output_stream())
```

この設定により、以下のような出力が得られます：

```
StreamResponse(predict_name='self', signature_field_name='answer', chunk='To')
StreamResponse(predict_name='self', signature_field_name='answer', chunk=' get')
StreamResponse(predict_name='self', signature_field_name='answer', chunk=' to')
StreamResponse(predict_name='self', signature_field_name='answer', chunk=' the')
StreamResponse(predict_name='self', signature_field_name='answer', chunk=' other')
StreamResponse(predict_name='self', signature_field_name='answer', chunk=' side of the frying pan!')
Prediction(
    answer='To get to the other side of the frying pan!'
)
```

注意：`dspy.streamify`は非同期ジェネレータを返すため、非同期コンテキスト内で使用する必要があります（JupyterやGoogle Colabなど、既にイベントループを備えた環境を使用している場合、直接ジェネレータを使用できます）。

上記のストリーミング処理では、2つの異なるエンティティが扱われていることにお気づきでしょう：`StreamResponse`と`Prediction`です。`StreamResponse`は、監視対象フィールドのストリーミングトークンをラップするオブジェクトで、この例では`answer`フィールドに対応します。一方、`Prediction`はプログラムの最終的な出力結果を表します。DSPyにおけるストリーミング処理はサイドカー方式で実装されており、言語モデル（LM）側でトークンのストリーミング出力を有効にします。これらのトークンは専用のサイドチャネルを通じて送信され、ユーザー定義のリスナーによって継続的に読み取られます。リスナーはストリームを逐次解釈し、監視対象の`signature_field_name`が出現を開始し、完了したかどうかを判断します。フィールドの出現が確認された時点で、リスナーは非同期ジェネレータにトークンの出力を開始します。このジェネレータはユーザーが読み取ることが可能です。リスナーの内部処理機構は背後で動作するアダプターに応じて変化し、通常、次のフィールドが現れるまでフィールドの完了を確定できないため、リスナーは最終的なジェネレータに送信する前に出力トークンをバッファリングします。このため、`StreamResponse`型の最終チャンクに複数のトークンが含まれることがよくあります。プログラムの出力自体もこのストリームに書き込まれ、前述のサンプル出力における`Prediction`型のチャンクとして表示されます。

これらの異なるタイプを処理し、カスタムロジックを実装するには：

```python
import asyncio

async def read_output_stream():
  output_stream = stream_predict(question="なぜニワトリはキッチンを渡ったのか？")

  async for chunk in output_stream:
    return_value = None
    if isinstance(chunk, dspy.streaming.StreamResponse):
      print(f"フィールド {chunk.signature_field_name} の出力トークン: {chunk.chunk}")
    elif isinstance(chunk, dspy.Prediction):
      return_value = chunk


program_output = asyncio.run(read_output_stream())
print("最終出力: ", program_output)
```

### `StreamResponse` の理解

`StreamResponse`（`dspy.streaming.StreamResponse`）は、ストリーミングトークンを処理するためのラッパークラスです。このクラスには以下の3つの主要なフィールドがあります：

- `predict_name`: `signature_field_name`を保持する予測の名称。この名称は、`your_program.named_predictors()`を実行した際に得られるキー名と同一です。上記のコード例では、`answer`が`predict`オブジェクトから直接取得されるため、`predict_name`は`self`として表示されます。これは`predict.named_predictors()`を実行した際に得られる唯一のキーです。
- `signature_field_name`: これらのトークンが対応する出力フィールド名。`predict_name`と`signature_field_name`を組み合わせることで、当該フィールドの一意の識別子が形成されます。本ガイドの後半では、複数フィールドのストリーミング処理や重複フィールド名の扱い方について詳しく解説します。
- `chunk`: ストリーミングデータのチャンク値。

### キャッシュを利用したストリーミング処理

キャッシュされた結果が存在する場合、ストリーム処理は個々のトークンをスキップし、最終的な`Prediction`オブジェクトのみを返します。例えば：

```
Prediction(
    answer='ディナープレートの向こう側に移動するためです！'
)
```

### 複数フィールドのストリーミング処理

各フィールドに対して個別に`StreamListener`を作成することで、複数のフィールドを監視することが可能です。以下にマルチモジュールプログラムの具体例を示します：

```python
import asyncio

import dspy

# キャッシュを無効にした OpenAI GPT-4o-mini モデルを初期化
lm = dspy.LM("openai/gpt-4o-mini", cache=False)
dspy.configure(lm=lm)


class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()

        # 質問→回答の予測モデルを定義
        self.predict1 = dspy.Predict("question->answer")
        # 回答→簡略化回答の予測モデルを定義
        self.predict2 = dspy.Predict("answer->simplified_answer")

    def forward(self, question: str, **kwargs):
        # 質問に対する回答を生成
        answer = self.predict1(question=question)
        # 生成された回答を簡略化
        simplified_answer = self.predict2(answer=answer)
        # 最終的に簡略化された回答を返す
        return simplified_answer


# モジュールのインスタンスを作成
predict = MyModule()
# ストリーム処理用のリスナーを設定
stream_listeners = [
    dspy.streaming.StreamListener(signature_field_name="answer"),
    dspy.streaming.StreamListener(signature_field_name="simplified_answer"),
]
# ストリーム処理用に予測関数を設定
stream_predict = dspy.streamify(
    predict,
    stream_listeners=stream_listeners,
)

async def read_output_stream():
    # 質問「なぜ鶏はキッチンを渡ったのか？」に対する出力を取得
    output = stream_predict(question="why did a chicken cross the kitchen?")

    # 戻り値を初期化
    return_value = None
    # 出力ストリームを順次処理
    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk)
        elif isinstance(chunk, dspy.Prediction):
            return_value = chunk
    # 最終的な処理結果を返す
    return return_value

# 非同期処理を実行して出力を取得
program_output = asyncio.run(read_output_stream())
print("最終出力結果: ", program_output)
```

出力結果は以下のようになります。

```
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk='To')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' get')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' to')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' the')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' other side of the recipe!')
StreamResponse(predict_name='predict2', signature_field_name='simplified_answer', chunk='To')
StreamResponse(predict_name='predict2', signature_field_name='simplified_answer', chunk=' reach')
StreamResponse(predict_name='predict2', signature_field_name='simplified_answer', chunk=' the')
StreamResponse(predict_name='predict2', signature_field_name='simplified_answer', chunk=' other side of the recipe!')
最終出力:  Prediction(
    simplified_answer='To reach the other side of the recipe!'
)
```

### 同一フィールドの複数回ストリーミング処理（dspy.ReAct と同様の動作）

デフォルトでは、`StreamListener` は単一のストリーミングセッションが完了すると自動的に終了します。
この設計思想は、パフォーマンス問題の発生を防止することを目的としています。すべてのトークンは設定されたすべてのストリームリスナーにブロードキャストされるため、
アクティブなリスナーが多すぎると重大なオーバーヘッドが生じる可能性があるためです。

ただし、`dspy.ReAct` のように DSPy モジュールをループ内で繰り返し使用する場合、各予測ごとに同一のフィールドを毎回ストリーミングしたいケースがあります。
このような動作を実現するには、`StreamListener` 作成時に allow_reuse=True を設定してください。以下に具体例を示します：

```python
import asyncio

import dspy

# GPT-4o-mini モデルをキャッシュ無効で初期化
lm = dspy.LM("openai/gpt-4o-mini", cache=False)
dspy.configure(lm=lm)


def fetch_user_info(user_name: str):
    """ユーザーの氏名や生年月日などの情報を取得する"""
    return {
        "name": user_name,
        "birthday": "2009-05-16",
    }


def get_sports_news(year: int):
    """指定された年のスポーツニュースを取得する"""
    if year == 2009:
        return "ウサイン・ボルトが100m走で世界記録を更新した"
    return None


# ReAct エージェントを設定し、必要なツールを指定
react = dspy.ReAct("question->answer", tools=[fetch_user_info, get_sports_news])

# ストリーミング処理用のリスナーを設定
# ReAct にはデフォルトで「next_thought」という出力フィールドが用意されています
stream_listeners = [
    dspy.streaming.StreamListener(signature_field_name="next_thought", allow_reuse=True),
]
stream_react = dspy.streamify(react, stream_listeners=stream_listeners)


async def read_output_stream():
    # 質問に対する出力ストリームを取得
    output = stream_react(question="Adamが生まれた年に起こったスポーツニュースは何ですか？")
    return_value = None
    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk)
        elif isinstance(chunk, dspy.Prediction):
            return_value = chunk
    return return_value


# 非同期処理を実行して結果を表示
print(asyncio.run(read_output_stream()))
```

この例では、StreamListenerで`allow_reuse=True`を設定することで、"next_thought"フィールドのストリーミング処理が初回だけでなく全ての反復処理で利用可能になります。このコードを実行すると、`next_thought`フィールドが生成されるたびにストリーミングトークンが出力されるのが確認できるでしょう。

#### 重複するフィールド名の処理方法

異なるモジュールから同名のフィールドをストリーミングする場合、`StreamListener`内で`predict`と`predict_name`の両方を指定する必要があります：

```python
import asyncio

import dspy

# キャッシュを無効にした OpenAI GPT-4o-mini モデルを初期化
lm = dspy.LM("openai/gpt-4o-mini", cache=False)
dspy.configure(lm=lm)


class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()

        # 質問→回答の予測モデルを定義
        self.predict1 = dspy.Predict("question->answer")
        # 回答→簡略化回答・スコアの予測モデルを定義
        self.predict2 = dspy.Predict("question, answer->answer, score")

    def forward(self, question: str, **kwargs):
        # 第一段階の予測を実行
        answer = self.predict1(question=question)
        # 第二段階の予測を実行して簡略化回答を取得
        simplified_answer = self.predict2(answer=answer)
        return simplified_answer


# モジュールインスタンスを作成
predict = MyModule()
# ストリーム処理用リスナーを設定
stream_listeners = [
    dspy.streaming.StreamListener(
        signature_field_name="answer",
        predict=predict.predict1,
        predict_name="predict1"
    ),
    dspy.streaming.StreamListener(
        signature_field_name="answer",
        predict=predict.predict2,
        predict_name="predict2"
    ),
]
# ストリーム処理機能を適用
stream_predict = dspy.streamify(
    predict,
    stream_listeners=stream_listeners,
)


async def read_output_stream():
    # 質問「なぜ鶏はキッチンを渡ったのか？」に対する出力を取得
    output = stream_predict(question="why did a chicken cross the kitchen?")

    return_value = None
    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk)
        elif isinstance(chunk, dspy.Prediction):
            return_value = chunk
    return return_value


# 非同期処理を実行して最終出力を取得
program_output = asyncio.run(read_output_stream())
print("最終出力: ", program_output)
```

出力結果は以下のようになります。

```
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk='To')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' get')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' to')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' the')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' other side of the recipe!')
StreamResponse(predict_name='predict2', signature_field_name='answer', chunk="I'm")
StreamResponse(predict_name='predict2', signature_field_name='answer', chunk=' ready')
StreamResponse(predict_name='predict2', signature_field_name='answer', chunk=' to')
StreamResponse(predict_name='predict2', signature_field_name='answer', chunk=' assist')
StreamResponse(predict_name='predict2', signature_field_name='answer', chunk=' you')
StreamResponse(predict_name='predict2', signature_field_name='answer', chunk='! Please provide a question.')
最終出力:  Prediction(
    answer="I'm ready to assist you! Please provide a question.",
    score='N/A'
)
```

## 中間ステータスストリーミング

ステータスストリーミング機能は、プログラムの実行状況をユーザーに随時通知する機能であり、ツール呼び出しや複雑なAIパイプラインといった長時間実行される処理において特に有用です。ステータスストリーミングを実装する手順は以下の通りです：

1. `dspy.streaming.StatusMessageProvider` クラスを継承したカスタムステータスメッセージプロバイダを作成する
2. 必要なフックメソッドをオーバーライドし、独自のステータスメッセージを提供する
3. 作成したプロバイダオブジェクトを `dspy.streamify` 関数に渡す

具体例：

```python
class MyStatusMessageProvider(dspy.streaming.StatusMessageProvider):
    def lm_start_status_message(self, instance, inputs):
        return f"言語モデルに入力 {inputs} を送信しています..."

    def lm_end_status_message(self, outputs):
        return f"処理が完了しました。出力結果: {outputs}!"
```

利用可能なフック機能：

- lm_start_status_message：dspy.LM を呼び出す開始時に表示されるステータスメッセージ
- lm_end_status_message：dspy.LM を呼び出す終了時に表示されるステータスメッセージ
- module_start_status_message：dspy.Module を呼び出す開始時に表示されるステータスメッセージ
- module_end_status_message：dspy.Module を呼び出す終了時に表示されるステータスメッセージ
- tool_start_status_message：dspy.Tool を呼び出す開始時に表示されるステータスメッセージ
- tool_end_status_message：dspy.Tool を呼び出す終了時に表示されるステータスメッセージ

各フック機能は、ステータスメッセージを含む文字列を返す必要があります。

メッセージプロバイダを作成した後は、単に `dspy.streamify` に渡すだけで、以下の機能を有効にすることができます：
- ステータスメッセージのストリーミング
- 出力トークンのストリーミング

以下に使用例を示します。中間ステータスメッセージは `dspy.streaming.StatusMessage` クラスで表現されるため、これを取得するためには追加の条件分岐が必要です。

```python
import asyncio

import dspy

# OpenAIのgpt-4o-miniモデルをロードし、キャッシュを無効に設定
lm = dspy.LM("openai/gpt-4o-mini", cache=False)
dspy.configure(lm=lm)


class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()

        # 入力値を2倍にするツールモジュールを定義
        self.tool = dspy.Tool(lambda x: 2 * x, name="double_the_number")
        # 思考過程を記録するChain of Thoughtモジュールを定義
        self.predict = dspy.ChainOfThought("num1, num2->sum")

    def forward(self, num, **kwargs):
        # ツールモジュールを用いて入力値を処理
        num2 = self.tool(x=num)
        # 処理結果をChain of Thoughtモジュールに入力
        return self.predict(num1=num, num2=num2)


class MyStatusMessageProvider(dspy.streaming.StatusMessageProvider):
    def tool_start_status_message(self, instance, inputs):
        # ツール実行開始時のステータスメッセージを生成
        return f"{instance.name}ツールを呼び出し中... 入力値: {inputs}"

    def tool_end_status_message(self, outputs):
        # ツール実行終了時のステータスメッセージを生成
        return f"ツール処理完了: 出力結果: {outputs}!"


# メインモジュールのインスタンスを作成
predict = MyModule()
# 処理結果を監視するためのストリームリスナーを設定
stream_listeners = [
    # Chain of Thoughtモジュールの内部出力フィールド「reasoning」を監視対象に指定
    dspy.streaming.StreamListener(signature_field_name="reasoning"),
]
# モジュールをストリーム処理用に設定
stream_predict = dspy.streamify(
    predict,
    stream_listeners=stream_listeners,
    status_message_provider=MyStatusMessageProvider(),
)


async def read_output_stream():
    # ストリーム処理を実行し、入力値3に対する予測結果を取得
    output = stream_predict(num=3)

    return_value = None
    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk)
        elif isinstance(chunk, dspy.Prediction):
            return_value = chunk
        elif isinstance(chunk, dspy.streaming.StatusMessage):
            print(chunk)
    return return_value


# 非同期処理を実行し、最終的な出力を取得
program_output = asyncio.run(read_output_stream())
print("最終出力結果: ", program_output)
```

出力サンプル：

```
StatusMessage(message='ツール double_the_number を起動中...')
StatusMessage(message='ツール呼び出し処理完了！LLMに呼び出し結果をクエリしています...')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk='To')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' find')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' the')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' sum')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' of')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' the')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' two')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' numbers')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=',')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' we')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' simply')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' add')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' them')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' together')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk='.')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' 以下が')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' ')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk='3')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' に')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' 6')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' を')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' 加算')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' した')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' 結果')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=': 9')
最終出力:  Prediction(
    reasoning='2つの数値の合計を求めるには、単純にそれらを加算します。具体的には、3に6を加えると9になります',
    sum='9'
)
```

## 同期ストリーミング処理

デフォルトでは、ストリーム化済みのDSPyプログラムを呼び出すと非同期ジェネレータが生成されます。同期ジェネレータを取得したい場合は、`async_streaming=False` というフラグを設定します：


```python
import os

import dspy

os.environ["OPENAI_API_KEY"] = "your_api_key"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

predict = dspy.Predict("question->answer")

# 'answer'フィールドのストリーミング機能を有効にする
stream_predict = dspy.streamify(
    predict,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
    async_streaming=False,
)

output = stream_predict(question="なぜ鶏はキッチンを渡ったのか？")

program_output = None
for chunk in output:
    if isinstance(chunk, dspy.streaming.StreamResponse):
        print(chunk)
    elif isinstance(chunk, dspy.Prediction):
        program_output = chunk
print(f"プログラム出力結果: {program_output}")
```
