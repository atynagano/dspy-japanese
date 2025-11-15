# DSPyキャッシュの利用方法とカスタマイズ

本チュートリアルでは、DSPyのキャッシュ機構の設計原理を解説するとともに、効果的な活用方法とカスタマイズ手法について具体的に説明します。

## DSPyキャッシュの構成

DSPyのキャッシュシステムは、以下の3つの独立した階層構造で構成されています：

1.  **メモリ内キャッシュ**：`cachetools.LRUCache`を用いて実装されており、頻繁にアクセスされるデータへの高速なアクセスを提供します。
2.  **ディスクキャッシュ**：`diskcache.FanoutCache`を活用することで、キャッシュされたデータを永続的に保存します。
3.  **プロンプトキャッシュ（サーバー側キャッシュ）**：この層はLLMサービスプロバイダー（例：OpenAI、Anthropic）によって管理されます。

DSPyはサーバー側のプロンプトキャッシュを直接制御することはできませんが、ユーザーがメモリ内キャッシュとディスクキャッシュの有効/無効を切り替えたり、必要に応じてカスタマイズしたりできる柔軟性を備えています。

## DSPyキャッシュの使用方法

デフォルトでは、DSPyではメモリ内キャッシュとディスクキャッシュの両方が自動的に有効化されています。キャッシュを使用するために特別な操作は必要ありません。キャッシュヒットが発生した場合、モジュール呼び出しの実行時間が大幅に短縮されることが確認できます。さらに、使用状況の追跡が有効になっている場合、キャッシュされた呼び出しの使用統計値は`None`として記録されます。

以下に具体的な使用例を示します：

```python
import dspy
import os
import time

os.environ["OPENAI_API_KEY"] = "{your_openai_key}"  # 実際のOpenAI APIキーに置き換えてください

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), track_usage=True)

predict = dspy.Predict("question->answer")

start = time.time()
result1 = predict(question="Who is the GOAT of basketball?")
print(f"処理時間: {time.time() - start: 2f}秒\n\n総使用量: {result1.get_lm_usage()}")

start = time.time()
result2 = predict(question="Who is the GOAT of basketball?")
print(f"処理時間: {time.time() - start: 2f}秒\n\n総使用量: {result2.get_lm_usage()}")
```

出力結果のサンプルは以下の通りです：

```
経過時間:  4.384113秒
総使用状況: {'openai/gpt-4o-mini': {'completion_tokens': 97, 'prompt_tokens': 144, 'total_tokens': 241, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0, 'text_tokens': None}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0, 'text_tokens': None, 'image_tokens': None}}}

経過時間:  0.000529秒
総使用状況: {}
```

## プロバイダ側プロンプトキャッシュの活用

DSPyに標準搭載されているキャッシュ機構に加え、AnthropicやOpenAIといったLLMプロバイダが提供するプロバイダ側プロンプトキャッシュ機能も利用可能です。この機能は、`dspy.ReAct()`モジュールなどのように類似したプロンプトを繰り返し送信する場合に特に有効で、プロバイダのサーバー側でプロンプトのプレフィックスをキャッシュすることで、レイテンシとコストの両方を削減できます。

プロンプトキャッシュを有効にするには、`dspy.LM()`関数に`cache_control_injection_points`パラメータを指定します。この機能はAnthropicやOpenAIなどの対応プロバイダで利用可能です。本機能の詳細については、[LiteLLMプロンプトキャッシュに関するドキュメント](https://docs.litellm.ai/docs/tutorials/prompt_caching#configuration)を参照してください。

```python
import dspy
import os

os.environ["ANTHROPIC_API_KEY"] = "{your_anthropic_key}"  # 実際のAnthropic APIキーに置き換えてください
lm = dspy.LM(
    "anthropic/claude-sonnet-4-5-20250929",
    cache_control_injection_points=[
        {
            "location": "message",
            "role": "system",
        }
    ],
)
dspy.configure(lm=lm)

# DSPyの任意のモジュールで使用可能
predict = dspy.Predict("question->answer")
result = predict(question="What is the capital of France?")
```

特に以下の場合に有効です：

- `dspy.ReAct()` 関数で同じ指示文を使用する場合
- システムプロンプトが長く、かつその内容が固定されている場合
- 類似したコンテキストで複数回のリクエストを行う場合

## DSPy キャッシュの無効化/有効化

キャッシュを完全に無効化したい場合、あるいはメモリキャッシュとディスクキャッシュを個別に制御したい場合が考えられます。具体的には以下のようなケースです：

- 同一の言語モデル（LM）リクエストに対して異なる応答が必要な場合
- ディスクへの書き込み権限がなく、ディスクキャッシュを無効化する必要がある場合
- メモリリソースが限られているため、メモリキャッシュを無効化したい場合

DSPyではこの目的のために `dspy.configure_cache()` ユーティリティ関数を提供しています。各キャッシュタイプの有効/無効状態は、対応するフラグを使用して制御できます：

```python
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)
```

さらに、インメモリキャッシュとディスクキャッシュの容量管理も可能です：

```python
dspy.configure_cache(
    enable_disk_cache=True,
    enable_memory_cache=True,
    disk_size_limit_bytes=YOUR_DESIRED_VALUE,
    memory_max_entries=YOUR_DESIRED_VALUE,
)
```

`disk_size_limit_bytes`はディスク上のキャッシュの最大サイズをバイト単位で指定するパラメータであるのに対し、`memory_max_entries`はメモリ上のキャッシュに保持可能なエントリの最大数を指定するパラメータです。

## キャッシュの理解とカスタマイズ

特定のユースケースにおいては、キャッシュキーの生成方法をより詳細に制御したい場合など、カスタムキャッシュを実装する必要があるかもしれません。デフォルトでは、キャッシュキーは`litellm`に送信されるすべてのリクエスト引数のハッシュ値から生成され、`api_key`などの認証情報は除外されます。

カスタムキャッシュを作成するには、`dspy.clients.Cache`クラスをサブクラス化し、必要なメソッドをオーバーライドする必要があります：

```python
class CustomCache(dspy.clients.Cache):
    def __init__(self, **kwargs):
        {ここに独自のコンストラクタを実装してください}

    def cache_key(self, request: dict[str, Any], ignored_args_for_cache_key: Optional[list[str]] = None) -> str:
        {キャッシュキーを計算するロジックを実装してください}

    def get(self, request: dict[str, Any], ignored_args_for_cache_key: Optional[list[str]] = None) -> Any:
        {キャッシュからデータを読み込むロジックを実装してください}

    def put(
        self,
        request: dict[str, Any],
        value: Any,
        ignored_args_for_cache_key: Optional[list[str]] = None,
        enable_memory_cache: bool = True,
    ) -> None:
        {キャッシュにデータを書き込むロジックを実装してください}
```

DSPyの他の機能とシームレスに連携させるためには、カスタムキャッシュクラスを実装する際に、基本クラスと同じメソッドシグネチャを使用するか、少なくともメソッド定義に`**kwargs`を含めることをお勧めします。これにより、キャッシュの読み書き操作時に実行時エラーが発生するのを防止できます。

カスタムキャッシュクラスを定義した後は、DSPyに対してこのクラスを使用するよう指示できます：

```python
dspy.cache = CustomCache()
```

具体的な例を用いて説明しよう。例えば、キャッシュキーの計算を、呼び出される特定の言語モデル（LM）などの他のパラメータを考慮せず、リクエストメッセージの内容のみに依存させたい場合がある。このような場合、以下のようにカスタムキャッシュを作成することが可能である：

```python
class CustomCache(dspy.clients.Cache):

    def cache_key(self, request: dict[str, Any], ignored_args_for_cache_key: Optional[list[str]] = None) -> str:
        messages = request.get("messages", [])
        # SHA-256ハッシュを生成し、ソート済みのキー順でJSON文字列をエンコードした後、その16進数表現を返す
        return sha256(ujson.dumps(messages, sort_keys=True).encode()).hexdigest()

dspy.cache = CustomCache(enable_disk_cache=True, enable_memory_cache=True, disk_cache_dir=dspy.clients.DISK_CACHE_DIR)
```

比較のために、以下のコードをカスタムキャッシュを使用しない状態で実行することを検討してください：

```python
import dspy
import os
import time

os.environ["OPENAI_API_KEY"] = "{your_openai_key}"  # 実際のOpenAI APIキーに置き換えてください

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

predict = dspy.Predict("question->answer")

start = time.time()
result1 = predict(question="Who is the GOAT of soccer?")
print(f"処理時間: {time.time() - start: 2f}秒")

start = time.time()
with dspy.context(lm=dspy.LM("openai/gpt-4.1-mini")):
    result2 = predict(question="Who is the GOAT of soccer?")
print(f"処理時間: {time.time() - start: 2f}秒")
```

経過時間から、2回目の呼び出し時にはキャッシュがヒットしなかったことが分かります。ただし、カスタムキャッシュを使用している場合：

```python
import dspy
import os
import time
from typing import Dict, Any, Optional
import ujson
from hashlib import sha256

os.environ["OPENAI_API_KEY"] = "{your_openai_key}"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class CustomCache(dspy.clients.Cache):

    def cache_key(self, request: dict[str, Any], ignored_args_for_cache_key: Optional[list[str]] = None) -> str:
        messages = request.get("messages", [])
        return sha256(ujson.dumps(messages, sort_keys=True).encode()).hexdigest()

dspy.cache = CustomCache(enable_disk_cache=True, enable_memory_cache=True, disk_cache_dir=dspy.clients.DISK_CACHE_DIR)

predict = dspy.Predict("question->answer")

start = time.time()
result1 = predict(question="Who is the GOAT of volleyball?")
print(f"処理時間: {time.time() - start: 2f}秒")

start = time.time()
with dspy.context(lm=dspy.LM("openai/gpt-4.1-mini")):
    result2 = predict(question="Who is the GOAT of volleyball?")
print(f"処理時間: {time.time() - start: 2f}秒")
```

2回目の呼び出し時にキャッシュヒットが発生することを確認できるでしょう。これは、カスタムキャッシュキーロジックの効果が発揮されている証拠です。