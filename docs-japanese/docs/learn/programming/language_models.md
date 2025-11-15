---
sidebar_position: 2
---

# 言語モデル

DSPyコードを実装する際の最初のステップは、使用する言語モデルを設定することです。例えば、OpenAIのGPT-4o-miniをデフォルトの言語モデルとして以下のように設定できます。

```python linenums="1"
# `OPENAI_API_KEY` 環境変数を使用して認証: import os; os.environ['OPENAI_API_KEY'] = 'ここにAPIキーを記述'
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)
```

!!! info "各種言語モデルの利用方法"

    === "OpenAI"
        環境変数 `OPENAI_API_KEY` を設定するか、以下のコードで `api_key` を直接指定することで認証できます。

        ```python linenums="1"
        import dspy
        lm = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_OPENAI_API_KEY')
        dspy.configure(lm=lm)
        ```

    === "Gemini (AI Studio)"
        環境変数 `GEMINI_API_KEY` を設定するか、以下のコードで `api_key` を直接指定することで認証可能です。

        ```python linenums="1"
        import dspy
        lm = dspy.LM('gemini/gemini-2.5-pro-preview-03-25', api_key='GEMINI_API_KEY')
        dspy.configure(lm=lm)
        ```

    === "Anthropic"
        環境変数 `ANTHROPIC_API_KEY` を設定するか、以下のコードで `api_key` を直接指定することで認証できます。

        ```python linenums="1"
        import dspy
        lm = dspy.LM('anthropic/claude-sonnet-4-5-20250929', api_key='YOUR_ANTHROPIC_API_KEY')
        dspy.configure(lm=lm)
        ```

    === "Databricks"
        Databricksプラットフォームをご利用の場合、SDK経由で自動的に認証されます。それ以外の場合は、環境変数 `DATABRICKS_API_KEY` および `DATABRICKS_API_BASE` を設定するか、以下のコードで `api_key` と `api_base` を直接指定してください。

        ```python linenums="1"
        import dspy
        lm = dspy.LM('databricks/databricks-meta-llama-3-1-70b-instruct')
        dspy.configure(lm=lm)
        ```

    === "GPUサーバー上のローカル言語モデル"
          まず[SGLang](https://sgl-project.github.io/start/install.html)をインストールし、言語モデルを指定してサーバーを起動してください。

          ```bash
          > pip install "sglang[all]"
          > pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/ 

          > CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --port 7501 --model-path meta-llama/Meta-Llama-3-8B-Instruct
          ```

          その後、DSPyコードからOpenAI互換のエンドポイントとして接続します。

          ```python linenums="1"
          lm = dspy.LM("openai/meta-llama/Meta-Llama-3-8B-Instruct",
                           api_base="http://localhost:7501/v1",  # 必ず適切なポートを指定してください
                           api_key="", model_type='chat')
          dspy.configure(lm=lm)
          ```

    === "ノートPC上のローカル言語モデル"
          まず[Ollama](https://github.com/ollama/ollama)をインストールし、言語モデルを指定してサーバーを起動します。

          ```bash
          > curl -fsSL https://ollama.ai/install.sh | sh
          > ollama run llama3.2:1b
          ```

          その後、DSPyコードから接続します。

        ```python linenums="1"
        import dspy
        lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
        dspy.configure(lm=lm)
        ```

    === "その他のプロバイダー"
        DSPyでは、[LiteLLMがサポートする数十種類のLLMプロバイダー](https://docs.litellm.ai/docs/providers)のいずれかを利用できます。各プロバイダーの指示に従い、適切な `{PROVIDER}_API_KEY` の設定方法と、`{provider_name}/{model_name}` のコンストラクタへの指定方法を確認してください。 

        具体的な使用例を以下に示す：

        - `anyscale/mistralai/Mistral-7B-Instruct-v0.1` モデルを使用する場合、`ANYSCALE_API_KEY` 環境変数を設定する
        - `together_ai/togethercomputer/llama-2-70b-chat` モデルを使用する場合、`TOGETHERAI_API_KEY` 環境変数を設定する
        - `sagemaker/<your-endpoint-name>` エンドポイントを使用する場合、`AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`、および `AWS_REGION_NAME` 環境変数を設定する
        - `azure/<your_deployment_name>` エンドポイントを使用する場合、`AZURE_API_KEY`、`AZURE_API_BASE`、`AZURE_API_VERSION` 環境変数を設定する。環境変数を設定せずに外部モデルを呼び出す場合は、以下の形式を使用する：
        `lm = dspy.LM('azure/<your_deployment_name>', api_key = 'AZURE_API_KEY' , api_base = 'AZURE_API_BASE', api_version = 'AZURE_API_VERSION')`


        
        プロバイダーが OpenAI 互換のエンドポイントを提供している場合、モデル名の先頭に `openai/` プレフィックスを追加するだけでよい。

        ```python linenums="1"
        import dspy
        lm = dspy.LM('openai/your-model-name', api_key='PROVIDER_API_KEY', api_base='YOUR_PROVIDER_URL')
        dspy.configure(lm=lm)
        ```
エラーが発生した場合は、[LiteLLM ドキュメント](https://docs.litellm.ai/docs/providers) を参照し、使用している変数名が正しいか、適切な手順に従っているかを確認してください。

## 直接的に LM を呼び出す方法

上記で設定した `lm` モデルを直接呼び出すことは簡単です。これにより統一された API インターフェースが提供され、自動キャッシュ機能などのユーティリティを活用できるようになります。

```python linenums="1"       
lm("これはテストです!", temperature=0.7)  # => ['これはテストです!']
lm(messages=[{"role": "user", "content": "これはテストです!"}])  # => ['これはテストです!']
``` 

## DSPyモジュールを用いたLMの活用

自然なDSpyプログラミングでは、モジュールの活用が不可欠です。この詳細については、次章で解説します。

```python linenums="1" 
# 思考連鎖モジュールを定義し、「質問→回答」という処理フローを指定する。
qa = dspy.ChainOfThought('question -> answer')

# 上記の設定でデフォルト言語モデルを使用して処理を実行する。
response = qa(question="デイヴィッド・グレゴリーが相続した城には何階建ての部分がありますか？")
print(response.answer)
```
**想定される出力例:**
```text
デイヴィッド・グレゴリーが継承した城は7階建てである。
```

## 複数の言語モデルの利用

デフォルトの言語モデルは、`dspy.configure` コマンドでグローバルに設定するか、`dspy.context` コマンドを使用してコードブロック内で個別に設定することが可能です。

!!! ヒント
    `dspy.configure` および `dspy.context` の使用はスレッドセーフな操作です！


```python linenums="1" 
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))
response = qa(question="David Gregoryが相続した城には何階建ての部分がありますか？")
print('GPT-4o-mini:', response.answer)

with dspy.context(lm=dspy.LM('openai/gpt-3.5-turbo')):
    response = qa(question="David Gregoryが相続した城には何階建ての部分がありますか？")
    print('GPT-3.5-turbo:', response.answer)
```
**想定される出力例:**
```text
GPT-4o-mini: デイヴィッド・グレゴリーが相続した城の階数については、提供された情報からは特定できません。
GPT-3.5-turbo: デイヴィッド・グレゴリーが相続した城は7階建てです。
```

## 言語モデル（LM）の設定構成

任意の言語モデルに対して、以下の属性を初期化時または各呼び出し時に任意に設定することが可能です。

```python linenums="1" 
gpt_4o_mini = dspy.LM('openai/gpt-4o-mini', temperature=0.9, max_tokens=3000, stop=None, cache=False)
```

デフォルトでは、DSPyの言語モデル（LM）はキャッシュ機能が有効になっています。同じ呼び出しを繰り返す場合、常に同じ出力が得られます。ただし、`cache=False`と設定することでキャッシュ機能を無効化することが可能です。

キャッシュ機能を維持したまま新規リクエストを強制したい場合（例えば多様な出力を得たい場合）は、
呼び出し時に一意の`rollout_id`を指定し、`temperature`に非ゼロ値を設定してください。DSPyはキャッシュエントリを参照する際に、入力データと`rollout_id`の両方をハッシュ化します。したがって、異なる値を指定すると新規のLMリクエストが実行されますが、同じ入力データと`rollout_id`を持つ後続の呼び出しについてはキャッシュが保持されます。このIDは`lm.history`にも記録されるため、実験中に異なるロールアウトを追跡・比較する際などに便利です。`temperature=0`のまま`rollout_id`のみを変更した場合でも、LMの出力には影響しません。

```python linenums="1"
lm("これはテストです!", rollout_id=1, temperature=1.0)
```

これらの LM 引数（kwargs）は DSPy モジュールに直接渡すことも可能です。初期化時にこれらを指定すると、すべての呼び出しに対してデフォルト値が設定されます：

```python linenums="1"
predict = dspy.Predict("question -> answer", rollout_id=1, temperature=1.0)
```

単一の実行時にこれらの設定を上書きする場合は、モジュールを呼び出す際に``config``辞書を指定してください：

```python linenums="1"
predict = dspy.Predict("question -> answer")
predict(question="What is 1 + 52?", config={"rollout_id": 5, "temperature": 1.0})
```

いずれの場合も、``rollout_id`` は基盤となる大規模言語モデル（LM）に渡され、そのキャッシュ動作に影響を与えます。また、各応答とともに保存されるため、後で特定のロールアウトを再生したり分析したりすることが可能になります。


## 出力内容と使用状況メタデータの確認

すべての LM オブジェクトは、入力、出力、トークン使用量（および対応するコスト）、およびメタデータを含むインタラクション履歴を保持しています。

```python linenums="1" 
len(lm.history)  # 例: 言語モデルへの呼び出し回数 3 回

lm.history[-1].keys()  # 言語モデルへの最後の呼び出し内容にアクセス（すべてのメタデータを含む）
```

**出力結果:**
```text
dict_keys(['prompt', 'messages', 'kwargs', 'response', 'outputs', 'usage', 'cost', 'timestamp', 'uuid', 'model', 'response_model', 'model_type'])
```

## Responses APIの使用方法

デフォルトでは、DSPyは言語モデル（LM）の呼び出しにLiteLLMの[Chat Completions API](https://docs.litellm.ai/docs/completion)を使用しています。このAPIはほとんどの標準的なモデルとタスクに適しています。ただし、OpenAIの推論モデル（例：`gpt-5`または今後リリースされる他のモデル）などの高度なモデルの場合、[Responses API](https://docs.litellm.ai/docs/response_api)経由でアクセスすることで、品質の向上や追加機能を活用できる可能性があります。Responses APIはDSPyでサポートされています。

**Responses APIを使用するべきケース:**

- OpenAIの推論モデルなど、`responses`エンドポイントをサポートまたは必須とするモデルを使用する場合
- 特定のモデルが提供する高度な推論機能、マルチターン対話機能、またはより豊富な出力機能を活用したい場合

**DSPyでResponses APIを有効にする方法:**

Responses APIを使用するには、`dspy.LM`インスタンスを作成する際に`model_type="responses"`と指定してください。

```python
import dspy

# DSPy を設定して、指定した言語モデルの Responses API を使用する
dspy.configure(
    lm=dspy.LM(
        "openai/gpt-5-mini",
        model_type="responses",
        temperature=1.0,
        max_tokens=16000,
    ),
)
```

なお、すべてのモデルやプロバイダーが Responses API に対応しているわけではない点にご注意ください。詳細については、[LiteLLM の公式ドキュメント](https://docs.litellm.ai/docs/response_api) をご参照ください。


## 上級編：カスタム言語モデルの構築と独自のアダプターの作成

使用頻度は低いものの、`dspy.BaseLM` を継承することでカスタム言語モデルを実装することが可能です。DSPy エコシステムにおけるもう一つの高度な機能として、_アダプター_ が挙げられます。これは DSPy のシグネチャと言語モデルの間に介在する中間層です。本ガイドの将来版では、これらの高度な機能について詳しく解説する予定ですが、現時点では必ずしも必要ではない機能です。

