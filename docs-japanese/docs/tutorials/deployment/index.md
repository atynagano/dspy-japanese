# チュートリアル：DSPyプログラムのデプロイ方法

本ガイドでは、DSPyプログラムを本番環境にデプロイするための2つの主要な方法について解説します。軽量なデプロイ環境にはFastAPIを、より本格的なプロダクション環境にはMLflowを使用したプログラムのバージョン管理と運用管理方法をご紹介します。

以下では、デプロイ対象として以下のシンプルなDSPyプログラムを想定しています。この例をより高度な実装に置き換えることも可能です。

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
dspy_program = dspy.ChainOfThought("question -> answer")
```

## FastAPIを使用したデプロイ方法

FastAPIは、DSPyプログラムをREST APIとして簡単に公開するための優れた手段を提供します。プログラムコードに直接アクセス可能で、軽量なデプロイソリューションが必要な場合に特に適しています。

```bash
> pip install fastapi uvicorn
> export OPENAI_API_KEY="your-openai-api-key"
```

上記で定義した`dspy_program`を実行するためのFastAPIアプリケーションを作成しましょう。

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import dspy

app = FastAPI(
    title="DSPy Program API",
    description="DSPyのChain of Thoughtプログラムを提供するシンプルなAPI",
    version="1.0.0"
)

# ドキュメント生成と入力検証を効率化するためのリクエストモデルを定義
class Question(BaseModel):
    text: str

# 使用する言語モデルを設定し、DSPyプログラムを非同期処理対応に構成する
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm, async_max_workers=4) # デフォルト値は8
dspy_program = dspy.ChainOfThought("question -> answer")
dspy_program = dspy.asyncify(dspy_program)

@app.post("/predict")
async def predict(question: Question):
    try:
        result = await dspy_program(question=question.text)
        return {
            "status": "success",
            "data": result.toDict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

上記のコードでは、`dspy.asyncify`関数を使用してdspyプログラムを非同期モードに変換し、FastAPIによる高スループットなデプロイメントを実現しています。現在の実装では、dspyプログラムを別スレッドで実行し、その結果を待機する方式を採用しています。

デフォルトでは、同時に起動可能なスレッド数は8に制限されています。これはワーカープールの概念に似ています。実行中のプログラムが8件ある状態で再度呼び出した場合、9番目の呼び出しは既存の8件のいずれかが終了するまで待機します。
非同期処理の最大スレッド数は、新たに追加された`async_max_workers`設定パラメータで変更可能です。

??? "ストリーミング機能（DSPy 2.6.0以降で対応）"

    DSPy 2.6.0以降ではストリーミング機能も利用可能です。`pip install -U dspy`コマンドで最新バージョンをインストールすることで使用できます。

    `dspy.streamify`関数を使用すると、dspyプログラムをストリーミングモードに変換できます。これは、最終的な予測結果が準備される前に、中間処理結果（O1スタイルの推論結果など）をクライアントに逐次送信したい場合に有用です。この機能は内部的に`asyncify`機能を利用し、実行時の動作仕様を継承しています。

    ```python
    dspy_program = dspy.asyncify(dspy.ChainOfThought("question -> answer"))
    streaming_dspy_program = dspy.streamify(dspy_program)

    @app.post("/predict/stream")
    async def stream(question: Question):
        async def generate():
            async for value in streaming_dspy_program(question=question.text):
                if isinstance(value, dspy.Prediction):
                    data = {"prediction": value.labels().toDict()}
                elif isinstance(value, litellm.ModelResponse):
                    data = {"chunk": value.json()}
                yield f"data: {ujson.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # DSPyプログラムの処理結果をサーバー送信イベントとしてストリーミングするケースが多いため、
    # 上記コードと同等の機能を提供するヘルパー関数を用意しています。

    from dspy.utils.streaming import streaming_response

    @app.post("/predict/stream")
    async def stream(question: Question):
        stream = streaming_dspy_program(question=question.text)
        return StreamingResponse(streaming_response(stream), media_type="text/event-stream")
    ```

作成したコードを`fastapi_dspy.py`などのファイルに保存してください。その後、以下のコマンドでアプリケーションをサーバー上で実行できます：

```bash
> uvicorn fastapi_dspy:app --reload
```

ローカルサーバーが `http://127.0.0.1:8000/` で起動します。以下のPythonコードで動作を確認できます：

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"text": "フランスの首都はどこですか？"}
)
print(response.json())
```

以下のようなレスポンスが表示されるはずです：

```json
{
  "status": "success",
  "data": {
    "reasoning": "フランスの首都は地理の授業で一般的に教えられる周知の事実であり、様々な文脈で言及される著名な都市です。パリは世界的に首都として認知されており、同国の政治・文化・経済の中心地として機能しています。",
    "answer": "フランスの首都はパリです。"
  }
}
```

## MLflowを用いたデプロイ方法

DSPyプログラムをパッケージ化し、隔離された環境でデプロイする必要がある場合には、MLflowの使用を推奨します。
MLflowは、機械学習ワークフローの管理において広く利用されているプラットフォームであり、バージョン管理、実験追跡、デプロイメント機能などを包括的に提供しています。

```bash
> pip install mlflow>=2.18.0
```

MLflowトラッキングサーバーを起動し、DSPyプログラムの実行環境を設定しましょう。以下のコマンドを実行すると、ローカルサーバーが
`http://127.0.0.1:5000/` で起動します。

```bash
> mlflow ui
```

次に、DSPyプログラムを定義し、MLflowサーバーにログ記録します。MLflowにおいて「log」という用語は多義的ですが、基本的には
プログラム情報と環境要件をMLflowサーバーに保存することを意味します。これは`mlflow.dspy.log_model()`関数を使用して実行します。
以下に具体的なコード例を示します：

> [!NOTE]
> MLflow 2.22.0以降では、DSPyプログラムをMLflowでデプロイする際に、カスタムDSPyモジュールクラスでラップする必要がありますのでご注意ください。
> これは、MLflowが位置引数を必要とするのに対し、DSPyのプリビルドモジュール（例：`dspy.Predict`や`dspy.ChainOfThought`）では位置引数が許可されていないためです。
> この問題を回避するには、`dspy.Module`を継承したラッパークラスを作成し、`forward()`メソッド内でプログラムのロジックを実装してください。

```python
import dspy
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("deploy_dspy_program")

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

class MyProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought("question -> answer")

    def forward(self, messages):
        return self.cot(question=messages[0]["content"])

dspy_program = MyProgram()

with mlflow.start_run():
    mlflow.dspy.log_model(
        dspy_program,
        "dspy_program",
        input_example={"messages": [{"role": "user", "content": "LLMエージェントとは何ですか？"}]},
        task="llm/v1/chat",
    )
```

デプロイされたプログラムがOpenAIのチャットAPIと同様の形式で入力を受け取り、出力を生成するよう、`task="llm/v1/chat"`を設定することを強く推奨します。これはLMアプリケーションで広く使用されている標準的なインターフェースです。上記のコードを`mlflow_dspy.py`などのファイルに記述し、実行してください。

プログラムのログ記録が完了したら、MLflow UIで保存された情報を確認できます。`http://127.0.0.1:5000/`にアクセスし、`deploy_dspy_program`実験を選択した後、作成した最新のランを選択してください。`Artifacts`タブを開くと、以下のようなスクリーンショットと同様の形式で、ログ記録されたプログラム情報が表示されます：

![MLflow UI](./dspy_mlflow_ui.png)

UIから（または`mlflow_dspy.py`を実行した際のコンソール出力から）ランIDを取得した後、以下のコマンドを使用してログ記録済みのプログラムをデプロイできます：

```bash
> mlflow models serve -m runs:/{run_id}/model -p 6000
```

プログラムのデプロイが完了したら、以下のコマンドでテストを実施できます：

```bash
> curl http://127.0.0.1:6000/invocations -H "Content-Type:application/json" --data '{"messages": [{"content": "2 + 2 の計算結果は？", "role": "user"}]}'
```

以下のようなレスポンスが表示されるはずです：

```json
{
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\"reasoning\": \"本問は2と2の和を求めるものです。解答を得るためには、単純に2つの数値を加算します: 2 + 2 = 4。\", \"answer\": \"4\"}"
      },
      "finish_reason": "stop"
    }
  ]
}
```

DSPyプログラムをMLflowでデプロイする方法、およびデプロイ環境をカスタマイズするための詳細な手順については、以下の公式
[MLflowドキュメント](https://mlflow.org/docs/latest/llms/dspy/index.html)
を参照してください。

### MLflowデプロイにおけるベストプラクティス

1. **環境管理**: 必ず`conda.yaml`または`requirements.txt`ファイルでPython依存関係を明確に定義してください。
2. **バージョン管理**: モデルバージョンには意味のあるタグと説明を付与してください。
3. **入力検証**: 明確な入力スキーマと使用例を定義してください。
4. **モニタリング**: 本番環境へのデプロイ時には、適切なロギングとモニタリングシステムを構築してください。

本番環境へのデプロイにおいては、MLflowとコンテナ化技術の組み合わせを検討してください：

```bash
> mlflow models build-docker -m "runs:/{run_id}/model" -n "dspy-program"
> docker run -p 6000:8080 dspy-program
```

プロダクション環境へのデプロイに関する詳細なガイドおよびベストプラクティスについては、
[MLflow公式ドキュメント](https://mlflow.org/docs/latest/llms/dspy/index.html)
を参照してください。
