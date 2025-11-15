# DSPyアダプターの理解

## アダプターとは何か？

アダプターは、`dspy.Predict`インターフェースと実際の言語モデル（Language Model: LM）をつなぐ橋渡し役です。DSPyモジュールを呼び出すと、
アダプターはユーザーから提供されたシグネチャ、入力データ、および`demos`（few-shot例）などの属性を受け取り、それらを
マルチターン形式のメッセージに変換してLMに送信します。

アダプターシステムの主な役割は以下の通りです：

- DSPyのシグネチャを、タスク定義とリクエスト/レスポンス構造を規定するシステムメッセージに変換する
- DSPyのシグネチャで定義されたリクエスト構造に従って入力データを整形する
- LMからの応答を解析し、`dspy.Prediction`インスタンスなどの構造化されたDSPy出力形式に変換する
- 会話履歴と関数呼び出しを管理する
- 事前に構築されたDSPyタイプ（`dspy.Tool`、`dspy.Image`など）をLM用プロンプトメッセージに変換する

## アダプターの設定

`dspy.configure(adapter=...)`を使用してPythonプロセス全体で使用するアダプターを選択するか、
`with dspy.context(adapter=...):`構文を用いて特定の名前空間にのみ影響を与えるように設定できます。

DSPyワークフローでアダプターが指定されていない場合、各`dspy.Predict.__call__`メソッドはデフォルトで`dspy.ChatAdapter`を使用します。したがって、以下の2つのコードスニペットは同等の動作となります：

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

predict = dspy.Predict("question -> answer")
result = predict(question="What is the capital of France?")
```

```python
import dspy

dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini"),
    adapter=dspy.ChatAdapter(),  # これはデフォルト設定値です
)

predict = dspy.Predict("question -> answer")
result = predict(question="フランスの首都はどこですか？")
```

## アダプターがシステムアーキテクチャにおいて果たす役割

処理の流れは以下の通りである：

1. ユーザーがDSPyエージェント（通常は入力機能を備えた`dspy.Module`インスタンス）を呼び出す。
2. 内部の`dspy.Predict`関数が実行され、言語モデル（LM）からの応答を取得する。
3. `dspy.Predict`は**Adapter.format()**メソッドを呼び出し、自身のシグネチャ情報、入力データ、およびデモデータを複数ターンにわたるメッセージ形式に変換する。これらのメッセージは`dspy.LM`コンポーネントに送信される。`dspy.LM`は`litellm`ライブラリをラップした軽量なインターフェースであり、実際のLMエンドポイントとの通信を担う。
4. LMは受信したメッセージを処理し、応答を生成する。
5. **Adapter.parse()**メソッドがLMの応答を構造化されたDSPy出力形式に変換する。この変換は元のシグネチャ仕様に従って行われる。
6. 最終的に、`dspy.Predict`を呼び出したクライアント側に、構造化された出力データが返される。

必要に応じて、`Adapter.format()`メソッドを直接呼び出すことで、LMに送信されるメッセージの内容を明示的に確認することが可能である。

```python
# 簡略化したフローの実行例
signature = dspy.Signature("question -> answer")
inputs = {"question": "2+2の答えは何ですか？"}
demos = [{"question": "1+1の答えは何ですか？", "answer": "2"}]

adapter = dspy.ChatAdapter()
print(adapter.format(signature, demos, inputs))
```

出力結果は以下のようになるべきです：

```
{'role': 'system', 'content': '入力フィールドは以下の通りです：\n1. `question` (文字列):\n出力フィールドは以下の通りです：\n1. `answer` (文字列):\nすべてのやり取りは以下の形式で行われ、適切な値が埋め込まれます。\n\n[[ ## question ## ]]\n{question}\n\n[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]\nこの形式を遵守した上で、あなたの目的は次の通りです：\n        与えられた`question`フィールドに基づき、対応する`answer`フィールドを生成してください。'}
{'role': 'user', 'content': '[[ ## question ## ]]\n1+1の答えは何ですか？'}
{'role': 'assistant', 'content': '[[ ## answer ## ]]\n2\n\n[[ ## completed ## ]]\n'}
{'role': 'user', 'content': '[[ ## question ## ]]\n2+2の答えは何ですか？\n\n対応する出力フィールドを回答してください。まず`[[ ## answer ## ]]`フィールドから始め、その後`[[ ## completed ## ]]`のマーカーで終了してください。'}
```

また、`adapter.format_system_message(signature)` を呼び出すことでシステムメッセージのみを取得することも可能です。

```python
import dspy

signature = dspy.Signature("question -> answer")
system_message = dspy.ChatAdapter().format_system_message(signature)
print(system_message)
```

出力結果は以下のようになるべきです：

```
入力フィールドは以下の通りです：
1. `question`（文字列型）：
出力フィールドは以下の通りです：
1. `answer`（文字列型）：
すべてのインタラクションは以下の形式で行われ、適切な値が埋め込まれます。

[[ ## question ## ]]
{question}
[[ ## answer ## ]]
{answer}
[[ ## completed ## ]]
この形式を遵守した上で、あなたのタスクは以下のように定義されます：
        入力フィールド `question` が与えられた場合、対応する出力フィールド `answer` を生成すること。
```

## アダプタの種類

DSPyでは用途に応じて複数のアダプタタイプを提供しており、それぞれ特定のユースケースに最適化されています：

### ChatAdapter

**ChatAdapter**はデフォルトのアダプタであり、あらゆる言語モデルで使用可能です。フィールドベースのフォーマットを採用しており、特別なマーカー記号を使用しています。

#### フォーマット構造

ChatAdapterは`[[ ## field_name ## ]]`というマーカー記号を用いてフィールドを区切っています。非プリミティブ型のPythonオブジェクトについては、その型のJSONスキーマ情報も併せて表示します。以下の例では、`dspy.inspect_history()`関数を使用して、`dspy.ChatAdapter`によってフォーマットされたメッセージを明確に表示しています。

```python
import dspy
import pydantic

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), adapter=dspy.ChatAdapter())


class ScienceNews(pydantic.BaseModel):
    text: str
    scientists_involved: list[str]


class NewsQA(dspy.Signature):
    """指定された科学分野に関するニュースを取得する"""

    science_field: str = dspy.InputField()
    year: int = dspy.InputField()
    num_of_outputs: int = dspy.InputField()
    news: list[ScienceNews] = dspy.OutputField(desc="科学関連ニュース")


predict = dspy.Predict(NewsQA)
predict(science_field="コンピュータ理論", year=2022, num_of_outputs=1)
dspy.inspect_history()
```

出力結果は以下のようになります。

```
[2025年8月15日 22:24:29.378666]

システムメッセージ:

入力フィールドは以下の通りです:
1. `science_field` (str):
2. `year` (int):
3. `num_of_outputs` (int):
出力フィールドは以下の通りです:
1. `news` (list[ScienceNews]): 科学ニュース
すべてのインタラクションは以下の形式で構造化され、適切な値が埋め込まれます。

[[ ## science_field ## ]]
{science_field}

[[ ## year ## ]]
{year}

[[ ## num_of_outputs ## ]]
{num_of_outputs}

[[ ## news ## ]]
{news}        # 注: 生成する値はJSONスキーマに準拠する必要があります: {"type": "array", "$defs": {"ScienceNews": {"type": "object", "properties": {"scientists_involved": {"type": "array", "items": {"type": "string"}, "title": "関与した科学者"}, "text": {"type": "string", "title": "本文"}}, "required": ["text", "scientists_involved"], "title": "ScienceNews"}}, "items": {"$ref": "#/$defs/ScienceNews"}}

[[ ## completed ## ]]
この構造に従うことで、あなたの目的は以下の条件を満たすことです:
        指定された科学分野に関するニュースを取得する


ユーザーメッセージ:

[[ ## science_field ## ]]
計算機理論

[[ ## year ## ]]
2022年

[[ ## num_of_outputs ## ]]
1

対応する出力フィールドを以下の順序で回答してください: まずフィールド `[[ ## news ## ]]` から開始し (有効なPythonリスト[ScienceNews]形式でフォーマットする必要があります)、その後 `[[ ## completed ## ]]` のマーカーで終了してください。


回答:

[[ ## news ## ]]
[
    {
        "scientists_involved": ["John Doe", "Jane Smith"],
        "text": "2022年、研究者たちは量子コンピューティングアルゴリズムにおいて重要な進展を遂げ、古典的コンピュータよりも高速に複雑な問題を解決できる可能性を示しました。このブレークスルーは、暗号学や最適化などの分野に革命をもたらす可能性があります。"
    }
]

[[ ## completed ## ]]
```

!!! info "実践課題: 印刷されたLM履歴から署名情報の位置を特定する"

    署名の調整を試み、変更が印刷されたLMメッセージにどのように反映されるかを観察してください。


各フィールドの前にはマーカー `[[ ## field_name ## ]]` が付与されています。出力フィールドのデータ型がプリミティブ型でない場合、指示文にはその型のJSONスキーマが含まれ、出力はそれに従って整形されます。出力フィールドはChatAdapterによって定義された構造に従っているため、自動的に構造化データとして解析可能です。

#### ChatAdapterを使用するべき場面

`ChatAdapter`には以下の利点があります:

- **汎用性の高さ**: あらゆる言語モデルで使用可能ですが、小規模なモデルでは要求された形式に合致しない応答が生成される場合があります。
- **フォールバック機能**: `ChatAdapter`が失敗した場合、自動的に`JSONAdapter`による再試行が行われます。

特に要件がない場合、`ChatAdapter`は信頼性の高い選択肢と言えます。

#### ChatAdapterを使用しないべき場面

以下の場合には`ChatAdapter`の使用を避けてください:

- **遅延が許容できないシステム**: `ChatAdapter`は他のアダプタと比べて定型出力トークンが多いため、遅延に敏感なシステムを構築する場合は別種のアダプタを検討することをお勧めします。

### JSONAdapter

**JSONAdapter**は、LMに対して署名で指定されたすべての出力フィールドを含むJSONデータを返すようプロンプトします。`response_format`パラメータを介して構造化出力をサポートするモデルに対して有効であり、ネイティブのJSON生成機能を活用することで、より高い信頼性での解析が可能になります。

#### フォーマット構造

`JSONAdapter`によってフォーマットされたプロンプトの入力部分は、`ChatAdapter`と類似していますが、出力部分は以下のように異なります:

```python
import dspy
import pydantic

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), adapter=dspy.JSONAdapter())


class ScienceNews(pydantic.BaseModel):
    text: str
    scientists_involved: list[str]


class NewsQA(dspy.Signature):
    """指定された科学分野に関するニュースを取得する"""

    science_field: str = dspy.InputField()
    year: int = dspy.InputField()
    num_of_outputs: int = dspy.InputField()
    news: list[ScienceNews] = dspy.OutputField(desc="科学関連ニュース")


predict = dspy.Predict(NewsQA)
predict(science_field="コンピュータ理論", year=2022, num_of_outputs=1)
dspy.inspect_history()
```

```
システムメッセージ:

入力フィールドは以下の通りです:
1. `science_field` (str):
2. `year` (int):
3. `num_of_outputs` (int):
出力フィールドは以下の通りです:
1. `news` (list[ScienceNews]): 科学ニュース
すべてのインタラクションは以下の形式で構造化され、適切な値が入力されます。

入力データの形式は以下の通りです:

[[ ## science_field ## ]]
{science_field}

[[ ## year ## ]]
{year}

[[ ## num_of_outputs ## ]]
{num_of_outputs}

出力は以下に示すフィールドを含むJSONオブジェクトとなります。

{
  "news": "{news}        # 注: 生成する値はJSONスキーマに準拠している必要があります: {\"type\": \"array\", \"$defs\": {\"ScienceNews\": {\"type\": \"object\", \"properties\": {\"scientists_involved\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}, \"title\": \"Scientists Involved\"}, \"text\": {\"type\": \"string\", \"title\": \"Text\"}}, \"required\": [\"text\", \"scientists_involved\"], \"title\": \"ScienceNews\"}}, \"items\": {\"$ref\": \"#/$defs/ScienceNews\"}}"
}
この構造に従うことで、あなたの目的は以下の条件を満たすことです:
        指定された科学分野に関するニュースを取得する


ユーザーメッセージ:

[[ ## science_field ## ]]
計算機理論

[[ ## year ## ]]
2022年

[[ ## num_of_outputs ## ]]
1

以下のフィールド順にJSONオブジェクト形式で応答してください: `news` (有効なPythonリスト[ScienceNews]としてフォーマットされている必要があります)。


応答:

{
  "news": [
    {
      "text": "2022年、研究者たちは量子コンピューティングアルゴリズムにおいて重要な進展を遂げ、量子システムが特定のタスクにおいて古典コンピュータを上回る性能を発揮できることを実証しました。このブレークスルーは、暗号学や複雑系シミュレーションなどの分野に革命をもたらす可能性があります。",
      "scientists_involved": [
        "Dr. Alice Smith",
        "Dr. Bob Johnson",
        "Dr. Carol Lee"
      ]
    }
  ]
}
```

#### JSONAdapterの適切な使用場面

`JSONAdapter`は以下の場合に特に有効です：

- **構造化出力が必要な場合**：モデルが`response_format`パラメータをサポートしている場合
- **低遅延が求められる場合**：LM応答における定型文部分が最小限で済むため、応答速度が向上します

#### JSONAdapterの使用を避けるべき場面

以下の場合には`JSONAdapter`の使用を避けてください：

- Ollama上でホストされている小規模なオープンソースモデルなど、構造化出力をネイティブにサポートしていないモデルを使用する場合

## まとめ

アダプターはDSPyにおいて重要な要素であり、構造化されたDSPyのシグネチャと言語モデルAPIの間の橋渡し役として機能します。
各種アダプターの適切な使用場面を理解することで、より信頼性が高く効率的なDSPyプログラムを構築することが可能になります。
