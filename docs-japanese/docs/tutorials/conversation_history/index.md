# 会話履歴の管理について

チャットボットなどのAIアプリケーションを構築する際、会話履歴の保持は基本的な機能要件となります。DSPyでは`dspy.Module`内で直接的な会話履歴管理機能は提供していませんが、効果的な会話履歴管理を支援する`dspy.History`ユーティリティを用意しています。

## `dspy.History`を用いた会話履歴管理の方法

`dspy.History`クラスは入力フィールド型として利用可能で、`messages: list[dict[str, Any]]`という属性を持ちます。このリストの各要素は辞書型であり、辞書のキーはユーザーが定義したシグネチャで指定したフィールド名に対応します。以下に具体例を示します：

```python
import dspy
import os

os.environ["OPENAI_API_KEY"] = "{your_openai_api_key}"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class QA(dspy.Signature):
    question: str = dspy.InputField()  # 質問入力フィールド
    history: dspy.History = dspy.InputField()  # 対話履歴フィールド
    answer: str = dspy.OutputField()  # 回答出力フィールド

predict = dspy.Predict(QA)  # 予測処理の設定
history = dspy.History(messages=[])  # 対話履歴の初期化

while True:
    question = input("質問を入力してください（'finish' と入力すると会話を終了します）: ")
    if question == "finish":
        break
    outputs = predict(question=question, history=history)  # 予測処理の実行
    print(f"\n{outputs.answer}\n")  # 出力された回答を表示
    history.messages.append({"question": question, **outputs})  # 履歴に記録を追加

dspy.inspect_history()  # 対話履歴の確認
```

会話履歴を使用する際には、以下の2つの重要な手順があります：

- **Signatureに`dspy.History`型のフィールドを含める**
- **実行時に履歴インスタンスを保持し、新しい会話ターンを逐次追加していく**。各エントリには、関連するすべての入力フィールド情報と出力フィールド情報を含める必要があります。

具体的な実行例は以下の通りです：

```
質問を入力してください。会話を終了するには「終了」と入力してください：PyTorchとTensorFlowの競合状況についてご存知ですか？

はい、PyTorchとTensorFlowの間には顕著な競合関係が存在します。これらは現在最も広く利用されている深層学習フレームワークの2つです。Facebookが開発したPyTorchは、動的計算グラフを採用している点が特徴で、特に研究用途において柔軟性と使いやすさに優れています。Googleが開発したTensorFlowは当初静的計算グラフを採用していましたが、その後Eager実行機能を導入することで使いやすさを向上させました。TensorFlowはスケーラビリティとデプロイメント機能に優れているため、本番環境でよく採用されています。両フレームワークとも活発なコミュニティと豊富なライブラリを有しており、選択の際には具体的なプロジェクト要件や個人の好みが重要な要素となります。

質問を入力してください。会話を終了するには「終了」と入力してください：どちらが勝利したのでしょうか？結果のみを教えてください。理由の説明は不要です、よろしくお願いします！

明確な勝者は存在しません。PyTorchとTensorFlowはいずれも広く使用されており、それぞれ独自の強みを持っています。

質問を入力してください。会話を終了するには「終了」と入力してください：終了




[2025-07-11T16:35:57.592762]

システムメッセージ：

入力フィールドは以下の通りです：
1. `question`（str）：
2. `history`（History）：
出力フィールドは以下の通りです：
1. `answer`（str）：
すべてのやり取りは以下の形式で構造化され、適切な値が埋め込まれます。

[[ ## question ## ]]
{question}

[[ ## history ## ]]
{history}

[[ ## answer ## ]]
{answer}

[[ ## completed ## ]]
この構造に従う場合、あなたの目的は以下の通りです：
        `question`と`history`のフィールドが与えられたとき、`answer`のフィールドを生成すること。


ユーザーメッセージ：

[[ ## question ## ]]
PyTorchとTensorFlowの競合状況についてご存知ですか？


アシスタントメッセージ：

[[ ## answer ## ]]
はい、PyTorchとTensorFlowの間には顕著な競合関係が存在します。これらは現在最も広く利用されている深層学習フレームワークの2つです。Facebookが開発したPyTorchは、動的計算グラフを採用している点が特徴で、特に研究用途において柔軟性と使いやすさに優れています。Googleが開発したTensorFlowは当初静的計算グラフを採用していましたが、その後Eager実行機能を導入することで使いやすさを向上させました。TensorFlowはスケーラビリティとデプロイメント機能に優れているため、本番環境でよく採用されています。両フレームワークとも活発なコミュニティと豊富なライブラリを有しており、選択の際には具体的なプロジェクト要件や個人の好みが重要な要素となります。

[[ ## completed ## ]]


ユーザーメッセージ：

[[ ## question ## ]]
どちらが勝利したのでしょうか？結果のみを教えてください。理由の説明は不要です、よろしくお願いします！

対応する出力フィールドを、`[[ ## answer ## ]]`フィールドから始めて、`[[ ## completed ## ]]`マーカーで終わる形式で回答してください。


回答：

[[ ## answer ## ]]
明確な勝者は存在しません。PyTorchとTensorFlowはいずれも広く使用されており、それぞれ独自の強みを持っています。

[[ ## completed ## ]]
```

各ユーザー入力とアシスタント応答がどのように履歴に追加されていくかに注目してください。これにより、モデルは会話の文脈をターン間で維持することが可能になります。

実際に言語モデルに送信されるプロンプトは、`dspy.inspect_history`の出力が示すように複数ターンにわたるメッセージです。各会話ターンは、ユーザーメッセージに続いてアシスタントメッセージが配置される形式で表現されます。

## 少数ショット例における履歴の扱い

プロンプトの入力フィールド欄に`history`が記載されていないことにお気づきかもしれません（例えばシステムメッセージ中の「2. `history`（履歴）：」といった入力フィールドのリストに含まれています）。これは意図的な設計です。会話履歴を含む少数ショット例をフォーマットする際、DSPyは履歴を複数ターンに展開しません。代わりに、OpenAIの標準フォーマットとの互換性を保つため、各少数ショット例は単一ターンとして表現されます。

具体例を挙げると：

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


class QA(dspy.Signature):
    question: str = dspy.InputField()  # 質問文
    history: dspy.History = dspy.InputField()  # 対話履歴
    answer: str = dspy.OutputField()  # 回答文


predict = dspy.Predict(QA)
history = dspy.History(messages=[])  # 初期状態の対話履歴

predict.demos.append(
    dspy.Example(
        question="What is the capital of France?",  # 質問例
        history=dspy.History(
            messages=[{"question": "What is the capital of Germany?", "answer": "The capital of Germany is Berlin."}]
        ),
        answer="The capital of France is Paris.",  # 期待される回答
    )
)

predict(question="What is the capital of America?", history=dspy.History(messages=[]))  # テスト実行
dspy.inspect_history()  # 対話履歴の確認
```

生成される履歴は以下のようになります。

```
[2025年7月11日 16:53:10.994111]

システムメッセージ:

入力フィールドは以下の通りです:
1. `question` (str): 
2. `history` (History):
出力フィールドは以下の通りです:
1. `answer` (str):
すべてのインタラクションは以下の形式で構造化され、適切な値が埋め込まれます。

[[ ## question ## ]]
{question}

[[ ## history ## ]]
{history}

[[ ## answer ## ]]
{answer}

[[ ## completed ## ]]
この構造に従うことで、あなたの目的は以下の条件を満たすことです:
        `question` と `history` のフィールドが与えられた場合、`answer` のフィールドを生成すること。


ユーザーメッセージ:

[[ ## question ## ]]
フランスの首都はどこですか？

[[ ## history ## ]]
{"messages": [{"question": "ドイツの首都はどこですか？", "answer": "ドイツの首都はベルリンです。"}]}


アシスタントメッセージ:

[[ ## answer ## ]]
フランスの首都はパリです。

[[ ## completed ## ]]


ユーザーメッセージ:

[[ ## question ## ]]
ドイツの首都はどこですか？

対応する出力フィールドを回答してください。まずフィールド `[[ ## answer ## ]]` から始め、その後 `[[ ## completed ## ]]` のマーカーで終了してください。


回答:

[[ ## answer ## ]]
ドイツの首都はベルリンです。

[[ ## completed ## ]]
```

ご覧の通り、few-shot例では会話履歴を複数ターンにわたって展開することはありません。代わりに、履歴情報をJSON形式のデータとしてセクション内に表現します：

```
[[ ## history ## ]]
{"messages": [{"question": "ドイツの首都はどこですか？", "answer": "ドイツの首都はベルリンです。"}]}
```

この手法により、標準的なプロンプト形式との互換性を確保しつつ、モデルに対して適切な会話文脈を提供することが可能となる。

