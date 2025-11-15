---
sidebar_position: 2
---

# シグネチャについて

DSPyにおいてLMにタスクを割り当てる際、必要な動作仕様を「シグネチャ」として定義します。

**シグネチャとは、DSPyモジュールの入出力動作を宣言的に記述した仕様です。** これにより、LMに対して「何をすべきか」を明確に指示することができ、動作の「方法」を詳細に指定する必要がなくなります。

関数のシグネチャ（引数とその型を定義したもの）に馴染みのある方なら、DSPyのシグネチャも同様の概念だとご理解いただけるでしょう。ただし、いくつかの重要な違いがあります。一般的な関数シグネチャが単に「動作を記述する」ものであるのに対し、DSPyのシグネチャは「モジュールの動作を宣言的に定義し、初期化する」ものです。さらに、DSPyのシグネチャではフィールド名そのものが重要となります。入力/出力の意味的役割を平易な英語で表現します：`question`と`answer`は異なる概念であり、`sql_query`と`python_code`も明確に区別されます。

## DSPyシグネチャを使用するべき理由

モジュール化され整理されたコードを実現し、LMへの呼び出しを高品質なプロンプト（あるいは自動ファインチューニング）に最適化できるためです。多くの開発者は、長大で脆弱なプロンプトを試行錯誤しながら作成したり、ファインチューニング用のデータを収集・生成したりすることでLMにタスクを実行させています。しかし、シグネチャを記述する方法は、プロンプトの試行錯誤やファインチューニングに比べて、はるかにモジュール化されており、適応性が高く、再現性に優れています。DSPyコンパイラは、データとパイプラインの範囲内で、指定したシグネチャに最適なプロンプトを自動生成します（あるいは小規模LMのファインチューニングを行います）。多くの場合、コンパイルによって生成されるプロンプトは人間が作成したものよりも優れた性能を発揮します。これはDSPyの最適化アルゴリズムが人間よりも創造的だからではなく、単により多くの可能性を試行でき、直接的に評価指標を調整できるからです。

## **インライン** DSPyシグネチャ

シグネチャは短い文字列として定義でき、入力/出力の意味的役割を定義する引数名とオプションの型を指定します。

1. 質問応答：`"question -> answer"` これはデフォルト型が常に`str`であるため、`"question: str -> answer: str"`と同等です

2. 感情分類：`"sentence -> sentiment: bool"` 例えば肯定的な場合`True`を返す

3. 要約生成：`"document -> summary"`

複数の入力/出力フィールドと型を持つシグネチャも定義可能です：

4. 検索拡張型質問応答：`"context: list[str], question: str -> answer: str"`

5. 推論を伴う多肢選択式質問応答：`"question, choices: list[str] -> reasoning: str, selection: int"`

**ヒント：** フィールド名については、有効な変数名であれば何でも使用できます。フィールド名は意味的に意味のあるものであるべきですが、最初はシンプルに保ち、キーワードを過度に最適化しようとしないでください。このような微調整はDSPyコンパイラに任せましょう。例えば要約生成の場合、`"document -> summary"`、`"text -> gist"`、あるいは`"long_context -> tldr"`といった表現で十分な場合が多いでしょう。

インラインシグネチャには実行時に変数を使用できる指示文を追加することも可能です。`instructions`キーワード引数を使用して、シグネチャに指示文を追加してください。

```python
toxicity = dspy.Predict(
    dspy.Signature(
        "comment -> toxic: bool",
        instructions="コメントに侮辱的表現、嫌がらせ、または皮肉的な侮蔑的発言が含まれている場合、「toxic」として判定してください。",
    )
)
comment = "あなたは美しい。"
toxicity(comment=comment).toxic
```

**出力結果:**
```text
False
```


### 事例A：感情分類

```python
sentence = "it's a charming and often moving journey."  # SST-2データセットからのサンプル文

classify = dspy.Predict('sentence -> sentiment: bool')  # Literal[]を使用した具体例は後ほど説明します
classify(sentence=sentence).sentiment
```
**出力結果:**
```text
True
```

### 事例B：要約

```python
# XSumデータセットからの具体例
document = """21歳のこの選手はハマーズで7試合に出場し、昨シーズンのヨーロッパリーグ予選ラウンドでアンドラのFCルストランズ戦において唯一の得点を記録した。リーは前シーズン、リーグ1のブラックプールとコルチェスター・ユナイテッドでそれぞれ2回の期限付き移籍を経験している。コルチェスターでは2得点を挙げたものの、チームの降格を阻止することはできなかった。昇格組であるタイクスとのリーの契約期間については現時点で明らかにされていない。最新のサッカー移籍情報については、専用ページを参照されたい。"""

summarize = dspy.ChainOfThought('document -> summary')
response = summarize(document=document)

print(response.summary)
```
**想定される出力例:**
```text
21歳の李選手は昨季、ウェストハム・ユナイテッドで7試合に出場し1得点を記録した。リーグ1のブラックプールとコルチェスター・ユナイテッドへの期限付き移籍も経験しており、後者では2得点を挙げている。現在はバーンズリーFCと契約を結んだが、契約期間の詳細は公表されていない。
```

多くの DSPy モジュール（`dspy.Predict` を除く）は、内部処理としてシグネチャを拡張することで補助的な情報を返す仕組みを備えています。

具体例として、`dspy.ChainOfThought` モジュールでは、出力 `summary` を生成する前に言語モデルが辿った推論過程を含む `reasoning` フィールドが追加されます。

```python
print("推論過程:", response.reasoning)
```
**想定される出力例:**
```text
考察：本論文では、リーがウェストハム・ユナイテッドで示したパフォーマンス、リーグ1への期限付き移籍期間における活躍、およびバーンズリーFCと締結した新契約について重点的に論じる必要がある。なお、契約期間の詳細は明らかにされていない点にも言及すべきである。
```

## **クラスベース** DSPy シグネチャ

高度なタスクを実行する場合、より詳細なシグネチャが必要となる場合があります。これは主に以下の目的のために使用されます：

1. タスクの性質に関する詳細情報を明確に記述する（以下の `docstring` として表現）

2. `dspy.InputField` の `desc` キーワード引数として入力フィールドの性質に関するヒントを提供する

3. `dspy.OutputField` の `desc` キーワード引数として出力フィールドに対する制約条件を指定する

### 使用例 C：分類タスク

```python
from typing import Literal

class Emotion(dspy.Signature):
    """感情分類クラス"""
    
    sentence: str = dspy.InputField()
    sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] = dspy.OutputField()

sentence = "i started feeling a little vulnerable when the giant spotlight started blinding me"  # 出典: dair-ai/emotion

classify = dspy.Predict(Emotion)
classify(sentence=sentence)
```
**想定される出力例:**
```text
Prediction(
    sentiment='fear'
)
```

**アドバイス:** 言語モデルに対して要求をより明確に指定することに問題はありません。クラスベース署名はこの目的に有効です。ただし、手動で署名のキーワードを過度に調整するのは避けるべきです。DSPyの最適化アルゴリズムの方が優れた結果をもたらす可能性が高く（さらに、異なる言語モデル間での転移性も向上します）。

### 事例D：引用への忠実性を評価する指標

```python
class CheckCitationFaithfulness(dspy.Signature):
    """提供された文脈に基づいてテキストが作成されていることを検証します。"""

    context: str = dspy.InputField(desc="ここに記載される事実は真であると仮定されます")
    text: str = dspy.InputField()
    faithfulness: bool = dspy.OutputField()
    evidence: dict[str, list[str]] = dspy.OutputField(desc="主張を裏付ける証拠")

context = "21歳の選手は昨シーズン、ハマーズで7試合に出場し、UEFAヨーロッパリーグ予選ラウンドでアンドラのFC Lustrains戦において唯一の得点を挙げました。リーは前シーズン、リーグ1のブラックプールとコルチェスター・ユナイテッドでそれぞれ1回ずつの期限付き移籍を経験しています。コルチェスターでは2得点を記録したものの、チームの降格を防ぐことはできませんでした。昇格組であるタイクスとのリーの契約期間については公表されていません。最新のサッカー移籍情報はすべて専用ページでご確認いただけます。"

text = "リーはコルチェスター・ユナイテッドで3得点を記録した。"

faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
faithfulness(context=context, text=text)
```
**想定される出力例:**
```text
予測結果：
    推論過程：「主張内容を文脈と照合する。本文ではリー・アダムスがコルチェスター・ユナイテッド戦で3得点を挙げたと記載されているが、文脈では『彼はU's（コルチェスター・ユナイテッドの愛称）戦で2得点を記録した』と明確に記述されている。これは明白な矛盾である。」
    忠実性評価：偽
    根拠証拠：{'得点数': ["2得点を記録した"]}
```

### 事例 E：マルチモーダル画像分類

```python
class DogPictureSignature(dspy.Signature):
    """画像に写っている犬の犬種を出力します。"""
    image_1: dspy.Image = dspy.InputField(desc="犬の画像")
    answer: str = dspy.OutputField(desc="画像に写っている犬の犬種")

image_url = "https://picsum.photos/id/237/200/300"
classify = dspy.Predict(DogPictureSignature)
classify(image_1=dspy.Image.from_url(image_url))
```

**想定される出力例:**

```text
Prediction(
    answer='ラブラドール・レトリーバー'
)
```

## シグネチャにおける型解決

DSPyのシグネチャでは、以下の多様なアノテーション型をサポートしています：

1. **基本型**：`str`、`int`、`bool` など
2. **型付けモジュール型**：`list[str]`、`dict[str, int]`、`Optional[float]`、`Union[str, int]` など
3. **カスタム型**：ユーザーコード内で定義された型
4. **ドット表記**：適切な設定を行うことで、入れ子構造の型を表現可能
5. **特殊データ型**：`dspy.Image`、`dspy.History` などの専用データ型

### カスタム型の取り扱い

```python
# シンプルなカスタム型定義
class QueryResult(pydantic.BaseModel):
    text: str  # クエリ結果のテキストデータ
    score: float  # クエリ結果のスコア値

signature = dspy.Signature("query: str -> result: QueryResult")

class MyContainer:
    class Query(pydantic.BaseModel):
        text: str  # クエリオブジェクトのテキストデータ

    class Score(pydantic.BaseModel):
        score: float  # クエリスコアオブジェクトのスコア値

signature = dspy.Signature("query: MyContainer.Query -> score: MyContainer.Score")
```

## 署名を用いたモジュール構築とコンパイル方法

署名は構造化された入力/出力を伴うプロトタイピングにおいて非常に便利ですが、これが唯一の用途ではありません！

複数の署名を組み合わせてより大きな[DSPyモジュール](modules.md)を構築し、さらにこれらのモジュールを[最適化されたプロンプト](../optimization/optimizers.md)やファインチューニング用にコンパイルすることが可能です。
