---
sidebar_position: 1
---

!!! warning "本ページの情報は古くなっており（DSPy 2.5では完全ではない可能性があります）"

# DSPyの使用例

DSPyを使用する際には、学習データセット、開発データセット、およびテストデータセットを扱います。これは従来の機械学習と同様の概念ですが、DSPyを効果的に活用するためには、通常必要となるラベル数が大幅に少ない（あるいは全く必要ない）点が特徴です。

DSPyにおけるデータの基本データ型は`Example`です。学習データセットおよびテストデータセット内の個々のデータを表現するために、**Exampleオブジェクト**を使用します。

DSPyの**Exampleオブジェクト**はPythonの`dict`と類似していますが、いくつかの有用な拡張機能を備えています。DSPyモジュールが返す値の型は`Prediction`であり、これは`Example`の特殊なサブクラスとして定義されています。

## `Example`オブジェクトの作成方法

DSPyを使用する際には、多くの評価実行や最適化処理を行うことになります。個々のデータポイントはすべて`Example`型として扱われます：

```python
qa_pair = dspy.Example(question="これは質問ですか？", answer="これは回答です。")

print(qa_pair)
print(qa_pair.question)
print(qa_pair.answer)
```
**出力結果:**
```text
Example({'question': 'これは質問ですか？', 'answer': 'これは回答です。'}) (input_keys=None)
これは質問ですか？
これは回答です。
```

具体例では、任意のフィールドキーと値の型を使用できるが、通常は値が文字列である場合が多い。

```text
object = Example(field1=value1, field2=value2, field3=value3, ...)
```

## 入力キーの指定方法

従来の機械学習（ML）フレームワークでは、「入力データ」と「ラベル」は明確に分離されている。

DSPyでは、`Example`オブジェクトに`with_inputs()`メソッドが用意されており、特定のフィールドを入力データとして指定することが可能である（それ以外のフィールドは単なるメタデータまたはラベルとして扱われる）。

```python
# 単一入力の場合
print(qa_pair.with_inputs("question"))

# 複数入力の場合; 意図がない限り、ラベルを明示的に入力としてマークしないよう注意してください
print(qa_pair.with_inputs("question", "answer"))
```

この柔軟性により、`Example`オブジェクトを様々なDSPyシナリオに合わせてカスタマイズすることが可能です。

`with_inputs()`メソッドを呼び出すと、元の`Example`オブジェクトの新しいコピーが作成されます。元のオブジェクト自体は変更されずに保持されます。


## 要素へのアクセスと更新

値は`.`（ドット）演算子を使用してアクセスできます。定義済みオブジェクト`Example(name="John Doe", job="sleep")`において、キー`name`の値にアクセスする場合、`object.name`と記述します。

特定のキーの値を取得または除外するには、`inputs()`メソッドと`labels()`メソッドを使用して、それぞれ入力キーのみまたは非入力キーのみを含む新しい`Example`オブジェクトを取得してください。

```python
article_summary = dspy.Example(article= "これは記事の本文です。", summary= "これは要約文です。").with_inputs("article")

input_key_only = article_summary.inputs()
non_input_key_only = article_summary.labels()

print("入力フィールドのみを含む例オブジェクト:", input_key_only)
print("非入力フィールドのみを含む例オブジェクト:", non_input_key_only)
```

**出力結果**
```
入力フィールドのみを含むサンプルオブジェクト: Example({'article': 'これは記事です。'})（input_keys=None）
非入力フィールドのみを含むサンプルオブジェクト: Example({'summary': 'これは要約です。'})（input_keys=None）
```

キーを除外する場合は `without()` メソッドを使用してください：

```python
article_summary = dspy.Example(context="これは記事です。", question="これは質問ですか？", answer="これは回答です。", rationale="これは根拠です。").with_inputs("context", "question")

print("回答および根拠キーを除外したExampleオブジェクト:", article_summary.without("answer", "rationale"))
```

**出力結果**
```
回答キーおよび根拠キーを含まないサンプルオブジェクトの例: Example({'context': 'これは記事です。', 'question': 'これは質問ですか？'})（input_keys=None）
```

値の更新は、`.`演算子を用いて新しい値を代入する操作に他ならない。

```python
article_summary.context = "新しいコンテキスト"
```

## 例オブジェクトの反復処理

`Example`クラスにおける反復処理は辞書と同様の機能を持ち、`keys()`、`values()`などの辞書操作メソッドをサポートします： 

```python
for k, v in article_summary.items():
    print(f"{k} = {v}")
```

**出力結果**

```text
context = This is an article.
question = This is a question?
answer = This is an answer.
rationale = This is a rationale.
```
