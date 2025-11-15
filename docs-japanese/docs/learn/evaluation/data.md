---
sidebar_position: 5
---

# データ

DSPyは機械学習フレームワークであるため、その操作には訓練データセット、開発データセット、およびテストデータセットが関与します。各データサンプルについて、通常以下の3種類の値を区別します：入力データ、中間ラベル、および最終ラベルです。DSPyを使用する際に中間ラベルや最終ラベルがなくても問題ありませんが、少なくともいくつかの入力データサンプルは必要となります。


## DSPyの`Example`オブジェクト

DSPyにおけるデータの基本データ型は`Example`です。訓練データセットおよびテストデータセットの各要素を表現するために、**Exampleオブジェクト**を使用します。

DSPyの**Exampleオブジェクト**はPythonの`dict`と類似していますが、いくつかの有用な拡張機能を備えています。DSPyモジュールが返す値は`Prediction`型であり、これは`Example`クラスの特殊なサブクラスとして定義されています。

DSPyを使用する際には、多くの評価および最適化処理を実行することになります。個々のデータポイントはすべて`Example`型となります：

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

訓練データセットは以下のように表現できます：

```python
trainset = [dspy.Example(report="LONG REPORT 1", summary="short summary 1"), ...]
```


### 入力キーの指定方法

従来の機械学習（ML）フレームワークでは、「入力データ」と「ラベル」は明確に分離されています。

DSPyでは、`Example`オブジェクトに`with_inputs()`メソッドが用意されており、特定のフィールドを入力データとして指定することが可能です（それ以外のフィールドはメタデータまたはラベルとして扱われます）。

```python
# 単一入力の場合
print(qa_pair.with_inputs("question"))

# 複数入力の場合; 意図がない限り、ラベルを明示的に入力としてマークしないよう注意してください
print(qa_pair.with_inputs("question", "answer"))
```

値へのアクセスには `.`（ドット）演算子を使用します。定義済みオブジェクト `Example(name="John Doe", job="sleep")` においてキー `name` の値を取得するには、`object.name` と記述します。

特定のキーへのアクセスまたは除外を行う場合は、`inputs()` および `labels()` メソッドを使用して、それぞれ入力キーのみまたは非入力キーのみを含む新しい Example オブジェクトを取得してください。

```python
article_summary = dspy.Example(article= "これは記事の本文です。", summary= "これは要約文です。").with_inputs("article")

input_key_only = article_summary.inputs()
non_input_key_only = article_summary.labels()

print("入力フィールドのみを含む例オブジェクト:", input_key_only)
print("非入力フィールドのみを含む例オブジェクト:", non_input_key_only)
```

**出力結果**
```
入力フィールドのみを含むサンプルオブジェクト: Example({'article': 'これは記事です。'})（input_keys={'article'}）
入力フィールド以外のフィールドのみを含むサンプルオブジェクト: Example({'summary': 'これは要約です。'})（input_keys=None）
```

<!-- ## データセットの読み込み方法

DSPyでデータセットをインポートする最も便利な方法の一つが`DataLoader`の使用です。最初のステップとしてオブジェクトを宣言し、このオブジェクトを用いて各種データ形式に対応したユーティリティ関数を呼び出すことでデータセットを読み込めます：

```python
from dspy.datasets import DataLoader

dl = DataLoader()
```

ほとんどのデータ形式において、対応するファイルパスを適切なメソッドに渡すだけで、そのデータセットの`Example`リストを取得できます：

```python
import pandas as pd

csv_dataset = dl.from_csv(
    "sample_dataset.csv",
    fields=("instruction", "context", "response"),
    input_keys=("instruction", "context")
)

json_dataset = dl.from_json(
    "sample_dataset.json",
    fields=("instruction", "context", "response"),
    input_keys=("instruction", "context")
)

parquet_dataset = dl.from_parquet(
    "sample_dataset.parquet",
    fields=("instruction", "context", "response"),
    input_keys=("instruction", "context")
)

pandas_dataset = dl.from_pandas(
    pd.read_csv("sample_dataset.csv"),    # DataFrame形式
    fields=("instruction", "context", "response"),
    input_keys=("instruction", "context")
)
```

これらは`DataLoader`が直接ファイルから読み込めるデータ形式の一例です。内部的にはこれらのメソッドの多くが`datasets`ライブラリの`load_dataset`メソッドを利用して各形式のデータセットを読み込みます。ただし、テキストデータを扱う場合、HuggingFaceのデータセットを使用することが一般的です。HuggingFaceデータセットを`Example`形式のリストとしてインポートするには、`from_huggingface`メソッドを使用します：

```python
blog_alpaca = dl.from_huggingface(
    "intertwine-expel/expel-blog",
    input_keys=("title",)
)
```

データセットの分割データにアクセスするには、対応するキーを参照します：

```python
train_split = blog_alpaca['train']

# このデータセットには分割データが1つしかないため、手動で訓練用とテスト用に分割できます。
# 例えば、訓練データから75行をサンプリングしてテストセットを作成します。
testset = train_split[:75]
trainset = train_split[75:]
```

`load_dataset` を使用して HuggingFace データセットを読み込む方法は、`from_huggingface` 経由でデータを読み込む場合と完全に同一です。具体的には、特定の分割データ（split）、サブ分割データ（subsplit）、読み込み指示（read instructions）などを指定する場合も同様です。コード例については、HF からのデータ読み込みに関する [チートシートのコードスニペット](/cheatsheet/#dspy-dataloaders) を参照してください。 -->