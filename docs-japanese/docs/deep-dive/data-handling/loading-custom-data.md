---
sidebar_position: 3
---

!!! warning "本ページの情報は古くなっており（DSPy 2.5では）完全に正確ではない可能性があります"

# カスタムデータセットの作成方法

`Example`オブジェクトの操作方法と、`HotPotQA`クラスを使用してHuggingFaceのHotPotQAデータセットを`Example`オブジェクトのリストとして読み込む方法については既に説明しました。しかし実際の業務環境では、このような構造化されたデータセットは稀です。むしろ独自のカスタムデータセットを扱う場面が多く、「どのように独自のデータセットを作成すればよいのか」「どのような形式にすべきか」といった疑問が生じることでしょう。

DSPyにおいて、データセットは`Example`オブジェクトのリストとして定義されます。この実装方法には2つの主要なアプローチがあります：

* **推奨方法：Pythonらしいアプローチ**：Python標準のユーティリティ機能とロジックを活用する方法
* **上級手法：DSPyの`Dataset`クラスを使用する方法**

## 推奨方法：Pythonらしいアプローチ

`Example`オブジェクトのリストを作成するには、ソースデータを読み込んでPythonのリスト形式に整形するだけで十分です。ここでは例として、3つのフィールド（**コンテキスト**、**質問**、**要約**）を含むサンプルCSVファイル`sample.csv`をPandasで読み込みます。その後、このデータを用いてデータリストを構築します。

```python
import pandas as pd

df = pd.read_csv("sample.csv")
print(df.shape)
```
**出力結果:**
```text
(1000, 3)
```
```python
dataset = []

for context, question, answer in df.values:
    dataset.append(dspy.Example(context=context, question=question, answer=answer).with_inputs("context", "question"))

print(dataset[:3])
```
**出力結果:**
```python
[Example({'context': nan, 'question': '次のうち魚の一種はどれですか？ トペまたはロープ', 'answer': 'トペ'}) (input_keys={'question', 'context'}),
 Example({'context': nan, 'question': 'ラクダが長期間水なしで生存できるのはなぜですか？', 'answer': 'ラクダは背中のこぶに蓄えた脂肪をエネルギー源および水分源として利用することで、長期間の生存を可能にしている。'}) (input_keys={'question', 'context'}),
 Example({'context': nan, 'question': "アリスの両親には3人の娘がいます：エイミー、ジェシー、そして3人目の娘の名前は何ですか？", "answer": '3人目の娘の名前はアリスである'}) (input_keys={'question', 'context'})]
```

この処理は比較的単純ですが、DSPyにおけるデータセットの読み込み方法を、DSPythonicなアプローチで具体的に見ていきましょう。

## 上級編：DSPyの`Dataset`クラスの活用（オプション）

前回の記事で定義した`Dataset`クラスを活用して、前処理を実装します：

* CSVファイルからデータをデータフレームに読み込む
* データを訓練用、開発用、テスト用に分割する
* `_train`、`_dev`、`_test`というクラス属性にデータを格納する。これらの属性は、HuggingFace Datasetのようなマッピングオブジェクトのリストまたはイテレータ形式である必要があります（これにより正しく機能します）。

これらの処理はすべて`__init__`メソッド内で実装します。このメソッドは、実際に実装が必要な唯一のメソッドです。

```python
import pandas as pd
from dspy.datasets.dataset import Dataset

class CSVDataset(Dataset):
    def __init__(self, file_path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        df = pd.read_csv(file_path)
        self._train = df.iloc[0:700].to_dict(orient='records')

        self._dev = df.iloc[700:].to_dict(orient='records')

dataset = CSVDataset("sample.csv")
print(dataset.train[:3])
```
**出力結果:**
```text
[例({'context': nan, 'question': '以下のうち魚の種類に該当するのはどれか？ トペまたはロープ', 'answer': 'トペ'}) (入力キー={'question', 'context'}),
 例({'context': nan, 'question': 'ラクダが長期間水なしで生存できるのはなぜか？', 'answer': 'ラクダは背中の脂肪を利用して、長期間にわたってエネルギーと水分を保持している'}) (入力キー={'question', 'context'}),
 例({'context': nan, 'question': "アリスの両親には3人の娘がいる：エイミー、ジェシー、そして3人目の娘の名前は何か？", "answer': '3人目の娘の名前はアリスである'}) (入力キー={'question', 'context'})]
```

コードの動作を段階的に解説します：

* DSPyの基本`Dataset`クラスを継承しています。これにより、データの読み込み/処理に関する有用な機能がすべて利用可能になります。
* CSV形式のデータをDataFrameに読み込みます。
* **訓練データ**としてDataFrameの先頭700行を取得し、`to_dict(orient='records')`メソッドを用いて辞書型リストに変換した後、`self._train`に代入します。
* **開発データ**としてDataFrameの先頭300行を取得し、同様に辞書型リストに変換した後、`self._dev`に代入します。

このDatasetベースクラスを使用することで、カスタムデータセットの読み込みが非常に容易になり、新しいデータセットごとに毎回定型的なコードを記述する必要がなくなります。

!!! 注意

    上記コードでは`_test`属性を初期化していませんが、これは問題を引き起こしません。ただし、テスト分割データにアクセスしようとするとエラーが発生します。

    ```python
    dataset.test[:5]
    ```
    ****
    ```text
    ---------------------------------------------------------------------------
    AttributeError                            Traceback (most recent call last)
    <ipython-input-59-5202f6de3c7b> in <cell line: 1>()
    ----> 1 dataset.test[:5]

    /usr/local/lib/python3.10/dist-packages/dspy/datasets/dataset.py in test(self)
        51     def test(self):
        52         if not hasattr(self, '_test_'):
    ---> 53             self._test_ = self._shuffle_and_sample('test', self._test, self.test_size, self.test_seed)
        54 
        55         return self._test_

    AttributeError: 'CSVDataset' object has no attribute '_test'
    ```

    この問題を回避するには、`_test`が`None`ではなく、適切なデータで初期化されていることを確認するだけで十分です。

`Dataset`クラスのメソッドをオーバーライドすることで、さらにクラスをカスタマイズすることも可能です。 

要約すると、Dataset 基底クラスは、最小限のコードでカスタムデータセットを読み込み・前処理するための簡潔な方法を提供する！
