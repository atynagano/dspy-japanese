---
sidebar_position: 2
---

!!! warning "本ページの情報は古くなっており（DSPy 2.5では完全ではない可能性があります）"

# 組み込みデータセットの活用

DSPyで独自のデータを使用するのは簡単です。データセットは単に`Example`オブジェクトのリストとして定義されます。DSPyを効果的に活用するには、既存のデータセットを見つけ出し、それらを自身の処理パイプラインに合わせて新たな用途で再利用する能力が不可欠です。DSPyはこの戦略を特に強力にサポートする設計となっています。

利便性を考慮し、DSPyでは現在以下のデータセットを標準でサポートしています：

* **HotPotQA**（多段階推論型質問応答）
* **GSM8k**（数学問題データセット）
* **Color**（基本色データセット）


## HotPotQAの読み込み方法

HotPotQAは、質問と回答のペアからなるデータセットです。

```python
from dspy.datasets import HotPotQA

dataset = HotPotQA(train_seed=1, train_size=5, eval_seed=2023, dev_size=50, test_size=0)

print(dataset.train)
```
**出力結果:**
```text
[Example({'question': '『窓辺にて』を発表したアメリカのシンガーソングライターは誰か？', 'answer': 'ジョン・タウンズ・ヴァン・ザント'}) (input_keys=None),
 Example({'question': 'キャンディス・キタがゲスト出演したアメリカの俳優は誰か？', 'answer': 'ビル・マーレイ'}) (input_keys=None),
 Example({'question': 「Who Put the Bomp」と「Self」のうち、より最近発行されたのはどの出版物か？', 'answer': 'Self'}) (input_keys=None),
 Example({'question': 『ヴィクトリア朝時代 - 写真で綴るその歴史』の著者が生まれた年は何年か？', 'answer': '1950年'}) (input_keys=None),
 Example({'question': スコット・ショウの記事が掲載された雑誌は、『Tae Kwon Do Times』と『Southwest Art』のどちらか？', 'answer': 'Tae Kwon Do Times'}) (input_keys=None)]
```

訓練データセット（5例）と開発データセット（50例）を読み込んだ。各訓練データ例は、質問文とその（人間による注釈付き）回答のみで構成されている。ご覧の通り、これらのデータは`Example`オブジェクトのリストとして読み込まれている。ただし、注意すべき点として、入力キーが自動的に設定されるわけではないため、この点については手動で設定する必要がある！

```python
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]

print(trainset)
```
**出力結果:**
```text
[例({'question': '『窓辺にて』を発表したアメリカのシンガーソングライターは誰か？', 'answer': 'ジョン・タウンズ・ヴァン・ザント'}) (入力キー={'question'}),
 例({'question': 'キャンディス・キタがゲスト出演したアメリカの俳優は誰か？', 'answer': 'ビル・マーレイ'}) (入力キー={'question'}),
 例({'question': '『Who Put the Bomp』と『Self』のうち、より最近出版されたのはどれか？', 'answer': 'Self'}) (入力キー={'question'}),
 例({'question': '『ヴィクトリア朝時代 - 写真で綴るその歴史』の著者が生まれた年は何年か？', 'answer': '1950年'}) (入力キー={'question'}),
 例({'question': 'スコット・ショウの記事が掲載された雑誌は、『Tae Kwon Do Times』と『Southwest Art』のどちらか？', 'answer': 'Tae Kwon Do Times'}) (入力キー={'question'})]
```

DSPyでは通常、最小限のラベル付けしか必要としません。一般的なパイプラインでは6～7段階の複雑な処理が必要になる場合がありますが、DSPyでは基本的に最初の質問文と最終的な回答文にのみラベルを付与すれば十分です。DSPyはパイプラインの実行に必要な中間ラベルを自動的に生成します。パイプラインの構成を変更した場合、それに応じてデータのブートストラップ処理も自動的に更新されます！


## 上級編：DSPyの`Dataset`クラスの内部構造（オプション）

`HotPotQA`データセットクラスの使用方法とHotPotQAデータセットの読み込み方法について説明してきましたが、実際にはどのように動作しているのでしょうか？`HotPotQA`クラスは`Dataset`クラスを継承しており、このクラスがデータソースから読み込んだデータを訓練用・テスト用・開発用の分割データ（すべて*例のリスト*として扱われます）に変換する処理を担当します。`HotPotQA`クラスでは、`__init__`メソッドを実装するだけでよく、ここではHuggingFaceから取得したデータを`_train`、`_test`、`_dev`という変数に割り当てます。それ以外の処理はすべて`Dataset`クラスのメソッドによって自動的に行われます。

![HotPotQAクラスにおけるデータ読み込み処理の流れ](./img/data-loading.png)

では、`Dataset`クラスのメソッドはどのようにしてHuggingFaceからのデータを変換しているのでしょうか？深呼吸して段階的に考えてみましょう...意図したダジャレです。上記の例では、`.train`、`.dev`、`.test`メソッドによってアクセスされるデータ分割が確認できるので、`train()`メソッドの実装内容を詳しく見てみましょう：

```python
@property
def train(self):
    if not hasattr(self, '_train_'):
        self._train_ = self._shuffle_and_sample('train', self._train, self.train_size, self.train_seed)

    return self._train_
```

ご覧の通り、`train()` メソッドは通常のメソッドではなくプロパティとして機能します。このプロパティ内では、まず `_train_` 属性が存在するかどうかを確認します。存在しない場合は、`_shuffle_and_sample()` メソッドを呼び出して、HuggingFace データセットが読み込まれている `self._train` を処理します。次に、`_shuffle_and_sample()` メソッドの内容を見てみましょう：

```python
def _shuffle_and_sample(self, split, data, size, seed=0):
    data = list(data)
    base_rng = random.Random(seed)

    if self.do_shuffle:
        base_rng.shuffle(data)

    data = data[:size]
    output = []

    for example in data:
        output.append(Example(**example, dspy_uuid=str(uuid.uuid4()), dspy_split=split))
    
        return output
```

`_shuffle_and_sample()` メソッドは2つの主要な処理を行います：

* `self.do_shuffle` が True に設定されている場合、入力データをシャッフルします。
* シャッフル後のデータから指定されたサイズ `size` のサンプルを抽出します。
* 抽出したサンプルデータに対してループ処理を行い、`data` 内の各要素を `Example` オブジェクトに変換します。この `Example` オブジェクトには、対応するサンプルデータとともに、一意の識別子（ID）と分割名（split name）が含まれています。

生の例データを `Example` オブジェクトに変換することで、Dataset クラスはこれらのデータを後処理段階で統一的な方法で扱えるようになります。例えば、PyTorch DataLoader で使用される collate メソッドは、各アイテムが `Example` オブジェクトであることを前提としています。

要約すると、`Dataset` クラスは必要なすべてのデータ処理を内部で処理し、異なる分割データにアクセスするためのシンプルな API を提供します。これは、生データの読み込み方法のみを定義すればよい HotpotQA などのデータセットクラスとは異なる特徴です。
