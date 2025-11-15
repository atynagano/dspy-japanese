# チュートリアル：DSPyプログラムの保存と読み込み方法

本ガイドでは、DSPyプログラムの保存と読み込み方法について詳しく解説します。大別すると、DSPyプログラムを保存する方法は以下の2通りがあります：

1. PyTorchにおける重みのみの保存と同様に、プログラムの状態のみを保存する方法
2. アーキテクチャと状態の両方を含むプログラム全体を保存する方法（`dspy>=2.6.0`でサポート）

## 状態のみの保存

プログラムの状態とは、DSPyプログラムの内部状態を指し、具体的には以下の要素を含みます：
- プログラムのシグネチャ
- デモ用データセット（few-shot examples）
- 各`dspy.Predict`呼び出しで使用する`lm`（言語モデル）などの関連情報
- `dspy.retrievers.Retriever`の`k`パラメータなど、他のDSPyモジュールの設定可能な属性

プログラムの状態を保存するには、`save`メソッドを使用し、`save_program=False`を指定します。状態の保存先としては、JSONファイルまたはpickleファイルのいずれかを選択できます。安全性と可読性の観点から、JSONファイルへの保存を推奨します。ただし、プログラム内に`dspy.Image`や`datetime.datetime`など、シリアライズできないオブジェクトが含まれている場合は、pickleファイルへの保存が必要となります。

例えば、何らかのデータを用いてコンパイルしたプログラムを、将来の利用に備えて保存したい場合の手順は以下の通りです：

```python
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

gsm8k = GSM8K()
gsm8k_trainset = gsm8k.train[:10]
dspy_program = dspy.ChainOfThought("question -> answer")

optimizer = dspy.BootstrapFewShot(metric=gsm8k_metric, max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=5)
compiled_dspy_program = optimizer.compile(dspy_program, trainset=gsm8k_trainset)
```

プログラムの状態をJSONファイルに保存するには：

```python
compiled_dspy_program.save("./dspy_program/program.json", save_program=False)
```

プログラムの状態をpickleファイルに保存するには：

!!! danger "セキュリティ警告：pickleファイルは任意のコードを実行可能"
    `.pkl`ファイルを読み込む際には、任意のコードが実行される可能性があり、危険を伴う場合があります。信頼できるソースからのファイルのみを、安全な環境下で読み込むようにしてください。**可能な限り`.json`ファイルの使用を推奨します**。pickleファイルを使用する必要がある場合は、必ずソースの信頼性を確認し、読み込み時に`allow_pickle=True`パラメータを設定してください。

```python
compiled_dspy_program.save("./dspy_program/program.pkl", save_program=False)
```

保存した状態を読み込むには、**同じプログラムを再構築**した後、`load` メソッドを使用して状態を読み込む必要があります。

```python
loaded_dspy_program = dspy.ChainOfThought("question -> answer") # 同一のプログラムを再作成します。
loaded_dspy_program.load("./dspy_program/program.json")

assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
    # ロードされたデモは辞書型であるのに対し、元のデモは dspy.Example 型です。
    assert original_demo.toDict() == loaded_demo
assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)
```

または、pickleファイルから状態を読み込むこともできます：

!!! danger "セキュリティ警告"
    pickleファイルを読み込む際は必ず`allow_pickle=True`を指定し、信頼できるソースからのみ読み込むようにしてください。

```python
loaded_dspy_program = dspy.ChainOfThought("question -> answer") # 同一のプログラムを再作成します。
loaded_dspy_program.load("./dspy_program/program.pkl", allow_pickle=True)

assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
    # ロードされたデモは辞書型であるのに対し、元のデモは dspy.Example 型です。
    assert original_demo.toDict() == loaded_demo
assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)
```

## プログラム全体の保存機能

!!! warning "セキュリティに関する重要な通知: プログラム全体の保存には pickle 形式を使用します"
    プログラム全体の保存には `cloudpickle` によるシリアライゼーションを採用していますが、これは pickle ファイルと同様のセキュリティリスクを伴います。安全な環境下で、信頼できるソースからのみプログラムを読み込むようにしてください。

`dspy>=2.6.0` バージョン以降、DSPy ではプログラム全体（アーキテクチャ情報および状態情報を含む）の保存が可能になりました。この機能は `cloudpickle` ライブラリによって実現されており、Python オブジェクトのシリアライズ/デシリアライズを行うための専用ツールです。

プログラム全体の保存を行うには、`save` メソッドを使用し、`save_program=True` を指定した上で、プログラムを保存するディレクトリパスを指定してください。ファイル名ではなくディレクトリパスを必要とする理由は、プログラム本体に加えて、依存関係のバージョン情報などのメタデータも同時に保存するためです。

```python
compiled_dspy_program.save("./dspy_program/", save_program=True)
```

保存済みプログラムを読み込むには、`dspy.load` メソッドを直接使用します：

```python
loaded_dspy_program = dspy.load("./dspy_program/")

assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
    # ロードされたデモは辞書型であるのに対し、元のデモは dspy.Example オブジェクトです。
    assert original_demo.toDict() == loaded_demo
assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)
```

全プログラム保存機能を使用すると、プログラムを再作成する必要はなく、状態情報とともにアーキテクチャを直接読み込むことができます。
ご自身のニーズに応じて、適切な保存方法を選択してください。

### インポート済みモジュールのシリアライズ処理

`save_program=True` を指定してプログラムを保存する場合、プログラムが依存しているカスタムモジュールを含める必要がある場合があります。
これは、プログラムがこれらのモジュールに依存している場合に必須となりますが、`dspy.load` を呼び出す時点では、これらのモジュールはインポートされていない状態です。

`save` 関数を呼び出す際に `modules_to_serialize` パラメータを指定することで、プログラムと一緒にシリアライズすべきカスタムモジュールを指定できます。
これにより、プログラムが依存しているすべての外部モジュールがシリアライズ処理に含まれ、後でプログラムを読み込む際に利用可能となります。

内部的には、cloudpickle の `cloudpickle.register_pickle_by_value` 関数を使用して、モジュールを値ベースでシリアライズ可能なものとして登録しています。
このようにモジュールを登録すると、cloudpickle は参照ではなく値ベースでモジュールをシリアライズするため、保存されたプログラムと共にモジュールの内容が確実に保持されます。

例えば、プログラムが以下のカスタムモジュールを使用している場合：

```python
import dspy
import my_custom_module

compiled_dspy_program = dspy.ChainOfThought(my_custom_module.custom_signature)

# カスタムモジュールを含むプログラムを保存
compiled_dspy_program.save(
    "./dspy_program/",
    save_program=True,
    modules_to_serialize=[my_custom_module]
)
```

これにより、必要なモジュールが正しくシリアライズされ、プログラムを後でロードする際に確実に利用可能となります。`modules_to_serialize`には任意の数のモジュールを指定可能です。`modules_to_serialize`を指定しない場合、追加のモジュールはシリアライズ用に登録されません。

## 後方互換性について

`dspy<3.0.0`以降では、保存済みプログラムの後方互換性を保証していません。例えば、`dspy==2.5.35`でプログラムを保存した場合、ロード時には必ず同じバージョンのDSPyを使用してください。そうしないと、プログラムが期待通りに動作しない可能性があります。異なるバージョンのDSPyで保存済みファイルをロードしてもエラーは発生しない場合がありますが、パフォーマンスが保存時と異なる可能性があります。

`dspy>=3.0.0`以降では、メジャーリリース間で保存済みプログラムの後方互換性を保証します。つまり、`dspy==3.0.0`で保存したプログラムは`dspy==3.7.10`でもロード可能であることを保証します。
