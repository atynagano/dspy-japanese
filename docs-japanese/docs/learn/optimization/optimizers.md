---
sidebar_position: 1
---

# DSPyオプティマイザ（旧称：テレプロンプター）


**DSPyオプティマイザ**とは、DSPyプログラム（プロンプトおよび/または言語モデルの重み）のパラメータを調整し、精度などの指定指標を最大化するアルゴリズムです。


一般的なDSPyオプティマイザでは、以下の3つの要素が必要です：

- **DSPyプログラム**。これは単一モジュール（例：`dspy.Predict`）でも、複数モジュールから構成される複雑なプログラムでも構いません。

- **評価指標**。これはプログラムの出力を評価し、スコアを付与する関数です（スコアが高いほど望ましい結果となります）。

- 少量の**学習用入力データ**。これは非常に小規模（例：5～10例）かつ不完全なもの（プログラムへの入力データのみで、ラベル情報は含まない）でも構いません。

大量のデータがある場合、DSPyはそれを活用できます。ただし、小規模なデータからでも十分な結果を得ることが可能です。

**注記：** 旧称は「テレプロンプター」でした。現在、公式名称の変更を実施しており、この変更はライブラリ全体およびドキュメントに反映されます。


## DSPyオプティマイザは何を調整し、どのように調整するのか？

DSPyに搭載されている各種オプティマイザは、異なる方法でプログラムの品質向上を図ります：
- **`dspy.BootstrapRS`<sup>[1](https://arxiv.org/abs/2310.03714)</sup>**のようなモジュールに対して、**高品質なfew-shot例を合成**することでプログラムの品質を向上させます。
- **`dspy.MIPROv2`<sup>[2](https://arxiv.org/abs/2406.11695)</sup>**や**`dspy.GEPA`<sup>[3](https://arxiv.org/abs/2507.19457)</sup>**のように、各プロンプトに対して**より適切な自然言語指示を提案**し、それらを知的に探索することで品質を向上させます。
- **`dspy.BootstrapFinetune`<sup>[4](https://arxiv.org/abs/2407.10930)</sup>**のように、モジュール用のデータセットを構築し、それらを用いてシステム内の言語モデル重みをファインチューニングすることで品質を向上させます。

??? "DSPyオプティマイザの具体例と、各オプティマイザの動作原理を教えてください？"

    `dspy.MIPROv2`オプティマイザを例に説明します。まず、MIPROは**ブートストラッピング段階**から開始します。この段階では、最適化前の状態にある可能性のあるプログラムを実行し、様々な入力データに対して複数回処理を行うことで、各モジュールの入力/出力動作のトレースを収集します。これらのトレースから、指定した評価指標で高評価を得た軌跡に該当するもののみを選択的に保持します。次に、MIPROは**接地型提案段階**に移行します。ここでは、DSPyプログラムのコード、データ、およびプログラム実行時のトレース情報を分析し、プログラム内の各プロンプトに対して多数の潜在的な指示案を作成します。最後に、MIPROは**離散探索段階**を開始します。学習データセットからミニバッチをサンプリングし、各プロンプト構築に使用する指示案とトレースの組み合わせを提案します。提案されたプログラム候補をミニバッチデータで評価し、その結果に基づいてスコアを算出します。このスコア情報を活用し、MIPROは時間の経過とともに提案品質が向上するよう補助モデルを更新していきます。

    DSPyオプティマイザの強力な特徴の一つは、その組み合わせ可能性にあります。`dspy.MIPROv2`を実行して生成されたプログラムを再度`dspy.MIPROv2`の入力として使用する、あるいは`dspy.BootstrapFinetune`の入力として用いることで、より優れた結果を得ることが可能です。これは`dspy.BetterTogether`の核心的な機能の一部です。別のアプローチとして、オプティマイザを実行した後に上位5つの候補プログラムを抽出し、それらを用いて`dspy.Ensemble`を構築することもできます。これにより、推論時の計算リソース（アンサンブル処理など）とDSPy独自の事前推論計算リソース（最適化予算）の両方を、高度に体系的な方法で拡張することが可能になります。



## 現在利用可能なDSPyオプティマイザにはどのようなものがありますか？

オプティマイザは`from dspy.teleprompt import *`コマンドでアクセス可能です。

### 自動Few-Shot学習

これらのオプティマイザは、プロンプト送信時にモデルが自動的に最適化された例を生成・組み込み、Few-Shot学習を実現するように設計されています。

1. [**`LabeledFewShot`**](../../api/optimizers/LabeledFewShot.md): 提供されたラベル付き入力データと出力データポイントから、Few-Shot例（デモ）を直接構築します。プロンプトに必要な例数`k`と、`trainset`からランダムに`k`個の例を選択する機能が必要です。

2. [**`BootstrapFewShot`**](../../api/optimizers/BootstrapFewShot.md): デフォルトではユーザーのプログラムが使用される`teacher`モジュールを活用し、プログラムの各段階に対する完全なデモンストレーションを生成します。`trainset`に含まれるラベル付き例も使用します。パラメータには`max_labeled_demos`（`trainset`からランダムに選択するデモンストレーション数）と`max_bootstrapped_demos`（`teacher`によって生成される追加例数）が含まれます。ブートストラッピングプロセスでは、デモンストレーションの妥当性を検証するために評価指標を使用し、指標を満たすもののみをコンパイルされたプロンプトに含めます。上級者向け機能として、互換性のある構造を持つ異なるDSPyプログラムを`teacher`として使用することも可能です（より困難なタスクに対応）。

3. [**`BootstrapFewShotWithRandomSearch`**](../../api/optimizers/BootstrapFewShotWithRandomSearch.md): `BootstrapFewShot`を複数回実行し、生成されたデモンストレーションに対してランダムサーチを適用します。最適化プロセス全体を通じて最も優れたプログラムを選択します。パラメータは`BootstrapFewShot`と同様ですが、さらに`num_candidate_programs`が追加されており、最適化中に評価されるランダムプログラムの数を指定します。この数には、未コンパイルプログラム、`LabeledFewShot`で最適化されたプログラム、例がシャッフルされていない`BootstrapFewShot`コンパイルプログラム、およびランダム化された例セットを持つ`BootstrapFewShot`コンパイルプログラム`num_candidate_programs`個が含まれます。

4. [**`KNNFewShot`**](../../api/optimizers/KNNFewShot.md): k近傍法アルゴリズムを用いて、与えられた入力例に最も近い訓練例のデモンストレーションを特定します。これらの近傍デモンストレーションは、`BootstrapFewShot`最適化プロセスにおける訓練セットとして使用されます。


### 自動指示最適化

これらのオプティマイザは、プロンプトに対して最適な命令を生成し、MIPROv2の場合は少数ショットデモンストレーションのセットも最適化することが可能です。

5. [**`COPRO`**](../../api/optimizers/COPRO.md): 各ステップごとに新規命令を生成・改良し、座標上昇法（メトリック関数と`trainset`を用いたヒルクライミング）によって最適化を行います。パラメータとして`depth`があり、これはオプティマイザがプロンプト改善を行う反復回数を指定します。

6. [**`MIPROv2`**](../../api/optimizers/MIPROv2.md): 各ステップにおいて命令と少数ショット例の両方を生成します。命令生成はデータ特性とデモンストレーション内容を認識して行われ、ベイズ最適化を用いてモジュール間の生成命令/デモンストレーション空間を効果的に探索します。

7. [**`SIMBA`**](../../api/optimizers/SIMBA.md): 確率的ミニバッチサンプリングを用いて出力変動が大きい困難な事例を特定し、LLMを適用して失敗要因を内省的に分析した上で、自己反省的な改善ルールを生成するか、成功したデモンストレーションを追加します。

8. [**`GEPA`**](../../api/optimizers/GEPA/overview.md): LMを使用してDSPyプログラムの実行軌跡を内省し、何が効果的で何が効果的でなかったかを特定した上で、ギャップを埋めるためのプロンプトを提案します。さらに、GEPAはドメイン固有のテキストフィードバックを活用してDSPyプログラムを迅速に改善することが可能です。GEPAの詳細な使用ガイドは[dspy.GEPA Tutorials](../../tutorials/gepa_ai_program/index.md)で提供されています。

### 自動ファインチューニング

このオプティマイザは、基盤となるLLMのファインチューニングに使用されます。

9. [**`BootstrapFinetune`**](/api/optimizers/BootstrapFinetune): プロンプトベースのDSPyプログラムを重み更新に変換します。出力結果として、同じステップ構成を持ちながら、プロンプトLMではなくファインチューニング済みモデルによって各ステップが実行されるDSPyプログラムが得られます。完全な使用例については、[分類タスク向けファインチューニングチュートリアル](https://dspy.ai/tutorials/classification_finetuning/)を参照してください。


### プログラム変換

10. [**`Ensemble`**](../../api/optimizers/Ensemble.md): 複数のDSPyプログラムをアンサンブルし、全セットを使用するか、ランダムにサブセットを選択して単一のプログラムに統合します。


## どのオプティマイザを使用するべきか？

最終的に「最適な」オプティマイザの選択とタスクに最適な設定を見つけるには、試行錯誤が必要です。DSPyにおける成功は依然として反復的なプロセスであり、タスクで最高のパフォーマンスを得るためには、探索と反復を繰り返す必要があります。

ただし、開始する際の一般的な指針として以下の点を考慮してください：

- **サンプル数が非常に少ない**場合（約10例程度）は、まず `BootstrapFewShot` を使用してください。
- **より多くのデータ**がある場合（50例以上）は `BootstrapFewShotWithRandomSearch` をお試しください。
- **指示最適化のみ**を行いたい場合（つまりプロンプトを0-shotのまま維持したい場合）は、`MIPROv2` [0-shot最適化用に設定済み](../../api/optimizers/MIPROv2.md) を使用してください。
- **より長い最適化実行**（例えば40試行以上）を行うために追加の推論呼び出しを許容でき、かつ十分なデータ量（過学習を防ぐため200例以上）がある場合、`MIPROv2` の使用を検討してください。
- **大規模言語モデル（7Bパラメータ以上など）**でこれらの手法を既に使用しており、非常に**効率的なプログラム**が必要な場合には、`BootstrapFinetune` を使用してタスク専用の小規模言語モデルをファインチューニングしてください。

## 最適化器の使用方法について

これらの最適化器はすべて共通のインターフェースを採用しており、キーワード引数（ハイパーパラメータ）に若干の違いがあります。完全な一覧は [APIリファレンス](../../api/optimizers/BetterTogether.md) で確認できます。

最も一般的な `BootstrapFewShotWithRandomSearch` を例に説明しましょう。

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

# 最適化設定：プログラムの各処理ステップについて、8事例分の「ブートストラッピング」（自己生成）サンプルを作成します。
# 最適化器はこのプロセスを10回繰り返し（初期試行分を含む）、その後開発セット上で最良の結果を選択します。
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4, num_candidate_programs=10, num_threads=4)

teleprompter = BootstrapFewShotWithRandomSearch(metric=YOUR_METRIC_HERE, **config)
optimized_program = teleprompter.compile(YOUR_PROGRAM_HERE, trainset=YOUR_TRAINSET_HERE)
```


!!! info "入門編 III：DSPyプログラムにおけるLMプロンプトまたは重みの最適化"
    標準的な簡易最適化処理のコストは2米ドル程度で、実行時間は約10分です。ただし、非常に大規模な言語モデルやデータセットを使用する場合は注意が必要です。
    最適化処理のコストは、使用する言語モデル、データセット、および設定によって、数セントから数十ドルまで変動します。

    === "ReActエージェント向けプロンプトの最適化"
        これは、Wikipedia検索による質問応答を行う`dspy.ReAct`エージェントを最小限の構成で実装した完全な実行例です。`HotPotQA`データセットからサンプリングした500組の質問-回答データセットを用いて、安価な`light`モードで`dspy.MIPROv2`による最適化を行っています。

        ```python linenums="1"
        import dspy
        from dspy.datasets import HotPotQA

        dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

        def search(query: str) -> list[str]:
            """Wikipediaから抄録を取得する関数"""
            results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
            return [x['text'] for x in results]

        trainset = [x.with_inputs('question') for x in HotPotQA(train_seed=2024, train_size=500).train]
        react = dspy.ReAct("question -> answer", tools=[search])

        tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", num_threads=24)
        optimized_react = tp.compile(react, trainset=trainset)
        ```

        DSPy 2.5.29を使用した簡易的な実行例では、この手法によりReActエージェントのスコアが24%から51%に向上しました。

    === "RAGシステム向けプロンプトの最適化"
        検索インデックス`search`、お好みの`dspy.LM`、および小規模な質問-正解応答データセットが与えられている場合、以下のコードスニペットを使用して、DSPyモジュールとして実装された組み込み`dspy.SemanticF1`評価指標を用いて、長文出力に対応するRAGシステムを最適化できます。

        ```python linenums="1"
        class RAG(dspy.Module):
            def __init__(self, num_docs=5):
                self.num_docs = num_docs
                self.respond = dspy.ChainOfThought('context, question -> response')

            def forward(self, question):
                context = search(question, k=self.num_docs)   # このスニペットでは定義されていません。上記リンクを参照してください
                return self.respond(context=context, question=question)

        tp = dspy.MIPROv2(metric=dspy.SemanticF1(), auto="medium", num_threads=24)
        optimized_rag = tp.compile(RAG(), trainset=trainset, max_bootstrapped_demos=2, max_labeled_demos=2)
        ```

        実際に実行可能な完全なRAG例については、[このチュートリアル](../../tutorials/rag/index.ipynb)をご覧ください。StackExchangeコミュニティのサブセットにおいて、RAGシステムの品質を53%から61%に向上させる効果があります。

    === "分類タスク向け重みの最適化"
        <details><summary>データセット設定コードを表示するにはクリックしてください</summary>

        ```python linenums="1"
        import random
        from typing import Literal

        from datasets import load_dataset

        import dspy
        from dspy.datasets import DataLoader

        # Banking77データセットを読み込む
        CLASSES = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True).features["label"].names
        kwargs = {"fields": ("text", "label"), "input_keys": ("text",), "split": "train", "trust_remote_code": True}

        # データセットから最初の2000例を読み込み、各訓練例にヒントを付与する
        trainset = [
            dspy.Example(x, hint=CLASSES[x.label], label=CLASSES[x.label]).with_inputs("text", "hint")
            for x in DataLoader().from_huggingface(dataset_name="PolyAI/banking77", **kwargs)[:2000]
        ]
        random.Random(0).shuffle(trainset)
        ```
        </details>

        ```python linenums="1"
        import dspy
        lm=dspy.LM('openai/gpt-4o-mini-2024-07-18')

        # 分類用DSPyモジュールを定義する。訓練時には利用可能な場合、ヒントを使用する
        signature = dspy.Signature("text, hint -> label").with_updated_fields('label', type_=Literal[tuple(CLASSES)])
        classify = dspy.ChainOfThought(signature)
        classify.set_lm(lm)

        # BootstrapFinetuneによる最適化を実行する
        optimizer = dspy.BootstrapFinetune(metric=(lambda x, y, trace=None: x.label == y.label), num_threads=24)
        optimized = optimizer.compile(classify, trainset=trainset)

        optimized(text="pending cash withdrawal"の意味は何ですか？)

        # 完全なファインチューニングチュートリアルについては、以下を参照：https://dspy.ai/tutorials/classification_finetuning/
        ```

        **出力例（最後の行から）：**
        ```text
        Prediction(
            reasoning='pending cash withdrawalとは、現金引き出しのリクエストが開始されたものの、まだ完了または処理されていない状態を指します。このステータスは、取引が進行中であり、資金がまだ口座から引き落とされていないか、ユーザーが利用できる状態になっていないことを意味します。',
            label='pending_cash_withdrawal'
        )
        ```

        DSPy 2.5.29を使用した同様の簡易的な実行では、GPT-4o-miniのスコアが66%から87%に向上しました。


## 最適化結果の保存と読み込み

最適化プログラムを実行した後、その結果を保存しておくことは有用です。後日、ファイルからプログラムを読み込んで推論に使用することができます。この目的のために、`load`および`save`メソッドを使用することができます。

```python
optimized_program.save(YOUR_SAVE_PATH)
```

生成されるファイルはプレーンテキスト形式のJSONフォーマットです。ソースプログラム内のすべてのパラメータと処理ステップが含まれています。いつでもこのファイルを読み込んで、オプティマイザが生成した内容を確認することが可能です。


ファイルからプログラムを読み込むには、当該クラスのインスタンスを生成した後、loadメソッドを呼び出す必要があります。

```python
loaded_program = YOUR_PROGRAM_CLASS()
loaded_program.load(path=YOUR_SAVE_PATH)
```

