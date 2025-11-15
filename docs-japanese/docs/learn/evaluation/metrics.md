---
sidebar_position: 5
---

# 評価指標

DSPyは機械学習フレームワークであるため、評価（学習進捗の追跡）と最適化（プログラムの性能向上）のための**自動評価指標**を適切に設計することが不可欠です。


## 評価指標とは何か？ 特定のタスクに適した指標をどのように定義すればよいか？

評価指標とは、データセットから抽出したサンプルとシステムの出力結果を受け取り、出力の品質を定量的に評価するスコアを返す単なる関数です。システムの出力が「良い」または「悪い」と判断される基準は何でしょうか？

単純なタスクの場合、これは「正解率」「完全一致率」「F1スコア」といった基本的な指標で十分な場合があります。単純な分類タスクや短文回答型の質問応答タスクなどがこれに該当します。

しかし、ほとんどの実用的なアプリケーションでは、システムは長文形式の出力を生成します。このような場合、評価指標としては、出力の複数の特性を検証するより複雑なDSPyプログラムを用いるべきです（多くの場合、大規模言語モデル（LM）からのAIフィードバックを活用する形になります）。

初回の設計で最適な指標を完成させることは難しいかもしれませんが、まずはシンプルな指標から始め、反復的に改善していくアプローチが有効です。


## 基本的な評価指標

DSPyにおける評価指標とは、Python関数として実装されたものであり、`example`（学習データまたは開発データからのサンプル）とDSPyプログラムの出力`pred`を受け取り、`float`型（または`int`型、`bool`型）のスコアを返すものです。

評価指標には、オプションとして第三の引数`trace`を指定できます。この引数は現時点では無視しても構いませんが、最適化目的で評価指標を使用する場合には強力な機能を利用可能になります。

以下に、`example.answer`と`pred.answer`を比較するシンプルな評価指標の例を示します。この特定の指標は`bool`型のスコアを返します。

```python
def validate_answer(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()
```

以下の組み込みユーティリティは一部のユーザーにとって便利です：

- `dspy.evaluate.metrics.answer_exact_match`
- `dspy.evaluate.metrics.answer_passage_match`

使用するメトリクスはより複雑なものでも構いません。例えば複数の特性を同時に評価することも可能です。以下に示すメトリクスは、`trace`が`None`の場合（評価または最適化時）には`float`値を返し、それ以外の場合（デモンストレーションのブートストラッピング時）には`bool`値を返すように設計されています。

```python
def validate_context_and_answer(example, pred, trace=None):
    # 正解ラベルと予測回答が一致しているか確認
    answer_match = example.answer.lower() == pred.answer.lower()

    # 予測回答が検索されたコンテキストのいずれかに含まれているか確認
    context_match = any(pred.answer.lower() in c for c in pred.context)

    if trace is None: # 評価または最適化処理を実行している場合
        return (answer_match + context_match) / 2.0
    else: # ブートストラッピング処理を実行している場合（各ステップの良質なデモンストレーションを自己生成する場合）
        return answer_match and context_match
```

適切な評価指標の定義は反復的なプロセスであるため、初期段階での評価実施とデータ・出力結果の分析が重要となる。


## 評価方法

評価指標を設定した後は、シンプルなPythonループを用いて評価を実行することが可能である。

```python
scores = []
for x in devset:
    pred = program(**x.inputs())
    score = metric(x, pred)
    scores.append(score)
```

必要に応じて、組み込みの `Evaluate` ユーティリティも利用可能です。このユーティリティは、並列評価（複数スレッド処理）や、入力/出力のサンプル表示、評価指標スコアの確認などに活用できます。

```python
from dspy.evaluate import Evaluate

# 評価器を設定します。このオブジェクトはコード内で再利用可能です。
evaluator = Evaluate(devset=YOUR_DEVSET, num_threads=1, display_progress=True, display_table=5)

# 評価を実行します。
evaluator(YOUR_PROGRAM, metric=YOUR_METRIC)
```


## 中級編：AIフィードバックを活用した評価指標の最適化

多くのアプリケーションでは、システムは長文形式の出力を生成するため、評価指標は大規模言語モデル（LM）から得られるAIフィードバックを用いて、出力の複数の側面を検証する必要があります。

このシンプルな評価指標の枠組みが役立つ場合があります。

```python
# 自動評価用のシグネチャを定義します。
class Assess(dspy.Signature):
    """指定された評価軸に基づいてツイートの品質を評価する。"""

    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer: bool = dspy.OutputField()
```

例えば、以下に示すのは、生成されたツイートが（1）与えられた質問に正しく回答しており、かつ（2）エンゲージメントを高める内容であるかを検証するシンプルなメトリクスである。さらに、（3）`len(tweet) <= 280` 文字という制約を満たしていることも確認する。

```python
def metric(gold, pred, trace=None):
    question, answer, tweet = gold.question, gold.answer, pred.output

    engagement = "評価対象テキストは自己完結型で読者を引きつけるツイートとして成立しているか?"
    accuracy = f"テキストは`{question}`という質問に対して`{answer}`という回答を含んでいる必要がある。評価対象テキストにはこの回答が含まれているか?"

    accuracy = dspy.Predict(Assess)(assessed_text=tweet, assessment_question=accuracy)
    engagement = dspy.Predict(Assess)(assessed_text=tweet, assessment_question=engagement)

    correct, engaging = [m.assessment_answer for m in [accuracy, engagement]]
    score = (correct + engaging) if correct and (len(tweet) <= 280) else 0

    if trace is not None: return score >= 2
    return score / 2.0
```

コンパイル時に`trace`が`None`でない場合、かつ厳密な判定を行いたい場合には、`score`が`2`以上の場合にのみ`True`を返す。それ以外の場合は、スコアを1.0で除した値（すなわち`score / 2.0`）を返却する。


## 上級編：DSPyプログラムを評価指標として使用する場合

評価指標自体がDSPyプログラムである場合、最も効果的な反復手法の一つは、自らの評価指標自体をコンパイル（最適化）することである。これは通常容易な作業である。なぜなら、評価指標の出力は通常単純な値（例えば5段階評価のスコアなど）であるため、評価指標の評価指標を定義し最適化することは、少数の事例を収集することで比較的簡単に実現できるからである。



### 上級編：`trace`へのアクセス方法

評価実行時に評価指標が使用される場合、DSPyはプログラムの各処理ステップを追跡しようとしない。

しかし、コンパイル（最適化）処理時には、DSPyは言語モデル（LM）への呼び出しをトレースする。このトレースには、各DSPy予測器に対する入力/出力情報が含まれており、これを活用することで最適化プロセスにおける中間ステップの検証が可能となる。


```python
def validate_hops(example, pred, trace=None):
    hops = [example.question] + [outputs.query for *_, outputs in trace if 'query' in outputs]

    if max([len(h) for h in hops]) > 100: return False
    if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))): return False

    return True
```
