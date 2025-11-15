# 出力精緻化：BestOfN および Refine モジュール

`BestOfN` および `Refine` は、キャッシュを回避するために異なるロールアウト ID を用いて複数回 `LM` を呼び出すことで、予測の信頼性と品質を向上させるように設計された DSPy モジュールである。両モジュールは、`N` 回の試行回数に達するか、`reward_fn` が`threshold`値を超える報酬を返した時点で処理を終了する。

## BestOfN モジュール

`BestOfN` は、指定されたモジュールを最大`N`回異なるロールアウト ID で実行するモジュールである。指定された閾値を満たす最初の予測結果を返すか、あるいは閾値を満たす結果が存在しない場合には最高報酬を得た予測結果を返す。

### 基本的な使用方法

例えば、モデルから単一単語の回答を得る確率を最大化したい場合を考える。この場合、`BestOfN` を使用して複数のロールアウト ID を試み、最も適切な結果を返すことができる。

```python
import dspy

def one_word_answer(args, pred: dspy.Prediction) -> float:
    """
    予測された回答が単一の単語で構成されている場合に 1.0 を返し、それ以外の場合は 0.0 を返す。
    """
    return 1.0 if len(pred.answer.split()) == 1 else 0.0

best_of_3 = dspy.BestOfN(
    module=dspy.ChainOfThought("question -> answer"), 
    N=3, 
    reward_fn=one_word_answer, 
    threshold=1.0
)

result = best_of_3(question="What is the capital of Belgium?")
print(result.answer)  # 出力: Brussels
```

### エラー処理

デフォルトでは、モジュールが試行中にエラーを検出した場合、`N`回の試行回数に達するまで再試行を継続します。この動作は`fail_count`パラメータで変更可能です：

```python
best_of_3 = dspy.BestOfN(
    module=qa, 
    N=3, 
    reward_fn=one_word_answer, 
    threshold=1.0,
    fail_count=1
)

best_of_3(question="ベルギーの首都はどこですか？")
# 最初の失敗後にエラーを発生させる
```

## Refine

`Refine`は`BestOfN`の機能を拡張したアルゴリズムであり、自動フィードバックループ機構を追加しています。最終試行を除く各試行が失敗した後、モジュールの性能に関する詳細なフィードバックを自動的に生成し、このフィードバックを後続の試行におけるヒントとして利用します。

### 基本的な使用方法

```python
import dspy

def one_word_answer(args, pred: dspy.Prediction) -> float:
    """
    予測された回答が単一の単語で構成されている場合に 1.0 を返し、それ以外の場合は 0.0 を返す。
    """
    return 1.0 if len(pred.answer.split()) == 1 else 0.0

refine = dspy.Refine(
    module=dspy.ChainOfThought("question -> answer"), 
    N=3, 
    reward_fn=one_word_answer, 
    threshold=1.0
)

result = refine(question="What is the capital of Belgium?")
print(result.answer)  # Brussels
```

### エラー処理

`BestOfN`と同様に、`Refine`もデフォルトではエラーが発生した場合でも最大`N`回まで試行を行います。この動作は`fail_count`パラメータで制御可能です：

```python
# 最初のエラーが発生した時点で処理を停止
refine = dspy.Refine(
    module=qa, 
    N=3, 
    reward_fn=one_word_answer, 
    threshold=1.0,
    fail_count=1
)
```

## 比較：BestOfNとRefineの違い

両モジュールは類似した目的を持つものの、そのアプローチには明確な差異があります：

- `BestOfN`は単に異なるロールアウトIDを試行し、`reward_fn`によって定義された評価基準に基づいて最も優れた予測結果を選択する方式を採用しています。
- `Refine`はフィードバックループ機構を追加しており、前回の予測結果と`reward_fn`内のコードを用いて、モジュール自身のパフォーマンスに関する詳細なフィードバックを生成します。このフィードバックは後続の実行時にヒントとして活用されます。

## 実践的な使用例

### 事実的な正確性の確保

```python
import dspy

class FactualityJudge(dspy.Signature):
    """与えられた文が事実に基づいているかどうかを判断する"""
    statement: str = dspy.InputField()  # 入力文
    is_factual: bool = dspy.OutputField()  # 事実性判定結果

factuality_judge = dspy.ChainOfThought(FactualityJudge)

def factuality_reward(args, pred: dspy.Prediction) -> float:
    statement = pred.answer    # 予測された回答文を取得
    result = factuality_judge(statement)    # 事実性判定を実行
    return 1.0 if result.is_factual else 0.0  # 事実性判定結果に応じて報酬値を返す

refined_qa = dspy.Refine(
    module=dspy.ChainOfThought("question -> answer"),  # 推論モジュール設定
    N=3,  # 推論ステップ数
    reward_fn=factuality_reward,  # 報酬関数として事実性判定を使用
    threshold=1.0  # 報酬閾値
)

result = refined_qa(question="ベルギーの首都について教えてください。")
print(result.answer)
```

### 要約機能 - 応答長の制御

```python
import dspy

def ideal_length_reward(args, pred: dspy.Prediction) -> float:
    """
    要約文の長さが75語前後の場合に高い報酬を与え、長すぎる要約に対しては報酬を逓減させる。
    """
    word_count = len(pred.summary.split())
    distance = abs(word_count - 75)
    return max(0.0, 1.0 - (distance / 125))

optimized_summarizer = dspy.BestOfN(
    module=dspy.ChainOfThought("text -> summary"),
    N=50,
    reward_fn=ideal_length_reward,
    threshold=0.9
)

result = optimized_summarizer(
    text="[要約対象の長文テキスト...]"
)
print(result.summary)
```

## `dspy.Suggest` および `dspy.Assert` からの移行について

DSPy 2.6 以降では、`BestOfN` および `Refine` が `dspy.Suggest` および `dspy.Assert` の代替機能として提供されています。
