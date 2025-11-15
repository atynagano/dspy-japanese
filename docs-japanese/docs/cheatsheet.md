---
sidebar_position: 999
---

# DSPy チートシート

本ドキュメントでは、頻繁に使用されるパターンのコードスニペットを掲載します。

## DSPy プログラム例

### 最新の言語モデル出力を強制する場合

DSPyは言語モデルの呼び出し結果をキャッシュします。キャッシュを回避しつつ新規結果をキャッシュするには、一意の``rollout_id``を指定し、``temperature``値を非ゼロ値（例：1.0）に設定してください：

```python
predict = dspy.Predict("question -> answer")
predict(question="1+1", config={"rollout_id": 1, "temperature": 1.0})
```

### dspy.Signature クラス

```python
class BasicQA(dspy.Signature):
    """短文による事実ベースの質問応答を行う。"""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="通常1～5語程度の簡潔な回答")
```

### dspy.ChainOfThought

```python
generate_answer = dspy.ChainOfThought(BasicQA)

# 特定の入力に対して予測器を呼び出し、ヒントを併せて与える。
question='空の色は何色ですか？'
pred = generate_answer(question=question)
```

### dspy.ProgramOfThought

```python
pot = dspy.ProgramOfThought(BasicQA)

question = 'サラはリンゴを5個持っています。彼女は店でさらに7個のリンゴを購入しました。現在、サラは全部で何個のリンゴを持っているでしょうか？'
result = pot(question=question)

print(f"質問: {question}")
print(f"思考プロセス後の最終予測回答: {result.answer}")
```

### dspy.ReAct

```python
react_module = dspy.ReAct(BasicQA)

question = 'サラはリンゴを5個持っています。彼女は店でさらに7個のリンゴを購入しました。現在、サラは全部で何個のリンゴを持っているでしょうか？'
result = react_module(question=question)

print(f"質問: {question}")
print(f"ReAct処理後の最終予測回答: {result.answer}")
```

### dspy.Retrieve

```python
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.configure(rm=colbertv2_wiki17_abstracts)

# 検索モジュールの定義
retriever = dspy.Retrieve(k=3)

query = 'FIFAワールドカップが初めて開催されたのは何年ですか？'

# 特定のクエリに対して検索モジュールを実行
topK_passages = retriever(query).passages

for idx, passage in enumerate(topK_passages):
    print(f'{idx+1}]', passage, '\n')
```

### dspy.CodeAct

```python
from dspy import CodeAct

def factorial(n):
    """n の階乗を計算する"""
    if n == 1:
        return 1
    return n * factorial(n-1)

act = CodeAct("n->factorial", tools=[factorial])
result = act(n=5)
result # 結果は 120 となる
```

### dspy.Parallel

```python
import dspy

parallel = dspy.Parallel(num_threads=2)
predict = dspy.Predict("question -> answer")
result = parallel(
    [
        (predict, dspy.Example(question="1+1").with_inputs("question")),
        (predict, dspy.Example(question="2+2").with_inputs("question"))
    ]
)
result
```

## DSPy メトリクス

### メトリクスとしての関数定義

カスタムメトリクスを作成するには、数値または真偽値を返す関数を定義します：

```python
def parse_integer_answer(answer, only_first_line=True):
    try:
        if only_first_line:
            answer = answer.strip().split('\n')[0]

        # 数字を含む最後のトークンを抽出
        answer = [token for token in answer.split() if any(c.isdigit() for c in token)][-1]
        answer = answer.split('.')[0]
        answer = ''.join([c for c in answer if c.isdigit()])
        answer = int(answer)

    except (ValueError, IndexError):
        # print(answer)
        answer = 0

    return answer

# 評価関数
def gsm8k_metric(gold, pred, trace=None) -> int:
    return int(parse_integer_answer(str(gold.answer))) == int(parse_integer_answer(str(pred.answer)))
```

### LLMを判定器として用いる場合

```python
class FactJudge(dspy.Signature):
    """文脈に基づいて回答の事実的正確性を判定するクラス"""

    context = dspy.InputField(desc="予測対象の文脈情報")
    question = dspy.InputField(desc="回答すべき質問文")
    answer = dspy.InputField(desc="質問に対する回答")
    factually_correct: bool = dspy.OutputField(desc="回答は文脈に基づいて事実的に正しいか？")

judge = dspy.ChainOfThought(FactJudge)

def factuality_metric(example, pred):
    factual = judge(context=example.context, question=example.question, answer=pred.answer)
    return factual.factually_correct
```

## DSPyの評価

```python
from dspy.evaluate import Evaluate

evaluate_program = Evaluate(devset=devset, metric=your_defined_metric, num_threads=NUM_THREADS, display_progress=True, display_table=num_rows_to_display)

evaluate_program(your_dspy_program)
```

## DSPyオプティマイザ

### LabeledFewShot

```python
from dspy.teleprompt import LabeledFewShot

labeled_fewshot_optimizer = LabeledFewShot(k=8)
compiled_dspy_program = labeled_fewshot_optimizer.compile(student=your_dspy_program, trainset=trainset)
```

### BootstrapFewShot

```python
from dspy.teleprompt import BootstrapFewShot

fewshot_optimizer = BootstrapFewShot(metric=your_defined_metric, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, max_errors=10)

your_dspy_program_compiled = fewshot_optimizer.compile(student = your_dspy_program, trainset=trainset)
```

#### コンパイル時に別の言語モデルを使用する場合（teacher_settingsで指定）

```python
from dspy.teleprompt import BootstrapFewShot

fewshot_optimizer = BootstrapFewShot(metric=your_defined_metric, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, max_errors=10, teacher_settings=dict(lm=gpt4))

your_dspy_program_compiled = fewshot_optimizer.compile(student = your_dspy_program, trainset=trainset)
```

#### コンパイル済みプログラムの再コンパイル - ブートストラップされたプログラムの再ブートストラップ

```python
your_dspy_program_compiledx2 = teleprompter.compile(
    your_dspy_program,
    teacher=your_dspy_program_compiled,
    trainset=trainset,
)
```

#### コンパイル済みプログラムの保存/読み込み

```python
save_path = './v1.json'
your_dspy_program_compiledx2.save(save_path)
```

```python
loaded_program = YourProgramClass()
loaded_program.load(path=save_path)
```

### BootstrapFewShotWithRandomSearch

BootstrapFewShotWithRandomSearchに関する詳細なドキュメントは[こちら](api/optimizers/BootstrapFewShot.md)で参照可能です。

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

fewshot_optimizer = BootstrapFewShotWithRandomSearch(metric=your_defined_metric, max_bootstrapped_demos=2, num_candidate_programs=8, num_threads=NUM_THREADS)

your_dspy_program_compiled = fewshot_optimizer.compile(student = your_dspy_program, trainset=trainset, valset=devset)

```

その他のカスタム設定は、`BootstrapFewShot`オプティマイザをカスタマイズする場合と同様の手順で行います。

### アンサンブル

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.teleprompt.ensemble import Ensemble

fewshot_optimizer = BootstrapFewShotWithRandomSearch(metric=your_defined_metric, max_bootstrapped_demos=2, num_candidate_programs=8, num_threads=NUM_THREADS)
your_dspy_program_compiled = fewshot_optimizer.compile(student = your_dspy_program, trainset=trainset, valset=devset)

ensemble_optimizer = Ensemble(reduce_fn=dspy.majority)
programs = [x[-1] for x in your_dspy_program_compiled.candidate_programs]
your_dspy_program_compiled_ensemble = ensemble_optimizer.compile(programs[:3])
```

### BootstrapFinetune

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BootstrapFinetune

# 現在の dspy.settings.lm 環境設定でプログラムをコンパイル
fewshot_optimizer = BootstrapFewShotWithRandomSearch(metric=your_defined_metric, max_bootstrapped_demos=2, num_threads=NUM_THREADS)
your_dspy_program_compiled = tp.compile(your_dspy_program, trainset=trainset[:some_num], valset=trainset[some_num:])

# ファインチューニング用モデルの設定構成
config = dict(target=model_to_finetune, epochs=2, bf16=True, bsize=6, accumsteps=2, lr=5e-5)

# BootstrapFinetune 環境でプログラムをコンパイル
finetune_optimizer = BootstrapFinetune(metric=your_defined_metric)
finetune_program = finetune_optimizer.compile(your_dspy_program, trainset=some_new_dataset_for_finetuning_model, **config)

finetune_program = your_dspy_program

# プログラムをロードし、評価前にモデルパラメータを有効化
ckpt_path = "saved_checkpoint_path_from_finetuning"
LM = dspy.HFModel(checkpoint=ckpt_path, model=model_to_finetune)

for p in finetune_program.predictors():
    p.lm = LM
    p.activated = False
```

### COPRO

COPROに関する詳細なドキュメントは[こちら](api/optimizers/COPRO.md)で参照可能です。

```python
from dspy.teleprompt import COPRO

eval_kwargs = dict(num_threads=16, display_progress=True, display_table=0)

copro_teleprompter = COPRO(prompt_model=model_to_generate_prompts, metric=your_defined_metric, breadth=num_new_prompts_generated, depth=times_to_generate_prompts, init_temperature=prompt_generation_temperature, verbose=False)

compiled_program_optimized_signature = copro_teleprompter.compile(your_dspy_program, trainset=trainset, eval_kwargs=eval_kwargs)
```

### MIPROv2

注意：詳細なドキュメントは[こちら](api/optimizers/MIPROv2.md)を参照してください。`MIPROv2`は`MIPRO`の最新拡張版であり、以下の2つの主要な改良点を含んでいます：（1）命令提案アルゴリズムの性能向上、および（2）ミニバッチ処理によるより効率的な探索アルゴリズムの実装。

#### MIPROv2を用いた最適化の実行方法

本セクションでは、多くのハイパーパラメータを自動設定し、簡易的な最適化処理を実行する`auto=light`オプションの使用方法を説明します。より長時間の最適化処理を行いたい場合は、`auto=medium`または`auto=heavy`を設定することもできます。より詳細な`MIPROv2`のドキュメント[こちら](api/optimizers/MIPROv2.md)では、手動でハイパーパラメータを設定する方法についても詳しく解説しています。

```python
# オプティマイザモジュールをインポート
from dspy.teleprompt import MIPROv2

# オプティマイザの初期化
teleprompter = MIPROv2(
    metric=gsm8k_metric,
    auto="light", # 最適化処理の強度を「light」「medium」「heavy」から選択可能
)

# プログラムの最適化実行
print(f"MIPROを用いてプログラムを最適化中...")
optimized_program = teleprompter.compile(
    program.deepcopy(),
    trainset=trainset,
    max_bootstrapped_demos=3,
    max_labeled_demos=4,
)

# 最適化済みプログラムを保存（将来の再利用用）
optimized_program.save(f"mipro_optimized")

# 最適化済みプログラムの評価
print(f"最適化済みプログラムの評価を実施...")
evaluate(optimized_program, devset=devset[:])
```

#### MIPROv2のみを使用した最適化（0-Shot）

```python
# オプティマイザモジュールをインポート
from dspy.teleprompt import MIPROv2

# オプティマイザの初期化
teleprompter = MIPROv2(
    metric=gsm8k_metric,
    auto="light", # 最適化処理の強度を「light」「medium」「heavy」から選択可能
)

# プログラムの最適化実行
print(f"MIPROを用いてプログラムを最適化中...")
optimized_program = teleprompter.compile(
    program.deepcopy(),
    trainset=trainset,
    max_bootstrapped_demos=0,
    max_labeled_demos=0,
)

# 最適化済みプログラムを保存（将来の再利用用）
optimized_program.save(f"mipro_optimized")

# 最適化済みプログラムの評価
print(f"最適化済みプログラムの評価を実施...")
evaluate(optimized_program, devset=devset[:])
```

### KNNFewShot

```python
from sentence_transformers import SentenceTransformer
from dspy import Embedder
from dspy.teleprompt import KNNFewShot
from dspy import ChainOfThought

knn_optimizer = KNNFewShot(k=3, trainset=trainset, vectorizer=Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode))

qa_compiled = knn_optimizer.compile(student=ChainOfThought("質問 -> 回答"))
```

### BootstrapFewShotWithOptuna

```python
from dspy.teleprompt import BootstrapFewShotWithOptuna

fewshot_optuna_optimizer = BootstrapFewShotWithOptuna(metric=your_defined_metric, max_bootstrapped_demos=2, num_candidate_programs=8, num_threads=NUM_THREADS)

your_dspy_program_compiled = fewshot_optuna_optimizer.compile(student=your_dspy_program, trainset=trainset, valset=devset)
```

その他のカスタム設定は、`dspy.BootstrapFewShot`オプティマイザをカスタマイズする場合と同様の手法で実装可能です。


### SIMBA

SIMBA（Stochastic Introspective Mini-Batch Ascentの略称）は、任意のDSPyプログラムを受け付け、ミニバッチ単位で処理を行うプロンプト最適化手法です。本手法では、プロンプト指示文やfew-shot例に対して段階的な改善を繰り返し行うことで、最適化を進めていきます。

```python
from dspy.teleprompt import SIMBA

simba = SIMBA(metric=your_defined_metric, max_steps=12, max_demos=10)

optimized_program = simba.compile(student=your_dspy_program, trainset=trainset)
```


## DSPy ツールおよびユーティリティ

### dspy.Tool

```python
import dspy

def search_web(query: str) -> str:
    """ウェブ検索を実行して関連情報を取得する"""
    return f"検索クエリ '{query}' に対する検索結果"

tool = dspy.Tool(search_web)
result = tool(query="Python プログラミング")
```

### dspy.streamify

```python
import dspy
import asyncio

predict = dspy.Predict("question->answer")

stream_predict = dspy.streamify(
    predict,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
)

async def read_output_stream():
    output_stream = stream_predict(question="Why did a chicken cross the kitchen?")

    async for chunk in output_stream:
        print(chunk)

asyncio.run(read_output_stream())
```


### dspy.asyncify

```python
import dspy

dspy_program = dspy.ChainOfThought("question -> answer")
dspy_program = dspy.asyncify(dspy_program)

asyncio.run(dspy_program(question="What is DSPy?"))
```


### 使用状況の追跡

```python
import dspy
dspy.configure(track_usage=True)

result = dspy.ChainOfThought(BasicQA)(question="2+2の計算結果は？")
print(f"トークン使用量: {result.get_lm_usage()}")
```

### dspy.configure_cache

```python
import dspy

# キャッシュ設定の構成
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)
```

## DSPyにおける`Refine`と`BestofN`機能

> DSPy 2.6において、`dspy.Suggest`と`dspy.Assert`はそれぞれ`dspy.Refine`と`dspy.BestofN`に置き換えられました。

### BestofN機能

この機能は、異なるロールアウトIDを用いてモジュールを最大`N`回実行し（キャッシュをバイパス）、`reward_fn`で定義された最良の予測結果、あるいは`threshold`条件を満たす最初の予測結果を返します。

```python
import dspy

qa = dspy.ChainOfThought("question -> answer")
def one_word_answer(args, pred):
    return 1.0 if len(pred.answer) == 1 else 0.0
best_of_3 = dspy.BestOfN(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0)
best_of_3(question="What is the capital of Belgium?").answer
# Brussels
```

### Refine（改良）

モジュールを改良する機能で、異なるロールアウトIDを用いて最大`N`回の実行を行い（キャッシュをバイパス）、`reward_fn`で定義された最良の予測結果、または`threshold`条件を満たす最初の予測結果を返します。各試行後（最終試行を除く）、`Refine`はモジュールのパフォーマンスに関する詳細なフィードバックを自動生成し、このフィードバックを後続の実行時のヒントとして活用することで、反復的な改良プロセスを実現します。

```python
import dspy

qa = dspy.ChainOfThought("question -> answer")
def one_word_answer(args, pred):
    return 1.0 if len(pred.answer) == 1 else 0.0
best_of_3 = dspy.Refine(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0)
best_of_3(question="What is the capital of Belgium?").answer
# Brussels
```

#### エラー処理

デフォルトでは、`Refine` は閾値が満たされるまでモジュールを最大 N 回実行しようとします。モジュールがエラーに遭遇した場合、最大 N 回の失敗試行まで処理を継続します。この動作を変更するには、`fail_count` を `N` より小さい値に設定してください。

```python
refine = dspy.Refine(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0, fail_count=1)
...
refine(question="ベルギーの首都はどこですか？")
# もし1回の試行で失敗した場合、モジュールはエラーを発生させます。
```

エラー処理を一切行わずにモジュールを最大 N 回実行したい場合は、`fail_count` を `N` に設定してください。これがデフォルトの動作です。

```python
refine = dspy.Refine(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0, fail_count=3)
...
refine(question="ベルギーの首都はどこですか？")
```
