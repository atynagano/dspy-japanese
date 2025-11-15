# dspy.MIPROv2

`MIPROv2`（<u>M</u>ultiprompt <u>I</u>nstruction <u>PR</u>oposal <u>O</u>ptimizer Version 2）は、指示文と少数ショット例を統合的に最適化可能なプロンプト最適化アルゴリズムである。本アルゴリズムは以下の手法によりこれを実現している：少数ショット例の候補をブートストラップし、タスクの異なるダイナミクスに基づいて生成された指示文を提案し、さらにベイズ最適化を用いてこれらのオプションの最適組み合わせを探索する。本手法は、少数ショット例と指示文の同時最適化、あるいは0ショット最適化のための指示文単体最適化のいずれにも適用可能である。

<!-- START_API_REF -->
::: dspy.MIPROv2
    handler: python
    options:
        members:
            - compile
            - get_params
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
:::
<!-- END_API_REF -->

## 使用例

以下のプログラム例では、MIPROv2を用いた数学問題の最適化処理を示している

```python
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

# 最適化アルゴリズムのインポート
from dspy.teleprompt import MIPROv2

# 言語モデルの初期化
lm = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_OPENAI_API_KEY')
dspy.configure(lm=lm)

# 最適化アルゴリズムの初期化
teleprompter = MIPROv2(
    metric=gsm8k_metric,
    auto="medium", # 最適化処理の強度は「light」「medium」「heavy」から選択可能
)

# プログラムの最適化実行
print(f"MIPROv2 を使用してプログラムを最適化中...")
gsm8k = GSM8K()
optimized_program = teleprompter.compile(
    dspy.ChainOfThought("question -> answer"),
    trainset=gsm8k.train,
)

# 最適化済みプログラムの保存
optimized_program.save(f"optimized.json")
```

## `MIPROv2`の動作原理

大まかに言えば、`MIPROv2`は言語モデルプログラム内の各予測器に対して、少数ショット学習用の事例と新規指示文の両方を生成し、ベイズ最適化を用いてこれらの変数の最適な組み合わせを探索することで機能します。視覚的な説明が必要な場合は、[このTwitterスレッド](https://x.com/michaelryan207/status/1804189184988713065)を参照してください。

以下に各処理ステップをより詳細に説明します：

1) **少数ショット学習用事例のブートストラップ生成**：
   - 訓練データセットからランダムに事例を抽出し、言語モデルプログラムに入力します。
   - プログラムの出力がこの事例に対して正しい場合、その事例を有効な少数ショット学習用候補として保持します。
   - 正しくない場合には、指定された数の候補が得られるまで別の事例を試します。この処理により、`num_candidates`セット分の`max_bootstrapped_demos`個のブートストラップ事例と、訓練データセットからサンプリングした`max_labeled_demos`個の基本事例が生成されます。

2) **指示文候補の提案**：
   - 指示文提案器は以下の要素を提供します：
     1) 訓練データセットの特性に関する生成要約
     2) 言語モデルプログラムのコード概要と、指示文生成対象とする特定の予測器に関する説明
     3) 以前に生成した少数ショット事例（各予測器に対する参照入力/出力として機能）
     4) 生成可能な指示文の特徴空間を探索するためのランダムなヒント（例：「創造的に」「簡潔に」など）
   - これらの文脈情報は`prompt_model`に入力され、高品質な指示文候補が生成されます。

3) **少数ショット事例と指示文の最適化組み合わせの探索**：
   - 最終的に、ベイズ最適化を用いて、プログラム内の各予測器に対して最も効果的な指示文と事例の組み合わせを選択します。
   - このプロセスでは、`num_trials`回の試行を実施し、各試行ごとに検証セット上で新たなプロンプトセットを評価します。
   - 各試行では、`minibatch_size`サイズのミニバッチのみを評価対象とします（`minibatch`=`True`の場合）。
   - 最適な平均プロンプトセットは、`minibatch_full_eval_steps`ごとに検証セット全体で再評価されます。
   - 最適化プロセスの終了時には、検証セット全体で最高の性能を示したプロンプトセットを備えた言語モデルプログラムが返されます。

より詳細な情報に関心のある方に向けて、`MIPROv2`に関する追加情報と、他のDSPy最適化手法との比較研究については、[本論文](https://arxiv.org/abs/2406.11695)を参照されたい。
