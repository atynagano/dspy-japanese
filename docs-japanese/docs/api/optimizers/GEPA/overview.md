# dspy.GEPA: リフレクティブ・プロンプト最適化アルゴリズム

**GEPA**（Genetic-Pareto）は、論文「GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning」（Agrawal et al., 2025, [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)）で提案されたリフレクティブ型の最適化アルゴリズムであり、任意のシステムの_textualコンポーネント_（プロンプトなど）を適応的に進化させる機能を有する。GEPAは評価指標から得られるスカラースコアに加え、ユーザーがテキスト形式のフィードバックを提供することで、最適化プロセスを導くことができる。このテキスト形式のフィードバックにより、GEPAはシステムが特定のスコアを獲得した理由をより明確に把握可能となり、その結果としてスコア向上のための改善点を内省的に特定できる。これにより、GEPAは極めて少ない試行回数で高性能なプロンプトを提案することが可能となる。

<!-- START_API_REF -->
::: dspy.GEPA
    handler: python
    options:
        members:
            - auto_budget
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

GEPAの核心的な特徴の一つは、ドメイン固有のテキスト形式フィードバックを効果的に活用できる点にある。ユーザーはGEPAの評価指標としてフィードバック関数を提供する必要があり、この関数は以下の呼び出し規約に従う必要がある：
<!-- START_API_REF -->
::: dspy.teleprompt.gepa.gepa.GEPAFeedbackMetric
    handler: python
    options:
        members:
            - __call__
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

`track_stats=True`が設定されている場合、GEPAは提案された全ての候補について詳細な結果と最適化実行時のメタデータを返却する。これらの結果はGEPAによって最適化されたプログラムの`detailed_results`属性に格納され、以下の型を持つ：
<!-- START_API_REF -->
::: dspy.teleprompt.gepa.gepa.DspyGEPAResult
    handler: python
    options:
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

GEPAの使用例については、[GEPAチュートリアル](../../../tutorials/gepa_ai_program/index.md)を参照されたい。

### 推論時検索機能

GEPAはテスト時/推論時の検索メカニズムとしても機能する。`valset`を評価用バッチに設定し、`track_best_outputs=True`を指定することで、GEPAは各バッチ要素に対して、進化的探索プロセス中に発見された最高スコアの出力を生成する。

```python
gepa = dspy.GEPA(metric=metric, track_stats=True, ...)
new_prog = gepa.compile(student, trainset=my_tasks, valset=my_tasks)
各タスクにおける達成最高スコア = new_prog.detailed_results.highest_score_achieved_per_val_task
最適出力結果 = new_prog.detailed_results.best_outputs_valset
```

## GEPAの動作原理

### 1. **内省的プロンプト変異**

GEPAはLLMを用いて、構造化された実行トレース（入力データ、出力結果、失敗事例、フィードバック情報）について内省を行い、指定されたモジュールを対象に、実際に観測された失敗事例や詳細なテキスト／環境フィードバックに基づいて、新たな指示文やプログラムコードを提案します。

### 2. **最適化信号としての豊富なテキストフィードバック**

GEPAは、単なるスカラー報酬だけでなく、利用可能なあらゆるテキスト形式のフィードバックを活用できます。これには評価ログ、コード実行履歴、構文解析失敗事例、制約条件違反、エラーメッセージ文字列、さらには特定のサブモジュールに特化した個別フィードバックも含まれます。これにより、ドメイン知識を考慮した実用的な最適化が可能となります。

### 3. **パレート最適に基づく候補選択**

GEPAは、単に「最良の」グローバル候補を進化させる手法（局所最適解や停滞を招く可能性がある）ではなく、パレート最適集合（少なくとも1つの評価事例において最高スコアを達成する候補群）を維持します。各反復処理では、このパレートフロンティアから（網羅率に比例する確率で）次の変異対象候補をサンプリングし、探索と補完的な戦略の堅牢な保持を両立させます。

### アルゴリズムの概要

1. **初期候補プール**として、最適化前のプログラムを設定します。
2. **反復処理**：
   - **パレートフロンティアから候補をサンプリング**します。
   - **訓練データセットからミニバッチをサンプリング**します。
   - **ミニバッチに対するモジュール実行トレースとフィードバックを収集**します。
   - **対象モジュールを選定**し、重点的な改善を行います。
   - **LLMによる内省：** 収集したフィードバックを活用し、対象モジュール向けの新たな指示文／プロンプトを反射的メタプロンプティングによって提案します。
   - **新規候補をミニバッチに適用**し、**改善が確認された場合はパレート検証セットで評価**を行います。
   - **候補プール／パレートフロンティアを更新**します。
   - **[オプション]システム認識型のマージ／交叉操作**：異なる系統から最も性能の高いモジュールを統合します。
3. **ロールアウトまたは評価指標の予算が枯渇するまで**処理を継続します。
4. **検証セットにおける総合性能が最も高い候補を出力**します。

## フィードバック指標の実装方法

適切に設計された評価指標は、GEPAのサンプル効率と学習信号の豊富さにおいて極めて重要です。GEPAでは、指標が`dspy.Prediction(score=..., feedback=...)`形式の出力を返すことを想定しています。GEPAはLLMベースのワークフローから得られる自然言語形式のトレースを最適化に活用し、中間的な実行軌跡やエラー情報を数値報酬に変換することなく平易なテキスト形式で保持します。これは人間の診断プロセスを模倣するものであり、システムの挙動やボトルネックをより明確に特定することを可能にします。

GEPAに適したフィードバックを実装するための実践的ガイドライン：

- **既存の成果物の活用**：ログデータ、単体テスト結果、評価スクリプト、プロファイラ出力などを活用します。これらを可視化するだけで十分な場合が多いです。
- **成果物の分解分析**：スコアを各評価項目（正確性、応答遅延、コスト、安全性など）ごとに分解し、エラー発生箇所を工程ごとに特定します。
- **処理過程の可視化**：パイプラインの各工程にラベル付けを行い、通過/不合格の判定結果とともに、特に重要なエラー（例えばコード生成パイプラインにおけるエラーなど）を報告します。
- **検証チェックによる根拠付け**：自動検証ツール（単体テスト、スキーマ検証、シミュレータなど）を使用するか、検証が困難なタスクについてはLLMを判定者として活用します（PUPAプロジェクトの手法参照）。
- **明確性の優先**：技術的複雑さよりも、エラー網羅率と意思決定ポイントの明確化に重点を置きます。

### 具体例

- **文書検索タスク**（例：HotpotQA）：単なるRecall/F1スコアだけでなく、正しく取得された文書、誤った文書、未取得の文書を個別に列挙します。
- **多目的最適化タスク**（例：PUPA）：総合スコアを各評価項目ごとに分解し、各目的関数の寄与度を明らかにすることで、品質とプライバシー保護などのトレードオフ関係を可視化します。
- **積層型パイプライン**（例：コード生成パイプライン：構文解析→コンパイル→実行→プロファイリング→評価）：各工程における個別の失敗箇所を可視化します。LLMによる自己修正の場合、自然言語によるトレース情報のみで十分な場合が多いです。

## カスタム指示提案機能の提案

GEPAの指示提案メカニズムを高度にカスタマイズする場合、カスタム指示提案者やコンポーネント選択機能の実装については、[高度な機能](GEPA_Advanced.md)のセクションを参照してください。

## 関連資料

- [GEPA論文：arxiv:2507.19457](https://arxiv.org/abs/2507.19457)
- [GEPA GitHubリポジトリ](https://github.com/gepa-ai/gepa) - `dspy.GEPA`オプティマイザで使用されているGEPAのコア進化パイプラインが収録されています。
- [DSPyチュートリアル](../../../tutorials/gepa_ai_program/index.md)
