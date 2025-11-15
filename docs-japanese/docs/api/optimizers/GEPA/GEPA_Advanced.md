# dspy.GEPA - 高度な機能

## カスタム命令提案機能

### instruction_proposerとは何か？

`instruction_proposer`は、GEPA最適化プロセスにおいて`reflection_lm`を呼び出し、新たなプロンプトを提案する役割を担うコンポーネントである。GEPAがDSPyプログラム内の性能低下要因を特定すると、この命令提案機能は実行トレース、フィードバック、および失敗事例を分析し、観測された問題に対処するための最適化された命令を生成する。

### デフォルト実装

デフォルトでは、GEPAは[GEPAライブラリ](https://github.com/gepa-ai/gepa)に組み込まれた命令提案機能を使用する。この機能は[`ProposalFn`](https://github.com/gepa-ai/gepa/blob/main/src/gepa/core/adapter.py)を実装している。[デフォルト提案機能](https://github.com/gepa-ai/gepa/blob/main/src/gepa/proposer/reflective_mutation/reflective_mutation.py#L53-L75)では、以下のプロンプトテンプレートが使用されている：

````
私はアシスタントに対し、以下の指示を与えて特定のタスクを実行させた：
```
<curr_instructions>
```

以下に、アシスタントに与えた各種タスク入力例と、それぞれの入力に対するアシスタントの応答、および応答内容を改善するための具体的なフィードバックを示す：
```
<inputs_outputs_feedback>
```

あなたの任務は、アシスタント向けの新たな指示文を作成することである。

各入力内容を慎重に精査し、入力形式を特定するとともに、私がアシスタントを用いて解決したいタスクの詳細な内容を正確に把握すること。

すべてのアシスタントの応答とそれに対応するフィードバックを熟読すること。タスクに関する専門的な知識や領域固有の事実情報を抽出し、それらを指示文に含めること。これらの情報は将来的にアシスタントが利用できない可能性があるため、網羅的に記載する必要がある。アシスタントが汎用的な解決戦略を採用している場合は、その戦略内容も指示文に反映させること。

新たな指示文は ``` ブロック内に記述すること。
````

このテンプレートには以下が自動的に入力されます：

- `<curr_instructions>`：現在最適化対象となっている指示文
- `<inputs_outputs_feedback>`：予測器への入力データ、生成された出力結果、および評価フィードバックを含む構造化Markdown形式のコンテンツ

デフォルト動作の具体例：

```python
# デフォルトの指示提案器が自動的に使用されます
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000, api_key=api_key),
    auto="medium"
)
optimized_program = gepa.compile(student, trainset=examples)
```

### カスタムinstruction_proposerを使用するべきケース

**注意:** カスタムinstruction proposerは高度な機能です。ほとんどのユーザーは、デフォルトのproposerで十分な性能を発揮できるため、まずはこちらを使用することを推奨します。

以下の要件がある場合にのみ、カスタムinstruction proposerの実装を検討してください：

- **マルチモーダル処理が必要な場合**：入力データにテキスト情報に加えて画像データ（dspy.Image形式）も含まれる場合
- **制限条件や長さ制約を細かく制御したい場合**：指示文の長さ、形式、構造的要件などについて、より詳細な制御が必要な場合
- **ドメイン固有の情報が必要な場合**：デフォルトのproposerではカバーされていない専門的な知識、専門用語、または文脈情報を組み込む必要がある場合。これは高度な機能であり、ほとんどのユーザーには不要です
- **LLMプロバイダー固有のプロンプティングガイドが必要な場合**：OpenAIやAnthropicなど、各LLMプロバイダーの独自のフォーマット要件に合わせて指示文を最適化したい場合
- **コンポーネント間の連携更新が必要な場合**：各コンポーネントを個別に最適化するのではなく、2つ以上のコンポーネントを協調的に更新する必要がある場合（関連する機能については[カスタムコンポーネント選択](#custom-component-selection)セクションのcomponent_selectorパラメータを参照）
- **外部知識の統合が必要な場合**：指示文生成時にデータベース、API、または知識ベースと連携する必要がある場合

### 利用可能なオプション

**組み込みオプション:**

- **デフォルトproposer**：標準のGEPA instruction proposer（`instruction_proposer=None`と指定した場合に使用される）。デフォルトのinstruction proposerはinstruction proposerとしての機能も有しています！これは最も汎用的なバージョンであり、GEPA論文やチュートリアルで報告されている多様な実験で使用されたものです。
- **MultiModalInstructionProposer**：`dspy.Image`形式の入力データと構造化されたマルチモーダルコンテンツを扱うことができます。

```python
from dspy.teleprompt.gepa.instruction_proposal import MultiModalInstructionProposer

# 画像やマルチモーダル入力を伴うタスクの場合
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000, api_key=api_key),
    instruction_proposer=MultiModalInstructionProposer(),
    auto="medium"
)
```

[GEPAライブラリ](https://github.com/gepa-ai/gepa)の機能拡張に伴い、特定分野に特化した新規指示提案者のコミュニティからの貢献を歓迎いたします。

### カスタム指示提案者の実装方法

カスタム指示提案者を実装するには、`ProposalFn`プロトコルに準拠した呼び出し可能なクラスまたは関数を定義する必要があります。GEPAは最適化処理中に自動的に提案者を呼び出します：

```python
from dspy.teleprompt.gepa.gepa_utils import ReflectiveExample

class CustomInstructionProposer:
    def __call__(
        self,
        candidate: dict[str, str],                          # 本ラウンドで更新対象となる候補コンポーネント名と対応する指示のマッピング
        reflective_dataset: dict[str, list[ReflectiveExample]],  # コンポーネントごとの構造化された例データ: {"入力": ..., "生成出力": ..., "フィードバック": ...}
        components_to_update: list[str]                     # 改善対象とするコンポーネント一覧
    ) -> dict[str, str]:                                    # 更新対象コンポーネントのみに対する新規指示マッピングを返す
        # ここに独自の指示生成ロジックを実装する
        return updated_instructions

# または関数形式で実装する場合:
def custom_instruction_proposer(candidate, reflective_dataset, components_to_update):
    # ここに独自の指示生成ロジックを実装する
    return updated_instructions
```

**反射型データセットの構造定義:**

- `dict[str, list[ReflectiveExample]]` - コンポーネント名をキーとし、その値として例示データのリストを保持する辞書型
- `ReflectiveExample` TypedDictは以下の要素を含む:
  - `Inputs: dict[str, Any]` - 予測器への入力データ（dspy.Imageオブジェクトを含む場合がある）
  - `Generated_Outputs: dict[str, Any] | str` - 成功時: 出力フィールドの辞書、失敗時: エラーメッセージ
  - `Feedback: str` - 常にメトリクス関数から取得される文字列、またはGEPAによって自動生成される文字列

#### 基本使用例: 単語制限提案器

```python
import dspy
from gepa.core.adapter import ProposalFn
from dspy.teleprompt.gepa.gepa_utils import ReflectiveExample

class GenerateWordLimitedInstruction(dspy.Signature):
    """現在の指示文とフィードバック事例に基づき、語数制限を考慮した改善版指示文を生成する"""

    current_instruction = dspy.InputField(desc="改善が必要な現在の指示文")
    feedback_summary = dspy.InputField(desc="肯定的・否定的事例を含む可能性のあるフィードバック情報")
    max_words = dspy.InputField(desc="新規指示文に許容される最大語数")

    improved_instruction = dspy.OutputField(desc="max_words制限を遵守しつつ問題点を修正した改善版指示文")

class WordLimitProposer(ProposalFn):
    def __init__(self, max_words: int = 1000):
        self.max_words = max_words
        self.instruction_improver = dspy.ChainOfThought(GenerateWordLimitedInstruction)

    def __call__(self, candidate: dict[str, str], reflective_dataset: dict[str, list[ReflectiveExample]], components_to_update: list[str]) -> dict[str, str]:
        updated_components = {}

        for component_name in components_to_update:
            if component_name not in candidate or component_name not in reflective_dataset:
                continue

            current_instruction = candidate[component_name]
            component_examples = reflective_dataset[component_name]

            # フィードバック要約の作成
            feedback_text = "\n".join([
                f"事例 {i+1}: {ex.get('Feedback', 'フィードバックなし')}"
                for i, ex in enumerate(component_examples)  # コンテキスト過多を防ぐため事例数を制限
            ])

            # モジュールを用いて指示文を改善
            result = self.instruction_improver(
                current_instruction=current_instruction,
                feedback_summary=feedback_text,
                max_words=self.max_words
            )

            updated_components[component_name] = result.improved_instruction

        return updated_components

# 使用例
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000, api_key=api_key),
    instruction_proposer=WordLimitProposer(max_words=700),
    auto="medium"
)
```

#### 高度な応用例：RAG強化型指示提案システム

```python
import dspy
from gepa.core.adapter import ProposalFn
from dspy.teleprompt.gepa.gepa_utils import ReflectiveExample

class GenerateDocumentationQuery(dspy.Signature):
    """事例とフィードバックを分析し、共通する問題パターンを特定した上で、関連する文書を取得するための具体的なデータベース検索クエリを生成する。

    本タスクでは、文書データベースから事例中に発見された問題パターンに対応するガイドラインを検索することを目的とする。フィードバックに頻出する問題、エラータイプ、あるいは失敗モードを特定し、これらのパターンを解決するのに役立つ文書を検索するための具体的な検索クエリを作成すること。"""

    current_instruction = dspy.InputField(desc="改善が必要な現在の指示文")
    examples_with_feedback = dspy.InputField(desc="問題の発生状況と繰り返し見られるパターンを示す事例とフィードバック")

    failure_patterns: str = dspy.OutputField(desc="事例から特定された共通の失敗パターンを要約したもの")

    retrieval_queries: list[str] = dspy.OutputField(desc="問題事例から特定された共通の問題パターンに対応する関連文書をデータベース内で検索するための具体的な検索クエリ")

class GenerateRAGEnhancedInstruction(dspy.Signature):
    """取得した文書情報と事例分析結果を活用して、改善された指示文を生成する。"""

    current_instruction = dspy.InputField(desc="改善が必要な現在の指示文")
    relevant_documentation = dspy.InputField(desc="専門文書から取得したガイドラインおよびベストプラクティス")
    examples_with_feedback = dspy.InputField(desc="現在の指示文で発生した問題を示す事例")

    improved_instruction: str = dspy.OutputField(desc="取得したガイドラインを反映し、事例で示された問題に対処した改良版指示文")

class RAGInstructionImprover(dspy.Module):
    """RAG技術を用いて専門文書を活用し、指示文を改善するモジュール。"""

    def __init__(self, retrieval_model):
        super().__init__()
        self.retrieve = retrieval_model  # dspy.Retrieveまたはカスタム検索モデルのいずれかを使用可能
        self.query_generator = dspy.ChainOfThought(GenerateDocumentationQuery)
        self.generate_answer = dspy.ChainOfThought(GenerateRAGEnhancedInstruction)

    def forward(self, current_instruction: str, component_examples: list):
        """取得した文書情報を活用して指示文を改善する。"""

        # LMに事例を分析させ、問題パターンに対応した具体的な検索クエリを生成させる
        query_result = self.query_generator(
            current_instruction=current_instruction,
            examples_with_feedback=component_examples
        )

        results = self.retrieve.query(
            query_texts=query_result.retrieval_queries,
            n_results=3
        )

        relevant_docs_parts = []
        for i, (query, query_docs) in enumerate(zip(query_result.retrieval_queries, results['documents'])):
            if query_docs:
                docs_formatted = "\n".join([f"  - {doc}" for doc in query_docs])
                relevant_docs_parts.append(
                    f"**検索クエリ #{i+1}**: {query}\n"
                    f"**取得ガイドライン**:\n{docs_formatted}"
                )

        relevant_docs = "\n\n" + "="*60 + "\n\n".join(relevant_docs_parts) + "\n" + "="*60

        # 取得した文脈情報を活用して改良版指示文を生成
        result = self.generate_answer(
            current_instruction=current_instruction

#### 統合パターン

**外部言語モデルと連携したカスタムプロポーザの使用：**

```python
class ExternalLMProposer(ProposalFn):
    def __init__(self):
        # 外部言語モデルインスタンスを独自に管理
        self.external_lm = dspy.LM('gemini/gemini-2.5-pro')

    def __call__(self, candidate, reflective_dataset, components_to_update):
        updated_components = {}

        with dspy.context(lm=self.external_lm):
            # self.external_lmを使用した独自の処理ロジックをここに記述
            for component_name in components_to_update:
                # ... 実装内容
                pass

        return updated_components

gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=None,  # カスタムプロポーザーを使用する場合のオプション引数
    instruction_proposer=ExternalLMProposer(),
    auto="medium"
)
```

**推奨プラクティス:**

- **DSPyの機能を最大限に活用する**: 直接言語モデルを呼び出すのではなく、`dspy.Module`、`dspy.Signature`、`dspy.Predict`などのDSPyコンポーネントを利用して指示提案器を構築することを推奨します。制約条件の充足には`dspy.Refine`を、複雑な推論タスクには`dspy.ChainOfThought`を使用するなど、複数のモジュールを組み合わせて高度な指示改善ワークフローを構築してください
- **包括的なフィードバック分析を有効にする**: dspy.GEPAの`GEPAFeedbackMetric`は1組の（正解データ、予測結果）ペアを逐次処理しますが、指示提案器はコンポーネントごとの全例を一括処理するため、例間のパターン検出や体系的な問題点の特定が可能になります
- **データのシリアライゼーションに注意する**: すべてのデータを文字列に変換するのは最適解ではない場合があります。`dspy.Image`のような複雑な入力タイプについては、その構造を保持したまま処理することで、言語モデルによる適切な処理が可能になります
- **徹底的なテストを実施する**: 代表的な失敗ケースを用いて、カスタム提案器の動作を十分に検証してください

## カスタムコンポーネント選択について

### component_selectorとは何か？

`component_selector`パラメータは、GEPAの各反復処理においてDSPyプログラム内のどのコンポーネント（予測器）を選択して最適化するかを制御します。デフォルトではラウンドロビン方式が採用されており、1回の反復ごとに1つのコンポーネントが更新されますが、最適化状態やパフォーマンスの推移、その他の文脈情報に基づいて、単一または複数のコンポーネントを選択するカスタム選択戦略を実装することも可能です。

### デフォルト動作

デフォルトでは、GEPAは**ラウンドロビン戦略**（`RoundRobinReflectionComponentSelector`）を採用しており、コンポーネントを順次循環させながら、1回の反復ごとに1つのコンポーネントを最適化します：

```python
# デフォルトのラウンドロビン方式によるコンポーネント選択
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000, api_key=api_key),
    # component_selector="round_robin"  # これはデフォルト設定
    auto="medium"
)
```

### 組み込み選択戦略

**文字列ベースのセレクタ：**

- `"round_robin"`（デフォルト）：コンポーネントを順次切り替えながら処理
- `"all"`：すべてのコンポーネントを同時に最適化対象として選択

```python
# すべてのコンポーネントを同時に最適化する場合
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=reflection_lm,
    component_selector="all",  # すべてのコンポーネントをまとめて更新
    auto="medium"
)

# 明示的なラウンドロビン選択を行う場合
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=reflection_lm,
    component_selector="round_robin",  # 各反復で1つのコンポーネントを選択
    auto="medium"
)
```

### カスタムコンポーネント選択を適用すべきケース

以下の要件がある場合、カスタムコンポーネント選択の実装を検討してください：

- **依存関係を考慮した最適化**：関連するコンポーネント群をまとめて更新する必要がある場合（例：分類器とその入力フォーマッタ）
- **LLMによる選択判断**：LLMに軌跡データを分析させ、どのコンポーネントに優先的に対処すべきかを判断させる場合
- **リソース制約を考慮した最適化**：最適化の徹底度と計算リソースのバランスを取る必要がある場合

### カスタムコンポーネント選択器プロトコル

カスタムコンポーネント選択器を実装するには、[`ReflectionComponentSelector`](https://github.com/gepa-ai/gepa/blob/main/src/gepa/proposer/reflective_mutation/base.py)プロトコルに準拠する必要があります。具体的には、呼び出し可能なクラスまたは関数を定義してください。GEPAは最適化処理中にこの選択器を自動的に呼び出します。

```python
from dspy.teleprompt.gepa.gepa_utils import GEPAState, Trajectory

class CustomComponentSelector:
    def __call__(
        self,
        state: GEPAState,                    # 履歴を含む完全な最適化状態
        trajectories: list[Trajectory],      # 現在のミニバッチからの実行トレース
        subsample_scores: list[float],       # 現在のミニバッチ内各例に対するスコア
        candidate_idx: int,                  # 現在最適化対象としているプログラム候補のインデックス
        candidate: dict[str, str],           # コンポーネント名 -> 命令マッピングの辞書
    ) -> list[str]:                          # 最適化対象とするコンポーネント名のリストを返す
        # ここに独自のコンポーネント選択ロジックを実装する
        return selected_components

# または関数として定義する場合:
def custom_component_selector(state, trajectories, subsample_scores, candidate_idx, candidate):
    # ここに独自のコンポーネント選択ロジックを実装する
    return selected_components
```

### カスタム実装例

以下に、コンポーネントの異なる半分を交互に最適化するシンプルな関数の実装例を示します：

```python
def alternating_half_selector(state, trajectories, subsample_scores, candidate_idx, candidate):
    """偶数反復ではコンポーネントの半数を、奇数反復では残りの半数を最適化する。"""
    components = list(candidate.keys())

    # コンポーネントが1つしかない場合は常に最適化する
    if len(components) <= 1:
        return components

    mid_point = len(components) // 2

    # 状態変数 state.i を用いて反復回数に応じて最適化対象を切り替える
    if state.i % 2 == 0:
        # 偶数反復時：前半のコンポーネントを最適化
        return components[:mid_point]
    else:
        # 奇数反復時：後半のコンポーネントを最適化
        return components[mid_point:]

# 使用例
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=reflection_lm,
    component_selector=alternating_half_selector,
    auto="medium"
)
```

### カスタム命令提案器との連携

コンポーネントセレクタはカスタム命令提案器とシームレスに連携して動作します。セレクタはどのコンポーネントを更新するかを決定し、その後命令提案器がそのコンポーネント向けの新規命令を生成します：

```python
# カスタムセレクタとカスタムプロポーザを組み合わせた GEPA インスタンス
gepa = dspy.GEPA(
    metric=my_metric,
    reflection_lm=reflection_lm,
    component_selector=alternating_half_selector,
    instruction_proposer=WordLimitProposer(max_words=500),
    auto="medium"
)
```
