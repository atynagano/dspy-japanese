# DSPy アサーション機能

!!! warning "アサーション機能は非推奨となっており、現在サポートされていません。代わりに `dspy.Refine` モジュール（または `dspy.Suggest` モジュール）のご利用を推奨します。"

以下の内容は非推奨となっており、今後削除される予定です。

## はじめに

言語モデル（LM）は機械学習とのインタラクション方法を革新し、自然言語理解・生成において多大な可能性をもたらしました。しかしながら、これらのモデルが特定のドメイン固有の制約条件を満たすことを保証することは依然として課題となっています。ファインチューニングや「プロンプトエンジニアリング」といった手法が発展しているとはいえ、これらのアプローチは極めて手間がかかり、LMを特定の制約条件に従わせるためには煩雑な手動操作が必要となります。DSpyのプログラミングプロンプティングパイプラインのモジュール化機能でさえ、これらの制約条件を効果的かつ自動的に強制する仕組みを備えていません。

この課題に対処するため、我々はDSpyフレームワーク内に「DSpy アサーション」機能を導入しました。DSpy アサーションは、LMに対する計算制約の自動強制を実現する機能です。開発者はこの機能を活用することで、最小限の手動介入でLMを所望の結果へと導くことが可能となり、LMの出力の信頼性・予測可能性・正確性を向上させることができます。

### dspy.Assert および dspy.Suggest API

DSpy アサーションでは以下の2つの主要な構成要素を導入しています：

- **`dspy.Assert`**:
  - **パラメータ**:
    - `constraint (bool)`: Pythonで定義されたブール値検証チェックの結果
    - `msg (Optional[str])`: フィードバックや修正ガイダンスを提供するユーザー定義のエラーメッセージ
    - `backtrack (Optional[module])`: 制約条件に失敗した場合の再試行対象モジュールを指定します。デフォルトのバックトラッキングモジュールは、アサーションの直前のモジュールとなります
  - **動作**: 失敗時には再試行を開始し、パイプラインの実行を動的に調整します。再試行を繰り返しても失敗が解消しない場合、実行を停止して `dspy.AssertionError` を発生させます

- **`dspy.Suggest`**:
  - **パラメータ**: `dspy.Assert` と同様
  - **動作**: 強制的な停止を行わずに再試行を通じて自己改善を促します。最大バックトラッキング試行回数に達した時点で失敗をログに記録し、その後の実行を継続します

- **dspy.Assert と Python のアサーションの違い**: 従来のPython `assert` 文が失敗時にプログラムを終了させるのに対し、`dspy.Assert` は洗練された再試行メカニズムを備えており、パイプラインが動的に調整を行うことが可能です。

具体的には、制約条件が満たされない場合：

- バックトラッキング機構: バックグラウンドでバックトラッキングが開始され、モデルに自己改善の機会が与えられます。これはシグネチャの修正を通じて実現されます。
- 動的シグネチャ修正: DSPyプログラムのシグネチャを内部的に修正し、以下のフィールドを追加します：
    - 過去の出力: 検証関数を通過しなかったモデルの過去の出力
    - 指示文: 何が問題だったのか、またどのように修正すべきかについてのユーザー定義のフィードバックメッセージ

`max_backtracking_attempts`の試行回数を超えてエラーが継続した場合、`dspy.Assert`はパイプラインの実行を停止し、`dspy.AssertionError`を発生させます。これにより、プログラムが「不適切な」言語モデル（LM）の動作状態で実行を継続することを防ぎ、同時にサンプルの失敗出力をユーザーに提示して評価を促すことができます。

- **`dspy.Suggest`と`dspy.Assert`の違い**：`dspy.Suggest`はより穏やかなアプローチを採用しています。基本的なリトライバックトラッキング機能は`dspy.Assert`と同様ですが、こちらはより控えめなガイダンス機能として機能します。`max_backtracking_attempts`の試行回数を超えてもモデルの出力がモデル制約を満たさない場合、`dspy.Suggest`はその継続的な失敗をログに記録し、残りのデータに対してプログラムの実行を継続します。これにより、LMパイプラインは「最善を尽くして」動作を継続することが可能になります。

- **`dspy.Suggest`**ステートメントは、評価フェーズにおいて「補助機能」として最適に活用できます。パイプラインを停止させることなく、ガイダンスの提供や潜在的な修正案を提示することが可能です。
- **`dspy.Assert`**ステートメントは、開発段階において「検証機能」としての使用が推奨されます。LMが期待通りに動作することを保証するための堅牢なメカニズムとして機能し、開発サイクルの早期段階でエラーを特定・修正するための有効な手段となります。


## 使用事例：DSPyプログラムへのアサーションの組み込み

はじめに、導入ガイドで定義されているマルチホップQA用SimplifiedBaleenパイプラインの具体例を用いて説明します。 

```python
class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=2, max_hops=2):
        super().__init__()

        # 各ホップで実行する思考連鎖生成器を初期化
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        # 検索結果取得モジュールを設定
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        # 回答生成用思考連鎖生成器を設定
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        # 最大ホップ数を設定
        self.max_hops = max_hops

    def forward(self, question):
        context = []
        prev_queries = [question]

        # 指定された最大ホップ数まで処理を繰り返す
        for hop in range(self.max_hops):
            # 現在のコンテキストと質問に基づいて検索クエリを生成
            query = self.generate_query[hop](context=context, question=question).query
            prev_queries.append(query)
            # 生成したクエリを用いて関連文書を取得
            passages = self.retrieve(query).passages
            # 取得した文書を既存のコンテキストと重複排除して更新
            context = deduplicate(context + passages)
        
        # 最終的なコンテキストに基づいて回答を生成
        pred = self.generate_answer(context=context, question=question)
        # 予測結果をコンテキストと回答を含む形式でラップ
        pred = dspy.Prediction(context=context, answer=pred.answer)
        return pred

# 簡易版バレーンモデルのインスタンスを作成
baleen = SimplifiedBaleen()

# 質問に対する回答を生成
baleen(question = "Gary Zukavの最初の著書が受賞した賞は何ですか？")
```

DSPyのアサーション機能を利用するには、検証用関数を定義し、モデル生成後に適切な形式でアサーションを宣言するだけでよい。

本使用例では、以下の制約条件を適用したい場合を想定する：
    1. 長さ制限 - 各クエリは100文字未満でなければならない
    2. 一意性 - 生成される各クエリは、既に生成されたクエリと重複してはならない

これらの検証チェックは、ブール値を返す関数として定義可能である：

```python
# クエリ長を簡易的に判定するブールチェック
len(query) <= 100

# 異なるクエリを検証するための Python 関数
def validate_query_distinction_local(previous_queries, query):
    """クエリが過去のクエリ群と異なるかどうかを確認する"""
    if previous_queries == []:
        return True
    if dspy.evaluate.answer_exact_match_str(query, previous_queries, frac=0.8):
        return False
    return True
```

これらの検証チェックは、`dspy.Suggest` ステートメントを通じて宣言することが可能です（本プログラムを最善の努力によるデモンストレーションとしてテストしたいため）。クエリ生成処理 `query = self.generate_query[hop](context=context, question=question).query` の後にもこれらのチェックを保持したいと考えています。

```python
dspy.Suggest(
    len(query) <= 100,
    "クエリは簡潔にし、100文字以内に収めてください",
    target_module=self.generate_query
)

dspy.Suggest(
    validate_query_distinction_local(prev_queries, query),
    "クエリは以下の内容と重複しないようにしてください: "
    + "; ".join(f"{i+1}) {q}" for i, q in enumerate(prev_queries)),
    target_module=self.generate_query
)
```

アサーションの効果を比較評価する場合、元のプログラムとは別にアサーションを組み込んだ専用プログラムを定義することをお勧めします。そうでない場合は、アサーションを無効に設定しても問題ありません。

アサーションを適用した場合のSimplifiedBaleenプログラムの外観について確認してみましょう：

```python
class SimplifiedBaleenAssertions(dspy.Module):
    def __init__(self, passages_per_hop=2, max_hops=2):
        super().__init__()
        # 各ホップで実行する思考連鎖生成器を初期化
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        # 検索結果取得モジュールを設定
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        # 回答生成用思考連鎖を設定
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        # 最大ホップ数を設定
        self.max_hops = max_hops

    def forward(self, question):
        context = []
        prev_queries = [question]

        for hop in range(self.max_hops):
            # 現在のホップで生成する検索クエリを取得
            query = self.generate_query[hop](context=context, question=question).query

            # クエリの長さが100文字以下であることを検証
            dspy.Suggest(
                len(query) <= 100,
                "クエリは簡潔にし、100文字以内に収めてください",
                target_module=self.generate_query
            )

            # クエリが直前のクエリ群と区別可能であることを検証
            dspy.Suggest(
                validate_query_distinction_local(prev_queries, query),
                "クエリは以下と区別可能である必要があります: "
                + "; ".join(f"{i+1}) {q}" for i, q in enumerate(prev_queries)),
                target_module=self.generate_query
            )

            # 現在のクエリを記録
            prev_queries.append(query)
            # 検索結果を取得
            passages = self.retrieve(query).passages
            # 文脈と検索結果を重複排除して更新
            context = deduplicate(context + passages)
        
        # 全てのクエリが互いに区別可能か確認
        if all_queries_distinct(prev_queries):
            self.passed_suggestions += 1

        # 最終的な回答を生成
        pred = self.generate_answer(context=context, question=question)
        # 予測結果を構造化して返す
        pred = dspy.Prediction(context=context, answer=pred.answer)
        return pred
```

DSPyアサーションを使用してプログラムを呼び出す場合、最後に1つの重要な手順が残っています。それは、内部アサーションによるバックトラッキングとリトライロジックを組み込むためのプログラム変換処理です。 

```python
from dspy.primitives.assertions import assert_transform_module, backtrack_handler

baleen_with_assertions = assert_transform_module(SimplifiedBaleenAssertions(), backtrack_handler)

# backtrack_handler はバックトラッキング機構の動作パラメータを複数設定可能
# 最大リトライ回数を変更する場合は、以下のように指定できる
baleen_with_assertions_retry_once = assert_transform_module(SimplifiedBaleenAssertions(), 
    functools.partial(backtrack_handler, max_backtracks=1))
```

別の方法として、デフォルトのバックトラッキング機構（`max_backtracks=2`）を使用して、`dspy.Assert/Suggest`ステートメントとともにプログラム上で直接`activate_assertions`を呼び出すことも可能です：

```python
baleen_with_assertions = SimplifiedBaleenAssertions().activate_assertions()
```

次に、LMクエリ生成履歴を調査することで、内部LMバックトラッキングの動作を確認する。検証チェック（100文字未満であること）に失敗した場合、バックトラッキング+リトライ処理中に内部関数`GenerateSearchQuery`のシグネチャが動的に変更され、過去のクエリと対応するユーザー定義指示（`"クエリは短く100文字未満である必要があります"`）が組み込まれる仕組みとなっている。


```text
複雑な質問に回答可能な簡潔な検索クエリを作成してください。

---

以下の形式に従ってください。

背景情報：関連する事実を含む場合があります

質問：${question}

推論プロセス：${query}を生成するため、段階的に考察します。まず...

検索クエリ：${query}

---

背景情報：
[1] «ケリー・コンドン | 1983年1月4日生まれのケリー・コンドンは...»
[2] «コロナ・リッカルド | 1878年頃～1917年10月15日没のコロナ・リッカルドは...»

質問：短編映画『ザ・ショア』に出演し、ロイヤル・シェイクスピア・カンパニー制作の『ハムレット』でオフィーリア役を演じた最年少女優は誰ですか？

推論プロセス：この質問に答えるため、段階的に考察します。まず第一に、ロイヤル・シェイクスピア・カンパニー制作の『ハムレット』でオフィーリアを演じた女優を特定する必要があります。次に、その女優が短編映画『ザ・ショア』にも出演しているかどうかを確認します。

検索クエリ：「ロイヤル・シェイクスピア・カンパニー版『ハムレット』におけるオフィーリア役の女優」＋「短編映画『ザ・ショア』出演女優」



複雑な質問に回答可能な簡潔な検索クエリを作成してください。

---

以下の形式に従ってください。

背景情報：関連する事実を含む場合があります

質問：${question}

過去のクエリ：エラーを含む以前の出力結果

指示事項：必ず遵守すべき要件

検索クエリ：${query}

---

背景情報：
[1] «ケリー・コンドン | 1983年1月4日生まれのケリー・コンドンは、HBO/BBC共同制作ドラマ『ローマ』におけるユリウス家のオクタヴィア役、AMCの『ベター・コール・ソウル』におけるステイシー・エアマントラウト役、そしてマーベル・シネマティック・ユニバースの各種作品でF.R.I.D.A.Y.の声を担当したことで知られるアイルランド出身のテレビ・映画女優です。また、ロイヤル・シェイクスピア・カンパニー制作の『ハムレット』でオフィーリアを演じた最年少女優としても記録されています。»
[2] «コロナ・リッカルド | 1878年頃～1917年10月15日没のコロナ・リッカルドは、イタリア生まれのアメリカ人女優で、ブロードウェイで短期間活動した後、結婚して家庭に入りました。ナポリ生まれの彼女は、1894年にエンパイア劇場で上演された戯曲でメキシコ人少女役を演じて女優デビューしました。ウィルソン・バレットに起用され、彼の作品『十字架の徴』でアメリカ各地を巡演するキャストに加わりました。リッカルドはアンカリア役を演じ、後に同作品でベレニケ役も演じました。1898年にはロバート・B・マンテルがその美貌に感銘を受け、『ロミオとジュリエット』と『オセロ』の2作品にもキャスティングしています。1899年に執筆したルイス・ストラングは、リッカルドを当時アメリカで最も有望な女優と評しました。1898年末、マンテルは彼女を再びシェイクスピア作品『ハムレット』のオフィーリア役に起用しました。その後、彼女はオーギュスト・ダリーの劇団に参加する予定でしたが、ダリーは1899年に死去しました。1899年、彼女は『ベン・ハー』の舞台初演でイラス役を演じ、最も大きな名声を得ました。»

質問：短編映画『ザ・ショア』に出演し、ロイヤル・シェイクスピア・カンパニー制作の『ハムレット』でオフィーリア役を演じた最年少女優は誰ですか？

過去のクエリ：「ロイヤル・シェイクスピア・カンパニー版『ハムレット』におけるオフィーリア役の女優」＋「短編映画『ザ・ショア』出演女優」

指示事項：クエリは簡潔にし、100文字未満に収めてください。

検索クエリ：「RSC版『ハムレット』オフィーリア役女優」＋「短編映画『ザ・ショア』出演女優」

```


## アサーション駆動型最適化手法

DSPyのアサーション機能は、DSPyが提供する各種最適化手法、特に`BootstrapFewShotWithRandomSearch`アルゴリズムと連携するように設計されている。以下に主要な設定項目を示す：

- コンパイル時のアサーション活用
    本設定では、コンパイルプロセスにおいて以下の2種類のアサーション駆動型手法を適用する：
    1. 例示ブートストラッピング：アサーションを用いて学習用の代表的な事例を生成
    2. 反例ブートストラッピング：アサーションを用いて学習プロセスの誤りを特定
    少数ショットデモンストレーション用の教師モデルは、DSPyのアサーション機能を活用することで、推論時に学生モデルが学習可能な堅牢な事例を生成できる。なお、この設定では学生モデルは推論時にアサーションを意識した最適化処理（バックトラッキングや再試行など）は行わない。
- コンパイル＋推論時のアサーション活用
    - 本設定では、コンパイル時だけでなく推論時にもアサーション駆動型最適化を実施する。具体的には、教師モデルがアサーションに基づく事例を提供する一方、学生モデルは推論時に独自のアサーションを用いてさらに最適化を行うことができる。 
```python
teleprompter = BootstrapFewShotWithRandomSearch(
    metric=validate_context_and_answer_and_hops,
    max_bootstrapped_demos=max_bootstrapped_demos,
    num_candidate_programs=6,
)

# アサーションを用いたコンパイル処理
compiled_with_assertions_baleen = teleprompter.compile(student = baleen, teacher = baleen_with_assertions, trainset = trainset, valset = devset)

# アサーションを用いたコンパイル＋推論処理
compiled_baleen_with_assertions = teleprompter.compile(student=baleen_with_assertions, teacher = baleen_with_assertions, trainset=trainset, valset=devset)

```
