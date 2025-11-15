---
sidebar_position: 1
hide:
  - navigation
  - toc

---

![DSPy](static/img/dspy_logo.png){ width="200", align=left }

# _プログラミング_—プロンプト作成ではなく_LMs_の_プログラミング_

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/dspy?period=monthly)](https://pepy.tech/projects/dspy)

DSPyはモジュール型AIソフトウェアを構築するための宣言型フレームワークです。脆弱な文字列ベースのアプローチではなく、**構造化されたコードを迅速に反復開発**することが可能で、**AIプログラムを効果的なプロンプトと重み付けパラメータにコンパイル**するアルゴリズムを提供します。これにより、単純な分類器から高度なRAGパイプライン、Agentループに至るまで、あらゆるタイプのAIシステムを構築できます。

プロンプトの調整やトレーニングジョブの管理に煩わされることなく、DSPy（Declarative Self-improving Python）では**自然言語モジュールからAIソフトウェアを構築**し、それらを**汎用的に異なるモデル、推論戦略、学習アルゴリズムと組み合わせて構成**することが可能です。これにより、AIソフトウェアの**信頼性、保守性、移植性**がモデルや戦略を超えて向上します。

*要約*: DSPyをAIプログラミングのための高レベル言語と考えてください（[講義資料](https://www.youtube.com/watch?v=JEMYuzrKLUw)）。アセンブリ言語からC言語への移行、あるいはポインタ演算からSQLへの移行と同様の進化です。コミュニティに参加し、サポートを求めるか、[GitHub](https://github.com/stanfordnlp/dspy)および[Discord](https://discord.gg/XCGy2WDCQB)を通じて貢献を開始してください。

<!-- このフレームワークの抽象化により、AIソフトウェアの信頼性と保守性が向上し、新たなモデルや学習手法が登場するにつれてより移植性の高いシステムを構築できるようになります。また、単に非常に洗練された設計であることも特徴です -->

!!! info "入門編 I: DSPyのインストールと言語モデルの設定"

    ```bash
    > pip install -U dspy
    ```

    === "OpenAI"
        環境変数`OPENAI_API_KEY`を設定するか、以下のコードで`api_key`を直接指定することで認証できます。

        ```python linenums="1"
        import dspy
        lm = dspy.LM("openai/gpt-4o-mini", api_key="YOUR_OPENAI_API_KEY")
        dspy.configure(lm=lm)
        ```

    === "Anthropic"
        環境変数`ANTHROPIC_API_KEY`を設定するか、以下のコードで`api_key`を直接指定することで認証できます。

        ```python linenums="1"
        import dspy
        lm = dspy.LM("anthropic/claude-sonnet-4-5-20250929", api_key="YOUR_ANTHROPIC_API_KEY")
        dspy.configure(lm=lm)
        ```

    === "Databricks"
        Databricksプラットフォームを使用している場合、SDK経由で認証が自動的に行われます。それ以外の場合は、環境変数`DATABRICKS_API_KEY`と`DATABRICKS_API_BASE`を設定するか、以下のコードで`api_key`と`api_base`を直接指定してください。

        ```python linenums="1"
        import dspy
        lm = dspy.LM(
            "databricks/databricks-llama-4-maverick",
            api_key="YOUR_DATABRICKS_ACCESS_TOKEN",
            api_base="YOUR_DATABRICKS_WORKSPACE_URL",  # 例: https://dbc-64bf4923-e39e.cloud.databricks.com/serving-endpoints
        )
        dspy.configure(lm=lm)
        ```

    === "Gemini"
        環境変数`GEMINI_API_KEY`を設定するか、以下のコードで`api_key`を直接指定することで認証できます。

        ```python linenums="1"
        import dspy
        lm = dspy.LM("gemini/gemini-2.5-flash", api_key="YOUR_GEMINI_API_KEY")
        dspy.configure(lm=lm)
        ```

    === "ノートPC上で動作させるローカルLM"
          まず[Ollama](https://github.com/ollama/ollama)をインストールし、指定した言語モデルを使用してサーバーを起動してください。

          ```bash
          > curl -fsSL https://ollama.ai/install.sh | sh
          > ollama run llama3.2:1b
          ```

          その後、DSPyのコードからこのサーバーに接続します。

        ```python linenums="1"
        import dspy
        lm = dspy.LM("ollama_chat/llama3.2:1b", api_base="http://localhost:11434", api_key="")
        dspy.configure(lm=lm)
        ```

    === "GPUサーバー上で動作させるローカルLM"
          まず[SGLang](https://docs.sglang.ai/get_started/install.html)をインストールし、指定した言語モデルを使用してサーバーを起動してください。

          ```bash
          > pip install "sglang[all]"
          > pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/ 

          > CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --port 7501 --model-path meta-llama/Llama-3.1-8B-Instruct
          ```
        
        `meta-llama/Llama-3.1-8B-Instruct`をMetaからダウンロードできない場合は、「Qwen/Qwen2.5-7B-Instruct」などを代わりに使用してください。

        次に、DSPyのコードからこのローカルLMをOpenAI互換のエンドポイントとして接続します。

          ```python linenums="1"
          lm = dspy.LM("openai/meta-llama/Llama-3.1-8B-Instruct",
                       api_base="http://localhost:7501/v1",  # 必ずこのポート番号を指定してください
                       api_key="local", model_type="chat")
          dspy.configure(lm=lm)
          ```

    === "その他のプロバイダー"
        DSPyでは、[LiteLLMがサポートする数十種類のLLMプロバイダー](https://docs.litellm.ai/docs/providers)のいずれかを使用できます。各プロバイダーの指示に従い、`{PROVIDER}_API_KEY`の設定方法および`{provider_name}/{model_name}`のコンストラクタへの渡し方を確認してください。

        具体例：

        - `anyscale/mistralai/Mistral-7B-Instruct-v0.1`の場合、`ANYSCALE_API_KEY`を使用
        - `together_ai/togethercomputer/llama-2-70b-chat`の場合、`TOGETHERAI_API_KEY`を使用
        - `sagemaker/<your-endpoint-name>`の場合、`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION_NAME`を使用
        - `azure/<your_deployment_name>`の場合、`AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION`、およびオプションの`AZURE_AD_TOKEN`と`AZURE_API_TYPE`を使用

        
        プロバイダーがOpenAI互換のエンドポイントを提供している場合、モデル名の先頭に`openai/`を付加するだけで使用できます。

        ```python linenums="1"
        import dspy
        lm = dspy.LM("openai/your-model-name", api_key="PROVIDER_API_KEY", api_base="YOUR_PROVIDER_URL")
        dspy.configure(lm=lm)
        ```

??? "LMを直接呼び出す方法"

     実用的なDSPyの利用には、モジュールの概念を理解する必要があります。これについては本ページの後半で詳しく説明します。ただし、先に設定した`lm`を直接呼び出すことも可能です。これにより統一されたAPIが提供され、自動キャッシュなどの便利な機能を活用できます。

     ```python linenums="1"       
     lm("これはテストです!", temperature=0.7)  # => ['これはテストです!']
     lm(messages=[{"role": "user", "content": "これはテストです!"}])  # => ['これはテストです!']
     ``` 


## 1) **モジュール**を使用することで、AIの動作を文字列ではなく**コードとして記述**できます。

信頼性の高いAIシステムを構築するには、迅速なイテレーションが不可欠です。しかし、プロンプトのメンテナンスには課題があります。LMや評価指標、パイプラインを変更するたびに、文字列やデータ構造を手動で調整しなければならないからです。2020年以降、私たちはトップクラスの複合LMシステムを10種類以上開発してきましたが、この問題を身をもって経験しました。そこでDSPyを開発し、AIシステム設計を特定のLMやプロンプティング戦略といった付随的な要素から切り離すことに成功したのです。

DSPyでは、プロンプト文字列の微調整から、**構造化され宣言的な自然言語モジュールを用いたプログラミング**へと焦点を移します。システム内の各AIコンポーネントについて、入力/出力の動作を_シグネチャ_として指定し、LMを呼び出すための戦略を割り当てるモジュールを選択します。DSPyはこれらのシグネチャをプロンプトに変換し、ユーザーが入力した出力を解析するため、異なるモジュールを組み合わせやすく、移植性が高く最適化可能なAIシステムを構築できます。


!!! info "入門編 II: 各種タスク向けのDSPyモジュール構築"
    上記で`lm`を設定した後、以下の例を試してみてください。各フィールドを調整することで、LMが標準状態でどの程度のタスクをこなせるか探索できます。以下の各タブでは、`dspy.Predict`、`dspy.ChainOfThought`、`dspy.ReAct`などのDSPyモジュールを、タスク固有の_シグネチャ_とともに設定しています。例えば、`question -> answer: float`というシグネチャは、モジュールに対して質問を受け取り`float`型の回答を生成するよう指示します。

    === "数学問題"

        ```python linenums="1"
        math = dspy.ChainOfThought("question -> answer: float")
        math(question="2つのサイコロを投げた時、合計が2になる確率は？")
        ```
        
        **想定出力例:**
        ```text
        Prediction(
            reasoning='2つのサイコロを投げた場合、各サイコロには6面あるため、合計36通りの結果が生じ得ます。2つのサイコロの目の合計が2になるのは、両方のサイコロが1を示す場合のみです。これは特定の1つの結果: (1, 1) に相当します。したがって、有利な結果は1通りのみです。合計が2になる確率は、有利な結果の数を可能な結果の総数で割った値、つまり1/36となります。',
            answer=0.0277776
        )
        ```

    === "RAGシステム"

        ```python linenums="1"       
        def search_wikipedia(query: str) -> list[str]:
            results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(query, k=3)
            return [x["text"] for x in results]
        
        rag = dspy.ChainOfThought("context, question -> response")

        question = "デイヴィッド・グレゴリーが相続した城の名前は何ですか？"
        rag(context=search_wikipedia(question), question=question)
        ```
        
        **想定出力例:**
        ```text
        Prediction(
            reasoning='文脈にはスコットランドの医師で発明家であるデイヴィッド・グレゴリーに関する情報が記載されています。特に1664年にキンナード城を相続したことが明記されており、これが質問に対する直接的な回答となります。',
            response='キンナード城'
        )
        ```

    === "分類タスク"

        ```python linenums="1"
        from typing import Literal

        class Classify(dspy.Signature):
            """与えられた文の感情分類を行う"""
            
            sentence: str = dspy.InputField()
            sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()
            confidence: float = dspy.OutputField()

        classify = dspy.Predict(Classify)
        classify(sentence="この本はとても楽しく読めましたが、最終章だけは例外でした。")
        ```
        
        **想定出力例:**

        ```text
        Prediction(
            sentiment='positive',
            confidence=0.75
        )
        ```

    === "情報抽出タスク"

        ```python linenums="1"        
        class ExtractInfo(dspy.Signature):
            """テキストから構造化情報を抽出する"""
            
            text: str = dspy.InputField()
            title: str = dspy.OutputField()
            headings: list[str] = dspy.OutputField()
            entities: list[dict[str, str]] = dspy.OutputField(desc="エンティティとそのメタデータのリスト")
        
        module = dspy.Predict(ExtractInfo)

        text = "Apple Inc. は本日、最新のiPhone 14を発表しました。" \
            "CEOのティム・クックはプレスリリースでその新機能を強調しました。"
        response = module(text=text)

        print(response.title)
        print(response.headings)
        print(response.entities)
        ```
        
        **想定出力例:**
        ```text
        Apple Inc. Announces iPhone 14
        ['導入部', "CEOの声明", '新機能']
        [{'name': 'Apple Inc.', 'type': '組織'}, {'name': 'iPhone 14', 'type': '製品'}, {'name': 'ティム・クック', 'type': '人物'}]
        ```

    === "エージェントシステム"

        ```python linenums="1"       
        def evaluate_math(expression: str):
            return dspy.PythonInterpreter({}).execute(expression)

        def search_wikipedia(query: str):
            results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(query, k=3)
            return [x["text"] for x in results]

        react = dspy.ReAct("question -> answer: float", tools=[evaluate_math, search_wikipedia])

        pred = react(question="キンネアディー城のデイヴィッド・グレゴリーの生年を9362158で割った値は何か？")
        print(pred.answer)
        ```
        
        **想定出力例:**

        ```text
        5761.328
        ```
    
    === "多段階パイプライン"

        ```python linenums="1"       
        class Outline(dspy.Signature):
            """トピックに関する包括的な概要を作成する"""
            
            topic: str = dspy.InputField()
            title: str = dspy.OutputField()
            sections: list[str] = dspy.OutputField()
            section_subheadings: dict[str, list[str]] = dspy.OutputField(desc="セクション見出しとサブ見出しの対応関係")

        class DraftSection(dspy.Signature):
            """記事の最上位セクションをドラフト作成する"""
            
            topic: str = dspy.InputField()
            section_heading: str = dspy.InputField()
            section_subheadings: list[str] = dspy.InputField()
            content: str = dspy.OutputField(desc="Markdown形式でフォーマットされたセクション内容")

        class DraftArticle(dspy.Module):
            def __init__(self):
                self.build_outline = dspy.ChainOfThought(Outline)
                self.draft_section = dspy.ChainOfThought(DraftSection)

            def forward(self, topic):
                outline = self.build_outline(topic=topic)
                sections = []
                for heading, subheadings in outline.section_subheadings.items():
                    section, subheadings = f"## {heading}", [f"### {subheading}" for subheading in subheadings]
                    section = self.draft_section(topic=outline.title, section_heading=section, section_subheadings=subheadings)
                    sections.append(section.content)
                return dspy.Prediction(title=outline.title, sections=sections)

        draft_article = DraftArticle()
        article = draft_article(topic="2002 FIFAワールドカップ")
        ```
        
        **想定出力例:**

        1500語程度の記事の例:

        ```text
        ## 予選プロセス

        2002 FIFAワールドカップの予選プロセスでは、以下のような一連の..... [紙面の都合上省略]。

        ### UEFA予選

        UEFA予選には50チームが参加し、13枠を争った..... [紙面の都合上省略]。

        .... [記事の続き]
        ```

        DSPyを使用することで、このような多段階モジュールの最適化が容易になります。システムの最終出力を評価できる限り、あらゆるDSPyオプティマイザが中間モジュールのチューニングを行うことが可能です。

??? "DSPyの実践的活用: 簡易スクリプト作成から高度なシステム構築まで"

    標準的なプロンプト設計では、インターフェース設計（「言語モデルにどのような動作をさせるべきか？」）と実装設計（「その動作をどのように指示するか？」）が混同されがちである。DSPyではこの問題に対処するため、前者を_シグネチャ_として分離することで、後者を推論したりデータから学習したりすることを可能にしている――これはより大きなプログラムの文脈においてである。

    
    最適化アルゴリズムを使用する前段階であっても、DSPyのモジュール機能により、直感的で移植性の高いコードとして効果的な言語モデルシステムをスクリプト化できる。多様なタスクや言語モデルに対応して、我々は_シグネチャテストスイート_を維持しており、これは組み込みのDSPyアダプタの信頼性を評価するものである。アダプタとは、最適化処理の前にシグネチャをプロンプトに変換するコンポーネントを指す。もし特定のタスクにおいて、単純なプロンプトが言語モデルの標準的なDSPy実装を常に上回る性能を示す場合、これはバグの可能性があるため、[問題報告](https://github.com/stanfordnlp/dspy/issues)を行っていただきたい。我々はこの情報を組み込みアダプタの改善に活用する予定である。

## 2) **最適化アルゴリズム**は、AIモジュールのプロンプトと重みを調整する。

DSPyは、自然言語注釈付きの高水準コードを、言語モデルをプログラムの構造や評価指標に整合させるための低水準計算処理、プロンプト、あるいは重み更新に変換するためのツール群を提供する。コードや評価指標を変更した場合でも、単に再コンパイルするだけで対応が可能である。

タスクの代表的な入力数十～数百件と、システム出力の品質を測定可能な_評価指標_があれば、DSPyの最適化アルゴリズムを使用できる。DSPyに搭載された各種最適化アルゴリズムは、**各モジュールに対して優れたfew-shot例を合成**することで機能する（例：`dspy.BootstrapRS`<sup>[1](https://arxiv.org/abs/2310.03714)</sup>）。さらに、**各プロンプトに対してより適切な自然言語指示を提案・知的に探索**する機能も備えている（例：[`dspy.GEPA`](https://dspy.ai/tutorials/gepa_ai_program/)<sup>[2](https://arxiv.org/abs/2507.19457)</sup>、`dspy.MIPROv2`<sup>[3](https://arxiv.org/abs/2406.11695)</sup>）。また、**モジュール用のデータセットを構築し、それらを用いて言語モデルの重みをファインチューニング**する機能も提供する（例：`dspy.BootstrapFinetune`<sup>[4](https://arxiv.org/abs/2407.10930)</sup>）。`dspy.GEPA`の実行に関する詳細なチュートリアルについては、[dspy.GEPAチュートリアル](https://dspy.ai/tutorials/gepa_ai_program/)を参照されたい。

!!! info "DSPyプログラムにおけるLMプロンプトまたは重みの最適化入門 III"
    標準的な最適化処理の実行コストは2米ドル程度で、所要時間は約20分である。ただし、非常に大規模な言語モデルやデータセットを使用する場合は注意が必要である。
    最適化コストは、使用する言語モデル、データセット、および設定に応じて、数セント程度から数十ドルに及ぶ場合がある。

    以下の例ではHuggingFace/datasetsライブラリを使用している。以下のコマンドでインストール可能である。

    ```bash
    > pip install -U datasets
    ```

    === "ReActエージェントのプロンプト最適化例"
        これは、Wikipedia検索を通じて質問に回答する`dspy.ReAct`エージェントを最小限の構成で完全に動作させる例であり、`HotPotQA`データセットからサンプリングした500件の質問-回答ペアを用いて、安価な`light`モードで`dspy.MIPROv2`による最適化を行うものである。

        ```python linenums="1"
        import dspy
        from dspy.datasets import HotPotQA

        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

        def search_wikipedia(query: str) -> list[str]:
            results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(query, k=3)
            return [x["text"] for x in results]

        trainset = [x.with_inputs('question') for x in HotPotQA(train_seed=2024, train_size=500).train]
        react = dspy.ReAct("question -> answer", tools=[search_wikipedia])

        tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", num_threads=24)
        optimized_react = tp.compile(react, trainset=trainset)
        ```

        このような簡易的な実行方法により、ReActのスコアが24%から51%に向上します。これは`gpt-4o-mini`に対してタスクの具体的な要件をより詳細に学習させることで実現しています。

    === "RAGシステム向けプロンプトの最適化"
        検索インデックス`search`、お好みの`dspy.LM`モデル、および質問と正解回答からなる小規模な`trainset`が与えられた場合、以下のコードスニペットを使用することで、DSpyモジュールとして実装されている組み込み`SemanticF1`評価指標を用いて、長文出力にも対応したRAGシステムの最適化が可能です。

        ```python linenums="1"
        class RAG(dspy.Module):
            def __init__(self, num_docs=5):
                self.num_docs = num_docs
                self.respond = dspy.ChainOfThought("context, question -> response")

            def forward(self, question):
                context = search(question, k=self.num_docs)   # 詳細は後述のチュートリアルを参照
                return self.respond(context=context, question=question)

        tp = dspy.MIPROv2(metric=dspy.evaluate.SemanticF1(decompositional=True), auto="medium", num_threads=24)
        optimized_rag = tp.compile(RAG(), trainset=trainset, max_bootstrapped_demos=2, max_labeled_demos=2)
        ```

        実際に実行可能な完全なRAGの例については、[このチュートリアル](tutorials/rag/index.ipynb)をご覧ください。この手法により、StackExchangeコミュニティの一部を対象としたRAGシステムの品質が相対的に10%向上します。

    === "分類タスクにおける重みの最適化"
        <details><summary>データセット設定コードを表示するにはクリックしてください。</summary>

        ```python linenums="1"
        import random
        from typing import Literal

        from datasets import load_dataset

        import dspy
        from dspy.datasets import DataLoader

        # Banking77データセットをロードします。
        CLASSES = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True).features["label"].names
        kwargs = {"fields": ("text", "label"), "input_keys": ("text",), "split": "train", "trust_remote_code": True}

        # データセットから最初の2,000例を読み込み、各*学習*例に対してヒントを付与する。
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

        # 分類タスク用のDSPyモジュールを定義する。学習時には利用可能な場合、ヒントを使用する。
        signature = dspy.Signature("text, hint -> label").with_updated_fields("label", type_=Literal[tuple(CLASSES)])
        classify = dspy.ChainOfThought(signature)
        classify.set_lm(lm)

        # BootstrapFinetuneによる最適化を実施する。
        optimizer = dspy.BootstrapFinetune(metric=(lambda x, y, trace=None: x.label == y.label), num_threads=24)
        optimized = optimizer.compile(classify, trainset=trainset)

        optimized(text="pending cash withdrawal"の意味を教えてください)

        # 完全なファインチューニングのチュートリアルについては、以下を参照：https://dspy.ai/tutorials/classification_finetuning/
        ```

        **出力例（最後の行から）：**
        ```text
        Prediction(
            reasoning='pending cash withdrawalとは、現金引き出しのリクエストが開始されたものの、まだ完了または処理されていない状態を指します。このステータスは、取引が進行中であり、資金が口座から引き落とされていないか、ユーザーが利用できる状態になっていないことを意味します。',
            label='pending_cash_withdrawal'
        )
        ```

        DSPy 2.5.29を使用した同様の簡易的なテストでは、GPT-4o-miniのスコアが66%から87%に向上しました。


??? "DSPyの最適化アルゴリズムの具体例を教えてください。異なる最適化手法はどのように機能しますか？"

    `dspy.MIPROv2`最適化アルゴリズムを例に説明します。まず、MIPROは**ブートストラッピング段階**から開始します。この段階では、最適化前のプログラムを受け取り、様々な入力に対して複数回実行することで、各モジュールの入力/出力動作のトレースを収集します。収集したトレースの中から、指定した評価指標で高いスコアを獲得した軌跡に関連するもののみを選択します。次に、MIPROは**接地提案段階**に移行します。ここでは、DSPyプログラムのコード、データ、およびプログラム実行時のトレース情報を参照し、プログラム内の各プロンプトに対して複数の潜在的な指示案を作成します。最後に、MIPROは**離散探索段階**を開始します。学習データセットからミニバッチをサンプリングし、各プロンプト構築に使用する指示案とトレースの組み合わせを提案します。提案されたプログラム候補はミニバッチ上で評価され、その結果に基づいてMIPROは時間の経過とともに提案の質を向上させる代理モデルを更新します。

    DSPyオプティマイザの強力な特徴の一つは、その組み合わせ可能性にある。`dspy.MIPROv2`を実行して生成されたプログラムを再度`dspy.MIPROv2`の入力として使用する、あるいは`dspy.BootstrapFinetune`の入力として用いることで、より優れた結果を得ることが可能である。これは`dspy.BetterTogether`の核心的な機能の一つである。別のアプローチとして、オプティマイザを実行した後に上位5つの候補プログラムを抽出し、それらを用いて`dspy.Ensemble`を構築することもできる。これにより、推論時の計算リソース（アンサンブル学習など）とDSPy独自の事前推論時計算リソース（最適化予算）の両方を、高度に体系的な方法で拡張することが可能となる。



<!-- 将来計画:
??? 環境におけるBootstrapRSまたはMIPRO（ローカルSGLang LMを使用）
MATHタスクにおけるBootstrapFS（Ollamaを使用したLlama-3.2などの小規模LMと、場合によっては大規模教師モデルを組み合わせて） -->



## 3) **DSPyのエコシステム**はオープンソースAI研究の発展に貢献する。

モノリシックな大規模言語モデル（LM）と比較して、DSPyのモジュール型アーキテクチャは、大規模なコミュニティがLMプログラムの構成的アーキテクチャ、推論時戦略、およびオプティマイザをオープンかつ分散型の方法で改善することを可能にする。これにより、DSPyユーザーはより高度な制御性を獲得し、反復作業を飛躍的に高速化できるとともに、最新のオプティマイザやモジュールを適用することでプログラムの性能を継続的に向上させることが可能となる。

DSPyの研究プロジェクトは2022年2月にスタンフォード大学NLP研究室で開始され、[ColBERT-QA](https://arxiv.org/abs/2007.00814)、[Baleen](https://arxiv.org/abs/2101.00436)、[Hindsight](https://arxiv.org/abs/2110.07752)といった初期の[複合LMシステム](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/)の開発から得られた知見を基盤としている。最初のバージョンは2022年12月に[DSP](https://arxiv.org/abs/2212.14024)として公開され、2023年10月までに[DSPy](https://arxiv.org/abs/2310.03714)へと進化を遂げた。[250名の貢献者](https://github.com/stanfordnlp/dspy/graphs/contributors)の協力により、DSPyはこれまで数万人もの人々にモジュール型LMプログラムの構築と最適化の手法を普及させてきた。

それ以来、DSPyコミュニティは最適化手法に関する広範な研究成果を生み出してきた。具体的には[MIPROv2](https://arxiv.org/abs/2406.11695)、[BetterTogether](https://arxiv.org/abs/2407.10930)、[LeReT](https://arxiv.org/abs/2410.23214)といった最適化アルゴリズム、[STORM](https://arxiv.org/abs/2402.14207)、[IReRa](https://arxiv.org/abs/2401.12178)、[DSPy Assertions](https://arxiv.org/abs/2312.13382)といったプログラムアーキテクチャ、[PAPILLON](https://arxiv.org/abs/2410.17127)、[PATH](https://arxiv.org/abs/2406.11706)、[WangLab@MEDIQA](https://arxiv.org/abs/2404.14544)、[UMDのプロンプティング事例研究](https://arxiv.org/abs/2406.06608)、[Haize Labsのレッドチーム演習プログラム](https://blog.haizelabs.com/posts/dspy/)といった新規問題への応用事例などが挙げられる。これらに加え、数多くのオープンソースプロジェクト、商用アプリケーション、および[多様なユースケース](community/use-cases.md)が開発されている。
