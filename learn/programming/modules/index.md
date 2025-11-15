# モジュール

**DSPyモジュール**とは、大規模言語モデル（LM）を利用するプログラムの構成要素となる基本単位です。

- 各組み込みモジュールは、**プロンプティング手法**（思考連鎖法やReActなど）を抽象化した機能を提供します。重要な点として、これらのモジュールは任意の入力形式に対応できるよう汎用的に設計されています。
- DSPyモジュールには**学習可能なパラメータ**（プロンプトを構成する要素やLMの重みパラメータなど）が含まれており、入力処理と出力生成のために呼び出し（実行）することが可能です。
- 複数のモジュールを組み合わせて、より大規模なモジュール（プログラム）を構築することができます。DSPyモジュールはPyTorchのニューラルネットワークモジュールから直接着想を得ていますが、これをLMベースのプログラムに適用した形で実装されています。

## `dspy.Predict`や`dspy.ChainOfThought`といった組み込みモジュールはどのように使用すればよいですか？

最も基本的なモジュールである`dspy.Predict`から説明しましょう。実際、他のすべてのDSPyモジュールは`dspy.Predict`を基盤として構築されています。ここでは、DSPyで使用するあらゆるモジュールの動作を定義するための宣言型仕様である[DSPyシグネチャ](../signatures/)について既にある程度の知識があることを前提とします。

モジュールを使用するには、まず**シグネチャを指定して宣言**します。その後、入力引数を与えてモジュールを**呼び出し**、最後に出力フィールドを抽出します！

```
sentence = "it's a charming and often moving journey."  # SST-2データセットからのサンプル文

# 1) 関数定義による宣言
classify = dspy.Predict('sentence -> sentiment: bool')

# 2) 入力引数を指定して関数を呼び出す
response = classify(sentence=sentence)

# 3) 出力結果を取得する
print(response.sentiment)
```

**出力結果:**

```
True
```

モジュールを宣言する際には、設定キーを渡すことが可能です。

以下の例では、`n=5`を指定して5つの補完結果を要求しています。また、`temperature`や`max_len`などのパラメータも指定できます。

ここでは`dspy.ChainOfThought`を使用してみましょう。多くの場合、`dspy.Predict`の代わりに`dspy.ChainOfThought`を使用するだけで、出力品質が大幅に向上します。

```
question = "ColBERT検索モデルの優れた特徴は何ですか？"

# 1) シグネチャを定義し、設定パラメータを指定してChainOfThoughtモデルを初期化する。
classify = dspy.ChainOfThought('question -> answer', n=5)

# 2) 入力引数を指定してモデルを呼び出す。
response = classify(question=question)

# 3) 出力結果を取得する。
response.completions.answer
```

**想定される出力例:**

```
['ColBERT検索モデルの最大の利点は、他のモデルと比較して優れた効率性と有効性を有している点である。',
 '大規模文書コレクションから関連性の高い情報を効率的に検索できる能力。',
 'ColBERT検索モデルの優れた特徴は、他のモデルと比較して優れた性能を発揮するとともに、事前学習済み言語モデルを効率的に活用している点にある。',
 'ColBERT検索モデルの優れた特徴は、他のモデルと比較して優れた効率性と精度を兼ね備えている点である。',
 'ColBERT検索モデルの優れた特徴は、ユーザーフィードバックを組み込める能力と、複雑なクエリに対応できる柔軟性を有している点である。']
```

ここで出力オブジェクトについて考察しましょう。`dspy.ChainOfThought`モジュールは通常、出力フィールド群の前に`reasoning`（推論過程）を追加します。

それでは、最初の推論プロセスと回答を詳しく見ていきましょう！

```
print(f"推論結果: {response.reasoning}")
print(f"回答: {response.answer}")
```

**想定される出力例:**

```
考察：ColBERTは、他の最先端検索モデルと比較して、効率性と有効性の両面で優れた性能を示すことが実証されている。文脈依存型埋め込み表現を採用しており、高精度かつスケーラブルな文書検索を実現している。
回答：ColBERT検索モデルの最大の特長は、他のモデルと比較して優れた効率性と有効性を兼ね備えている点にある。
```

これは、単一の補完結果を取得する場合でも、複数の補完結果を取得する場合でも、同様に利用可能である。

また、これらの異なる補完結果は、`Prediction`オブジェクトのリストとして、あるいは各フィールドごとに個別のリストとして取得することも可能である。

```
response.completions[3].reasoning == response.completions.reasoning[3]
```

**出力結果:**

```
True
```

## その他のDSPyモジュールにはどのような種類があり、どのように使用すればよいですか？

その他のモジュールも基本的な機能は似通っています。主に、署名機能の内部動作を変更する点が特徴です。

1. **`dspy.Predict`**: 基本的な予測モジュールです。署名機能自体の動作を変更するものではありません。学習プロセスにおける主要な機能（指示文やデモンストレーションの保存、言語モデルの更新など）を処理します。
1. **`dspy.ChainOfThought`**: 言語モデルに対して、署名機能の応答を決定する前に段階的な思考プロセスを踏ませるよう学習させます。
1. **`dspy.ProgramOfThought`**: 言語モデルに対して、実行時に特定の結果をもたらすコードを生成するよう学習させます。このコードの実行結果に基づいて最終的な応答が決定されます。
1. **`dspy.ReAct`**: 与えられた署名機能を実装するためにツールを使用できるエージェントモジュールです。
1. **`dspy.MultiChainComparison`**: `ChainOfThought`モジュールから複数の出力を生成し、それらを比較した上で最終的な予測を行うことができます。

また、関数型のモジュールもいくつか用意しています：

1. **`dspy.majority`**: 複数の予測結果から最も頻度の高い回答を返す、基本的な投票機能を提供します。

簡単なタスクにおけるDSPyモジュールの使用例

以下の例を試す前に、`lm`の設定を行ってください。フィールドを調整することで、お使いの言語モデルが標準状態でどの程度のタスクを得意とするかを確認できます。

```
math = dspy.ChainOfThought("question -> answer: float")
math(question="2つのサイコロを投げた時、合計が2になる確率は？")
```

**出力例:**

```
Prediction(
    reasoning='2つのサイコロを投げると、各サイコロには6面あるため、合計36通りの結果が生じ得ます。2つのサイコロの目の合計が2となるのは、両方のサイコロが1を示す場合のみです。これは唯一の特定結果です：(1, 1)。したがって、有利な結果は1通りのみです。合計が2となる確率は、有利な結果の数を全可能な結果の数で割った値、すなわち1/36となります。',
    answer=0.0277776
)
```

```
def search(query: str) -> list[str]:
    """Wikipediaから抄録を取得します。"""
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
    return [x['text'] for x in results]

rag = dspy.ChainOfThought('context, question -> response')

question = "デイヴィッド・グレゴリーが相続した城の名前は何ですか？"
rag(context=search(question), question=question)
```

**出力例:**

```
Prediction(
    reasoning='文脈にはスコットランドの医師で発明家であるデイヴィッド・グレゴリーに関する情報が含まれています。特に1664年に彼がキンネアディー城を相続したことが明記されています。この情報が直接、デイヴィッド・グレゴリーが相続した城の名前に関する質問に答えています。',
    response='キンネアディー城'
)
```

```
from typing import Literal

class Classify(dspy.Signature):
    """与えられた文の感情分類を行う"""

    sentence: str = dspy.InputField()  # 入力文
    sentiment: Literal['positive', 'negative', 'neutral'] = dspy.OutputField()  # 感情分類結果
    confidence: float = dspy.OutputField()  # 分類の信頼度

classify = dspy.Predict(Classify)
classify(sentence="この本はとても楽しく読めたが、最終章ではなかった")
```

**出力例:**

```
Prediction(
    sentiment='positive',
    confidence=0.75
)
```

```
text = "Apple Inc.は本日、最新モデルのiPhone 14を発表した。ティム・クックCEOはプレスリリースでその新機能を強調した"

module = dspy.Predict("text -> title, headings: list[str], entities_and_metadata: list[dict[str, str]]")
response = module(text=text)

print(response.title)
print(response.headings)
print(response.entities_and_metadata)
```

**出力例:**

```
Apple Unveils iPhone 14
['Introduction', 'Key Features', "CEO's Statement"]
[{'entity': 'Apple Inc.', 'type': 'Organization'}, {'entity': 'iPhone 14', 'type': 'Product'}, {'entity': 'Tim Cook', 'type': 'Person'}]
```

```
def evaluate_math(expression: str) -> float:
    return dspy.PythonInterpreter({}).execute(expression)

def search_wikipedia(query: str) -> str:
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
    return [x['text'] for x in results]

react = dspy.ReAct("question -> answer: float", tools=[evaluate_math, search_wikipedia])

pred = react(question="9362158をキンネアディー城のデイヴィッド・グレゴリー氏の生年に割った値は何か?")
print(pred.answer)
```

**出力例:**

```
5761.328
```

## 複数のモジュールを組み合わせてより大規模なプログラムを作成するにはどうすればよいですか？

DSPyは単なるPythonコードであり、任意の制御フローでモジュールを使用できます。内部的には`compile`時にLM呼び出しをトレースする若干の魔法が働いていますが、基本的にはモジュールを自由に呼び出せるようになっています。

[マルチホップ検索](https://dspy.ai/tutorials/multihop_search/)などのチュートリアルを参照してください。以下に例としてそのモジュールを再掲します。

```
class Hop(dspy.Module):
    def __init__(self, num_docs=10, num_hops=4):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought('claim, notes -> query')
        self.append_notes = dspy.ChainOfThought('claim, notes, context -> new_notes: list[str], titles: list[str]')

    def forward(self, claim: str) -> list[str]:
        notes = []
        titles = []

        for _ in range(self.num_hops):
            query = self.generate_query(claim=claim, notes=notes).query
            context = search(query, k=self.num_docs)
            prediction = self.append_notes(claim=claim, notes=notes, context=context)
            notes.extend(prediction.new_notes)
            titles.extend(prediction.titles)

        return dspy.Prediction(notes=notes, titles=list(set(titles)))
```

次に、カスタムモジュールクラス `Hop` のインスタンスを作成し、`__call__` メソッドを用いて呼び出します：

```
hop = Hop()
print(hop(claim="Stephen Curry は人類史上最も優れた 3 ポイントシューターである"))
```

## 言語モデル（LM）の使用状況をどのように追跡すればよいですか？

バージョン要件

LM使用状況の追跡機能は、DSPyバージョン2.6.16以降で利用可能です。

DSPyでは、すべてのモジュール呼び出しにおける言語モデルの使用状況を自動的に追跡できます。追跡機能を有効にするには：

```
dspy.configure(track_usage=True)
```

有効化後、任意の `dspy.Prediction` オブジェクトから使用統計情報にアクセス可能です：

```
usage = prediction_instance.get_lm_usage()
```

使用統計データは、各言語モデル名をその使用統計値に対応付ける辞書形式で返される。以下に完全な使用例を示す：

```
import dspy

# DSPyを設定し、使用状況トラッキングを有効にする
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini", cache=False),
    track_usage=True
)

# 複数の言語モデル呼び出しを行うシンプルなプログラムを定義する
class MyProgram(dspy.Module):
    def __init__(self):
        self.predict1 = dspy.ChainOfThought("質問 -> 回答")
        self.predict2 = dspy.ChainOfThought("質問, 回答 -> スコア")

    def __call__(self, question: str) -> str:
        answer = self.predict1(question=question)
        score = self.predict2(question=question, answer=answer)
        return score

# プログラムを実行し、使用状況を確認する
program = MyProgram()
output = program(question="フランスの首都はどこですか？")
print(output.get_lm_usage())
```

これにより、以下のような使用統計情報が出力されます：

```
{
    'openai/gpt-4o-mini': {
        'completion_tokens': 61,
        'prompt_tokens': 260,
        'total_tokens': 321,
        'completion_tokens_details': {
            'accepted_prediction_tokens': 0,
            'audio_tokens': 0,
            'reasoning_tokens': 0,
            'rejected_prediction_tokens': 0,
            'text_tokens': None
        },
        'prompt_tokens_details': {
            'audio_tokens': 0,
            'cached_tokens': 0,
            'text_tokens': None,
            'image_tokens': None
        }
    }
}
```

DSPyのキャッシュ機能（メモリ内キャッシュまたはlitellmを介したディスクキャッシュ）を使用する場合、キャッシュされた応答は使用統計には反映されません。具体例を挙げると：

```
# キャッシュ機能を有効にする
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini", cache=True),
    track_usage=True
)

program = MyProgram()

# 初回呼び出し時 - 使用統計情報が表示される
output = program(question="ザンビアの首都はどこですか？")
print(output.get_lm_usage())  # トークン使用量が表示される

# 2回目の呼び出し - 同一質問の場合、キャッシュが使用される
output = program(question="ザンビアの首都はどこですか？")
print(output.get_lm_usage())  # 空の辞書 {}が表示される
```
