# MLflowを用いたDSPyオプティマイザの追跡方法

本チュートリアルでは、DSPyの最適化プロセスを追跡・分析するためのMLflowの活用方法について解説します。MLflowが提供するDSPy向けの組み込み統合機能により、DSPyの最適化作業におけるトレーサビリティとデバッグ可能性が確保されます。これにより、最適化プロセス中の中間トライアルの把握、最適化済みプログラムとその実行結果の保存、およびプログラム実行状況の可視化が可能となります。

自動ロギング機能を通じて、MLflowは以下の情報を追跡します：

* **オプティマイザパラメータ**
    * 少数ショット例の数
    * 候補数
    * その他の設定パラメータ

* **プログラム状態**
    * 初期命令および少数ショット例
    * 最適化後の命令および少数ショット例
    * 最適化プロセス中の中間命令および少数ショット例

* **データセット**
    * 使用された学習データ
    * 使用された評価データ

* **性能推移**
    * 全体的な評価指標の推移
    * 各評価ステップにおける性能値

* **トレース情報**
    * プログラム実行トレース
    * モデルの応答内容
    * 中間プロンプト

## 導入手順

### 1. MLflowのインストール
まず初めに、MLflow（バージョン2.21.1以降）をインストールしてください：

```bash
pip install mlflow>=2.21.1
```

### 2. MLflowトラッキングサーバーの起動

以下のコマンドでMLflowトラッキングサーバーを起動します。これにより、ローカルサーバーが`http://127.0.0.1:5000/`で起動します：

```bash
# MLflow トレーシングを使用する場合、SQL ストアの使用を強く推奨します
mlflow server --backend-store-uri sqlite:///mydb.sqlite
```

### 3. Autologging機能の有効化

MLflowを設定してDSPyの最適化プロセスを追跡可能にします：

```python
import mlflow
import dspy

# 全機能を有効にした自動ロギングを設定
mlflow.dspy.autolog(
    log_compiles=True,    # 最適化処理の過程を記録
    log_evals=True,       # 評価結果を記録
    log_traces_from_compile=True  # 最適化処理中のプログラムトレースを記録
)

# MLflowトラッキングの設定
mlflow.set_tracking_uri("http://localhost:5000")  # ローカルMLflowサーバーを使用
mlflow.set_experiment("DSPy-Optimization")
```

### 4. プログラムの最適化

以下に、数学問題解法プログラムの最適化状況を追跡する完全な実装例を示す：

```python
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

# 言語モデルの設定
lm = dspy.LM(model="openai/gpt-4o")
dspy.configure(lm=lm)

# データセットの読み込み
gsm8k = GSM8K()
trainset, devset = gsm8k.train, gsm8k.dev

# プログラムの定義
program = dspy.ChainOfThought("question -> answer")

# 追跡機能を備えたオプティマイザの作成と実行
teleprompter = dspy.teleprompt.MIPROv2(
    metric=gsm8k_metric,
    auto="light",
)

# 最適化プロセスは自動的に追跡されます
optimized_program = teleprompter.compile(
    program,
    trainset=trainset,
)
```

### 5. 結果の確認

最適化処理が完了したら、MLflowのUIを使用して結果を分析できます。以下に、最適化実行結果を確認する手順を説明します。

#### ステップ1: MLflow UIへのアクセス
ウェブブラウザで `http://localhost:5000` にアクセスし、MLflowトラッキングサーバーのUIを表示します。

#### ステップ2: 実験構造の理解
実験ページを開くと、最適化プロセスの階層構造が表示されます。親実行は全体の最適化プロセスを表し、子実行は最適化過程で作成されたプログラムの各中間バージョンを示しています。

![実験ページ](./experiment.png)

#### ステップ3: 親実行の分析
親実行をクリックすると、最適化プロセスの全体像を確認できます。オプティマイザの設定パラメータや、評価指標の時間経過に伴う変化など、詳細な情報が表示されます。親実行には、最終的な最適化済みプログラム（命令文、シグネチャ定義、few-shot例など）が保存されています。さらに、最適化プロセスで使用された学習データも確認可能です。

![親実行詳細](./parent_run.png)

#### ステップ4: 子実行の調査
各子実行は、特定の最適化試行の詳細なスナップショットを提供します。実験ページから子実行を選択すると、その中間プログラムに関する様々な側面を確認できます。
実行パラメータタブまたはアーティファクトタブでは、中間プログラムで使用された命令文やfew-shot例を確認できます。
特に有用な機能として「トレース」タブがあり、ここではプログラムの実行過程をステップバイステップで確認できます。これにより、DSPyプログラムがどのように入力を処理し、出力を生成するのかを正確に理解できます。

![子実行詳細](./child_run.png)

### 6. 推論用モデルの読み込み
最適化済みプログラムは、MLflowトラッキングサーバーから直接読み込み、推論に使用することができます：

```python
model_path = mlflow.artifacts.download_artifacts("mlflow-artifacts:/path/to/best_model.json")
program.load(model_path)
```

## トラブルシューティング

- トレースが表示されない場合、`log_traces_from_compile=True` が設定されていることを確認してください。
- 大規模なデータセットを扱う場合、メモリ使用量を抑えるため `log_traces_from_compile=False` の設定を検討してください。
- MLflowの実行データにプログラム的にアクセスするには、`mlflow.get_run(run_id)` メソッドを使用してください。

その他の機能については、[MLflow公式ドキュメント](https://mlflow.org/docs/latest/llms/dspy) を参照してください。
