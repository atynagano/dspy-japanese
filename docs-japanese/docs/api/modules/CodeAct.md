# dspy.CodeAct

<!-- START_API_REF -->
::: dspy.CodeAct
    handler: python
    options:
        members:
            - __call__
            - batch
            - deepcopy
            - dump_state
            - get_lm
            - inspect_history
            - load
            - load_state
            - map_named_predictors
            - named_parameters
            - named_predictors
            - named_sub_modules
            - parameters
            - predictors
            - reset_copy
            - save
            - set_lm
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
<!-- END_API_REF -->

# CodeAct

CodeActはDSPyモジュールの一つであり、コード生成とツール実行を統合することで問題解決を実現する機能を提供する。本モジュールは、指定されたツール群とPython標準ライブラリを活用してタスクを遂行するためのPythonコードスニペットを生成する。

## 基本的な使用方法

以下にCodeActの簡単な使用例を示す：

```python
import dspy
from dspy.predict import CodeAct

# 単純なツール関数を定義
def factorial(n: int) -> int:
    """数値の階乗を計算する。"""
    if n == 1:
        return 1
    return n * factorial(n-1)

# CodeActインスタンスを作成
act = CodeAct("n->factorial_result", tools=[factorial])

# CodeActインスタンスを使用
result = act(n=5)
print(result) # 階乗(5) = 120 を計算して出力する
```

## 動作原理

CodeActは反復的な処理フローで動作します：

1. 入力パラメータと利用可能なツールを取得
2. これらのツールを使用するPythonコードスニペットを生成
3. Pythonサンドボックス環境上で生成コードを実行
4. 出力結果を収集し、タスクの完了判定を実施
5. 収集した情報に基づいて元の質問に回答

## ⚠️ 制約事項

### ツールとしては純粋な関数のみを受け付けます（呼び出し可能なオブジェクトは非対応）

以下の例では、呼び出し可能なオブジェクトを使用しているため動作しません。

```python
# ❌ 不適切な実装例
class Add():
    def __call__(self, a: int, b: int):
        return a + b

dspy.CodeAct("question -> answer", tools=[Add()])
```

### 外部ライブラリの使用は不可

以下の例では、外部ライブラリ `numpy` を使用しているため正常に動作しません。

```python
# ❌ 不適切な実装例
import numpy as np

def exp(i: int):
    return np.exp(i)

dspy.CodeAct("質問 -> 回答", tools=[exp])
```

### `CodeAct` にすべての依存関数を指定する必要がある

`CodeAct` に指定されていない他の関数やクラスに依存する関数は使用できません。以下の例では、ツール関数が `Profile` や `secret_function` など、`CodeAct` に指定されていない他の関数やクラスに依存しているため、正常に動作しません。

```python
# ❌ 不適切な実装例
from pydantic import BaseModel

class Profile(BaseModel):
    name: str
    age: int
    
def age(profile: Profile):
    return 

def parent_function():
    print("Hi!")

def child_function():
    parent_function()

dspy.CodeAct("question -> answer", tools=[age, child_function])
```

代わりに、以下の例では問題なく動作します。これは、必要なすべてのツール関数が `CodeAct` に渡されているためです：

```python
# ✅ 正常動作

def parent_function():
    print("こんにちは!")

def child_function():
    parent_function()

dspy.CodeAct("質問 -> 回答", tools=[parent_function, child_function])
```
