# dspy.SIMBA

<!-- START_API_REF -->
::: dspy.SIMBA
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
<!-- END_API_REF -->

## 使用例

```python
optimizer = dspy.SIMBA(metric=your_metric)
optimized_program = optimizer.compile(your_program, trainset=your_trainset)

# 最適化済みプログラムを保存して将来の利用に備える
optimized_program.save(f"optimized.json")
```

## SIMBAの動作原理
SIMBA（Stochastic Introspective Mini-Batch Ascent）は、LLMを活用して自身のパフォーマンスを分析し、改善規則を生成するDSPy最適化手法である。本手法では以下のプロセスを実行する：
1. ミニバッチをサンプリングする
2. 出力値の変動が大きい困難な事例を特定する
3. 特定された事例に対して、自己内省的な改善規則を生成するか、または成功した事例をデモンストレーションとして追加する
詳細な解説については、[Marius](https://x.com/rasmus1610)による[こちらの優れたブログ記事](https://blog.mariusvach.com/posts/dspy-simba)を参照されたい。