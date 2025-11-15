# DSPyの本番環境での活用

<div class="grid cards" style="text-align: left;" markdown>

- :material-earth:{ .lg .middle } __実運用事例__

    ---

    DSPyは多くの企業やスタートアップで本番環境に導入されています。実際の導入事例をご覧ください。

    [:octicons-arrow-right-24: 導入事例](../community/use-cases.md)

- :material-magnify-expand:{ .lg .middle } __モニタリング＆可観測性__

    ---

    OpenTelemetryベースの**MLflow Tracing**を使用して、DSPyプログラムのモニタリングを実施できます。

    [:octicons-arrow-right-24: 可観測性の設定](../tutorials/observability/index.md#tracing)

- :material-ab-testing: __再現性の確保__

    ---

    DSPyのネイティブなMLflow統合機能により、プログラム実行ログ、メトリクス、設定情報、実行環境を記録することで、完全な再現性を実現します。

    [:octicons-arrow-right-24: MLflow統合機能](https://mlflow.org/docs/latest/llms/dspy/index.html)

- :material-rocket-launch: __デプロイメント__

    ---

    本番環境への移行時には、DSPyのMLflow Model Serving統合機能を活用して、アプリケーションのデプロイを容易に行えます。

    [:octicons-arrow-right-24: デプロイメントガイド](../tutorials/deployment/index.md)

- :material-arrow-up-right-bold: __スケーラビリティ__

    ---

    DSPyはスレッドセーフ性を考慮して設計されており、高スループット環境向けにネイティブな非同期実行機能をサポートしています。

    [:octicons-arrow-right-24: 非同期プログラム](../api/utils/asyncify.md)

- :material-alert-rhombus: __ガードレール＆制御性__

    ---

    DSPyの**シグネチャ**、**モジュール**、**オプティマイザ**機能を活用することで、LLMの出力を適切に制御・誘導することが可能です。

    [:octicons-arrow-right-24: シグネチャの学習](../learn/programming/signatures.md)

</div>
