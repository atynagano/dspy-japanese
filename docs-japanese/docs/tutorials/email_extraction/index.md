# DSPyを用いた電子メールからの情報抽出

本チュートリアルでは、DSpyを活用したインテリジェントな電子メール処理システムの構築方法を説明します。本システムでは、各種電子メールから重要な情報を自動抽出し、その意図を分類し、さらに後続処理に適した形式でデータを構造化する機能を実現します。

## 構築するシステムの概要

本チュートリアルを修了すると、以下の機能を備えたDSpyベースの電子メール処理システムを構築できるようになります：

- **電子メールの種別分類**（注文確認メール、サポート依頼メール、会議招待メールなど）
- **主要エンティティの抽出**（日付、金額、製品名、連絡先情報など）
- **緊急度レベルの判定**と必要な対応アクションの特定
- **抽出データの構造化**による一貫したフォーマットへの変換
- **多様な電子メール形式への対応**（複数のフォーマット形式を堅牢に処理可能）

## 前提条件

- DSPyモジュールとシグネチャに関する基本的な知識
- Python 3.9以降のインストール済み環境
- OpenAI APIキー（または他のサポート対象LLMへのアクセス権）

## インストールと環境設定

```bash
pip install dspy
```

<details>
<summary>推奨設定：MLflow Tracingを有効にして、内部処理の詳細を把握しましょう</summary>

### MLflowとDSPyの統合について

<a href="https://mlflow.org/">MLflow</a>は、DSPyとネイティブに連携可能なLLMOpsツールであり、説明可能性と実験追跡機能を提供します。本チュートリアルでは、MLflowを使用してプロンプトや最適化の進捗をトレースとして可視化することで、DSPyの動作をより詳細に理解できます。以下の4つの簡単な手順に従って、MLflowを簡単にセットアップ可能です。

![MLflowトレース画面](./mlflow-tracing-email-extraction.png)

1. MLflowのインストール

```bash
%pip install mlflow>=3.0.0
```

2. 別のターミナルウィンドウでMLflow UIを起動する
```bash
mlflow ui --port 5000 --backend-store-uri sqlite:///mlruns.db
```

3. MLflowにノートブックを接続する
```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy")
```

4. トレース機能の有効化
```python
mlflow.dspy.autolog()
```


統合機能の詳細については、[MLflow DSPyドキュメント](https://mlflow.org/docs/latest/llms/dspy/index.html)も併せてご参照ください。
</details>

## ステップ1: データ構造の定義

まず、メールから抽出したい情報の種類を明確に定義します：

```python
import dspy
from typing import List, Optional, Literal
from datetime import datetime
from pydantic import BaseModel
from enum import Enum

class EmailType(str, Enum):
    ORDER_CONFIRMATION = "order_confirmation"
    SUPPORT_REQUEST = "support_request"
    MEETING_INVITATION = "meeting_invitation"
    NEWSLETTER = "newsletter"
    PROMOTIONAL = "promotional"
    INVOICE = "invoice"
    SHIPPING_NOTIFICATION = "shipping_notification"
    OTHER = "other"

class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ExtractedEntity(BaseModel):
    entity_type: str
    value: str
    confidence: float
```

## ステップ2: DSPyシグネチャの作成

次に、メール処理パイプラインのシグネチャを定義します：

```python
class ClassifyEmail(dspy.Signature):
    """メールの内容に基づいて、その種別と緊急度を分類する。"""

    email_subject: str = dspy.InputField(desc="メールの件名")
    email_body: str = dspy.InputField(desc="メールの本文内容")
    sender: str = dspy.InputField(desc="送信者情報")

    email_type: EmailType = dspy.OutputField(desc="分類されたメール種別")
    urgency: UrgencyLevel = dspy.OutputField(desc="メールの緊急度レベル")
    reasoning: str = dspy.OutputField(desc="分類の根拠となる簡潔な説明")

class ExtractEntities(dspy.Signature):
    """メール本文から主要なエンティティ情報を抽出する。"""

    email_content: str = dspy.InputField(desc="件名と本文を含むメールの全文内容")
    email_type: EmailType = dspy.InputField(desc="分類済みのメール種別")

    key_entities: list[ExtractedEntity] = dspy.OutputField(desc="抽出されたエンティティのリスト（種別・値・信頼度を含む）")
    financial_amount: Optional[float] = dspy.OutputField(desc="記載された金額情報（例：'$99.99'）")
    important_dates: list[str] = dspy.OutputField(desc="メール内で言及されている重要な日付リスト")
    contact_info: list[str] = dspy.OutputField(desc="抽出された関連連絡先情報")

class GenerateActionItems(dspy.Signature):
    """メール内容と抽出情報に基づき、必要なアクション項目を決定する。"""

    email_type: EmailType = dspy.InputField()
    urgency: UrgencyLevel = dspy.InputField()
    email_summary: str = dspy.InputField(desc="メール内容の簡潔な要約")
    extracted_entities: list[ExtractedEntity] = dspy.InputField(desc="メール内で検出された主要エンティティ")

    action_required: bool = dspy.OutputField(desc="何らかのアクションが必要かどうか")
    action_items: list[str] = dspy.OutputField(desc="実施すべき具体的なアクション項目リスト")
    deadline: Optional[str] = dspy.OutputField(desc="該当する場合のアクション期限")
    priority_score: int = dspy.OutputField(desc="1～10段階で示す優先度スコア")

class SummarizeEmail(dspy.Signature):
    """メール内容の要点を簡潔にまとめた要約を作成する。"""

    email_subject: str = dspy.InputField()
    email_body: str = dspy.InputField()
    key_entities: list[ExtractedEntity] = dspy.InputField()

    summary: str = dspy.OutputField(desc="メールの主要ポイントを2～3文でまとめた要約")
```

## ステップ3: メール処理モジュールの構築

次に、メインとなるメール処理モジュールを作成します：

```python
class EmailProcessor(dspy.Module):
    """DSPyフレームワークを活用した包括的な電子メール処理システム"""

    def __init__(self):
        super().__init__()

        # 処理コンポーネントの初期化
        self.classifier = dspy.ChainOfThought(ClassifyEmail)
        self.entity_extractor = dspy.ChainOfThought(ExtractEntities)
        self.action_generator = dspy.ChainOfThought(GenerateActionItems)
        self.summarizer = dspy.ChainOfThought(SummarizeEmail)

    def forward(self, email_subject: str, email_body: str, sender: str = ""):
        """電子メールを処理し、構造化された情報を抽出する"""

        # ステップ1: 電子メールの分類
        classification = self.classifier(
            email_subject=email_subject,
            email_body=email_body,
            sender=sender
        )

        # ステップ2: エンティティ抽出
        full_content = f"件名: {email_subject}\n\n送信元: {sender}\n\n{email_body}"
        entities = self.entity_extractor(
            email_content=full_content,
            email_type=classification.email_type
        )

        # ステップ3: 要約生成
        summary = self.summarizer(
            email_subject=email_subject,
            email_body=email_body,
            key_entities=entities.key_entities
        )

        # ステップ4: アクションの決定
        actions = self.action_generator(
            email_type=classification.email_type,
            urgency=classification.urgency,
            email_summary=summary.summary,
            extracted_entities=entities.key_entities
        )

        # ステップ5: 結果の構造化
        return dspy.Prediction(
            email_type=classification.email_type,
            urgency=classification.urgency,
            summary=summary.summary,
            key_entities=entities.key_entities,
            financial_amount=entities.financial_amount,
            important_dates=entities.important_dates,
            action_required=actions.action_required,
            action_items=actions.action_items,
            deadline=actions.deadline,
            priority_score=actions.priority_score,
            reasoning=classification.reasoning,
            contact_info=entities.contact_info
        )
```

## ステップ4: メール処理システムの実行

メール処理システムの動作を確認するための簡単な関数を作成します：

```python
import os
def run_email_processing_demo():
    """メール処理システムのデモンストレーション"""
    
    # DSPyの設定
    lm = dspy.LM(model='openai/gpt-4o-mini')
    dspy.configure(lm=lm)
    os.environ["OPENAI_API_KEY"] = "<YOUR OPENAI KEY>"
    
    # メール処理モジュールの作成
    processor = EmailProcessor()
    
    # テスト用サンプルメール
    sample_emails = [
        {
            "subject": "ご注文確認 #12345 - MacBook Pro 発送のご案内",
            "body": """拝啓 スミス様

このたびはご注文誠にありがとうございます。ご注文番号 #12345 の処理が完了いたしましたのでお知らせいたします。

ご注文内容:
- MacBook Pro 14インチ（スペースグレイ）
- 合計金額: 2,399.00ドル
- 発送予定: 2024年12月15日
- 追跡番号: 1Z999AA1234567890

ご不明な点がございましたら、support@techstore.com までお問い合わせください。

敬具
TechStore チーム""",
            "sender": "orders@techstore.com"
        },
        {
            "subject": "緊急: サーバー障害発生 - 至急対応が必要です",
            "body": """DevOpsチーム各位

現在、本番環境に影響を与える重大なサーバー障害が発生しております。

影響範囲: 全ユーザーがプラットフォームにアクセス不能
発生時刻: EST 14:30

直ちに緊急連絡会議にご参加ください: +1-555-123-4567

最優先事項としてご対応ください。

よろしくお願い申し上げます。
サイト信頼性チーム""",
            "sender": "alerts@company.com"
        },
        {
            "subject": "ミーティング案内: Q4計画策定セッション",
            "body": """チーム各位

Q4計画策定ミーティングへのご参加をご案内いたします。

日時: 2024年12月20日（金）14:00～16:00 EST
場所: 会議室A

12月18日までに参加可否をご返信ください。

よろしくお願い申し上げます。
サラ・ジョンソン""",
            "sender": "sarah.johnson@company.com"
        }
    ]
    
    # 各メールを処理し結果を表示
    print("🚀 メール処理デモンストレーション")
    print("=" * 50)
    
    for i, email in enumerate(sample_emails):
        print(f"\n📧 メール {i+1}: {email['subject'][:50]}...")
        
        # メール処理の実行
        result = processor(
            email_subject=email["subject"],
            email_body=email["body"],
            sender=email["sender"]
        )
        
        # 主要結果の表示
        print(f"   📊 種別: {result.email_type}")
        print(f"   🚨 緊急度: {result.urgency}")
        print(f"   📝 要約: {result.summary}")
        
        if result.financial_amount:
            print(f"   💰 金額: ${result.financial_amount:,.2f}")
        
        if result.action_required:
            print(f"   ✅ 対応要: あり")
            if result.deadline:
                print(f"   ⏰ 期限: {result.deadline}")
        else:
            print(f"   ✅ 対応要: なし")

# デモの実行
if __name__ == "__main__":
    run_email_processing_demo()
```

## 期待される出力結果
```
🚀 メール処理デモ
==================================================

📧 メール 1: 注文確認 #12345 - MacBook Pro の発送準備完了...
   📊 種別: order_confirmation
   🚨 緊急度: 低
   📝 概要: 本メールは、John Smith 様の注文番号 #12345 に関する確認通知です。Space Gray カラーの MacBook Pro 14インチを総額 $2,399.00 で受注いたしました。発送予定日は 2024年12月15日です。追跡番号とカスタマーサポート窓口の連絡先を記載しております。
   💰 金額: $2,399.00
   ✅ 対応要否: 不要

📧 メール 2: 緊急: サーバー障害発生 - 至急ご対応ください...
   📊 種別: other
   🚨 緊急度: 重大
   📝 概要: サイト信頼性チームより、米国東部標準時午後2時30分に発生した重大なサーバー障害について報告がありました。この障害により全ユーザーがプラットフォームにアクセスできない状態となっております。DevOpsチームには直ちに緊急会議にご参加いただき、本問題の解決にご協力くださいますようお願いいたします。
   ✅ 対応要否: 必要
   ⏰ 期限: 即時

📧 メール 3: 会議案内: Q4 事業計画策定ミーティング...
   📊 種別: meeting_invitation
   🚨 緊急度: 中
   📝 概要: Sarah Johnson より、2024年12月20日 米国東部標準時午後2時から4時まで、会議室Aにて Q4 事業計画策定ミーティングを開催いたします。ご参加予定の方は12月18日までに出欠のご連絡をお願いいたします。
   ✅ 対応要否: 必要
   ⏰ 期限: 12月18日
```

## 今後の展開

- **メール種別の追加分類**（ニュースレター、プロモーションメールなど）の実装と精度向上
- **メールサービスプロバイダとの連携**（Gmail API、Outlook、IMAPなど）の実現
- **各種LLMの比較検討**と最適化手法の検証
- **多言語対応機能**の追加による国際メール処理の実現
- **プログラム性能の最適化**による処理速度の向上
