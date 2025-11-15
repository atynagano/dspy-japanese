# DSPy ReActとMem0を活用した記憶機能搭載エージェントの構築

本チュートリアルでは、DSPyのReActフレームワークと[Mem0](https://docs.mem0.ai/)の記憶機能を組み合わせることで、対話履歴を保持しながら情報を記憶可能なインテリジェントな会話エージェントを構築する方法を解説します。ユーザーの文脈情報を保持・参照し、それを活用したパーソナライズされた一貫性のある応答を生成できるエージェントの作成方法を習得できます。

## 構築する内容

本チュートリアルを修了すると、以下の機能を備えた記憶機能搭載エージェントを構築できるようになります：

- **ユーザーの嗜好や過去の対話履歴**を保持する機能
- ユーザーやトピックに関する**事実情報の保存・検索**機能
- 記憶情報を活用して**意思決定**を行い、パーソナライズされた応答を生成する機能
- 文脈を考慮した**複雑な多ターン対話**を処理する機能
- 異なる種類の記憶（事実情報、嗜好、経験など）を管理する機能

## 前提条件

- DSPyおよびReActエージェントに関する基本的な知識
- Python 3.9以降のインストール環境
- 使用するLLMプロバイダーのAPIキー

## インストールと環境設定

```bash
pip install dspy mem0ai
```

## ステップ1：Mem0統合の理解

Mem0はAIエージェント向けの記憶層を提供し、記憶の保存、検索、および取得を可能にする。ここではDSPyとの連携方法について順を追って説明する：

```python
import dspy
from mem0 import Memory
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

# 環境設定
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Mem0メモリシステムの初期化
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.1
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small"
        }
    }
}
```

## ステップ2：メモリ認識型ツールの開発

次に、メモリシステムと相互作用可能なツールを開発します：

```python
import datetime

class MemoryTools:
    """Mem0メモリシステムと連携するためのツール群"""

    def __init__(self, memory: Memory):
        self.memory = memory

    def store_memory(self, content: str, user_id: str = "default_user") -> str:
        """メモリに情報を保存する"""
        try:
            self.memory.add(content, user_id=user_id)
            return f"メモリに保存しました: {content}"
        except Exception as e:
            return f"メモリ保存時にエラーが発生しました: {str(e)}"

    def search_memories(self, query: str, user_id: str = "default_user", limit: int = 5) -> str:
        """関連するメモリを検索する"""
        try:
            results = self.memory.search(query, user_id=user_id, limit=limit)
            if not results:
                return "該当するメモリは見つかりませんでした"

            memory_text = "関連するメモリが見つかりました:\n"
            for i, result in enumerate(results["results"]):
                memory_text += f"{i}. {result['memory']}\n"
            return memory_text
        except Exception as e:
            return f"メモリ検索時にエラーが発生しました: {str(e)}"

    def get_all_memories(self, user_id: str = "default_user") -> str:
        """特定ユーザーの全メモリを取得する"""
        try:
            results = self.memory.get_all(user_id=user_id)
            if not results:
                return "このユーザーのメモリは見つかりませんでした"

            memory_text = "ユーザーの全メモリ:\n"
            for i, result in enumerate(results["results"]):
                memory_text += f"{i}. {result['memory']}\n"
            return memory_text
        except Exception as e:
            return f"メモリ取得時にエラーが発生しました: {str(e)}"

    def update_memory(self, memory_id: str, new_content: str) -> str:
        """既存のメモリ内容を更新する"""
        try:
            self.memory.update(memory_id, new_content)
            return f"メモリの内容を更新しました: {new_content}"
        except Exception as e:
            return f"メモリ更新時にエラーが発生しました: {str(e)}"

    def delete_memory(self, memory_id: str) -> str:
        """特定のメモリを削除する"""
        try:
            self.memory.delete(memory_id)
            return "メモリの削除に成功しました"
        except Exception as e:
            return f"メモリ削除時にエラーが発生しました: {str(e)}"

def get_current_time() -> str:
    """現在の日時を取得する"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

## ステップ3: メモリ機能を備えたReActエージェントの構築

次に、メモリ機能を利用できるメインのReActエージェントを実装します：

```python
class MemoryQA(dspy.Signature):
    """
    あなたは有益なアシスタントとして機能し、メモリ機能を利用可能です。
    ユーザーからの入力に応答する際は、必ずその情報をメモリに保存してください。
    後で参照できるようにするためです。
    """
    user_input: str = dspy.InputField()
    response: str = dspy.OutputField()

class MemoryReActAgent(dspy.Module):
    """Mem0メモリ機能を強化したReActエージェント"""

    def __init__(self, memory: Memory):
        super().__init__()
        self.memory_tools = MemoryTools(memory)

        # ReAct用のツールリストを作成
        self.tools = [
            self.memory_tools.store_memory,
            self.memory_tools.search_memories,
            self.memory_tools.get_all_memories,
            get_current_time,
            self.set_reminder,
            self.get_preferences,
            self.update_preferences,
        ]

        # 作成したツールを用いてReActを初期化
        self.react = dspy.ReAct(
            signature=MemoryQA,
            tools=self.tools,
            max_iters=6
        )

    def forward(self, user_input: str):
        """メモリを活用した推論処理によるユーザー入力の処理"""
        
        return self.react(user_input=user_input)

    def set_reminder(self, reminder_text: str, date_time: str = None, user_id: str = "default_user") -> str:
        """ユーザーにリマインダーを設定する"""
        reminder = f"{date_time}に設定されたリマインダー: {reminder_text}"
        return self.memory_tools.store_memory(
            f"REMINDER: {reminder}", 
            user_id=user_id
        )

    def get_preferences(self, category: str = "general", user_id: str = "default_user") -> str:
        """特定カテゴリにおけるユーザーの設定情報を取得する"""
        query = f"ユーザー設定 {category}"
        return self.memory_tools.search_memories(
            query=query,
            user_id=user_id
        )

    def update_preferences(self, category: str, preference: str, user_id: str = "default_user") -> str:
        """ユーザーの設定情報を更新する"""
        preference_text = f"{category}に関するユーザー設定: {preference}"
        return self.memory_tools.store_memory(
            preference_text,
            user_id=user_id
        )
```

## ステップ4: メモリ拡張エージェントの実行

メモリ機能を備えたエージェントと対話するためのシンプルなインターフェースを作成します：

```python
import time
def run_memory_agent_demo():
    """メモリ拡張型 ReAct エージェントのデモンストレーション"""

    # DSPy の設定
    lm = dspy.LM(model='openai/gpt-4o-mini')
    dspy.configure(lm=lm)

    # メモリシステムの初期化
    memory = Memory.from_config(config)

    # エージェントの作成
    agent = MemoryReActAgent(memory)

    # メモリ機能を実証する会話例
    print("🧠 メモリ拡張型 ReAct エージェントデモンストレーション")
    print("=" * 50)

    conversations = [
        "こんにちは、私はアリスです。イタリア料理、特にカルボナーラパスタが大好きです",
        "私はアリスです。朝7時頃に運動するのを好みます",
        "私はアリスです。私の食の好みについて覚えていることはありますか？",
        "私はアリスです。明日の買い物を忘れないようにリマインダーを設定してください",
        "私はアリスです。私の運動の好みは何ですか？",
        "私はアリスです。週末にはハイキングも楽しみます",
        "私はアリスです。これまでに私についてどんなことを知っていますか？"
    ]

    for i, user_input in enumerate(conversations, 1):
        print(f"\n📝 ユーザー: {user_input}")

        try:
            response = agent(user_input=user_input)
            print(f"🤖 エージェント: {response.response}")
            time.sleep(1)

        except Exception as e:
            print(f"❌ エラー: {e}")

# デモンストレーションの実行
if __name__ == "__main__":
    run_memory_agent_demo()
```

## 期待される出力結果
```
🧠 記憶強化型 ReAct エージェントのデモンストレーション
==================================================

📝 ユーザー: こんにちは、私はアリスと申します。イタリア料理、特にカルボナーラパスタが大好きです。

🤖 エージェント: アリスさん、はじめまして！カルボナーラパスタは素晴らしい選択ですね。お気に入りのお店はありますか？それともご自宅で作られるのがお好きですか？

📝 ユーザー: 私はアリスです。朝7時頃に運動するのを好んでいます。

🤖 エージェント: アリスさん、はじめまして！朝7時頃に運動されるのがお好きだと記録しました。朝のトレーニングに関するご質問やアドバイスが必要な際は、いつでもお気軽にお尋ねください。

📝 ユーザー: 私はアリスです。私の食の好みについて覚えていることはありますか？

🤖 エージェント: アリスさんがイタリア料理を好まれており、特にカルボナーラパスタがお気に入りであることを記憶しています。

📝 ユーザー: 私はアリスです。明日の食料品買い物のリマインダーを設定してください。

🤖 エージェント: アリスさん、明日の食料品買い物のリマインダーを設定いたしました。

📝 ユーザー: 私はアリスです。私の運動に関する好みは何ですか？

🤖 エージェント: アリスさんの運動に関する好みは、朝7時頃にトレーニングを行うことです。

📝 ユーザー: 私はアリスです。週末にはハイキングも楽しみます。

🤖 エージェント: アリスさん、はじめまして！週末にハイキングをお楽しみとのこと、素晴らしいですね。おすすめのハイキングコースやハイキングのコツについて知りたいことがあれば、いつでもお尋ねください。

📝 ユーザー: 私はアリスです。これまでのところ、私についてどのようなことを把握していますか？

🤖 エージェント: 現時点で把握している情報は以下の通りです：アリスさんは週末にハイキングを楽しまれており、朝7時頃の運動を好まれ、イタリア料理、特にカルボナーラパスタが大好物です。また、2023年10月7日に食料品買い物のリマインダーが設定されています。
```

## 今後の展開

- **データベース（PostgreSQL/MongoDB）を用いたメモリ永続化機能**の実装
- **メモリ内容の分類機能**とタグ付け機構の追加による情報整理の効率化
- **データ管理のためのメモリ有効期限ポリシー**の設定
- **本番環境向けのマルチユーザーメモリ分離機構**の構築
- **メモリ分析機能**とインサイトレポートの追加
- **ベクトルデータベースとの連携**による高度な意味検索の実現
- **長期保存効率向上のためのメモリ圧縮機能**の実装

本チュートリアルでは、DSPyのReActフレームワークにMem0のメモリ機能を統合することで、インテリジェントかつ文脈理解能力を備えたエージェントを構築する手法を解説します。これにより、エージェントは対話を通じて情報を学習・記憶できるようになり、実世界アプリケーションにおける有用性が大幅に向上します。