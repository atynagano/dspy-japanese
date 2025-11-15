---
sidebar_position: 3
---

# モデルコンテキストプロトコル（MCP）

[モデルコンテキストプロトコル（MCP）](https://modelcontextprotocol.io/)は、アプリケーションが言語モデルに対してコンテキストを提供する方法を標準化するオープンプロトコルです。DSPyはこのMCPをサポートしており、MCP対応サーバーが提供する任意のツールをDSPyエージェントと連携して使用することが可能です。

## インストール方法

MCP対応版DSPyをインストールするには：

```bash
pip install -U "dspy[mcp]"
```

## 概要

MCP（Multi-Client Protocol）を利用することで、以下の機能が実現可能となります：

- **標準化されたツールの活用** - 任意のMCP対応サーバーに接続可能
- **スタック間ツールの共有** - 異なるフレームワーク間で同一のツールセットを利用可能
- **統合プロセスの簡素化** - MCPツールを1行のコードでDSPyツールへ変換可能

DSPy自体はMCPサーバーへの直接接続機能をサポートしていません。MCPライブラリのクライアントインターフェースを使用して接続を確立し、`mcp.ClientSession`オブジェクトを`dspy.Tool.from_mcp_tool`関数に渡すことで、MCPツールをDSPyツール形式に変換できます。

## DSPyにおけるMCPの利用方法

### 1. HTTPサーバー（リモート接続）

HTTPプロトコルを使用したリモートMCPサーバーの場合、ストリーム可能なHTTPトランスポート方式を採用してください：

```python
import asyncio
import dspy
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    # HTTP MCPサーバーに接続
    async with streamablehttp_client("http://localhost:8000/mcp") as (read, write):
        async with ClientSession(read, write) as session:
            # セッションを初期化
            await session.initialize()

            # ツール一覧を取得してdspy形式に変換
            response = await session.list_tools()
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in response.tools
            ]

            # ReActエージェントの作成と使用
            class TaskSignature(dspy.Signature):
                task: str = dspy.InputField()
                result: str = dspy.OutputField()

            react_agent = dspy.ReAct(
                signature=TaskSignature,
                tools=dspy_tools,
                max_iters=5
            )

            result = await react_agent.acall(task="東京の天気予報を確認")
            print(result.result)

asyncio.run(main())
```

### 2. Stdioサーバ（ローカルプロセス）

MCPを使用する最も一般的な方法は、ローカルプロセスとして動作するstdio通信用サーバを利用する方法である：

```python
import asyncio
import dspy
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # stdioサーバーの設定
    server_params = StdioServerParameters(
        command="python",                    # 実行するコマンド
        args=["path/to/your/mcp_server.py"], # サーバースクリプトのパス
        env=None,                            # オプションの環境変数
    )

    # サーバーに接続
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # セッションを初期化
            await session.initialize()

            # 利用可能なツールの一覧を取得
            response = await session.list_tools()

            # MCPツールをDSPyツールに変換
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in response.tools
            ]

            # 変換したツールを使用してReActエージェントを作成
            class QuestionAnswer(dspy.Signature):
                """利用可能なツールを用いて質問に回答する"""
                question: str = dspy.InputField()
                answer: str = dspy.OutputField()

            react_agent = dspy.ReAct(
                signature=QuestionAnswer,
                tools=dspy_tools,
                max_iters=5
            )

            # エージェントを使用
            result = await react_agent.acall(
                question="25 + 17 の結果は?"
            )
            print(result.answer)

# 非同期関数を実行
asyncio.run(main())
```

## ツール変換処理

DSPy は MCP ツールから DSPy ツールへの自動変換処理を以下のように行います：

```python
# セッションから取得した MCP ツールオブジェクト
mcp_tool = response.tools[0]

# MCP ツールオブジェクトを DSPy ツールオブジェクトに変換
dspy_tool = dspy.Tool.from_mcp_tool(session, mcp_tool)

# DSPy ツールオブジェクトは以下の情報を保持します:
# - ツール名および説明
# - パラメータスキーマとデータ型
# - 引数の詳細な説明
# - 非同期実行機能のサポート

# DSPy ツールと同様に使用可能
result = await dspy_tool.acall(param1="value", param2=123)
```

## 詳細情報

- [MCP公式ドキュメント](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [DSPy MCPチュートリアル](https://dspy.ai/tutorials/mcp/)
- [DSPyツールズドキュメント](./tools.md)

DSPyにおけるMCP統合機能により、任意のMCPサーバーが提供する標準化ツールを容易に利用可能となり、最小限の設定で強力なエージェント機能を実現できます。
