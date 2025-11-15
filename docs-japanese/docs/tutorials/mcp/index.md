# チュートリアル：DSPy環境でMCPツールを活用する方法

MCP（Model Context Protocol）とは、アプリケーションがLLM（大規模言語モデル）に対してコンテキストを提供するための標準化プロトコルである。開発における一定のオーバーヘッドはあるものの、MCPを採用することで、使用している技術スタックに関わらず、他の開発者とツールやリソース、プロンプトを共有できる貴重な機会が得られる。同様に、他の開発者が作成したツールを、コードを書き換えることなくそのまま利用することも可能だ。

本ガイドでは、DSPy環境でMCPツールを活用する方法について順を追って解説する。デモンストレーションとして、ユーザーが航空券の予約や既存予約の変更・キャンセルを行える航空サービスエージェントを構築する。このシステムはカスタムツールを備えたMCPサーバーに依存するが、[コミュニティによって構築されたMCPサーバー](https://modelcontextprotocol.io/examples)にも容易に応用できる内容となっている。

??? "本チュートリアルの実行方法"
    本チュートリアルは、Google ColabやDatabricksノートブックなどのホスト型IPythonノートブック環境では実行できない。
    コードを実行するには、ローカルデバイス上でコードを記述するためのガイドに従う必要がある。
    本コードはmacOS環境でテスト済みであり、Linux環境でも同様の動作が期待される。

## 必要な依存関係のインストール

作業を開始する前に、以下の必須依存関係をインストールしておく必要がある：

```shell
pip install -U "dspy[mcp]"
```

## MCPサーバーのセットアップ

まず、航空会社エージェント向けのMCPサーバーを設定します。このサーバーには以下のコンポーネントが含まれます：

- データベース群
  - ユーザーデータベース：ユーザー情報を格納
  - フライトデータベース：フライト情報を格納
  - チケットデータベース：顧客のチケット情報を格納
- ツール群
  - fetch_flight_info：特定の日付におけるフライト情報を取得
  - fetch_itinerary：予約済みの旅程情報を取得
  - book_itinerary：ユーザーに代わってフライトを予約
  - modify_itinerary：フライト変更またはキャンセルによる旅程の修正
  - get_user_info：ユーザー情報を取得
  - file_ticket：人的対応が必要なバックログチケットを登録

作業ディレクトリ内に`mcp_server.py`ファイルを作成し、以下の内容を貼り付けてください：

```python
import random
import string

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

# MCPサーバーの初期化
mcp = FastMCP("Airline Agent")


class Date(BaseModel):
    # LLMは`datetime.datetime`オブジェクトの正確な指定が苦手であるため、明示的な構造化を行う
    year: int
    month: int
    day: int
    hour: int


class UserProfile(BaseModel):
    user_id: str
    name: str
    email: str


class Flight(BaseModel):
    flight_id: str
    date_time: Date
    origin: str
    destination: str
    duration: float
    price: float


class Itinerary(BaseModel):
    confirmation_number: str
    user_profile: UserProfile
    flight: Flight


class Ticket(BaseModel):
    user_request: str
    user_profile: UserProfile


user_database = {
    "Adam": UserProfile(user_id="1", name="Adam", email="adam@gmail.com"),
    "Bob": UserProfile(user_id="2", name="Bob", email="bob@gmail.com"),
    "Chelsie": UserProfile(user_id="3", name="Chelsie", email="chelsie@gmail.com"),
    "David": UserProfile(user_id="4", name="David", email="david@gmail.com"),
}

flight_database = {
    "DA123": Flight(
        flight_id="DA123",
        origin="SFO",
        destination="JFK",
        date_time=Date(year=2025, month=9, day=1, hour=1),
        duration=3,
        price=200,
    ),
    "DA125": Flight(
        flight_id="DA125",
        origin="SFO",
        destination="JFK",
        date_time=Date(year=2025, month=9, day=1, hour=7),
        duration=9,
        price=500,
    ),
    "DA456": Flight(
        flight_id="DA456",
        origin="SFO",
        destination="SNA",
        date_time=Date(year=2025, month=10, day=1, hour=1),
        duration=2,
        price=100,
    ),
    "DA460": Flight(
        flight_id="DA460",
        origin="SFO",
        destination="SNA",
        date_time=Date(year=2025, month=10, day=1, hour=9),
        duration=2,
        price=120,
    ),
}

itinery_database = {}
ticket_database = {}


@mcp.tool()
def fetch_flight_info(date: Date, origin: str, destination: str):
    """指定された日付における出発地から目的地までのフライト情報を取得"""
    flights = []

    for flight_id, flight in flight_database.items():
        if (
            flight.date_time.year == date.year
            and flight.date_time.month == date

サーバーを起動する前に、まずコード内容を確認しておこう。

最初に、`FastMCP`インスタンスを作成する。これはMCPサーバーを迅速に構築するための便利なユーティリティである：

```python
mcp = FastMCP("Airline Agent")
```

次に、実際のアプリケーションで使用されるデータベーススキーマに相当するデータ構造を定義します。例えば：

```python
class Flight(BaseModel):
    flight_id: str
    date_time: Date
    origin: str
    destination: str
    duration: float
    price: float
```

その後、データベースインスタンスを初期化する。実際のアプリケーションでは、これらのインスタンスは実際のデータベースへの接続オブジェクトとなるが、ここでは簡略化のため辞書型を使用する：

```python
user_database = {
    "Adam": UserProfile(user_id="1", name="Adam", email="adam@gmail.com"),
    "Bob": UserProfile(user_id="2", name="Bob", email="bob@gmail.com"),
    "Chelsie": UserProfile(user_id="3", name="Chelsie", email="chelsie@gmail.com"),
    "David": UserProfile(user_id="4", name="David", email="david@gmail.com"),
}
```

次のステップでは、使用するツールを定義し、`@mcp.tool()` デコレータでマークします。これにより、MCPクライアントがMCPツールとして認識できるようになります：

```python
@mcp.tool()
def fetch_flight_info(date: Date, origin: str, destination: str):
    """指定された日付における出発地から目的地までの航空便情報を取得する"""
    flights = []

    for flight_id, flight in flight_database.items():
        if (
            flight.date_time.year == date.year
            and flight.date_time.month == date.month
            and flight.date_time.day == date.day
            and flight.origin == origin
            and flight.destination == destination
        ):
            flights.append(flight)
    return flights
```

最終ステップとして、サーバーを起動します：

```python
if __name__ == "__main__":
    mcp.run()
```

これでサーバーの実装が完了しました！それでは起動してみましょう：

```shell
python 作業ディレクトリパス/mcp_server.py
```

## MCPサーバーのツールを活用したDSPyプログラムの作成

サーバーの起動が完了したら、実際にMCPサーバー上のツールを利用してユーザーを支援する航空サービスエージェントを実装していきます。作業ディレクトリ内に`dspy_mcp_agent.py`というファイルを作成し、そのファイルにコードを追加していく手順に従ってください。

### MCPサーバーからのツール取得

まず最初に、MCPサーバー上に存在するすべての利用可能なツールを取得し、DSPyから使用可能な状態にする必要があります。DSPyでは標準ツールインターフェースとしてAPI [`dspy.Tool`](https://dspy.ai/api/primitives/Tool/)が提供されています。すべてのMCPツールを`dspy.Tool`形式に変換していきましょう。

MCPサーバーと通信するためのMCPクライアントインスタンスを作成し、利用可能なすべてのツールを取得します。その後、静的メソッド`from_mcp_tool`を使用してこれらのツールを`dspy.Tool`形式に変換します：

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# stdio接続用のサーバーパラメータを設定
server_params = StdioServerParameters(
    command="python",  # 実行コマンド
    args=["path_to_your_working_directory/mcp_server.py"],  # コマンド引数
    env=None,  # 環境変数設定
)

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 接続を初期化
            await session.initialize()
            # 利用可能なツール一覧を取得
            tools = await session.list_tools()

            # MCPツールをDSPyツールに変換
            dspy_tools = []
            for tool in tools.tools:
                dspy_tools.append(dspy.Tool.from_mcp_tool(session, tool))

            print(len(dspy_tools))
            print(dspy_tools[0].args)

if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
```

上記のコードにより、利用可能なすべてのMCPツールを収集し、それらをDSPyツールに変換することに成功しました。


### 顧客対応用DSPyエージェントの構築

次に、`dspy.ReAct`を使用して顧客対応用エージェントを構築します。`ReAct`とは「推論と実行」を意味する概念であり、LLMに対してツールを呼び出すか否か、あるいは処理を終了するか否かの判断を求めます。ツールが必要な場合、LLMはどのツールを呼び出すべきか、また適切な引数をどのように設定すべきかを決定する責任を負います。

通常の手順として、エージェントの入力形式と出力形式を定義するために`dspy.Signature`を作成する必要があります：

```python
import dspy

class DSPyAirlineCustomerService(dspy.Signature):
    """あなたは航空会社のカスタマーサービス担当者です。ユーザーからの問い合わせに対応するため、複数のツールが提供されています。ユーザーの要求を適切に処理するために、使用する最適なツールを選択してください。"""

    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
            "処理結果の概要およびユーザーが必要な情報をまとめたメッセージ。"
            "例えば、フライト予約リクエストの場合は確認番号などが該当します。"
        )
    )
```

次に、エージェントに使用する言語モデル（LM）を選択します：

```python
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
```

次に、`dspy.ReAct` API にツールと署名パラメータを渡すことで ReAct エージェントを構築する。これにより、以下の完全なコードスクリプトを組み立てることが可能となる：

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import dspy

# stdio接続用のサーバーパラメータを設定
server_params = StdioServerParameters(
    command="python",  # 実行コマンド
    args=["script_tmp/mcp_server.py"],  # オプションのコマンドライン引数
    env=None,  # オプションの環境変数設定
)


class DSPyAirlineCustomerService(dspy.Signature):
    """あなたは航空会社のカスタマーサービス担当者です。ユーザーからのリクエストに対応する複数のツールが与えられています。
    ユーザーの要求を適切に満たすために、使用する最適なツールを選択する必要があります。"""

    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
            "処理結果の要約およびユーザーが必要な情報を含むメッセージ。"
            "例えば、フライト予約リクエストの場合は確認番号などが該当します。"
        )
    )


dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


async def run(user_request):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 接続を初期化
            await session.initialize()
            # 利用可能なツールを一覧表示
            tools = await session.list_tools()

            # MCPツールをDSPyツールに変換
            dspy_tools = []
            for tool in tools.tools:
                dspy_tools.append(dspy.Tool.from_mcp_tool(session, tool))

            # エージェントを作成
            react = dspy.ReAct(DSPyAirlineCustomerService, tools=dspy_tools)

            result = await react.acall(user_request=user_request)
            print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(run("2025年9月1日にSFOからJFKまでのフライトを予約してください。私の名前はアダムです"))
```

注意：MCPツールはデフォルトで非同期動作するため、`react.acall`を呼び出す必要がある。以下のスクリプトを実行してみよう：

```shell
python 作業ディレクトリパス/dspy_mcp_agent.py
```

出力結果は以下のようになるはずです：

```
Prediction(
    trajectory={'thought_0': '2025年9月1日にSFOからJFKへのアダム向け航空券情報を取得し、予約可能な便を特定する必要がある。', 'tool_name_0': 'fetch_flight_info', 'tool_args_0': {'date': {'year': 2025, 'month': 9, 'day': 1, 'hour': 0}, 'origin': 'SFO', 'destination': 'JFK'}, 'observation_0': ['{"flight_id": "DA123", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 1}, "origin": "SFO", "destination": "JFK", "duration": 3.0, "price": 200.0}', '{"flight_id": "DA125", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 7}, "origin": "SFO", "destination": "JFK", "duration": 9.0, "price": 500.0}'], ..., 'tool_name_4': 'finish', 'tool_args_4': {}, 'observation_4': '完了しました。'},
    reasoning="2025年9月1日のSFOからJFKへのアダム向け航空券予約を正常に完了した。2つの予約可能便を確認し、より経済的な選択肢である午前1時発の便DA123（200ドル）を選定した。アダムのユーザープロファイルを取得し、予約手続きを完了した。航空券の確認番号は8h7clk3qである。",
    process_result='2025年9月1日のSFOからJFKへの航空券予約が正常に完了しました。確認番号は8h7clk3qです。'
)
```

`trajectory`フィールドには、思考プロセスと行動プロセスの全履歴が記録されています。内部でどのような処理が行われているか詳細を知りたい場合は、[可観測性ガイド](https://dspy.ai/tutorials/observability/)を参照し、MLflowを設定してください。これにより、`dspy.ReAct`内部で実行される各ステップを可視化できます！


## まとめ

本ガイドでは、カスタムMCPサーバーと`dspy.ReAct`モジュールを活用した航空サービスエージェントを構築しました。MCPサポートの観点から、DSPyはMCPツールと連携するためのシンプルなインターフェースを提供しており、必要な機能を自由に実装できる柔軟性を備えています。
