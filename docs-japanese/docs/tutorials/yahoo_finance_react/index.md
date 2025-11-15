# DSPy ReActとYahoo Financeニュースを活用した金融分析

本チュートリアルでは、[LangChainのYahoo Financeニュースツール](https://python.langchain.com/docs/integrations/tools/yahoo_finance_news/)と連携したDSPy ReActを用いて、リアルタイム市場分析が可能な金融分析エージェントを構築する方法を解説します。

## 構築するシステム

ニュース記事を取得し、感情分析を実施した上で投資判断に役立つインサイトを提供する金融エージェントを構築します。

## セットアップ手順

```bash
pip install dspy langchain langchain-community yfinance
```

## ステップ1: LangChainツールをDSPyに変換する

```python
import dspy
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from dspy.adapters.types.tool import Tool
import json
import yfinance as yf

# DSPy の設定
lm = dspy.LM(model='openai/gpt-4o-mini')
dspy.configure(lm=lm, allow_tool_async_sync_conversion=True)

# LangChain の Yahoo Finance ツールを DSPy 形式に変換
yahoo_finance_tool = YahooFinanceNewsTool()
finance_news_tool = Tool.from_langchain(yahoo_finance_tool)
```

## ステップ2：補助的な財務ツールの作成

```python
def get_stock_price(ticker: str) -> str:
    """指定した銘柄の最新株価および基本情報を取得する。"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1d")
        
        if hist.empty:
            return f"{ticker} のデータ取得に失敗しました"
        
        current_price = hist['Close'].iloc[-1]
        prev_close = info.get('previousClose', current_price)
        change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
        
        result = {
            "ticker": ticker,
            "price": round(current_price, 2),
            "change_percent": round(change_pct, 2),
            "company": info.get('longName', ticker)
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"エラー: {str(e)}"

def compare_stocks(tickers: str) -> str:
    """カンマ区切りで指定された複数銘柄を比較する。"""
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(',')]
        comparison = []
        
        for ticker in ticker_list:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1d")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_close = info.get('previousClose', current_price)
                change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
                
                comparison.append({
                    "ticker": ticker,
                    "price": round(current_price, 2),
                    "change_percent": round(change_pct, 2)
                })
        
        return json.dumps(comparison)
    except Exception as e:
        return f"エラー: {str(e)}"
```

## ステップ3：Financial ReActエージェントの構築

```python
class FinancialAnalysisAgent(dspy.Module):
    """Yahoo Financeデータを利用した金融分析用ReActエージェント"""
    
    def __init__(self):
        super().__init__()
        
        # 全てのツールを統合
        self.tools = [
            finance_news_tool,  # LangChain Yahoo Financeニュース取得ツール
            get_stock_price,
            compare_stocks
        ]
        
        # ReActフレームワークの初期化
        self.react = dspy.ReAct(
            signature="financial_query -> analysis_response",
            tools=self.tools,
            max_iters=6
        )
    
    def forward(self, financial_query: str):
        return self.react(financial_query=financial_query)
```

## ステップ4：財務分析の実施

```python
def run_financial_demo():
    """金融分析エージェントのデモンストレーション機能"""
    
    # エージェントの初期化
    agent = FinancialAnalysisAgent()
    
    # サンプルクエリの定義
    queries = [
        "Apple Inc. (AAPL) に関する最新のニュースと、株価への影響について教えてください",
        "AAPL、GOOGL、MSFTのパフォーマンス比較を行ってください",
        "Teslaの最近のニュースを検索し、センチメント分析を実施してください"
    ]
    
    for query in queries:
        print(f"クエリ: {query}")
        response = agent(financial_query=query)
        print(f"分析結果: {response.analysis_response}")
        print("-" * 50)

# デモの実行
if __name__ == "__main__":
    run_financial_demo()
```

## 出力例

「Appleに関する最新のニュースは？」といったクエリでエージェントを実行すると、以下の処理が行われます：

1. Yahoo Finance Newsツールを使用して、Appleに関する最新のニュース記事を取得
2. 現在の株価データを取得
3. 取得した情報を分析し、有益なインサイトを提供

**サンプル応答例：**
```
分析：現在のアップル社（AAPL）の株価が196.58ドルで、0.48%の小幅な上昇を示していることから、同銘柄は市場において安定的な推移を続けていると推測される。ただし、最新のニュース情報が入手できない状況であるため、投資家心理や株価動向に影響を及ぼす可能性のある重要な動向については把握できていない。投資家は、今後発表される予定の重要事項や、マイクロソフト社（MSFT）など他のテクノロジー株との比較においてアップルの業績に影響を与える可能性のある市場動向について、引き続き注視する必要がある。
```

## 非同期ツールの活用について

多くのLangchainツールでは、パフォーマンス向上のために非同期処理が採用されています。非同期ツールの詳細については、[ツールのドキュメント](../../learn/programming/tools.md#async-tools)を参照してください。

## 主な利点

- **ツール連携**：LangChainツールとDSPy ReActをシームレスに統合可能
- **リアルタイムデータ**：最新の市場データやニュース情報にアクセス可能
- **拡張性**：金融分析ツールの追加が容易
- **インテリジェントな推論**：ReActフレームワークによる段階的な分析プロセス

本チュートリアルでは、DSPyのReActフレームワークがLangChainの金融分析ツールと連携し、インテリジェントな市場分析エージェントを構築する方法について解説します。
