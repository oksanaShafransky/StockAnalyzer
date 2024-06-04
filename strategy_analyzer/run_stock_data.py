import datetime
from strategy_analyzer.yt_macd_rsi import YT_MACD_RSI_strategy
from strategy_analyzer.sma_strategy import SMAStrategy
from strategy_analyzer.stock_manager import StockManager
import pandas as pd
import yfinance as yf
def get_sp500_tickers():
    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(sp500_url)
    sp500_df = table[0]
    return sp500_df['Symbol'].tolist()

# Function to get market cap for a ticker
def get_market_cap(ticker):
    try:
        stock = yf.Ticker(ticker)
        market_cap = stock.info['marketCap']
        return market_cap
    except Exception as e:
        print(f"Error fetching market cap for {ticker}: {e}")
        return None

# Get tickers

if __name__ == "__main__":
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=5*365)

    #tickers = ['AAPL','ABNB','AEG','SEDG']
    #tickers = ['AAPL','AMZN']
    tickers = get_sp500_tickers()

    stock_manager = StockManager()
    stocks_default = stock_manager.get_stocks_by_ticker(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    stocks_for_strategy_130 = stocks_default

    sma_strategy_default = YT_MACD_RSI_strategy()
    sma_strategy_130 = YT_MACD_RSI_strategy(window=150)

    stock_manager.run_strategy(stocks_default, sma_strategy_default)
    stock_manager.calc_profit_for_stocks(stocks_default)
    #stock_manager.visualize(stocks_default, sma_strategy_default)

    #stock_manager.send_email_with_plots(stocks_default, sma_strategy_default, ['vladiks@gmail.com','oksi78@yahoo.com','oksi.shafransky@gmail.com'])
