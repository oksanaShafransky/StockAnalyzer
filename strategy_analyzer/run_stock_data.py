import datetime
from strategy_analyzer.yt_macd_rsi import YT_MACD_RSI_strategy
from strategy_analyzer.sma_strategy import SMAStrategy
from strategy_analyzer.stock_manager import StockManager
import pandas as pd

import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd


# Function to scrape NASDAQ tickers
def scrape_nasdaq_tickers():
    url = 'https://www.advfn.com/nasdaq/nasdaq.asp'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    tickers = []

    # Find the table containing the tickers
    table = soup.find('table', {'class': 'market tab1'})

    if table:
        rows = table.find_all('tr')[1:]  # Skip the header row
        for row in rows:
            ticker = row.find_all('td')[1].text.strip()  # Second column contains the ticker symbol
            tickers.append(ticker)

    return tickers


# Function to get market cap of a ticker
def get_market_cap(ticker):
    try:
        stock = yf.Ticker(ticker)
        market_cap = stock.info['marketCap']
        return market_cap
    except Exception as e:
        print(f"Could not retrieve data for {ticker}: {e}")
        return None




if __name__ == "__main__":
    # Get NASDAQ tickers dynamically
    nasdaq_tickers = pd.read_csv('../profits/nasdaq_screener.csv')[['Symbol','MarketCap']]
    sp_tickers = pd.read_csv('../profits/sp_list.csv')[['Symbol', 'MarketCap']]
    top_stocks = 1500

    merged_df = pd.concat([nasdaq_tickers, sp_tickers])

    # Sort by 'Market Cap' in descending order
    sorted_df = merged_df.sort_values(by='MarketCap', ascending=False)

    unique_sorted_df = sorted_df.drop_duplicates(subset='Symbol')

    # Select the top 1500 rows with unique names
    top_unique_df = unique_sorted_df.head(top_stocks)

    # Convert to list (if you want to convert the entire row, including all columns)
    top_list = top_unique_df.values.tolist()

    # If you want to convert only the 'Company' column to a list
    top_companies_list = top_unique_df['Symbol'].tolist()

    # Create a DataFrame to store the results
    # data = {'Ticker': [], 'Market Cap': []}
    # nasdaq_tickers_list = nasdaq_tickers['Symbol'].tolist() if nasdaq_tickers is not None else None
    # # Retrieve market cap for each ticker
    # for ticker in nasdaq_tickers_list:
    #     market_cap = get_market_cap(ticker)
    #     if market_cap:
    #         data['Ticker'].append(ticker)
    #         data['Market Cap'].append(market_cap)
    # data.to_csv('../portfolio/nasdaq_tickers.csv', index=False)
    #
    # df = pd.DataFrame(data)
    # print(df)
    #
    # # Sort by Market Cap
    # df_sorted = df.sort_values(by='Market Cap', ascending=False)

    # Print the top 10 biggest NASDAQ stocks
    print(top_companies_list)

    end_date = datetime.datetime.now() - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=2*365)
    to_email = ['vladiks@gmail.com', 'oksi78@yahoo.com', 'oksi.shafransky@gmail.com']
    portfolio_df = pd.read_csv('../portfolio/portfolio_dash.csv')
    #portfolio_df = None
    tickers = portfolio_df['Ticker'].tolist() if portfolio_df is not None else None
    faang_tickers = ['AAPL','NVDA','META','GOOG','AMZN']
    subject = f'Portfolio Notifications for {end_date}' if tickers else f'Stock Change Notifications for {end_date}'

    stock_manager = StockManager()
    # stocks_default = stock_manager.get_stocks_by_tickers(None, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), load_from_disk=False, top_stocks=1000)
    # stocks_for_strategy_130 = stocks_default
    #
    # sma_strategy_default = YT_MACD_RSI_strategy()
    # sma_strategy_130 = YT_MACD_RSI_strategy(window=150)
    #
    # stock_manager.run_strategy(stocks_default, sma_strategy_default)
    #
    # #stock_manager.visualize(stocks_default, sma_strategy_default)
    #
    # all_profits = stock_manager.calc_profit_for_stocks(stocks_default)
    # all_profits.to_csv('../profits/profits_for_YT_MACD.csv', index=False)

    #sma_strategy_default = SMAStrategy()
    #stock_manager.run_process_for_strategy_tickers(None, sma_strategy_default, '../profits/profits_for_sma.csv', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), load_from_disk=False, top_stocks=1000)
    #stock_manager.get_all_trend_changers_tickers(['AMZN'], subject, start_date, end_date, to_email)
    stock_manager.get_all_trend_changers_tickers(top_companies_list, subject, start_date, end_date, to_email, top_stocks=top_stocks)


    #stock_manager.send_mail_with_trend_change_signals(stocks_default, '2024-05-28', to_email)
    #stock_manager.send_email_with_plots(stocks_default, sma_strategy_default, ['vladiks@gmail.com','oksi78@yahoo.com','oksi.shafransky@gmail.com'])
