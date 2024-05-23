import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import logging as logger

big_tickers = ['MSFT','AAPL','NVDA','GOOG','GOOGL','AMZN','META','BRK/A','BRK/B','TSM','LLY','AVGO','NVO','TSLA','JPM','WMT','V','UNH','XOM','MA','PG','ASML','JNJ','COST','ORCL']
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
def get_top_tickers(num:int=10):
    # Use pandas to read the table directly from the URL
    tables = pd.read_html(url)

    # The first table on the page contains the S&P 500 tickers
    sp500_table = tables[0]

    # Get the ticker symbols
    tickers = sp500_table['Symbol'].tolist()
    return tickers


def plot_ticker_trend(all_data_df, ticker
                      ):
    if ticker not in all_data_df.columns:
        print(f"Ticker {ticker} not found in the data.")
        return

    data = all_data_df[ticker].dropna()
    if data.empty:
        print(f"No data available for ticker {ticker}.")
        return

    # Prepare data for linear regression
    x = np.arange(len(data)).reshape(-1, 1)  # Day indices
    y = data.values.reshape(-1, 1)           # Close prices

    # Perform linear regression
    model = LinearRegression().fit(x, y)
    trend_line = model.predict(x)

    # Determine the trend direction
    trend_slope = model.coef_[0][0]
    trend_direction = "positive" if trend_slope > 0 else "negative"

    # Plot the data and the trend line
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data.values, label=f"{ticker} Close Price")
    plt.plot(data.index, trend_line, label='Trend Line', linestyle='--', color='red')
    plt.title(f"{ticker} Close Price and {trend_direction.capitalize()} Trend over Last 30 Days")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.show()
def get_all_data(tickers):
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=30)

    # Step 3: Fetch historical data for each ticker
    all_data = {}

    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
            if not data.empty:
                all_data[ticker] = data['Close']
            else:
                all_data[ticker] = None
        except Exception as e:
            all_data[ticker] = None
            print(f"Failed to get data for {ticker}: {e}")

    # Step 4: Convert the collected data to a DataFrame
    return pd.DataFrame(all_data)

if __name__ == '__main__':
    try:
        tickers = big_tickers #get_top_tickers()
        all_data_df = get_all_data(tickers)
        daily_returns = all_data_df.pct_change()
        # Calculate cumulative returns over the period
        cumulative_returns = (1 + daily_returns).cumprod() - 1
        plot_ticker_trend(all_data_df, 'AAPL')
    except Exception as e:
        logger.exception('error on main')