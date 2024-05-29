import datetime
import numpy as np

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def get_stock_data(ticker, start_date, end_date):
    """Fetch historical data for a given ticker."""
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)
    return hist

def calculate_moving_average(data, window):
    """Calculate the moving average for the given data."""
    data['Moving_Avg'] = data['Close'].rolling(window=window).mean()
    return data

def calculate_derivative(data):
    """Calculate the first derivative of the moving average."""
    data['MA_Derivative'] = data['Moving_Avg'].diff()
    return data

def identify_trend_changes(data):
    """Identify points where the trend changes."""
    data['Trend_Change'] = 0  # Initialize the column with zeros
    data.loc[(data['MA_Derivative'] > 0) & (data['MA_Derivative'].shift(1) <= 0), 'Trend_Change'] = 1  # Negative to positive
    data.loc[(data['MA_Derivative'] < 0) & (data['MA_Derivative'].shift(1) >= 0), 'Trend_Change'] = -1  # Positive to negative
    return data

def generate_notifications(data, current_day):
    """Generate BUY and SELL notifications based on trend and price conditions."""
    data['Trend_10'] = data['Close'].rolling(window=100).mean()
    data['Trend_11'] = data['Close'].rolling(window=15).mean()
    data['Trend_Positive_10'] = data['Trend_10'].diff(15) > 0.3  #data['Trend_11'] - data['Trend_10'] > 0  # Positive trend for last 10 days
    data['Trend_Negative_10'] = data['Trend_10'].diff(15) < -0.3 #data['Trend_11'] - data['Trend_10'] < 0  # Negative trend for last 10 days


    data['BUY'] = (#(data['Date'] == current_day) &
                   (data['Trend_Positive_10']) &
                   (data['Close'] > data['Moving_Avg'])
                  # &(data['Trend_10']<=data['Moving_Avg'])
    )
                  # (data['Close'].shift(1) < data['Moving_Avg'].shift(1)))

    data['SELL'] = (#(data['Date'] == current_day) &
                    (data['Trend_Negative_10']) &
                    (data['Close'] < data['Moving_Avg'])
                    #(data['Close'] < data['Trend_10'])
                    #& (data['Trend_10'] > data['Moving_Avg'])
                    )
                   # (data['Close'].shift(1) > data['Moving_Avg'].shift(1)))
    print(data)
    return data


def plot_moving_average(subplot, data, ticker, window):
    # subplot = plt.subplots(1, 1, figsize=(14, 5 * 1), sharex=True)
    # ax1 = subplot[0]
    # ax2 = subplot[0]  # Using the same axis to ensure perfect overlap
    # ax3 = subplot[0]
    ax1, ax2, ax3 = plt.figure(figsize=(14, 7))

    ax1.plot(data['Close'], label='Close Price', color='blue')
    ax1.plot(data['Moving_Avg'], label=f'{window}-Day Moving Average', color='orange')
    ax1.scatter(data[data['Trend_Change'] == 1].index, data[data['Trend_Change'] == 1]['Moving_Avg'], color='green',
                marker='^', label='Uptrend')
    ax1.scatter(data[data['Trend_Change'] == -1].index, data[data['Trend_Change'] == -1]['Moving_Avg'], color='red',
                marker='v', label='Downtrend')
    ax1.scatter(data[data['BUY']].index, data[data['BUY']]['Close'], color='blue', marker='o', label='BUY', s=100)
    ax1.scatter(data[data['SELL']].index, data[data['SELL']]['Close'], color='red', marker='X', label='SELL', s=100)
    ax1.grid(True)
    # Plot the same moving average on the same y-axis
    ax2.plot(data.index, data['Trend_10'], color='orange', linestyle='--')
    ax3.plot(data.index, data['Trend_10'].diff(20), color='red', linestyle='--')

    ax1.set_title(f'{ticker} Stock Price, Moving Average, and Trend Changes')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax2.set_ylabel('Trend_10')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    ax1.legend(lines_1, labels_1, loc='upper left')


def calculate_profits(data):
    """Calculate the profits from first BUY-SELL pairs."""
    buys = data[data['BUY']]
    sells = data[data['SELL']]

    profits = []
    buy_indices = buys.index
    sell_indices = sells.index

    buy_it = iter(buy_indices)
    sell_it = iter(sell_indices)

    try:
        buy_index = next(buy_it)
        sell_index = next(sell_it)

        while True:
            while sell_index < buy_index:
                sell_index = next(sell_it)
            buy_price = data.loc[buy_index, 'Close']
            sell_price = data.loc[sell_index, 'Close']
            profit = sell_price - buy_price
            profits.append((buy_index, buy_price, sell_index, sell_price, profit))
            buy_index = next(buy_it)
            while buy_index < sell_index:
                buy_index = next(buy_it)
    except StopIteration:
        pass

    return profits

def main(tickers, start_date, end_date, window):
    num_stocks = len(tickers)
    #fig, axs = plt.subplots(num_stocks, 1, figsize=(14, 5 * num_stocks), sharex=True)

    #if num_stocks == 1:
    #    axs = [axs]  # Ensure axs is iterable if there's only one stock

    all_profits = []
    buy_actions = []

    for i, ticker in enumerate(tickers):
        data = get_stock_data(ticker, start_date, end_date)
        data_with_ma = calculate_moving_average(data, window)
        data_with_ma = calculate_derivative(data_with_ma)
        data_with_ma = identify_trend_changes(data_with_ma)
        print(data_with_ma.columns)
        current_day = '2024-05-21 00:00:00-05:00'
        last_month = datetime.date.today() - datetime.timedelta(days=30)
        data_with_notifications = generate_notifications(data_with_ma, current_day)
        print(f"{ticker} Moving Average with Trend Changes and Notifications:")
        print(data_with_notifications[['Close', 'Moving_Avg', 'MA_Derivative', 'Trend_Change', 'BUY', 'SELL']].tail(20))  # Print the last 20 rows

        profits = calculate_profits(data_with_notifications)
        print(f"{ticker} Moving Average with Trend Changes and Notifications:")
        print(data_with_notifications[['Close', 'Moving_Avg', 'MA_Derivative', 'Trend_Change', 'BUY', 'SELL']].tail(
            20))  # Print the last 20 rows
        print(f"Profits for {ticker}:")

        for buy_index, buy_price, sell_index, sell_price, profit in profits:
            print(
                f"BUY at {buy_index.date()} (Price: {buy_price}) and SELL at {sell_index.date()} (Price: {sell_price}): Profit = {profit:.2f}")
            all_profits.append([ticker, buy_index, buy_price, sell_index, sell_price, profit])
        #plot_moving_average(axs[i], data_with_notifications, ticker, window)

        recent_buys = data_with_notifications[(data_with_notifications.index >= pd.Timestamp(last_month, tz='America/New_York')) & (data_with_notifications['BUY'])]
        # Collect buy actions per day
        buy_actions.extend([date.strftime('%Y-%m-%d') for date in recent_buys.index])

    # Convert profit data to DataFrame
    profit_df = pd.DataFrame(all_profits, columns=['Ticker', 'Buy_Date', 'Buy_Price', 'Sell_Date', 'Sell_Price', 'Profit'])
    #profit_df['Buy_Date'] = profit_df['Buy_Date'].dt.date  # Convert to date for better readability
    #profit_df['Sell_Date'] = profit_df['Sell_Date'].dt.date  # Convert to date for better readability
    #profit_df.to_csv('../profits/profits_all.csv', index=False)
    print("\nProfits saved to profits.csv")
    print(profit_df)
    buy_actions_per_day = pd.Series(buy_actions).value_counts().sort_index()
    print(buy_actions_per_day)

    # Optionally, save the results to a CSV file
    buy_actions_per_day.to_csv('../profits/buy_actions_per_day.csv', header=['Buy Actions'])

    #plt.tight_layout()
    #plt.show()

def get_all_nasdaq_tickers():
    # URL to fetch all NASDAQ listed companies
    url = 'https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download'
    nasdaq_stocks = pd.read_csv(url)
    return nasdaq_stocks['Symbol'].tolist()

def is_ticker_trading(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Fetch a small amount of historical data to check if the ticker is trading
        hist = stock.history(period="1d")
        if hist.empty:
            return False
        return True
    except Exception as e:
        print(f"Error checking ticker {ticker}: {e}")
        return False

def get_top_tickers_by_market_cap(tickers, top_n=1000):
    ticker_data = []

    for ticker in tickers:
        if not is_ticker_trading(ticker):
            print(f"{ticker} is not trading or is outdated")
            continue

        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            market_cap = info.get('marketCap', None)
            if market_cap:
                ticker_data.append((ticker, market_cap))
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    sorted_tickers = sorted(ticker_data, key=lambda x: x[1], reverse=True)
    return [ticker for ticker, _ in sorted_tickers[:top_n]]


def read_and_sort_tickers(file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Ensure the 'Market Cap' column is treated as strings and handle NaN values
    df['Market Cap'] = df['Market Cap'].astype(str).str.replace('$', '').str.replace(',', '')

    # Convert the 'Market Cap' column to numeric values, forcing errors to NaN
    df['Market Cap'] = pd.to_numeric(df['Market Cap'], errors='coerce')

    # Drop rows where 'Market Cap' is NaN (if necessary)
    df = df.dropna(subset=['Market Cap'])

    # Sort the DataFrame by 'Market Cap' in descending order
    df_sorted = df.sort_values(by='Market Cap', ascending=False)

    # Select the top 1000 tickers
    top_1000_tickers = df_sorted.head(1000)

    # Optionally, save the result to a new CSV file
    top_1000_tickers.to_csv('../profits/top_1000_nasdaq_tickers.csv', index=False)

    return top_1000_tickers


# Function to process tickers and collect buy actions
def process_tickers(tickers):
    buy_actions = []

    for ticker in tickers:
        try:
            ticker_df = get_ticker(ticker)
            result = fetch_and_analyze_ticker(ticker_df, ticker)
            #buy_actions.append(result)
        except Exception as e:
            print(f"Error processing ticker {ticker}: {e}")

    #return pd.concat(buy_actions)


def read_and_sort_tickers(file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Ensure the 'Market Cap' column is treated as strings and handle NaN values
    df['Market Cap'] = df['Market Cap'].astype(str).str.replace('$', '').str.replace(',', '')

    # Convert the 'Market Cap' column to numeric values, forcing errors to NaN
    df['Market Cap'] = pd.to_numeric(df['Market Cap'], errors='coerce')

    # Drop rows where 'Market Cap' is NaN (if necessary)
    df = df.dropna(subset=['Market Cap'])

    # Sort the DataFrame by 'Market Cap' in descending order
    df_sorted = df.sort_values(by='Market Cap', ascending=False)

    # Select the top 1000 tickers
    top_1000_tickers = df_sorted.head(1000)

    return top_1000_tickers['Symbol'].tolist()


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
def plot_buy_actions_per_day(buy_actions_per_ticker):
    for idx, row in buy_actions_per_ticker.iterrows():
        fig, ax = plt.subplots()
        ax.bar(row['Buy_Actions_Per_Day'].index, row['Buy_Actions_Per_Day'].values)

        # Customize the x-axis ticks
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        plt.xlabel('Date')
        plt.ylabel('Number of Buy Actions')
        plt.title(f"{row['Ticker']} Buy Actions per Day")
        plt.xticks(rotation=45, ha='right')  # Rotate the x-axis labels for better readability
        plt.tight_layout()
        plt.show()


def get_ticker(ticker_name):
    print(f'Read data for ticker {ticker_name}')
    return pd.read_csv(f'../tickers/{ticker_name}.csv')

def download_data(ticker_list):
    count = 0
    for i, ticker in enumerate(ticker_list):
        try:
            print(f'Downloding {ticker} {i} of 1000')
            stock_data = yf.download(ticker, start="1969-01-01", end=datetime.date.today().strftime("%Y-%m-%d"))
            stock_data.to_csv(f'../tickers/{ticker}.csv')
            count += 1
        except Exception as e:
            print(f'Cannot download ticker {ticker}')
            continue
    print(f'Downloaded successfully {count} tickers')


def fetch_and_analyze_ticker(stock_data, ticker, window=150):
    # Ensure the Date column is in datetime format and set it as the index

    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data['Date_i'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date_i', inplace=True)

    stock_data['Moving_Avg'] = stock_data['Close'].rolling(window=window).mean()
    stock_data['Moving_Avg_100'] = stock_data['Close'].rolling(window=10).mean()
    stock_data['Moving_Avg_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['Moving_Avg_30'] = stock_data['Close'].rolling(window=30).mean()
    stock_data['Moving_Avg_14'] = stock_data['Close'].rolling(window=14).mean()

    stock_data['Trend'] = stock_data['Close'] > stock_data['Moving_Avg']
    stock_data['Trend_Change'] = stock_data['Trend'].diff().fillna(0).astype(int)
    stock_data['Trend_Change_100'] = stock_data['Moving_Avg_100'].diff().fillna(0).astype(float)

    stock_data['Trend_Change_100'] = stock_data['Moving_Avg_100'].diff().fillna(0).astype(float)
    stock_data['time_int'] = stock_data.index.astype('int64')  # Convert datetime index to int64
    stock_data['close_diff'] = np.diff(stock_data['Close'], prepend=np.nan)
    stock_data['time_diff'] = np.diff(stock_data['time_int'], prepend=np.nan)
    stock_data['close_derivative'] = stock_data['close_diff'] / stock_data['time_diff']



    stock_data['Trend_10'] = stock_data['Close'].rolling(window=100).mean()
    stock_data['Trend_11'] = stock_data['Close'].rolling(window=25).mean()
    stock_data['Trend_Positive_10'] = stock_data['Trend_10'].diff(25) > 0.2  #data['Trend_11'] - data['Trend_10'] > 0  # Positive trend for last 10 days
    stock_data['Trend_Negative_10'] = stock_data['Trend_10'].diff(25) < -0.2 #data['Trend_11'] - data['Trend_10'] < 0  # Negative trend for last 10 days

    # Identify buy/sell signals
    stock_data['BUY'] = (
                #(stock_data['Trend_Change']>0) &
                #(stock_data['Trend_Change_100'] > 0) &
                (stock_data['Close'] > stock_data['Moving_Avg']) &
                (stock_data['Close'].shift(1) > stock_data['Moving_Avg'].shift(1)) &
                (stock_data['Trend_Positive_10']))
    stock_data['SELL'] = (#stock_data['Trend_Change'] == -1) &
                          #(stock_data['Close'].shift(1) < stock_data['Moving_Avg'].shift(1)) &
                          (stock_data['Close'] < stock_data['Moving_Avg']) &
                          (stock_data['Trend_Negative_10'])
                          )

    in_sequence = False
    buy_actions = []
    plot_stock_data(stock_data, ticker)


    # for idx, row in stock_data.iterrows():
    #     if row['BUY'] and not in_sequence:
    #         in_sequence = True
    #         buy_start_idx = idx
    #     elif row['SELL'] and in_sequence:
    #         in_sequence = False
    #         buy_end_idx = idx
    #         buy_actions.append(stock_data.loc[buy_start_idx:buy_end_idx])
    #
    # if buy_actions:
    #     result = pd.concat(buy_actions)
    #     result['Ticker'] = ticker
    #     result['Num_Buy_Actions'] = 1
    #     return result[['Ticker', 'Date', 'Num_Buy_Actions']]
    # else:
    #     return pd.DataFrame(columns=['Ticker', 'Date', 'Num_Buy_Actions'])


def identify_first_signals(stock_data):
    # Identify sequences of BUY and SELL signals
    stock_data['BUY_SEQ'] = (stock_data['BUY'] & ~stock_data['BUY'].shift(1).fillna(False)).astype(float)
    stock_data['SELL_SEQ'] = (stock_data['SELL'] & ~stock_data['SELL'].shift(1).fillna(False)).astype(float)

    # Keep only the first BUY and SELL in each sequence
    first_buys = stock_data[stock_data['BUY_SEQ'] == 1]
    first_sells = stock_data[stock_data['SELL_SEQ'] == 1]

    return first_buys, first_sells

def plot_stock_data(stock_data, ticker, window=150):

    # Ensure the Date column is in datetime format and set it as the index


    # Identify first BUY and SELL signals
    first_buys, first_sells = identify_first_signals(stock_data)
    plt.figure(figsize=(14, 7))

    plt.plot(stock_data['Close'], label='Close Price', color='blue')
    plt.plot(stock_data['Moving_Avg'], label=f'{window}-Day Moving Average', color='orange')
    plt.plot(stock_data['Moving_Avg_50'], label='50-Day Moving Average', color='yellow')
    plt.plot(stock_data['Moving_Avg_30'], label='20-Day Moving Average', color='brown')
    plt.plot(stock_data['Moving_Avg_14'], label='14-Day Moving Average', color='purple')
    plt.scatter(stock_data[stock_data['Trend_Change'] == 1].index, stock_data[stock_data['Trend_Change'] == 1]['Moving_Avg'], color='green',
                marker='^', label='Uptrend')
    plt.scatter(stock_data[stock_data['Trend_Change'] == -1].index, stock_data[stock_data['Trend_Change'] == -1]['Moving_Avg'], color='red',
                marker='v', label='Downtrend')
    plt.scatter(first_buys.index, first_buys['Close'], color='blue', marker='o', label='BUY', s=100)
    plt.scatter(first_sells.index, first_sells['Close'], color='red', marker='X', label='SELL', s=100)
    plt.grid(True)

    # Plot the same moving average on the same y-axis
    plt.plot(stock_data.index, stock_data['Moving_Avg'], color='orange', linestyle='--')
    plt.plot(stock_data.index, stock_data['Trend_Change'].diff(20), color='red', linestyle='--')

    plt.title(f'{ticker} Stock Price, Moving Average, and Trend Changes')
    plt.xlabel('Date')
    plt.ylabel('Price')

    plt.legend()
    plt.show(block=True)

    # plt.plot(stock_data['Close'], label='Close Price', color='purple')
    # plt.plot(stock_data['Moving_Avg'], label=f'{window}-Day Moving Average', color='orange')
    # plt.scatter(stock_data[stock_data['Trend_Change'] == 1].index, stock_data[stock_data['Trend_Change'] == 1]['Moving_Avg'], color='green',
    #             marker='^', label='Uptrend')
    # plt.scatter(stock_data[stock_data['Trend_Change'] == -1].index, stock_data[stock_data['Trend_Change'] == -1]['Moving_Avg'], color='red',
    #             marker='v', label='Downtrend')
    #
    # #plt.scatter(stock_data[stock_data['BUY']].index, stock_data[stock_data['BUY']]['Close'], color='blue', marker='o', label='BUY', s=100)
    # #plt.scatter(stock_data[stock_data['SELL']].index, stock_data[stock_data['SELL']]['Close'], color='red', marker='X', label='SELL', s=100)
    #
    # plt.scatter(first_buys.index, first_buys['Close'], color='blue', marker='o', label='BUY', s=100)
    # plt.scatter(first_sells.index, first_sells['Close'], color='red', marker='X', label='SELL', s=100)
    #
    # plt.grid(True)
    #
    # # Plot the same moving average on the same y-axis
    # plt.plot(stock_data.index, stock_data['Trend_10'], color='orange', linestyle='--')
    # plt.plot(stock_data.index, stock_data['Trend_10'].diff(20), color='red', linestyle='--')
    #
    # plt.title(f'{ticker} Stock Price, Moving Average, and Trend Changes')
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.ylabel('Trend_10')
    #
    # plt.legend()
    # plt.show(block=True)


if __name__ == "__main__":
    # List of NASDAQ stock tickers
    # big_tickers = ['MSFT', 'AAPL', 'NVDA', 'GOOG', 'GOOGL', 'AMZN', 'META', 'BRK/A', 'BRK/B', 'TSM', 'LLY', 'AVGO',
    #                'NVO', 'TSLA', 'JPM', 'WMT', 'V', 'UNH', 'XOM', 'MA', 'PG', 'ASML', 'JNJ', 'COST', 'ORCL']
    # nasdaq_stocks = ['JPM']  # Updated FB to META
    start_date = '2022-01-01'
    end_date = '2024-05-22'
    window = 150

    # nasdaq_tickers = get_all_nasdaq_tickers()
    # top_tickers = get_top_tickers_by_market_cap(nasdaq_tickers, top_n=1000)
    # print(top_tickers)
    # # Save to CSV if needed
    # top_tickers_df = pd.DataFrame(top_tickers, columns=['Ticker'])
    # top_tickers_df.to_csv('../profits/top_1000_nasdaq_tickers.csv', index=False)

    #top_1000_tickers = read_and_sort_tickers('../profits/nasdaq_screener_1716360525447.csv')
    # download_data(top_1000_tickers)
    # exit()


    #top_1000_nasdaq_tickers_df = pd.read_csv('../profits/top_1000_nasdaq_tickers.csv')
    #top_list = top_1000_nasdaq_tickers_df['Symbol'].head(1).tolist()
    top_list = ['JPM']
    print(top_list)


    # Process the tickers and get buy actions per day for the last month
    buy_actions_per_ticker = process_tickers(top_list)

    # Print the results
    print(buy_actions_per_ticker)

    #plot_buy_actions_per_day(buy_actions_per_ticker)


    # Optionally, save the results to a CSV file
    #buy_actions_per_ticker.to_csv('../profits/buy_actions_per_ticker.csv', index=False)

    #main(top_1000_nasdaq_tickers_df['Symbol'].tolist(), start_date, end_date, window)
