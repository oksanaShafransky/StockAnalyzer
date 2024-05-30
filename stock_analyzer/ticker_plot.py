import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
MOVING_AVG1=120
MOVING_AVG2=60
MOVING_AVG3=3


def calculate_percentage_difference(stock_data, days_back=3):
    """
    Calculate the percentage difference between every day and 'days_back' days back.

    Parameters:
    stock_data (pd.DataFrame): The stock data with a 'Close' column.
    days_back (int): The number of days to look back for calculating the difference.

    Returns:
    pd.Series: The percentage differences.
    """
    # Shift the 'Close' column by the specified number of days
    stock_data[f'SMA{days_back}'] = stock_data['Close'].rolling(window=days_back).mean()
    stock_data[f'SMA_{days_back}_days_back'] = stock_data[f'SMA{days_back}'].shift(days_back)

    # Calculate the percentage difference
    stock_data[f'SMA_Diff_{days_back}_days'] = ((stock_data[f'SMA{days_back}'] - stock_data[f'SMA_{days_back}_days_back']) /
                                                stock_data[f'SMA_{days_back}_days_back']) * 100


    stock_data[f'Close_{days_back}_days_back'] = stock_data['Close'].shift(days_back)

    # Calculate the percentage difference
    stock_data[f'Pct_Diff_{days_back}_days'] = ((stock_data['Close'] - stock_data[f'Close_{days_back}_days_back']) /
                                                stock_data[f'Close_{days_back}_days_back']) * 100

    return stock_data

def identify_signals(stock_data, window=MOVING_AVG1, neutral_threshold=0.0005):
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)

    stock_data[f'Moving_Avg_{MOVING_AVG1}'] = stock_data['Close'].rolling(window=window).mean()
    stock_data[f'MA_Diff_{MOVING_AVG1}'] = stock_data[f'Moving_Avg_{MOVING_AVG1}'].diff()
    stock_data[f'MA_Trend_{MOVING_AVG1}'] = np.where(stock_data[f'MA_Diff_{MOVING_AVG1}'] > 0, 1, -1)

    # Identify trend type
    stock_data[f'Trend_Type_{MOVING_AVG1}'] = np.where(
        abs(stock_data[f'MA_Diff_{MOVING_AVG1}'] / stock_data[f'Moving_Avg_{MOVING_AVG1}']) <= neutral_threshold, 'Neutral',
        np.where(stock_data[f'MA_Diff_{MOVING_AVG1}'] > 0, 'Positive', 'Negative')
    )
    stock_data[f'Trend_Change_{MOVING_AVG1}'] = stock_data[f'Trend_Type_{MOVING_AVG1}'].ne(stock_data[f'Trend_Type_{MOVING_AVG1}'].shift()).astype(int)

    # Identify sell signals: negative trend change and price below moving average
    stock_data[f'Moving_Avg_{MOVING_AVG1}'] = stock_data['Close'].rolling(window=MOVING_AVG1).mean()
    stock_data[f'Moving_Avg_{MOVING_AVG2}'] = stock_data['Close'].rolling(window=MOVING_AVG2).mean()
    stock_data = calculate_percentage_difference(stock_data, days_back=3)
    stock_data = calculate_percentage_difference(stock_data, days_back=2)
    stock_data = calculate_percentage_difference(stock_data, days_back=120)
    stock_data = calculate_percentage_difference(stock_data, days_back=10)

    stock_data[f'MA_Diff_{MOVING_AVG2}'] = stock_data[f'Moving_Avg_{MOVING_AVG2}'].diff()
    stock_data[f'MA_Trend_{MOVING_AVG2}'] = np.where(stock_data['MA_Diff_60'] > 0, 1, -1)
    stock_data[f'Trend_Type_{MOVING_AVG2}'] = np.where(
        abs(stock_data[f'MA_Diff_{MOVING_AVG2}'] / stock_data[f'Moving_Avg_{MOVING_AVG2}']) <= neutral_threshold, 'Neutral',
        np.where(stock_data[f'MA_Diff_{MOVING_AVG2}'] > 0, 'Positive', 'Negative')
    )
    stock_data[f'Trend_Change_{MOVING_AVG1}'] = stock_data[f'Trend_Type_{MOVING_AVG1}'].ne(stock_data[f'Trend_Type_{MOVING_AVG1}'].shift()).astype(int)
    stock_data[f'Trend_Change_{MOVING_AVG2}'] = stock_data[f'Trend_Type_{MOVING_AVG2}'].ne(stock_data[f'Trend_Type_{MOVING_AVG2}'].shift()).astype(int)
    stock_data['Close_Exp'] = stock_data['Close'].ewm(span=3, adjust=True).mean()


    stock_data['Distance'] = stock_data['Close'] - stock_data[f'Moving_Avg_{MOVING_AVG1}']
    stock_data['Distance_Per'] = (stock_data['Distance'] * 100)/stock_data[f'Moving_Avg_{MOVING_AVG1}']
    stock_data['Distance_MA'] = stock_data['Distance_Per'].rolling(window=MOVING_AVG1).mean()
    stock_data['Distance_STD_PER'] = stock_data['Distance_Per'].rolling(window=MOVING_AVG1).std()
    stock_data['Distance_STD_HIGH'] = stock_data['Distance_MA'] + stock_data['Distance_STD_PER']
    stock_data['Distance_STD_LOW'] = stock_data['Distance_MA'] - stock_data['Distance_STD_PER']
    stock_data['BUY'] = ((stock_data['Close'] > stock_data[f'Moving_Avg_{MOVING_AVG1}'])
        & (stock_data['Distance_Per']<stock_data['Distance_MA']))
    stock_data['SELL'] = ((stock_data['Distance_Per'] > stock_data['Distance_MA'])
        & (stock_data['Distance_STD_HIGH']>stock_data['Distance_MA']))


    # Identify buy signals: positive trend change and price above moving average
    #stock_data['BUY'] = (((stock_data[f'Trend_Type_{MOVING_AVG1}'] == 'Positive') & (stock_data[f'Trend_Change_{MOVING_AVG1}'] == 1))
    #                     & (stock_data['Close'] >= stock_data[f'Moving_Avg_{MOVING_AVG1}']))
    # stock_data['SELL'] = (stock_data['Trend_Change'] == 1) & (
    #             stock_data['Close'] < stock_data['Moving_Avg'])
    #stock_data['BUY'] =  ((stock_data[f'Trend_Type_{MOVING_AVG1}'] == 'Positive') & (stock_data[f'Trend_Change_{MOVING_AVG1}'] == 1) & stock_data['Close'] > stock_data[f'Moving_Avg_{MOVING_AVG1}'])
    # stock_data['BUY'] = ((stock_data['Close'] >= stock_data[f'Moving_Avg_{MOVING_AVG1}'])
    #                      & (stock_data['Close_Exp'] > stock_data['Close']  )
    #                      & (stock_data['Pct_Diff_2_days']>0)
    #                      & (stock_data['SMA_Diff_3_days']>0)
    #                      & (stock_data['SMA_Diff_10_days']>0))

                        #& ((stock_data[f'Trend_Type_{MOVING_AVG1}'] == 'Positive') & (stock_data[f'Trend_Change_{MOVING_AVG1}'] == 1)))
    #stock_data['SELL'] = (stock_data['Close'] <= stock_data[f'Moving_Avg_{MOVING_AVG2}'])
                          #& ((stock_data[f'Trend_Type_{MOVING_AVG2}'] == 'Negative') & (stock_data[f'Trend_Change_{MOVING_AVG2}'] == 1)))
                          #| (stock_data['Close'] <= stock_data[f'Moving_Avg_{MOVING_AVG2}']))


    # Filter first BUY and SELL in each sequence
    buy_signals = []
    sell_signals = []
    in_buy = False
    for i, row in stock_data.iterrows():
        if row['BUY'] and not in_buy:
            buy_signals.append(i)
            in_buy = True
        elif row['SELL'] and in_buy:
            sell_signals.append(i)
            in_buy = False

    stock_data['Filtered_BUY'] = stock_data.index.isin(buy_signals)
    stock_data['Filtered_SELL'] = stock_data.index.isin(sell_signals)



    return stock_data

    return stock_data
    # stock_data['BUY'] = ((stock_data['MA_Trend'] == 1)
    #                      & (stock_data['MA_Trend'].shift(1) == -1)
    #                      )
    # stock_data['SELL'] = ((stock_data['MA_Trend'] == -1)
    #                       & (stock_data['MA_Trend'].shift(1) == 1)
    #                       & (stock_data['Close']<stock_data['Moving_Avg'])
    #                       & (stock_data['Close']<stock_data['Moving_Avg_50'])
    #                       & (stock_data['Close']<stock_data['Moving_Avg_30'])
    #                       & (stock_data['Close']<stock_data['Moving_Avg_14'])
    #                       )


def plot_stock_data(stock_data, ticker, window=MOVING_AVG1):
    plt.figure(figsize=(14, 7))
    # stock_data['Moving_Avg_50'] = stock_data['Close'].rolling(window=50).mean()
    # stock_data['Moving_Avg_30'] = stock_data['Close'].rolling(window=30).mean()
    # stock_data['Moving_Avg_14'] = stock_data['Close'].rolling(window=14).mean()
    # plt.plot(stock_data['Moving_Avg_50'], label='50-Day Moving Average', color='brown')
    # plt.plot(stock_data['Moving_Avg_30'], label='30-Day Moving Average', color='yellow')
    plt.plot(stock_data[f'Moving_Avg_{MOVING_AVG2}'], label=f'{MOVING_AVG2}-Day Moving Average', color='purple')

    # Plot positive trend change points
    plt.plot(stock_data['Close'], label='Close Price', color='blue')
    plt.plot(stock_data[f'Moving_Avg_{MOVING_AVG1}'], label=f'{MOVING_AVG1}-Day Moving Average', color='orange')
    plt.scatter(stock_data[stock_data['Filtered_BUY']].index, stock_data[stock_data['Filtered_BUY']]['Close'],
                color='green', marker='^', label='BUY', s=100)
    plt.scatter(stock_data[stock_data['Filtered_SELL']].index, stock_data[stock_data['Filtered_SELL']]['Close'],
                color='red', marker='v', label='SELL', s=100)

    # Plot positive trend change points
    positive_trend_changes = stock_data[(stock_data['Trend_Type'] == 'Positive') & (stock_data['Trend_Change'] == 1)]
    negative_trend_changes = stock_data[(stock_data['Trend_Type'] == 'Negative') & (stock_data['Trend_Change'] == 1)]
    neutral_trend_changes = stock_data[(stock_data['Trend_Type'] == 'Neutral') & (stock_data['Trend_Change'] == 1)]

    plt.scatter(positive_trend_changes.index, positive_trend_changes['Moving_Avg'], color='purple', marker='x',
                label='Positive Trend Change', s=100)
    plt.scatter(negative_trend_changes.index, negative_trend_changes['Moving_Avg'], color='brown', marker='o',
                label='Negative Trend Change', s=100)
    plt.scatter(neutral_trend_changes.index, neutral_trend_changes['Moving_Avg'], color='grey', marker='s',
                label='Neutral Trend Change', s=100)

    plt.grid(True)

    plt.title(f'{ticker} Stock Price, Moving Average, and Buy/Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')

    plt.legend()
    plt.show(block=True)


def calculate_profits(stock_data):
    buy_indices = stock_data[stock_data['Filtered_BUY']].index
    sell_indices = stock_data[stock_data['Filtered_SELL']].index

    profits = []

    for i in range(min(len(buy_indices), len(sell_indices))):
        buy_date = buy_indices[i]
        sell_date = sell_indices[i]

        buy_price = stock_data.loc[buy_date, 'Close']
        sell_price = stock_data.loc[sell_date, 'Close']

        profit = sell_price - buy_price
        profits.append((buy_date, sell_date, buy_price, sell_price, profit))

    profit_df = pd.DataFrame(profits, columns=['Buy Date', 'Sell Date', 'Buy Price', 'Sell Price', 'Profit'])
    total_profit = profit_df['Profit'].sum()

    return profit_df, total_profit

def calculate_recent_profits(stock_data, recent_days=50):
    recent_data = stock_data[-recent_days:]
    buy_indices = recent_data[recent_data['Filtered_BUY']].index
    sell_indices = recent_data[recent_data['Filtered_SELL']].index

    profits = []

    for i in range(min(len(buy_indices), len(sell_indices))):
        buy_date = buy_indices[i]
        sell_date = sell_indices[i]

        buy_price = recent_data.loc[buy_date, 'Close']
        sell_price = recent_data.loc[sell_date, 'Close']

        profit = sell_price - buy_price
        profits.append((buy_date, sell_date, buy_price, sell_price, profit))

    recent_profit_df = pd.DataFrame(profits, columns=['Buy Date', 'Sell Date', 'Buy Price', 'Sell Price', 'Profit'])
    total_recent_profit = recent_profit_df['Profit'].sum()

    return recent_profit_df, total_recent_profit


def calculate_last_5_years_profits(stock_data):
    # Filter data for the last 5 years
    end_date = stock_data.index[-1]
    start_date = end_date - pd.DateOffset(years=5)
    stock_data_5_years = stock_data[start_date:end_date]

    buy_indices = stock_data_5_years[stock_data_5_years['Filtered_BUY']].index
    sell_indices = stock_data_5_years[stock_data_5_years['Filtered_SELL']].index

    profits = []

    for i in range(min(len(buy_indices), len(sell_indices))):
        buy_date = buy_indices[i]
        sell_date = sell_indices[i]

        buy_price = stock_data_5_years.loc[buy_date, 'Close']
        sell_price = stock_data_5_years.loc[sell_date, 'Close']

        profit = sell_price - buy_price
        profits.append((buy_date, sell_date, buy_price, sell_price, profit))

    profit_df = pd.DataFrame(profits, columns=['Buy Date', 'Sell Date', 'Buy Price', 'Sell Price', 'Profit'])
    total_profit = profit_df['Profit'].sum()

    return profit_df, total_profit

def plot_stock_data_last_5_years(stock_data, ticker, window=MOVING_AVG1):
    # Filter data for the last 5 years
    end_date = stock_data.index[-1]
    start_date = end_date - pd.DateOffset(years=5)
    stock_data_5_years = stock_data[start_date:end_date]
    #stock_data_5_years['Moving_Avg_60'] = stock_data_5_years['Close'].rolling(window=60).mean()

    #plt.figure(figsize=(14, 7))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(stock_data_5_years['Close'], label='Close Price', color='blue')
    ax1.plot(stock_data_5_years['Close_Exp'], label='Exp Close Price', color='red')

    ax1.plot(stock_data_5_years[f'Moving_Avg_{MOVING_AVG1}'], label=f'{MOVING_AVG1}-Day Moving Average', color='orange')
    ax1.plot(stock_data_5_years[f'Moving_Avg_{MOVING_AVG2}'], label=f'{MOVING_AVG2}-Day Moving Average', color='green')
    ax1.scatter(stock_data_5_years[stock_data_5_years['Filtered_BUY']].index, stock_data_5_years[stock_data_5_years['Filtered_BUY']]['Close'], color='green', marker='^', label='BUY', s=100)
    ax1.scatter(stock_data_5_years[stock_data_5_years['Filtered_SELL']].index, stock_data_5_years[stock_data_5_years['Filtered_SELL']]['Close'], color='red', marker='v', label='SELL', s=100)

    # Plot positive trend change points
    positive_trend_changes = stock_data_5_years[(stock_data_5_years[f'Trend_Type_{MOVING_AVG1}'] == 'Positive') & (stock_data_5_years[f'Trend_Change_{MOVING_AVG1}'] == 1)]
    negative_trend_changes = stock_data_5_years[(stock_data_5_years[f'Trend_Type_{MOVING_AVG1}'] == 'Negative') & (stock_data_5_years[f'Trend_Change_{MOVING_AVG1}'] == 1)]
    neutral_trend_changes = stock_data_5_years[(stock_data_5_years[f'Trend_Type_{MOVING_AVG1}'] == 'Neutral') & (stock_data_5_years[f'Trend_Change_{MOVING_AVG1}'] == 1)]

    ax1.scatter(positive_trend_changes.index, positive_trend_changes[f'Moving_Avg_{MOVING_AVG1}'], color='purple', marker='x', label='Positive Trend Change', s=100)
    ax1.scatter(negative_trend_changes.index, negative_trend_changes[f'Moving_Avg_{MOVING_AVG1}'], color='brown', marker='o', label='Negative Trend Change', s=100)
    ax1.scatter(neutral_trend_changes.index, neutral_trend_changes[f'Moving_Avg_{MOVING_AVG1}'], color='grey', marker='s', label='Neutral Trend Change', s=100)

    ax1.grid(True)

    ax1.set_title(f'{ticker} Stock Price, Moving Average, and Buy/Sell Signals (Last 5 Years)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()

    # Plot the additional data on the second subplot
    ax2.plot(stock_data_5_years.index, stock_data_5_years['Distance_Per'], label='Distance_Per', color='purple')
    ax2.plot(stock_data_5_years.index, stock_data_5_years['Distance_STD_HIGH'], label='Distance_STD_HIGH', color='red')
    ax2.plot(stock_data_5_years.index, stock_data_5_years['Distance_STD_LOW'], label='Distance_STD_LOW', color='blue')
    ax2.plot(stock_data_5_years.index, stock_data_5_years['Distance_MA'], label='Distance_Per', color='green')
    ax2.scatter(stock_data_5_years[stock_data_5_years['Filtered_BUY']].index, stock_data_5_years[stock_data_5_years['Filtered_BUY']]['Close'], color='green', marker='^', label='BUY', s=100)
    ax2.scatter(stock_data_5_years[stock_data_5_years['Filtered_SELL']].index, stock_data_5_years[stock_data_5_years['Filtered_SELL']]['Close'], color='red', marker='v', label='SELL', s=100)

    ax2.set_title(f'{ticker} Percentage Difference')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Percentage Difference')
    ax2.legend()
    ax2.grid(True)


    plt.tight_layout()
    plt.show()


# Example usage
file_path = 'path_to_your_csv_file.csv'  # Replace with the path to your CSV file
ticker = 'CAH'  # Replace with your desired ticker
stock_data = pd.read_csv(f'../tickers/{ticker}.csv')
stock_data = identify_signals(stock_data, MOVING_AVG1)
#plot_stock_data(stock_data, ticker)




profit_df_5_years, total_profit_5_years = calculate_last_5_years_profits(stock_data)

# Print the profit DataFrame and total profit for the last 5 years
print(profit_df_5_years)
print(f"Total Profit (Last 5 Years): {total_profit_5_years}")


plot_stock_data_last_5_years(stock_data, ticker, MOVING_AVG1)