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
    ax1 = subplot
    ax2 = subplot  # Using the same axis to ensure perfect overlap
    ax3 = subplot

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
    fig, axs = plt.subplots(num_stocks, 1, figsize=(14, 5 * num_stocks), sharex=True)

    if num_stocks == 1:
        axs = [axs]  # Ensure axs is iterable if there's only one stock

    all_profits = []
    for i, ticker in enumerate(tickers):
        data = get_stock_data(ticker, start_date, end_date)
        data_with_ma = calculate_moving_average(data, window)
        data_with_ma = calculate_derivative(data_with_ma)
        data_with_ma = identify_trend_changes(data_with_ma)
        print(data_with_ma.columns)
        current_day = '2024-05-21 00:00:00-05:00'
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

    # Convert profit data to DataFrame
    profit_df = pd.DataFrame(all_profits, columns=['Ticker', 'Buy_Date', 'Buy_Price', 'Sell_Date', 'Sell_Price', 'Profit'])
    profit_df['Buy_Date'] = profit_df['Buy_Date'].dt.date  # Convert to date for better readability
    profit_df['Sell_Date'] = profit_df['Sell_Date'].dt.date  # Convert to date for better readability
    profit_df.to_csv('../profits/profits_all.csv', index=False)
    print("\nProfits saved to profits.csv")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # List of NASDAQ stock tickers
    big_tickers = ['MSFT', 'AAPL', 'NVDA', 'GOOG', 'GOOGL', 'AMZN', 'META', 'BRK/A', 'BRK/B', 'TSM', 'LLY', 'AVGO',
                   'NVO', 'TSLA', 'JPM', 'WMT', 'V', 'UNH', 'XOM', 'MA', 'PG', 'ASML', 'JNJ', 'COST', 'ORCL']
    nasdaq_stocks = ['TSLA']  # Updated FB to META
    start_date = '2022-01-01'
    end_date = '2024-05-22'
    window = 150

    main(big_tickers, start_date, end_date, window)
