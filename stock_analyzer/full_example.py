import yfinance as yf
import pandas_ta as ta
import pandas as pd
import matplotlib.pyplot as plt

# Fetch data
def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

# Calculate indicators
def calculate_indicators(data):
    data['SMA_50'] = ta.sma(data['Close'], length=50)
    data['SMA_200'] = ta.sma(data['Close'], length=200)
    data['RSI'] = ta.rsi(data['Close'], length=14)
    return data

# Generate buy/sell signals
def generate_signals(data):
    buy_signals = []
    sell_signals = []

    for i in range(1, len(data)):
        if data['SMA_50'][i] > data['SMA_200'][i] and data['SMA_50'][i-1] <= data['SMA_200'][i-1]:
            buy_signals.append(data.index[i])
        elif data['SMA_50'][i] < data['SMA_200'][i] and data['SMA_50'][i-1] >= data['SMA_200'][i-1]:
            sell_signals.append(data.index[i])

    return buy_signals, sell_signals

# Plot signals
def plot_signals(data, buy_signals, sell_signals):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price', alpha=0.5)
    plt.plot(data['SMA_50'], label='50-Day SMA', alpha=0.5)
    plt.plot(data['SMA_200'], label='200-Day SMA', alpha=0.5)
    plt.scatter(data.loc[buy_signals].index, data.loc[buy_signals]['Close'], marker='^', color='g', label='Buy Signal', alpha=1)
    plt.scatter(data.loc[sell_signals].index, data.loc[sell_signals]['Close'], marker='v', color='r', label='Sell Signal', alpha=1)
    plt.title("Buy/Sell Signals")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Main execution
# ticker = 'CAH'  # Replace with your desired ticker
# stock_data = pd.read_csv(f'../tickers/{ticker}.csv')
data = get_stock_data("AAPL", "2020-01-01", "2023-01-01")
data = calculate_indicators(data)
buy_signals, sell_signals = generate_signals(data)
plot_signals(data, buy_signals, sell_signals)
