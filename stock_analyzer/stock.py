import matplotlib.pyplot as plt
import talib

class Stock:
    def __init__(self, name, ticker, data):
        self.name = name
        self.ticker = ticker
        self.data = data

    def calculate_profit(self):
        pass

    def plot_data(self):
        pass

    def calculate_indicators(self, data):
        data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
        data['SMA_200'] = talib.SMA(data['Close'], timeperiod=200)
        data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
        return data

    def generate_signals(self):
        buy_signals = []
        sell_signals = []

        for i in range(1, len(self.data)):
            if self.data['SMA_50'][i] > self.data['SMA_200'][i] and self.data['SMA_50'][i - 1] <= self.data['SMA_200'][i - 1]:
                buy_signals.append(self.data.index[i])
            elif self.data['SMA_50'][i] < self.data['SMA_200'][i] and self.data['SMA_50'][i - 1] >= self.data['SMA_200'][i - 1]:
                sell_signals.append(self.data.index[i])

        return buy_signals, sell_signals

    def backtest_strategy(self, buy_signals, sell_signals):
        initial_balance = 10000
        balance = initial_balance
        position = 0

        for date in buy_signals:
            price = self.data.loc[date]['Close']
            position = balance / price
            balance = 0

        for date in sell_signals:
            price = self.data.loc[date]['Close']
            balance = position * price
            position = 0

        return balance

    def plot_signals(self, buy_signals, sell_signals):
        plt.figure(figsize=(14, 7))
        plt.plot(self.data['Close'], label='Close Price', alpha=0.5)
        plt.plot(self.data['SMA_50'], label='50-Day SMA', alpha=0.5)
        plt.plot(self.data['SMA_200'], label='200-Day SMA', alpha=0.5)
        plt.scatter(self.data.loc[buy_signals].index, self.data.loc[buy_signals]['Close'], marker='^', color='g',
                    label='Buy Signal', alpha=1)
        plt.scatter(self.data.loc[sell_signals].index, self.data.loc[sell_signals]['Close'], marker='v', color='r',
                    label='Sell Signal', alpha=1)
        plt.title(f"{self.ticker} Buy/Sell Signals")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
