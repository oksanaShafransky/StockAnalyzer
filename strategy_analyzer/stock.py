import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class Stock:
    def __init__(self, name:str, ticker:str, stock_data, stock_info):
        self.name = name
        self.ticker = ticker
        self.stock_data = stock_data
        self.stock_info = stock_info
        self.profit = None

    def calculate_profit(self):
        buy_indices = self.stock_data[self.stock_data['Filtered_BUY']].index
        sell_indices = self.stock_data[self.stock_data['Filtered_SELL']].index

        profits = []
        for i in range(min(len(buy_indices), len(sell_indices))):
            buy_date = buy_indices[i]
            sell_date = sell_indices[i]

            buy_price = self.stock_data.loc[buy_date, 'Close']
            sell_price = self.stock_data.loc[sell_date, 'Close']

            profit = sell_price - buy_price
            profit_percentage = ((sell_price - buy_price) / buy_price) * 100
            profits.append((self.ticker, buy_date, sell_date, buy_price, sell_price, profit, profit_percentage))

        profit_df = pd.DataFrame(profits,
                                 columns=['Ticker', 'Buy Date', 'Sell Date', 'Buy Price', 'Sell Price', 'Profit',
                                          'Profit Percentage'])
        total_profit = profit_df['Profit'].sum()
        return profit_df, total_profit

    def is_ticker_trading(self):
        try:
            stock = yf.Ticker(self.ticker)
            # Fetch a small amount of historical data to check if the ticker is trading
            hist = stock.history(period="1d")
            if hist.empty:
                return False
            return True
        except Exception as e:
            print(f"Error checking ticker {self.ticker}: {e}")
            return False

    def get_trend_change_signal(self, current_date, window=120):
        self.stock_data['Moving_Avg'] = self.stock_data['Close'].rolling(window=window).mean()
        self.stock_data['Trend'] = self.stock_data['Moving_Avg'].diff()

        # Determine the trend change points
        self.stock_data['Trend_Change'] = 0
        self.stock_data.loc[self.stock_data['Trend'] > 0, 'Trend_Change'] = 1  # Positive trend
        self.stock_data.loc[self.stock_data['Trend'] < 0, 'Trend_Change'] = -1  # Negative trend

        # Shift the Trend_Change column to identify the points where the trend changes
        self.stock_data['Trend_Change_Shifted'] = self.stock_data['Trend_Change'].shift(1)

        # Identify the exact points where the trend changes
        self.stock_data['Trend_Change_Signal'] = self.stock_data.apply(
            lambda row: 'Positive to Negative' if row['Trend_Change'] == -1 and row['Trend_Change_Shifted'] == 1 else
            'Negative to Positive' if row['Trend_Change'] == 1 and row['Trend_Change_Shifted'] == -1 else
            'No Change',
            axis=1
        )
        print(self.stock_data['Trend_Change_Signal'])
        current_date = pd.to_datetime(current_date).date()
        # Check if the trend change happened on the specific date
        if current_date in self.stock_data.index.date:
            trend_change_on_specific_date = self.stock_data.loc[str(current_date), 'Trend_Change_Signal']
            print(f'Trend change on {current_date}: {trend_change_on_specific_date}')
            return trend_change_on_specific_date
        else:
            print(f'The date {current_date} does not exist in the stock data.')
            return None

    def plot_stock_data(self, specific_date, trend_change, window=120):
        plt.figure(figsize=(14, 7))
        plt.plot(self.stock_data['Close'], label='Close Price', color='blue')
        plt.plot(self.stock_data['Moving_Avg'], label=f'{window}-Day Moving Average', color='orange')
        positive_trend_changes = self.stock_data[self.stock_data['Trend_Change_Signal'] == 'Negative to Positive']
        negative_trend_changes = self.stock_data[self.stock_data['Trend_Change_Signal'] == 'Positive to Negative']
        plt.scatter(positive_trend_changes.index, positive_trend_changes['Close'], color='green', marker='^',
                    label='Positive Trend Change')
        plt.scatter(negative_trend_changes.index, negative_trend_changes['Close'], color='red', marker='v',
                    label='Negative Trend Change')
        plt.grid(True)
        plt.title(f'{self.ticker} Stock Price and Moving Average with Trend Changes ({trend_change})')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plot_filename = f'../plots/{self.ticker}_stock_plot_{specific_date}.png'
        plt.savefig(plot_filename)
        plt.close()
        return plot_filename
