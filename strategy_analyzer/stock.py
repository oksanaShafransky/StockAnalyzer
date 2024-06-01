import pandas as pd
import yfinance as yf
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
            print(f"Error checking ticker {ticker}: {e}")
            return False