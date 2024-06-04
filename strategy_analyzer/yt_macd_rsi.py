from strategy_analyzer.stock import Stock
from strategy_analyzer.strategy import Strategy
import pandas_ta as ta
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class YT_MACD_RSI_strategy(Strategy):
    name = 'YT_MACD_RSI_strategy'
    def __init__(self, params:{}=None, window:int=120, neutral_threshold=0.002):
        super().__init__(params)
        self.window = window
        self.neutral_threshold = neutral_threshold

    def normalize(self,series):
        min_val = series.min()
        max_val = series.max()
        return 2 * (series - min_val) / (max_val - min_val) - 1

    def calculate_dmi(self, data, window=14):
        """
        Calculates the Directional Movement Index (DMI) for a DataFrame containing high, low, and close prices.

        Args:
            data (pd.DataFrame): DataFrame with columns 'High', 'Low', and 'Close'.
            window (int, optional): The window length for calculating the DMI. Defaults to 14.

        Returns:
            pd.DataFrame: DataFrame with additional columns 'DX', 'ADX', 'ADXR'.
        """

        # # Add 'PriorClose' column with initial value (e.g., first close price)
        # data['PriorClose'] = data['Close'].shift(1)
        # data['PriorClose'].iloc[0] = data['Close'].iloc[0]  # Fill the first NaN value

        # Avoid chained assignment warnings (use .loc for efficient updates)
        data['PriorClose'] = data['Close'].shift(1)  # Create 'PriorClose' using .loc
        data.loc[0, 'PriorClose'] = data['Close'].iloc[0]  # Fill the first NaN value using .loc

        # Select relevant columns

        # Calculate True High (TRH)
        data['TRH'] = data[['High', 'PriorClose']].max(axis=1)

        # Calculate True Low (TRL)
        data['TRL'] = data[['Low', 'PriorClose']].min(axis=1)

        # Calculate Upward Movement (UM)
        data['UM'] = data['TRH'] - data['PriorClose']
        data['UM'] = data['UM'].where(data['UM'] > 0, 0)

        # Calculate Downward Movement (DM)
        data['DM'] = data['PriorClose'] - data['TRL']
        data['DM'] = data['DM'].where(data['DM'] > 0, 0)

        # Calculate Smoothed True Range (SMATR)
        data['SMATR'] = data['UM'].rolling(window=window).mean() + data['DM'].rolling(window=window).mean()

        # Calculate DX (Directional Movement Index)
        data['DX'] = 100 * data['UM'].rolling(window=window).mean() / data['SMATR']

        # Calculate ADX (Average Directional Index)
        data['ADX'] = data['DX'].rolling(window=window).mean()

        # Calculate ADXR (Average Directional Movement Index Rating)
        data['ADXR'] = (data['ADX'].shift(1) + data['ADX']) / 2

        # Drop temporary columns
        #data.drop(columns=['TRH', 'TRL', 'UM', 'DM', 'SMATR'], inplace=True)

        # Shift Prior Close for the next calculation
        data['PriorClose'] = data['Close'].shift(1)
        data.dropna(inplace=True)  # Drop rows with NaN values

        return data

    def calc_buy_sell(self, stock:Stock):
        stock_data = stock.stock_data

        stock_data['Moving_Avg'] = stock_data['Close'].rolling(window=self.window).mean()
        stock_data['MA_Diff'] = stock_data['Moving_Avg'].diff()
        stock_data['MA_Trend'] = np.where(stock_data['MA_Diff'] > 0, 1, -1)
        stock_data['Close_Exp'] = stock_data['Close'].ewm(span=3, adjust=True).mean()
        stock_data['Descent'] = (stock_data['Moving_Avg']  - stock_data['Moving_Avg'] .shift(12))
        # Normalize the Descent column
        stock_data['Descent'] = self.normalize(stock_data['Descent'])
        stock_data['Smoothed_Descent'] = stock_data['Descent'].rolling(window=5, center=True).mean()

        # Calculate the derivative of the Smoothed_Descent column
        stock_data['Descent_Derivative'] = stock_data['Smoothed_Descent'].diff()


        #stock_data['Descent_Derivative'] = stock_data['Descent'].diff()
        stock_data['Descent_Derivative'] = self.normalize(stock_data['Descent_Derivative'])

        # Identify trend type
        stock_data['Trend_Type'] = np.where(
            abs(stock_data['MA_Diff'] / stock_data['Moving_Avg']) <= self.neutral_threshold,
            'Neutral',
            np.where(stock_data['MA_Diff'] > 0, 'Positive', 'Negative')
        )

        NonMPStrategy = ta.Strategy(
            name="EMAs, BBs, and MACD",
            description="Non Multiprocessing Strategy by rename Columns",
            ta=[
                {"kind": "ema", "length": 8},
                {"kind": "ema", "length": 21},
                {"kind": "ema", "length": 30},
                {"kind": "ema", "length": 200},
                {"kind": "rsi", "length": 10},
                {"kind": "rsi", "length": 25},
                {"kind": "bbands", "length": 20, "col_names": ("BBL", "BBM", "BBU","BBM1", "BBU1")},
                {"kind": "macd", "fast": 12, "slow": 26, "signal":9, "col_names": ("MACD", "MACD_H", "MACD_S")}
            ]
        )
        # Run it
        stock_data.ta.strategy(NonMPStrategy)
        stock_data = self.calculate_dmi(stock_data)

        stock_data['Trend_Change'] = stock_data['Trend_Type'].ne(stock_data['Trend_Type'].shift()).astype(int)

        stock_data['Distance'] = stock_data['Close'] - stock_data['Moving_Avg']
        stock_data['Distance_Per'] = (stock_data['Distance'] * 100) / stock_data['Moving_Avg']
        stock_data['Distance_MA'] = stock_data['Distance_Per'].rolling(window=self.window).mean()
        stock_data['Distance_STD_PER'] = stock_data['Distance_Per'].rolling(window=self.window).std()
        stock_data['Distance_STD_HIGH'] = stock_data['Distance_MA'] + stock_data['Distance_STD_PER']
        stock_data['Distance_STD_LOW'] = stock_data['Distance_MA'] - stock_data['Distance_STD_PER']



        stock_data['BUY'] = (
                            # (stock_data['Trend_Type'] == 'Positive') &
                             (stock_data['Close'] > stock_data['Moving_Avg']*1.015 )
                            & (stock_data['Descent'] > 0.2 )

                             #& (stock_data['Close'] > stock_data['EMA_200']*1.015 )
                             & (stock_data['MACD'] < 0)
                             & (stock_data['MACD_S'] < 0)
                             & (stock_data['MACD_S'] < stock_data['MACD']))

        stock_data['SELL'] = ((stock_data['Close'] < stock_data['Moving_Avg']*0.985)
                              #| (stock_data['Close'] < stock_data['EMA_30'])
                              #|((stock_data['Distance_Per'] > stock_data['Distance_MA'])
                              #& (stock_data['Distance_STD_HIGH'] > stock_data['Distance_MA'])
                              | (stock_data['RSI_10'] > 75)
                              # |( (stock_data['MACD'] > 1)
                              # & (stock_data['MACD_S'] > 1)
                              # & (stock_data['MACD_S'] > stock_data['MACD']))
                              )


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

    def visualize(self, stock:Stock):
        bar_width = 0.8
        fig, (ax1, ax2, ax3,ax4,ax5) = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
        ax1.plot(stock.stock_data['Close'], label='Close Price', color='blue')

        #plt.bar(stock.stock_data.index, stock.stock_data['High'] - stock.stock_data['Low'],color='black')  # Body (high - low)

        # ax1.bar(stock.stock_data.index,stock.stock_data['Close'] - stock.stock_data['Open'], bar_width, bottom=stock.stock_data['Open'], color='black')
        # ax1.bar(stock.stock_data.index,stock.stock_data['High'] - stock.stock_data['Close'], bar_width, bottom=stock.stock_data['Close'], color='green')
        # ax1.bar(stock.stock_data.index,stock.stock_data['Low'] - stock.stock_data['Open'], bar_width, bottom=stock.stock_data['Low'], color='red')

        plt.xticks(rotation=45)
        #ax1.plot(stock.stock_data['Close_Exp'], label='Exp Close Price', color='red')

        # ax1.plot(stock.stock_data.index, stock.stock_data['Moving_Avg'], label='Moving_Avg-Day Moving Average',
        #          color='orange')
        # ax1.plot(stock.stock_data['Moving_Avg'], label='Moving_Avg-Day Moving Average',
        #          color='green')

        ax1.plot(stock.stock_data['Moving_Avg'], label='Moving_Avg-Day Moving Average',
                 color='orange')
        ax1.plot(stock.stock_data['EMA_200'], label='Exponential 200 Day Moving Average',
                 color='cyan')

        # ax1.plot(stock.stock_data['EMA_30'], label='Exponential 30 Day Moving Average',
        #          color='olive')


        ax1.scatter(stock.stock_data[stock.stock_data['Filtered_BUY']].index,
                    stock.stock_data[stock.stock_data['Filtered_BUY']]['Close'], color='green', marker='^',
                    label='BUY', s=100)
        ax1.scatter(stock.stock_data[stock.stock_data['Filtered_SELL']].index,
                    stock.stock_data[stock.stock_data['Filtered_SELL']]['Close'], color='red', marker='v',
                    label='SELL', s=100)

        # Plot positive trend change points
        positive_trend_changes = stock.stock_data[(stock.stock_data['Trend_Type'] == 'Positive') & (
                    stock.stock_data['Trend_Change'] == 1)]
        negative_trend_changes = stock.stock_data[(stock.stock_data['Trend_Type'] == 'Negative') & (
                    stock.stock_data['Trend_Change'] == 1)]
        neutral_trend_changes = stock.stock_data[(stock.stock_data['Trend_Type'] == 'Neutral') & (
                    stock.stock_data['Trend_Change'] == 1)]

        ax1.scatter(positive_trend_changes.index, positive_trend_changes['Moving_Avg'], color='purple',
                    marker='x', label='Positive Trend Change', s=100)
        ax1.scatter(negative_trend_changes.index, negative_trend_changes['Moving_Avg'], color='brown',
                    marker='o', label='Negative Trend Change', s=100)
        ax1.scatter(neutral_trend_changes.index, neutral_trend_changes['Moving_Avg'], color='grey',
                    marker='s', label='Neutral Trend Change', s=100)

        ax1.grid(True)

        ax1.set_title(f'{stock.ticker} Stock Price, Moving Average, and Buy/Sell Signals')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend()

        # Plot the additional data on the second subplot
        #ax2 = ax1.twinx()


        ax2.plot(stock.stock_data.index, stock.stock_data["MACD"], label="MACD", color="blue")
        ax2.plot(stock.stock_data.index, stock.stock_data["MACD_S"], label="Signal", color="red")

        ax2.bar(stock.stock_data.index, stock.stock_data["MACD_H"], label="MACD_H", color="purple")
        ax2.set_ylabel("MACD")


        ax3.plot(stock.stock_data.index, stock.stock_data["RSI_10"], label="RSI_10", color="blue")
        ax3.plot(stock.stock_data.index, stock.stock_data["RSI_25"], label="RSI_25", color="red")
        ax3.set_ylabel("RSI")

        ax4.plot(stock.stock_data.index, stock.stock_data["Descent"], label="Descent", color="blue")
        ax4.plot(stock.stock_data.index, stock.stock_data["Descent_Derivative"], label="Descent deriviative" , color="red")
        ax4.set_ylabel("Descent")

        ax5.plot(stock.stock_data.index, stock.stock_data["DX"], label="DX", color="blue")
        ax5.plot(stock.stock_data.index, stock.stock_data["ADXR"], label="ADXR", color="red")
        ax5.plot(stock.stock_data.index, stock.stock_data["ADX"], label="ADX", color="green")
        ax5.set_ylabel("DI")



        # ax2.plot(stock.stock_data.index, stock.stock_data['Distance_Per'], label='Distance_Per', color='purple')
        # ax2.plot(stock.stock_data.index, stock.stock_data['Distance_STD_HIGH'], label='Distance_STD_HIGH',
        #          color='red')
        # ax2.plot(stock.stock_data.index, stock.stock_data['Distance_STD_LOW'], label='Distance_STD_LOW',
        #          color='blue')
        # ax2.plot(stock.stock_data.index, stock.stock_data['Distance_MA'], label='Distance_MA', color='green')
        # ax2.scatter(stock.stock_data[stock.stock_data['Filtered_BUY']].index,
        #             stock.stock_data[stock.stock_data['Filtered_BUY']]['Close'], color='green', marker='^',
        #             label='BUY', s=100)
        # ax2.scatter(stock.stock_data[stock.stock_data['Filtered_SELL']].index,
        #             stock.stock_data[stock.stock_data['Filtered_SELL']]['Close'], color='red', marker='v',
        #             label='SELL', s=100)
        # ax2.scatter(stock.stock_data[stock.stock_data['Filtered_BUY']].index,
        #             0, color='green', marker='^',
        #             label='BUY', s=100)
        # ax2.scatter(stock.stock_data[stock.stock_data['Filtered_SELL']].index,
        #             0, color='red', marker='v',
        #             label='SELL', s=100)

        ax2.set_title(f'{stock.ticker} MACD')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('MACD')
        ax2.legend()
        ax2.grid(True)



        ax3.set_title(f'{stock.ticker} RSI')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('RSI')
        ax3.legend()
        ax3.grid(True)

        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(30))

        ax4.set_title(f'{stock.ticker} Descent')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Descent')
        ax4.legend()
        ax4.grid(True)

        ax5.set_title(f'{stock.ticker} DI')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('DI')
        ax5.legend()
        ax5.grid(True)




        plt.tight_layout()
        image_path = f"../plots/{stock.ticker}_stock_plot.png"
        plt.savefig(image_path)
        #plt.close(fig)  # Close the figure to save memory
        return image_path


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
        stock_data[f'SMA_Diff_{days_back}_days'] = ((stock_data[f'SMA{days_back}'] - stock_data[
            f'SMA_{days_back}_days_back']) /
                                                    stock_data[f'SMA_{days_back}_days_back']) * 100

        stock_data[f'Close_{days_back}_days_back'] = stock_data['Close'].shift(days_back)

        # Calculate the percentage difference
        stock_data[f'Pct_Diff_{days_back}_days'] = ((stock_data['Close'] - stock_data[f'Close_{days_back}_days_back']) /
                                                    stock_data[f'Close_{days_back}_days_back']) * 100

        return stock_data

