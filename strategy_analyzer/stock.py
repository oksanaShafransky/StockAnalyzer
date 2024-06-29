import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from pandas_ta.volume import kvo


class Stock:
    def __init__(self, name: str, ticker: str, stock_data, stock_info=None):
        self.name = name
        self.ticker = ticker
        self.stock_data = stock_data
        self.stock_info = stock_info
        self.profit = None

    def calculate_profit(self):
        print(f'going to calculate profits for ticker {self.ticker}')
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
        total_profit = profit_self.stock_data['Profit'].sum()
        return profit_df, total_profit

    def calculate_rsi(self, period=14):
        try:
            delta = self.stock_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)

            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            self.stock_data['RSI'] = rsi
        except Exception as e:
            print(f'Failed to get rsu: {e}')
            self.stock_data['RSI'] = None

    def compute_cmf(self, window=20):
        """
        This function computes the Chaikin Money Flow (CMF) for a given DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing at least 'High', 'Low', 'Close', and 'Volume' columns.
            window (int, optional): The window size for calculating the CMF. Defaults to 20.

        Returns:
            pd.DataFrame: The original DataFrame with an additional 'CMF' column.
        """
        # Money Flow Ratio (MFR)
        self.stock_data['MFR'] = ((self.stock_data['Close'] - self.stock_data['Low']) / (
                    self.stock_data['High'] - self.stock_data['Low'])) * self.stock_data['Volume']

        # Average Money Flow Ratio (AMFR)
        self.stock_data['AMFR'] = self.stock_data['MFR'].rolling(window=window).mean()

        # Chaikin Money Flow (CMF)
        self.stock_data['CMF'] = (self.stock_data['MFR'] - self.stock_data['AMFR']) / (
                    self.stock_data['High'] - self.stock_data['Low'])

    def calculate_maxima_statistics(self):
        # Find the indices of the zero values
        #zero_indices = self.stock_data.index[(self.stock_data['klinger_buy_signal'] or self.stock_data['klinger_sell_signal'])].tolist()
        zero_indices_sell = self.stock_data.index[self.stock_data['klinger_sell_signal']].tolist()
        zero_indices_buy = self.stock_data.index[self.stock_data['klinger_buy_signal']].tolist()
        zero_indices = zero_indices_buy + zero_indices_sell
        zero_indices.sort()
        #zero_indices = self.stock_data.index[self.stock_data[column_name] == 0].tolist()

        # If there are less than two zeroes, return NaN for both average and median
        if len(zero_indices) < 2:
            return np.nan, np.nan

        maxima = []

        # Iterate through the segments between zero values
        for i in range(len(zero_indices) - 1):
            start_idx = zero_indices[i]
            end_idx = zero_indices[i + 1]
            #segment = self.stock_data['Close'][start_idx:end_idx]

            segment = (self.stock_data['Close'][end_idx]-self.stock_data['Close'][start_idx])*100/self.stock_data['Close'][start_idx]


            # Find the absolute local maximum in the segment
            #if len(segment) > 0:
                #maxima.append(segment.abs().max())
            maxima.append(np.abs(segment))

        # Calculate the average and median of the maxima
        if len(maxima) > 0:
            avg_maxima = np.mean(maxima)
            median_maxima = np.median(maxima)
        else:
            avg_maxima = np.nan
            median_maxima = np.nan

        return avg_maxima, median_maxima

    def compute_cci(self, window=14):
        """
        This function computes the CCI for a given DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing at least 'High', 'Low', and 'Close' columns.
            window (int, optional): The window size for calculating the Typical Price (TP). Defaults to 14.

        Returns:
            pd.DataFrame: The original DataFrame with an additional 'CCI' column.
        """
        # Calculate the Typical Price (TP)
        self.stock_data['TP'] = (self.stock_data['High'] + self.stock_data['Low'] + self.stock_data['Close']) / 3

        # Calculate the Mean Deviation (MD)
        self.stock_data['MD'] = self.stock_data['TP'].rolling(window=window).std()

        # Avoid division by zero for the first 'window' rows with NaN in 'MD'
        self.stock_data['MD'] = self.stock_data['MD'].replace(0, pd.NA).ffill()

        # Calculate the CCI
        self.stock_data['CCI'] = (self.stock_data['Close'] - self.stock_data['TP']) / (0.015 * self.stock_data['MD'])


    #####################################################
    def positive_volume_index(self, ma_period=255):
        self.stock_data['previous_volume'] = self.stock_data['Volume'].shift(1)
        self.stock_data['previous_close'] = self.stock_data['Close'].shift(1)
        self.stock_data['pvi'] = 1000  # Starting value for PVI
        pvi_values = [1000]

        for i in range(1, len(self.stock_data)):
            if self.stock_data['Volume'].iloc[i] > self.stock_data['previous_volume'].iloc[i]:
                change = ((self.stock_data['Close'].iloc[i] - self.stock_data['previous_close'].iloc[i]) / self.stock_data['previous_close'].iloc[i]) * \
                         pvi_values[-1]
                pvi_values.append(pvi_values[-1] + change)
            else:
                pvi_values.append(pvi_values[-1])

        self.stock_data['pvi'] = pvi_values
        self.stock_data['pvi_ma'] = self.stock_data['pvi'].rolling(window=ma_period).mean()

        # Generate buy/sell signals
        self.stock_data['pvi_buy_signal'] = (self.stock_data['pvi'] > self.stock_data['pvi_ma']) & (self.stock_data['pvi'].shift(1) <= self.stock_data['pvi_ma'].shift(1))
        self.stock_data['pvi_sell_signal'] = (self.stock_data['pvi'] < self.stock_data['pvi_ma']) & (self.stock_data['pvi'].shift(1) >= self.stock_data['pvi_ma'].shift(1))


##############################################################
    def klinger_volume_oscillator_tv(self, trig_len=13, fast_x=34, slow_x=55):
        hlc3 = (self.stock_data['High'] + self.stock_data['Low'] + self.stock_data['Close']) / 3
        x_trend = (self.stock_data['Volume'] * 100).where(hlc3 > hlc3.shift(1), -self.stock_data['Volume'] * 100)

        x_fast = x_trend.ewm(span=fast_x, adjust=False).mean()
        x_slow = x_trend.ewm(span=slow_x, adjust=False).mean()

        x_kvo = x_fast - x_slow
        x_trigger = x_kvo.ewm(span=trig_len, adjust=False).mean()

        self.stock_data['KVO'] = x_kvo
        self.stock_data['Trigger'] = x_trigger

        # Buy and Sell signals
        self.stock_data['klinger_tv_buy_signal'] = (self.stock_data['KVO'] > self.stock_data['Trigger']) & (self.stock_data['KVO'].shift(1) <= self.stock_data['Trigger'].shift(1))
        self.stock_data['klinger_tv_sell_signal'] = (self.stock_data['KVO'] < self.stock_data['Trigger']) & (self.stock_data['KVO'].shift(1) >= self.stock_data['Trigger'].shift(1))

    def klinger_volume_oscillator7(self, trig_len=13, fast_x=34, slow_x=55):
        hlc3 = (self.stock_data['high7'] + self.stock_data['low7'] + self.stock_data['close7']) / 3
        x_trend = (self.stock_data['volume7'] * 100).where(hlc3 > hlc3.shift(1), -self.stock_data['volume7'] * 100)

        x_fast = x_trend.ewm(span=fast_x, adjust=False).mean()
        x_slow = x_trend.ewm(span=slow_x, adjust=False).mean()

        x_kvo = x_fast - x_slow
        x_trigger = x_kvo.ewm(span=trig_len, adjust=False).mean()

        self.stock_data['KVO7'] = x_kvo
        self.stock_data['Trigger7'] = x_trigger

        # Buy and Sell signals
        self.stock_data['klinger7_buy_signal'] = (self.stock_data['KVO7'] > self.stock_data['Trigger7']) & (self.stock_data['KVO7'].shift(1) <= self.stock_data['Trigger7'].shift(1))
        self.stock_data['klinger7_sell_signal'] = (self.stock_data['KVO7'] < self.stock_data['Trigger7']) & (self.stock_data['KVO7'].shift(1) >= self.stock_data['Trigger7'].shift(1))

    def klinger_oscillator(self, fast_period=34, slow_period=55, signal_period=13):
        # Calculate True Range High (TRH) and True Range Low (TRL)
        self.stock_data['previous_close'] = self.stock_data['Close'].shift(1)
        self.stock_data['trh'] = self.stock_data[['High', 'previous_close']].max(axis=1)
        self.stock_data['trl'] = self.stock_data[['Low', 'previous_close']].min(axis=1)

        # Calculate Volume Force (VF)
        self.stock_data['vf'] = ((2 * (self.stock_data['Close'] - self.stock_data['trl']) / (self.stock_data['trh'] - self.stock_data['trl']) - 1) * self.stock_data['Volume']).fillna(0)

        # Calculate Cumulative Volume Force (CVF)
        self.stock_data['cvf'] = self.stock_data['vf'].cumsum()

        # Calculate fast and slow EMAs of CVF
        self.stock_data['fast_cvf_ema'] = self.stock_data['cvf'].ewm(span=fast_period, adjust=False).mean()
        self.stock_data['slow_cvf_ema'] = self.stock_data['cvf'].ewm(span=slow_period, adjust=False).mean()

        # Calculate Klinger Oscillator (KO)
        self.stock_data['ko'] = self.stock_data['fast_cvf_ema'] - self.stock_data['slow_cvf_ema']

        # Calculate Signal Line (EMA of KO)
        self.stock_data['signal'] = self.stock_data['ko'].ewm(span=signal_period, adjust=False).mean()

        # Generate buy/sell signals
        self.stock_data['klinger_buy_signal'] = (self.stock_data['ko'] > self.stock_data['signal']) & (self.stock_data['ko'].shift(1) <= self.stock_data['signal'].shift(1))
        self.stock_data['klinger_sell_signal'] = (self.stock_data['ko'] < self.stock_data['signal']) & (self.stock_data['ko'].shift(1) >= self.stock_data['signal'].shift(1))


    ###################################

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
        self.stock_data[f'Moving_Avg_{window}'] = self.stock_data['Close'].rolling(window=window).mean()
        self.stock_data['Trend'] = self.stock_data[f'Moving_Avg_{window}'].diff()

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
            print(f'Trend change for ticker {self.name} {self.ticker} on {current_date} window={window}: {trend_change_on_specific_date}')
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


    def plot_stock_data_for_spy(self, specific_date, spy_stock):
        # plt.figure(figsize=(14, 7))
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        ax1.plot(self.stock_data['Close'], label='Close Price', color='blue')
        ax1.plot(self.stock_data['Moving_Avg_150'], label='Moving_Avg_150', color='green')
        ax1.grid(True)

        ax1.set_title(f'{self.ticker} Stock Price, Moving Average')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend()

        # Plot the additional data on the second subplot
        ax2.plot(self.stock_data.index, self.stock_data['signal'], label='signal', color='purple')
        ax2.plot(self.stock_data.index, self.stock_data['ko'], label='ko', color='red')
        ax2.set_title(f'{self.ticker} kringer')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Percentage Difference')
        ax2.legend()
        ax2.grid(True)


        ax3.plot(self.stock_data.index, self.stock_data['Trigger7'], label='Trigger7', color='purple')
        ax3.plot(self.stock_data.index, self.stock_data['KVO7'], label='KVO7', color='red')
        ax3.plot(self.stock_data.index, self.stock_data['Trigger'], label='Trigger', color='green')
        ax3.plot(self.stock_data.index, self.stock_data['KVO'], label='KVO', color='brown')
        # ax3.plot(self.stock_data.index, self.stock_data['klinger7_buy_signal'], label='klinger7_buy_signal', color='yellow')
        # ax3.plot(self.stock_data.index, self.stock_data['klinger7_sell_signal'], label='klinger7_sell_signal', color='olive')
        # ax3.plot(self.stock_data.index, self.stock_data['klinger_tv_buy_signal'], label='klinger_tv_buy_signal',
        #          color='green')
        # ax3.plot(self.stock_data.index, self.stock_data['klinger_tv_sell_signal'], label='klinger_tv_sell_signal',
        #          color='blue')

        ax3.set_title(f'{self.ticker} kringer')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Percentage Difference')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.show(block=True)
        plot_filename = f'../plots/{self.ticker}_stock_plot_{str(specific_date)[:10]}.png'
        #plt.savefig(plot_filename)
        #plt.close()
        return plot_filename
    def plot_stock_data_for_windows(self, specific_date, windows=[150,50,20,14,7], colors=['green','pink','yellow','orange','red']):
        plt.figure(figsize=(14, 7))
        plt.plot(self.stock_data['Close'], label='Close Price', color='blue')
        for i, win in enumerate(windows):
            plt.plot(self.stock_data[f'Moving_Avg_{win}'], label=f'{win}-Day Moving Average', color=colors[i])


        plt.grid(True)
        plt.title(f'{self.ticker} Stock Price and Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plot_filename = f'../plots/{self.ticker}_stock_plot_{str(specific_date)[:10]}.png'
        plt.savefig(plot_filename)
        plt.close()
        return plot_filename
