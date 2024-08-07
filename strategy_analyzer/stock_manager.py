import smtplib
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests
import csv
from strategy_analyzer.stock import Stock
from strategy_analyzer.strategy import Strategy
import datetime
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
#from yahoo_fin import stock_info as si
from strategy_analyzer.utils import calculate_consecutive_days, calculate_consecutive_days2

# Adjust pandas display options to show all rows
pd.set_option('display.max_rows', 100)  # Set the max number of rows to display
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Set the width to avoid wrapping
pd.set_option('display.colheader_justify', 'left')  # Left align column headers


class StockManager:
    def __init__(self):
        pass

    # def fetch_top_stocks(self, n):
    #     try:
    #         # Fetch top stock tickers by market cap
    #         tickers = si.get_day_gainers().head(n)['Symbol'].tolist()
    #         return tickers
    #     except Exception as e:
    #         print(f"Error fetching top stocks: {e}")
    #         return []

    def fetch_tickers_from_alphavantage(self, api_key):
        url = f"https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.text
            with open("../summary_tables/tickers.csv", "w") as file:
                file.write(data)
            tickers_df = pd.read_csv("../summary_tables/tickers.csv")
            return tickers_df
        else:
            print(f"Error fetching tickers from Alpha Vantage: {response.status_code}")
            return pd.DataFrame()

    def fetch_market_cap(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            market_cap = stock.info.get('marketCap', 0)
            return market_cap
        except Exception as e:
            print(f"Error fetching market cap for {ticker}: {e}")
            return 0

    def get_top_stocks(self, api_key, top_n=1000):
        tickers_df = self.fetch_tickers_from_alphavantage(api_key)
        tickers_df['marketCap'] = tickers_df['symbol'].apply(self.fetch_market_cap)
        tickers_df = tickers_df[tickers_df['marketCap'] > 0]  # Filter out stocks without market cap info
        tickers_df = tickers_df.sort_values(by=['marketCap', 'symbol'],
                                            ascending=[False, True])  # Sort by market cap and then alphabetically
        return tickers_df.head(top_n)

    def run_process_for_strategy_tickers(self, ticker_names:[], strategy, csv_file_path, start_date=None, end_date=None, load_from_disk=False, top_stocks=100)->[]:
        if not ticker_names:
            #ticker_names = self.fetch_top_stocks(top_stocks)
            #ticker_names = self.get_top_stocks('6GZV5H4NC27ZR564', top_stocks)
            ticker_names = pd.read_csv('../profits/top_1000_nasdaq_tickers.csv')['Symbol'][:top_stocks]
        stocks = []
        n = 0
        try:
            for ticker in ticker_names:
                try:
                    if load_from_disk:
                        stock_data = pd.read_csv(f'../tickers/{ticker}.csv')
                    else:
                        stock = yf.Ticker(ticker)
                        stock_data = stock.history(start=start_date, end=end_date)
                        stock_data.reset_index(inplace=True)
                        stock_info = stock.info

                    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
                    stock_data.set_index('Date', inplace=True)
                    stock_data_sliced = stock_data.loc[start_date:end_date] if start_date and end_date else stock_data
                    stock = Stock(stock.info['shortName'] if not load_from_disk else ticker, ticker, stock_data_sliced, stock_info if not load_from_disk else None)
                    strategy.calc_buy_sell(stock)
                    profit_data, total_profit = stock.calculate_profit()
                    print(f'Total Profit for ticker {stock.ticker} stock {stock.name}: {total_profit}')
                    print(profit_data)
                    n += 1
                    print(f'got data for ticker {ticker}: {stock.name}')
                    profit_data.to_csv(csv_file_path, mode='a', index=False, header=not pd.io.common.file_exists(csv_file_path))
                except Exception as e:
                    print(f'Failed get data for ticker {ticker}')
                    continue
                print(f'Got data for {n} tickers of {top_stocks}')

        except Exception as e:
            print(f'Error on get_stocks_by_ticker: {e}')
            raise Exception(e)

    def get_SPY_close(self, start_date, end_date):
        stock = yf.Ticker('SPY')
        stock_data = stock.history(start=start_date, end=end_date)
        stock_data.reset_index(inplace=True)
        stock_info = stock.info

        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data.set_index('Date', inplace=True)
        stock = Stock(stock.info['shortName'], 'SPY', stock_data, stock_info)
        stock.stock_data['Normalized_Close'] = self.rolling_normalize_to_base(stock.stock_data['Close'],150)
        stock.stock_data['Normalized_Close_30'] = self.rolling_normalize_to_base(stock.stock_data['Close'], 30)
        return stock

    def normalize_to_reference(self, target_data, reference_data):
        reference_min = reference_data.min()
        reference_max = reference_data.max()
        normalized_data = (target_data - reference_min) / (reference_max - reference_min)
        return normalized_data

    # Normalize both time series to start from 100
    def normalize_to_base(self, data, base_value=100):
        return (data / data.iloc[0]) * base_value

    def rolling_normalize_to_base(self, data, window, base_value=100):
        normalized_data = pd.Series(index=data.index)
        for i in range(len(data)):
            if i >= window - 1:
                window_data = data.iloc[i - window + 1:i + 1]
                normalized_data.iloc[i] = (window_data.iloc[-1] / window_data.iloc[0]) * base_value
            else:
                normalized_data.iloc[i] = (data.iloc[i] / data.iloc[0]) * base_value
        return normalized_data

    # Function to normalize data to [0, 1] with a moving window
    def moving_min_max_normalize(self, data, window=150):
        rolling = data.rolling(window=window)
        rolling_min = rolling.min()
        rolling_max = rolling.max()
        normalized_data = (data - rolling_min) / (rolling_max - rolling_min)
        return normalized_data

    def prepare_data_for_list_of_stocks(self, ticker_names, top_stocks, spy_stock, summary_table_data, watch_list_table, added_stocks, removed_stocks, added_stocks_watch_list, unusual_volume, buy_indicator, sell_indicator):
        for i, ticker in enumerate(ticker_names):
            try:
                stock = yf.Ticker(ticker)
                #stock_data = stock.history(start=start_date, end=end_date)
                #stock = yf.download(ticker, period=f'{345}d', interval='1d')
                try:
                    stock_data = stock.history(period='2y', interval='1d')
                except Exception as e:
                    stock_data = stock.history(period='max', interval='1d')
                stock_data.reset_index(inplace=True)
                stock_info = stock.info

                stock_data['Date'] = pd.to_datetime(stock_data['Date'])
                stock_data.set_index('Date', inplace=True)
                stock = Stock(stock.info['shortName'], ticker, stock_data, stock_info)

                last_close = stock.stock_data['Close'].iloc[-1]
                previous_close = stock.stock_data['Close'].iloc[-2]
                price_change_pct = (((last_close - previous_close) / previous_close) * 100).round(3)

                last_volume = stock.stock_data['Volume'].iloc[-1]
                previous_volume = stock.stock_data['Volume'].iloc[-2]
                volume_change_pct = (((last_volume - previous_volume) / previous_volume) * 100).round(3)

                stock.stock_data[f'Moving_Avg_{150}'] = stock.stock_data['Close'].rolling(window=150).mean()
                stock.stock_data[f'Moving_Avg_{50}'] = stock.stock_data['Close'].rolling(window=50).mean()
                stock.stock_data[f'Moving_Avg_{20}'] = stock.stock_data['Close'].rolling(window=20).mean()
                stock.stock_data[f'Moving_Avg_{7}'] = stock.stock_data['Close'].rolling(window=7).mean()

                stock.stock_data['low7'] = stock.stock_data['Low'].rolling(window=7).min()
                stock.stock_data['high7'] = stock.stock_data['High'].rolling(window=7).max()
                stock.stock_data['close7'] = stock.stock_data['Close']
                stock.stock_data['volume7'] = stock.stock_data['Volume'].rolling(window=7).sum()


                stock.stock_data['SMA_Volume150'] = stock.stock_data['Volume'].rolling(window=150).mean()
                stock.stock_data['STD_Volume150'] = stock.stock_data['Volume'].rolling(window=150).std()
                current_date = stock.stock_data.index[-1]
                close_price = stock.stock_data.loc[str(current_date), 'Close'].round(2)
                sma150_price = stock.stock_data.loc[str(current_date), f'Moving_Avg_{150}'].round(2)
                sma50_price = stock.stock_data.loc[str(current_date), f'Moving_Avg_{50}'].round(2)
                sma20_price = stock.stock_data.loc[str(current_date), f'Moving_Avg_{20}'].round(2)
                sma7_price = stock.stock_data.loc[str(current_date), f'Moving_Avg_{7}'].round(2)
                volume = stock.stock_data.loc[str(current_date), 'Volume']
                sma150_volume = stock.stock_data.loc[str(current_date), 'SMA_Volume150']
                std150_volume = stock.stock_data.loc[str(current_date), 'STD_Volume150']

                stock.stock_data['volume_change_prct'] = (stock.stock_data['Volume'] - stock.stock_data['Volume'].shift(1)) / stock.stock_data['Volume'].shift(1) * 100
                stock.stock_data['consecutive_volume_change_prct'] = calculate_consecutive_days2(stock.stock_data['volume_change_prct'])
                consecutive_volume_change_prct = stock.stock_data['consecutive_volume_change_prct'].iloc[-1]

                stock.stock_data['consecutive_volume_days'] = calculate_consecutive_days(stock.stock_data['Volume'], stock.stock_data['SMA_Volume150'])
                consecutive_volume_days = stock.stock_data['consecutive_volume_days'].iloc[-1]

                stock.stock_data['close_price_prct'] = (stock.stock_data['Close'] - stock.stock_data['Close'].shift(1)) / stock.stock_data['Close'].shift(1) * 100
                stock.stock_data['consecutive_up_days_prct'] = calculate_consecutive_days2(stock.stock_data['close_price_prct'])
                consecutive_up_days_prct = stock.stock_data['consecutive_up_days_prct'].iloc[-1]
                stock.stock_data['consecutive_up_days_above_sma150'] = calculate_consecutive_days(stock.stock_data['Close'], stock.stock_data['Moving_Avg_150'])
                consecutive_up_days_above_sma150 = stock.stock_data['consecutive_up_days_above_sma150'].iloc[-1]

                stock.calculate_rsi()
                stock.compute_cmf()
                stock.compute_cci()
                stock.positive_volume_index()
                stock.klinger_oscillator()
                stock.klinger_volume_oscillator7()
                stock.klinger_volume_oscillator_tv()
                stock.klinger_volume_oscillator_tv_weekly()
                stock.klinger_volume_oscillator_tv_3days()
                rsi = stock.stock_data.loc[str(current_date), 'RSI'].round(2)
                cmf = stock.stock_data.loc[str(current_date), 'CMF'].round(2)
                cci = stock.stock_data.loc[str(current_date), 'CCI'].round(2)
                avg_maxima, median_maxima = stock.calculate_maxima_statistics()

                avg_maxima_perc = (((avg_maxima - close_price) / avg_maxima) * 100).round(3)
                median_maxima_perc = (((median_maxima - close_price) / median_maxima) * 100).round(3)
                stock.stock_data['Normalized_Close'] = self.rolling_normalize_to_base(stock.stock_data['Close'], 150)
                stock.stock_data['Normalized_Close_30'] = self.rolling_normalize_to_base(stock.stock_data['Close'], 30)
                normalized_close_vs_spy = (stock.stock_data.loc[str(current_date.date()), 'Normalized_Close'] - spy_stock.stock_data.iloc[-1]['Normalized_Close']).round(3)
                print(f'Processing ticker {stock.name} {stock.ticker} counter = {i} of {top_stocks}')
                distance_percentage = ((close_price - sma150_price) * 100 / sma150_price).round(3)

                prev_sma150_price = stock.stock_data.loc[str(current_date), f'Moving_Avg_{150}'].round(2)

                klinger_sell = stock.stock_data['klinger_sell_signal'].iloc[-1]
                klinger_buy = stock.stock_data['klinger_buy_signal'].iloc[-1]

                klinger_tv_consecutive_buy_daily = stock.stock_data['klinger_tv_consecutive_buy_daily'].iloc[-1]
                klinger_tv_consecutive_sell_daily = stock.stock_data['klinger_tv_consecutive_sell_daily'].iloc[-1]

                klinger_consecutive_buy_3days = stock.stock_data['klinger_consecutive_buy_3days'].iloc[-1]
                klinger_consecutive_sell_3days = stock.stock_data['klinger_consecutive_sell_3days'].iloc[-1]

                klinger_tv_consecutive_sell_weekly = stock.stock_data['klinger_tv_consecutive_sell_weekly'].iloc[-1]
                klinger_tv_consecutive_buy_weekly = stock.stock_data['klinger_tv_consecutive_buy_weekly'].iloc[-1]

                klinger7_signal = stock.stock_data.loc[str(current_date), 'Trigger7'].round(2)
                klinger7_kvo = stock.stock_data.loc[str(current_date), 'KVO7'].round(2)
                klinger7_sell = stock.stock_data['klinger7_sell_signal'].iloc[-1]
                klinger7_buy = stock.stock_data['klinger7_buy_signal'].iloc[-1]
                klinger_tv_signal = stock.stock_data.loc[str(current_date), 'Trigger'].round(2)
                klinger_tv_kvo = stock.stock_data.loc[str(current_date), 'KVO'].round(2)
                klinger_tv_sell = stock.stock_data['klinger_tv_sell_signal'].iloc[-1]
                klinger_tv_buy = stock.stock_data['klinger_tv_buy_signal'].iloc[-1]
                klinger_tv_sell_weekly = stock.stock_data['klinger_tv_sell_signal_weekly'].iloc[-1]
                klinger_tv_buy_weekly = stock.stock_data['klinger_tv_buy_signal_weekly'].iloc[-1]
                klinger_tv_sell_3days = stock.stock_data['klinger_tv_sell_signal_3days'].iloc[-1]
                klinger_tv_buy_3days = stock.stock_data['klinger_tv_buy_signal_3days'].iloc[-1]
                pvi_sell = stock.stock_data['pvi_sell_signal'].iloc[-1]
                pvi_buy = stock.stock_data['pvi_buy_signal'].iloc[-1]

                #added and removed stocks
                if previous_close < prev_sma150_price and close_price > sma150_price:
                    added_stocks.append([stock.ticker, stock.name, previous_close, close_price, price_change_pct, prev_sma150_price, sma150_price, volume, volume_change_pct, sma150_volume, std150_volume])
                    added_stocks_watch_list.append([f'{stock.ticker}', '441.58', '2024/06/13', '16:00 EDT', '0.519989', '440.78', '443.39',
                         '439.37', '15858484', '', '', '', '', '', '', ''])
                    print(f'Ticker {stock.ticker}, {stock.name} added to the current day list')
                if previous_close > prev_sma150_price and close_price < sma150_price:
                    removed_stocks.append([stock.ticker, stock.name, previous_close, close_price, price_change_pct, prev_sma150_price, sma150_price, volume, volume_change_pct, sma150_volume, std150_volume])
                    print(f'Ticker {stock.ticker}, {stock.name} removed from the current day list')

                if (pvi_buy or klinger_tv_buy) and close_price > sma150_price:
                    buy_indicator.append([f'{stock.ticker}', '441.58', '2024/06/13', '16:00 EDT', '0.519989', '440.78', '443.39','439.37', '15858484', '', '', '', '', '', '', ''])
                    print(f'Ticker {stock.ticker}, {stock.name} added to buy indicator list')

                if (pvi_sell or klinger_tv_sell) and close_price < sma150_price:
                    sell_indicator.append([f'{stock.ticker}', '441.58', '2024/06/13', '16:00 EDT', '0.519989', '440.78', '443.39', '439.37', '15858484', '', '', '', '', '', '', ''])
                    print(f'Ticker {stock.ticker}, {stock.name} added to sell indicator list')

                #volume
                if volume > (sma150_volume + std150_volume):
                    unusual_volume.append([stock.ticker, stock.name, previous_close, close_price,price_change_pct, prev_sma150_price, sma150_price, volume, volume_change_pct, sma150_volume, std150_volume])
                    print(f'Ticker {stock.ticker}, {stock.name} added to unusual volume list')
                #plot_filename = stock.plot_stock_data_for_spy(current_date, spy_stock)
                if close_price > sma150_price:

                    # attachments.append(plot_filename)
                    link = f'https://finance.yahoo.com/quote/{stock.ticker}'
                    beta = stock.stock_info.get('beta', "")
                    recommendationKey = stock.stock_info.get('recommendationKey', "")
                    summary_table_data.append(
                        [stock.ticker, stock.name, close_price, price_change_pct, consecutive_up_days_prct,
                         sma150_price,consecutive_up_days_above_sma150,
                         volume, volume_change_pct, sma150_volume, consecutive_volume_days, consecutive_volume_change_prct,
                         distance_percentage, normalized_close_vs_spy,
                         klinger_tv_sell, klinger_tv_buy,
                         klinger_tv_sell_3days, klinger_tv_buy_3days,
                         klinger_tv_sell_weekly, klinger_tv_buy_weekly,
                         klinger_tv_consecutive_sell_daily, klinger_tv_consecutive_buy_daily,

                         pvi_sell, pvi_buy, avg_maxima_perc, median_maxima_perc,
                         rsi, cmf, cci, sma50_price, sma20_price, sma7_price,
                         beta, recommendationKey,
                         stock.stock_info['numberOfAnalystOpinions'], link])




                    short_list_columns = ['ticker', 'name', 'previous_close', 'close_price', 'price_change_pct',
                                          'prev_sma150_price',
                                          'sma150_price', 'volume', 'volume_change_pct', 'sma150_volume',
                                          'std150_volume']


                    watch_list_table.append(
                        [f'{stock.ticker}', '441.58', '2024/06/13', '16:00 EDT', '0.519989', '440.78', '443.39',
                         '439.37', '15858484', '', '', '', '', '', '', ''])
                    print(
                        f'=============Ticker {ticker} crossed SMA150: close_price = {close_price}, sma150_price = {sma150_price}================')
            except Exception as e:
                print(f'Failed get data for ticker {ticker}: {e}')
                continue

    def get_all_trend_changers_tickers(self, ticker_names:[], title, start_date, end_date, to_email, top_stocks=1000)->[]:
        spy_stock = self.get_SPY_close(start_date, end_date)

        attachments = []
        summary_table_data_last_day = []
        watch_list_table = []
        added_stocks = []
        removed_stocks = []
        added_stocks_watch_list = []
        unusual_volume = []
        indicator_sell_list = []
        indicator_buy_list = []
        try:
            self.prepare_data_for_list_of_stocks(ticker_names, top_stocks, spy_stock, summary_table_data_last_day, watch_list_table, added_stocks, removed_stocks, added_stocks_watch_list, unusual_volume, indicator_buy_list, indicator_sell_list)

            watch_list_columns = ['Symbol', 'Current Price', 'Date', 'Time', 'Change', 'Open',
             'High', 'Low', 'Volume', 'Trade Date',
             'Purchase Price', 'Quantity', 'Commission', 'High Limit',
             'Low Limit', 'Comment']
            summary_columns = ['Ticker', 'Name', 'Current Price', 'price_change_pct', 'consecutive_up_days_prct',
                    'sma150_price', 'consecutive_days_above_sma150',
                     'Volume', 'volume_change_pct', 'sma150_volume', 'consecutive_volume_days_above_sma150', 'consecutive_volume_change_prct',
                     'distance_percentage', 'vs_spy',
                     'kelinger_tv_sell_signal', 'kelinger_tv_buy_signal',
                     'kelinger_tv_sell_signal_sell_3days', 'kelinger_tv_buy_signal_buy_3days',
                     'kelinger_tv_sell_signal_sell_weekly', 'kelinger_tv_buy_signal_buy_weekly',
                     'klinger_tv_consecutive_sell_daily', 'klinger_tv_consecutive_buy_daily',

                     'pvi_sell_signal', 'pvi_buy_signal', 'avg_maxima_perc', 'median_maxima_perc',
                     'rsi', 'cmf', 'cci', 'sma50_price', 'sma20_price', 'sma7_price', 'beta', 'recommendationKey',
                     'numberOfAnalystOpinions', 'Link']
            short_list_columns = ['ticker', 'name', 'previous_close', 'close_price', 'price_change_pct','prev_sma150_price',
                                  'sma150_price', 'volume', 'volume_change_pct', 'sma150_volume', 'std150_volume']

            summary_table_last_day_df = pd.DataFrame(summary_table_data_last_day, columns=summary_columns).sort_values(by='rsi')

            self.save_chunks(pd.DataFrame(watch_list_table, columns=watch_list_columns), file_name=f'watch_list_table_{end_date.date()}', attachments=attachments)

            summary_file_name = f'../summary_tables/summary_table_{end_date.date()}.csv'
            summary_table_last_day_df.to_csv(summary_file_name, index=False)
            attachments.append(summary_file_name)
            if added_stocks:
                added_stocks_df = pd.DataFrame(added_stocks, columns=short_list_columns)
                added_stocks_file_name = f'../summary_tables/added_stocks_{end_date.date()}.csv'
                added_stocks_df.to_csv(added_stocks_file_name, index=False)
                attachments.append(added_stocks_file_name)
                self.save_chunks(pd.DataFrame(added_stocks_watch_list,
                                              columns=watch_list_columns),
                                 file_name=f'../summary_tables/added_stocks_watch_list_table_{end_date.date()}', attachments=attachments)

            if removed_stocks:
                removed_stocks_df = pd.DataFrame(removed_stocks, columns=short_list_columns)
                removed_stocks_file_name = f'../summary_tables/removed_stocks_{end_date.date()}.csv'
                removed_stocks_df.to_csv(removed_stocks_file_name, index=False)
                attachments.append(removed_stocks_file_name)


            if unusual_volume:
                unusual_volume_df = pd.DataFrame(unusual_volume, columns=short_list_columns)
                unusual_volume_file_name = f'../summary_tables/unusual_volume_{end_date.date()}.csv'
                unusual_volume_df.to_csv(unusual_volume_file_name, index=False)
                attachments.append(unusual_volume_file_name)

            if indicator_sell_list:
                self.save_chunks(pd.DataFrame(indicator_sell_list, columns=watch_list_columns), file_name=f'../summary_tables/indicator_sell_watch_list_table_{end_date.date()}', attachments=attachments)

            if indicator_buy_list:
                self.save_chunks(pd.DataFrame(indicator_buy_list, columns=watch_list_columns), file_name=f'../summary_tables/indicator_buy_watch_list_table_{end_date.date()}', attachments=attachments)

            #summary_table = summary_table_df.to_string(index=False)
            if attachments:
                body = f'Trend change detected on {end_date}. Please find the attached plots for more details.'
                self.send_email(title, body, to_email, attachments, summary_table_last_day_df)
            else:
                print(f'No significant trend changes on {end_date} for any stocks.')

                #print(f'got data for ticker {ticker}: {stock.name}')
        except Exception as e:
            print(f'Error on get_stocks_by_ticker: {e}')
            raise Exception(str(e))

    def save_chunks(self, df, file_name, attachments, chunk_size=250):
        num_chunks = (len(df) // chunk_size) + 1
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk = df[start_idx:end_idx]
            if not chunk.empty:
                chunk.to_csv(f'{file_name}_{i + 1}.csv', index=False)
                attachments.append(f'{file_name}_{i + 1}.csv')
                print(f'Saved {file_name}_{i + 1}.csv')

    def get_stocks_by_tickers(self, ticker_names:[], start_date=None, end_date=None, load_from_disk=False, top_stocks=100)->[]:
        if not ticker_names:
            #ticker_names = self.fetch_top_stocks(top_stocks)
            #ticker_names = self.get_top_stocks('6GZV5H4NC27ZR564', top_stocks)
            ticker_names = pd.read_csv('../profits/top_1000_nasdaq_tickers.csv')['Symbol'][:top_stocks]
        stocks = []
        n = 0
        try:
            for ticker in ticker_names:
                try:
                    if load_from_disk:
                        stock_data = pd.read_csv(f'../tickers/{ticker}.csv')
                    else:
                        stock = yf.Ticker(ticker)
                        stock_data = stock.history(start=start_date, end=end_date)
                        stock_data.reset_index(inplace=True)
                        stock_info = stock.info

                    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
                    stock_data.set_index('Date', inplace=True)
                    stock_data_sliced = stock_data.loc[start_date:end_date] if start_date and end_date else stock_data
                    stock = Stock(stock.info['shortName'] if not load_from_disk else ticker, ticker, stock_data_sliced, stock_info if not load_from_disk else None)
                    stocks.append(stock)
                    n += 1
                    print(f'got data for ticker {ticker}: {stock.name}')
                except Exception as e:
                    print(f'Failed get data for ticker {ticker}')
                    continue
                print(f'Got data for {n} tickers of {top_stocks}')
            return stocks
        except Exception as e:
            print(f'Error on get_stocks_by_ticker: {e}')
            raise Exception(e)

    def run_strategy(self, stocks:[], strategy:Strategy):
        for stock in stocks:
            strategy.calc_buy_sell(stock)




    def visualize(self, stocks, strategy:Strategy):

        # Plot each stock's data
        for stock in stocks:
            image_path = strategy.visualize(stock)
        plt.show(block=True)

    def send_mail_with_trend_change_signals(self, stocks, current_date, to_email):
        attachments = []
        summary_table_data = []
        for stock in stocks:
            trend_change = stock.get_trend_change_signal(current_date)
            if trend_change and trend_change != 'No Change':
                print(f'trend_change type {trend_change}')
                plot_filename = stock.plot_stock_data(current_date, trend_change)
                attachments.append(plot_filename)
                current_price = stock.stock_data.loc[current_date, 'Close']
                summary_table_data.append([stock.ticker, stock.name, current_price, trend_change])

        summary_table_df = pd.DataFrame(summary_table_data, columns=['Ticker', 'Name', 'Current Price', 'Trend Change'])
        summary_table = summary_table_df.to_string(index=False)
        if attachments:
            subject = f'Stock Trend Change Notifications for {current_date}'
            body = f'Trend change detected on {current_date}. Please find the attached plots for more details.'
            self.send_email(subject, body, to_email, attachments, summary_table_df)
        else:
            print(f'No significant trend changes on {current_date} for any stocks.')

    def send_email(self, subject, body, to_email, attachments, summary_table=None):
        try:
            sender_email = 'oksi.shafransky@gmail.com'
            sender_password = 'llyd uexw iaec yitp'
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ', '.join(to_email)
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))
            summary_table_html = "<html><body><h2>Summary Table</h2><table border='1'>"
            summary_table_html += "<tr>"
            for col_name in summary_table.columns:
                summary_table_html += f"<th>{col_name}</th>"
            summary_table_html += "</tr>"
            for index, row in summary_table.iterrows():
                summary_table_html += "<tr>"
                for value in row:
                    summary_table_html += f"<td>{value}</td>"
                summary_table_html += "</tr>"
            summary_table_html += "</table></body></html>"
            msg.attach(MIMEText(summary_table_html, 'html'))

            for attachment in attachments:
                with open(attachment, 'rb') as f:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header('Content-Disposition', f'attachment; filename={attachment}')
                    msg.attach(part)
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            print(f'Email sent to {to_email} with attachments {attachments}')
        except Exception as e:
            print(f'Error on sending email {e}')


    def send_email_with_plots(self, stocks, strategy, recipient_email):
        try:
            # Email configuration
            sender_email = 'oksi.shafransky@gmail.com'
            sender_password = 'llyd uexw iaec yitp'
            subject = 'Stock Plots'
            body = 'Please find the attached stock plots.'

            # Create a multipart message
            msg = MIMEMultipart()
            msg['From'] = sender_email # sender_email
            msg['To'] =  ', '.join(recipient_email)
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            # Attach the plots
            for stock in stocks:
                plot_path = strategy.visualize(stock)
                attachment = open(plot_path, 'rb')
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f"attachment; filename= {os.path.basename(plot_path)}")
                msg.attach(part)
                attachment.close()
                os.remove(plot_path)  # Remove the file after attaching

            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()  # Enable TLS
                server.login(sender_email, sender_password)
                text = msg.as_string()
                server.sendmail(sender_email, recipient_email, text)
        except Exception as e:
            print(f'Error on sending email {e}')

    def calc_profit_for_stocks(self, stocks):
        all_profits = []
        for stock in stocks:
            profit_data, total_profit = stock.calculate_profit()
            print(f'Total Profit for ticker {stock.ticker} stock {stock.name}: {total_profit}')
            print(profit_data)
            all_profits.append(profit_data)
        return pd.concat(all_profits, ignore_index=True)