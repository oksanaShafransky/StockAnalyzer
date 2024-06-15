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
            with open("tickers.csv", "w") as file:
                file.write(data)
            tickers_df = pd.read_csv("tickers.csv")
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

    def calculate_rsi(self, stock, period=14):
        delta = stock.stock_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        stock.stock_data['RSI'] = rsi
        return stock.stock_data

    def get_all_trend_changers_tickers(self, ticker_names:[], start_date, end_date, to_email, top_stocks=1000)->[]:
        if not ticker_names:
            ticker_names = pd.read_csv('../profits/top_1000_nasdaq_tickers.csv')['Symbol'][:top_stocks]

        spy_stock = self.get_SPY_close(start_date, end_date)
        spy_stock.stock_data['Normalized_Close'] = self.rolling_normalize_to_base(spy_stock.stock_data['Close'],150)
        spy_stock.stock_data['Normalized_Close_30'] = self.rolling_normalize_to_base(spy_stock.stock_data['Close'], 30)
        attachments = []
        summary_table_data = []
        watch_list_table = []
        watch_list_table.append(
            ['Symbol', 'Current Price', 'Date', 'Time', 'Change', 'Open', 'High', 'Low', 'Volume', 'Trade Date',
             'Purchase Price', 'Quantity', 'Commission', 'High Limit', 'Low Limit', 'Comment'])
        try:
            for i, ticker in enumerate(ticker_names):
                try:
                    stock = yf.Ticker(ticker)
                    stock_data = stock.history(start=start_date, end=end_date)
                    stock_data.reset_index(inplace=True)
                    stock_info = stock.info

                    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
                    stock_data.set_index('Date', inplace=True)
                    stock = Stock(stock.info['shortName'], ticker, stock_data, stock_info)
                    trend_change_SMA150 = stock.get_trend_change_signal(end_date, window=150)
                    trend_change_SMA50 =  stock.get_trend_change_signal(end_date, window=50)
                    trend_change_SMA20 =  stock.get_trend_change_signal(end_date, window=20)
                    trend_change_SMA14 =  stock.get_trend_change_signal(end_date, window=14)
                    trend_change_SMA7 =   stock.get_trend_change_signal(end_date, window=7)
                    stock.stock_data[f'Moving_Avg_{150}'] = stock.stock_data['Close'].rolling(window=150).mean()
                    stock.stock_data[f'Moving_Avg_{50}'] = stock.stock_data['Close'].rolling(window=50).mean()
                    stock.stock_data[f'Moving_Avg_{20}'] = stock.stock_data['Close'].rolling(window=20).mean()
                    stock.stock_data[f'Moving_Avg_{7}'] = stock.stock_data['Close'].rolling(window=7).mean()
                    current_date = pd.to_datetime(end_date).date()
                    close_price = stock.stock_data.loc[str(current_date), 'Close'].round(2)
                    sma150_price = stock.stock_data.loc[str(current_date), f'Moving_Avg_{150}'].round(2)
                    sma50_price = stock.stock_data.loc[str(current_date), f'Moving_Avg_{50}'].round(2)
                    sma20_price = stock.stock_data.loc[str(current_date), f'Moving_Avg_{20}'].round(2)
                    sma7_price = stock.stock_data.loc[str(current_date), f'Moving_Avg_{7}'].round(2)
                    self.calculate_rsi(stock)
                    rsi = stock.stock_data.loc[str(current_date), 'RSI'].round(2)
                    stock.stock_data['Normalized_Close'] = self.rolling_normalize_to_base(stock.stock_data['Close'],150)
                    stock.stock_data['Normalized_Close_30'] = self.rolling_normalize_to_base(stock.stock_data['Close'],30)
                    #stock.stock_data['Normalized_Close'] = self.normalize_to_reference(stock.stock_data['Close'], spy_stock.stock_data['Close'])
                    normalized_close_vs_spy = stock.stock_data.loc[str(current_date), 'Normalized_Close'] - spy_stock.stock_data.loc[str(current_date), 'Normalized_Close'].round(3)
                    print(f'Processing ticker {stock.name} {stock.ticker} counter = {i} of {top_stocks}')
                    #if trend_change_SMA150=='Negative to Positive':
                    distance_percentage = (close_price - sma150_price)*100/sma150_price
                    if close_price > sma150_price:
                        plot_filename = stock.plot_stock_data_for_spy(end_date, spy_stock)
                        attachments.append(plot_filename)
                        current_price = stock.stock_data.loc[str(end_date)[:10], 'Close'].round(2)
                        link = f'https://finance.yahoo.com/quote/{stock.ticker}'
                        summary_table_data.append([stock.ticker, stock.name, current_price, sma150_price,  distance_percentage, normalized_close_vs_spy, rsi, sma50_price,sma20_price, sma7_price, stock.stock_info['beta'], stock.stock_info['recommendationKey'], stock.stock_info['numberOfAnalystOpinions'], link])
                        watch_list_table.append([f'{stock.ticker}','441.58','2024/06/13','16:00 EDT','0.519989','440.78','443.39','439.37','15858484','','','','','','',''])
                        print(f'=============found trend change for ticker {ticker}: close_price = {close_price}, current_price = {current_price} ================')
                except Exception as e:
                    print(f'Failed get data for ticker {ticker}: {e}')
                    continue
            summary_table_df = pd.DataFrame(summary_table_data, columns=['Ticker', 'Name', 'Current Price', 'sma150_price', 'distance_percentage', 'vs_spy', 'rsi', 'sma50_price', 'sma20_price','sma7_price','beta','recommendationKey','numberOfAnalystOpinions','Link']).sort_values(by='rsi')

            self.save_chunks(pd.DataFrame(watch_list_table), file_name=f'watch_list_table_{end_date.date()}', attachments=attachments)
            # for i in range(max(1, int(len(watch_list_table)%300))):
            #     watch_list_file_name = f'watch_list_table_{end_date.date()}_{i}.csv'
            #     with open(watch_list_file_name, 'w', newline='') as file:
            #         writer = csv.writer(file)
            #         writer.writerows(watch_list_table)
            #         attachments.append(watch_list_file_name)

            summary_file_name = f'summary_table_{end_date.date()}.csv'
            summary_table_df.to_csv(summary_file_name, index=False)
            attachments.append(summary_file_name)

            #summary_table = summary_table_df.to_string(index=False)
            if attachments:
                subject = f'Stock Change Notifications for {end_date}'
                body = f'Trend change detected on {end_date}. Please find the attached plots for more details.'
                self.send_email(subject, body, to_email, attachments, summary_table_df)
            else:
                print(f'No significant trend changes on {end_date} for any stocks.')

                #print(f'got data for ticker {ticker}: {stock.name}')
        except Exception as e:
            print(f'Error on get_stocks_by_ticker: {e}')
            raise Exception(str(e))

    def save_chunks(self, df, file_name, attachments, chunk_size=300):
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