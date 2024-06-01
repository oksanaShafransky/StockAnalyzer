import smtplib
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
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

class StockManager:
    def __init__(self):
        pass

    def get_stocks_by_ticker(self, ticker_names:[], start_date=None, end_date=None)->[]:
        stocks = []
        try:
            for ticker in ticker_names:
                stock = yf.Ticker(ticker)
                stock_data = stock.history(start=start_date, end=end_date)
                stock_data.reset_index(inplace=True)
                stock_info = stock.info

                stock_data['Date'] = pd.to_datetime(stock_data['Date'])
                stock_data.set_index('Date', inplace=True)
                stock_data_sliced = stock_data.loc[start_date:end_date] if start_date and end_date else stock_data
                stock = Stock(yf.Ticker(ticker).info['shortName'], ticker, stock_data_sliced, stock_info)
                stocks.append(stock)
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

    def calc_profit_for_stocks(self, stock):
        pass

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