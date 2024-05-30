from stock_analyzer.stock import Stock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    start_date = '2022-01-01'
    end_date = '2024-05-22'
    window = 150
    ticker = 'AAPL'
    stock_data = pd.read_csv(f'../tickers/{ticker}.csv')
    stock = Stock('Apple', ticker, stock_data)
