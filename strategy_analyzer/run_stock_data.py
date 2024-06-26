import datetime
from strategy_analyzer.yt_macd_rsi import YT_MACD_RSI_strategy
from strategy_analyzer.sma_strategy import SMAStrategy
from strategy_analyzer.stock_manager import StockManager
import pandas as pd

if __name__ == "__main__":
    end_date = datetime.datetime.now() - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=2*365)
    to_email = ['vladiks@gmail.com', 'oksi78@yahoo.com', 'oksi.shafransky@gmail.com']
    tickers = ['NVDA']
    stock_manager = StockManager()
    # stocks_default = stock_manager.get_stocks_by_tickers(None, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), load_from_disk=False, top_stocks=1000)
    # stocks_for_strategy_130 = stocks_default
    #
    # sma_strategy_default = YT_MACD_RSI_strategy()
    # sma_strategy_130 = YT_MACD_RSI_strategy(window=150)
    #
    # stock_manager.run_strategy(stocks_default, sma_strategy_default)
    #
    # #stock_manager.visualize(stocks_default, sma_strategy_default)
    #
    # all_profits = stock_manager.calc_profit_for_stocks(stocks_default)
    # all_profits.to_csv('../profits/profits_for_YT_MACD.csv', index=False)

    #sma_strategy_default = SMAStrategy()
    #stock_manager.run_process_for_strategy_tickers(None, sma_strategy_default, '../profits/profits_for_sma.csv', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), load_from_disk=False, top_stocks=1000)
    stock_manager.get_all_trend_changers_tickers(None, start_date, end_date, to_email)


    #stock_manager.send_mail_with_trend_change_signals(stocks_default, '2024-05-28', to_email)
    #stock_manager.send_email_with_plots(stocks_default, sma_strategy_default, ['vladiks@gmail.com','oksi78@yahoo.com','oksi.shafransky@gmail.com'])
