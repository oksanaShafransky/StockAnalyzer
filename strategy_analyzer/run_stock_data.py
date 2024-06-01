import datetime
from strategy_analyzer.sma_strategy import SMAStrategy
from strategy_analyzer.stock_manager import StockManager

if __name__ == "__main__":
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=600)

    tickers = ['AAPL','AIG']
    stock_manager = StockManager()
    stocks_default = stock_manager.get_stocks_by_ticker(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    stocks_for_strategy_130 = stocks_default
    sma_strategy_default = SMAStrategy()
    sma_strategy_130 = SMAStrategy(window=130)
    stock_manager.run_strategy(stocks_default, sma_strategy_default)
    #stock_manager.visualize(stocks_default, sma_strategy_default)
    stock_manager.send_email_with_plots(stocks_default, sma_strategy_default, ['vladiks@gmail.com','oksi78@yahoo.com','oksi.shafransky@gmail.com'])
