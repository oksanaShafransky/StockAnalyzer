import pandas as pd
import pandas_ta as ta

df = pd.DataFrame() # Empty DataFrame

# Load data
df = pd.read_csv("/home/vlad/PycharmProjects/StockAnalyzer/tickers/AAPL.csv", sep=",")
# OR if you have yfinance installed
df = df.ta.ticker("aapl")

# VWAP requires the DataFrame index to be a DatetimeIndex.
# Replace "datetime" with the appropriate column from your DataFrame
#df.set_index(pd.DatetimeIndex(df["datetime"]), inplace=True)

# Calculate Returns and append to the df DataFrame
df.ta.sma(length=10)
df.ta.log_return(cumulative=True, append=True)
df.ta.percent_return(cumulative=True, append=True)


# New Columns with results
df.columns

# Take a peek
print(df.tail())