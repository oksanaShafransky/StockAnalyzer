import pandas as pd
def is_two_crows(data):
  """
  Identifies two crows candlestick pattern in a DataFrame.

  Args:
      data (pd.DataFrame): DataFrame with columns 'Open', 'High', 'Low', 'Close'.

  Returns:
      pd.Series: Series with True/False values indicating if the pattern is present.
  """

  # Conditions for Two Crows:
  # 1. First candle is long black candlestick (Open > Close)
  # 2. Second candle opens within the first candle's body (Second Open < First Open)
  # 3. Second candle closes lower than the first candle's body (Second Close < First Open)
  conditions = [
      data['Open'] > data['Close'],  # First candle black
      data['Open.1'] < data['Open'],   # Second Open lower than first Open
      data['Close.1'] < data['Open']    # Second Close lower than first Open body
  ]
  return data[conditions].any(axis=1)
