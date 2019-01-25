import iexfinance as iex
from datetime import datetime
import pandas as pd

azmn = iex.stocks.Stock("AMZN", output_format="pandas")

amzn_quote = azmn.get_quote(filter_=["openTime", "open", "closeTime", "close", "high", "low", "latestPrice", "latestUpdate"
                                 "latestTime", "latestVolume", "delayedPriceTime", "delayedPrice", "extendedPrice", 
                                 "extendedChange", "extendedChangePercent", "previousClose", "change", "percentChange"
                                 "marketCap", "wek52High", "week52Low", "ytdChange"])

print(amzn_quote)

start = datetime(2018, 12, 12)

end = datetime(2019, 1, 22)

dates = pd.date_range(start, end).tolist()

minute_data = iex.stocks.get_historical_intraday("AMZN", datetime(2018, 12, 11), output_format="pandas")

for date in dates:
    day_trades = iex.stocks.get_historical_intraday("AMZN", date, output_format="pandas")
    print(day_trades)
    minute_data = minute_data.append(day_trades)


minute_data.to_csv("mintue_trade_data.csv")
