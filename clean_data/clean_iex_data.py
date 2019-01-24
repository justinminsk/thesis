import pandas as pd 
from dateutil.parser import *

df = pd.read_csv("~/repo/thesis/iex/mintue_trade_data.csv", na_values=[-1])

df = df.fillna(method="ffill")

df.to_pickle("./iex_clean.pkl")

# TODO: Get time in python datetime and also only have averge saved on own to_pickle
# TODO: Look into if day by day is useful for non-trading days

date_df = pd.DataFrame({"date": df.date.astype(str) + " " + df.minute, "stock_price_col" : df.average})
date_df.date = date_df.date.apply(parse)

print(date_df.head())

date_df.to_pickle("./date_iex_data.pkl")
