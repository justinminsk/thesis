import pandas as pd 

df = pd.read_csv("~/repo/thesis/iex/mintue_trade_data.csv", na_values=[-1])

df = df.fillna(method="ffill")

print(df.head(50))

df.to_pickle("./iex_clean.pkl")

# TODO: Get time in python datetime and also only have averge saved on own to_pickle
# TODO: Look into if day by day is useful for non-trading days
