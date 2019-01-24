import pandas as pd 

df = pd.read_csv("~/repo/thesis/iex/mintue_trade_data.csv", na_values=[-1])

df = df.fillna(method="ffill")

print(df.head(50))

df.to_pickle("./iex_clean.pkl")
