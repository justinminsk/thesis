import fastparquet
import pandas as pd 
import numpy as np
from dateutil.parser import *

print("--Start--")
print("clean_iex_data")
# Get Data
df = pd.read_csv("./iex/mintue_trade_data.csv", na_values=[-1])

# Get date and  minute to timestamps
df.date = df.date.astype(str) + " " + df.minute
df = df.drop(columns=["minute"])
df.date = df.date.apply(parse)

# Fill Na with values above
df = df.fillna(method="ffill")
df = df.dropna()

df.average = df.average.shift(-1)
df = df[:-1]

# Save all data to a pickle
fastparquet.write("iex_clean.parquet", df)

# create a smaller dataframe to add to twitter and wallstreet journal
date_df = pd.DataFrame({"date_col": df.date, "stock_price_col" : df.average})

# save smaller df to a pickle
fastparquet.write("date_iex_data.parquet", date_df)

print("--End--")
