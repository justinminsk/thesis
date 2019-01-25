import pandas as pd 
import numpy as np
from dateutil.parser import *


# Get Data
df = pd.read_csv("~/repo/thesis/iex/mintue_trade_data.csv", na_values=[-1])

# Get date and  minute to timestamps
df.date = df.date.astype(str) + " " + df.minute
df = df.drop(columns=["minute"])
df.date = df.date.apply(parse)

# Get missing minutes
start = parse("2018-12-11 09:30:00")
end = parse("2019-01-22 15:59:00")

date_list = pd.date_range(start=start, end=end, freq="min")

df.index = df.date
df = df.reindex(date_list, fill_value=np.nan)

# Fill Na with values above
df = df.fillna(method="ffill")

# Save all data to a pickle
df.to_pickle("./iex_clean.pkl")

# create a smaller dataframe to add to twitter and wallstreet journal
date_df = pd.DataFrame({"date": df.date, "stock_price_col" : df.average})

# save smaller df to a pickle
date_df.to_pickle("./date_iex_data.pkl")
