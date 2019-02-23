import fastparquet
import os
import glob
import re
from scipy import stats
from tqdm import tqdm
import pandas as pd
import numpy as np
from dateutil.parser import parse


print("--Start--")
print("clean_wallstreet_data")

df = pd.read_csv("wallstreet/WSJ.csv")

date_df = pd.read_parquet("date_iex_data.parquet", engine="fastparquet")

print("df:", df.shape)
print("Date_df:", date_df.shape)

print("Data Imported")

df.date = df.date.apply(parse).map(lambda x: x.replace(second=0, microsecond=0, tzinfo=None))

df = df.sort_values("date")

print("Resampled To Get Articles Text Per Minute")

# https://stackoverflow.com/questions/46656718/merge-2-dataframes-on-closest-past-date
df = pd.merge_asof(df.set_index('date_col').sort_index(),
                   date_df.set_index('date', drop=False).sort_index(),
                   left_index=True, right_index=True, direction="forward")

df = df.reset_index(drop=True)

print("df:", df.shape)
print(df.head())

print("Get Articles Tied to Trading Times")

print("Data Merged")

del date_df

filename = "wallstreet/wallstreet_data.parquet"
fastparquet.write(filename, df)

print("Full DataFrame Saved")

print("--End--")
