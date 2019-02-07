import fastparquet
import os
import glob
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
from dateutil.parser import *

print("--Start--")
print("spilt_data_by_day")

# get minute by minute list 
start = parse("2018-12-11") # 2018-12-12  09:29:00 
end = parse("2019-01-24") # 2019-01-23  16:00:00

dates_list = pd.date_range(start=start, end=end, freq="D")

# https://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe
path = "./en_tweets"
allFiles = glob.glob(os.path.join(path,"*.csv"))

np_array_list = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    np_array_list.append(df.as_matrix())

comb_np_array = np.vstack(np_array_list)
df = pd.DataFrame(comb_np_array)

df.columns = ["created_at", "id_str", "text", "truncated", "verified", "followers_count", "favourites_count"]

df.drop(columns=["id_str", "truncated", "verified", "followers_count", "favourites_count"])

df["tweet_count"] = 1

date_df = pd.read_parquet("date_iex_data.parquet", engine="fastparquet")

print("Data Imported")

df.created_at = df.created_at.apply(parse)
df.created_at = df.created_at.map(lambda x: x.replace(tzinfo=None)).map(lambda x: x.replace(second=0, microsecond=0, tzinfo=None))
df.loc[:,'date_col'] = df.created_at

print("Changed DateTime to Minute By Minute")

df = df.set_index("created_at")

df = df.resample("1Min").agg({"text" : " ".join, "tweet_count" : sum})

print("Resampled To Get Tweet Text Per Minute")

df = df.merge(date_df, how="left", on="date_col")
df = df.drop("date_col", 1)

print("Data Merged")

del date_df

filename = "day_data/full_data.parquet"
fastparquet.write(filename, df)

print("Full DataFrame Saved")

for i in tqdm(range(1, len(dates_list)-1)):
    prev_date = dates_list[i - 1]
    date = dates_list[i]
    filename = "day_data/data"+str(date)+".parquet"
    temp_df = df.loc[(df["created_at"] > prev_date) & (df["created_at"] < date)]
    fastparquet.write(filename, temp_df)

print("--End--")
