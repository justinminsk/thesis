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
print("clean_twitter_data")

# https://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe
path = "./en_tweets"
allFiles = glob.glob(os.path.join(path,"*.csv"))

np_array_list = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    np_array_list.append(df.values)

comb_np_array = np.vstack(np_array_list)
df = pd.DataFrame(comb_np_array)

df.columns = ["created_at", "id_str", "text", "truncated", "verified", "followers_count", "favourites_count"]

df.drop(columns=["id_str", "truncated", "verified", "followers_count", "favourites_count"])

# For agg tweets 
df["tweet_count"] = 1

date_df = pd.read_parquet("iex_data/date_iex_data.parquet", engine="fastparquet")

print("df:", df.shape)
print("Date_df:", date_df.shape)

print("Data Imported")

df.created_at = df.created_at.apply(parse).map(lambda x: x.replace(second=0, microsecond=0, tzinfo=None))

print("Changed DateTime to Minute By Minute")

df = df.set_index("created_at")

df = df.resample("1Min").agg({"text" : " ".join, "tweet_count" : sum})

print("df:", df.shape)

df.loc[:,'date_col'] = df.index

df = df.reset_index(drop=True)

df = df.sort_values("date_col")

print("Resampled To Get Tweet Text Per Minute")

# https://stackoverflow.com/questions/46656718/merge-2-dataframes-on-closest-past-date
df = pd.merge_asof(df.set_index('date_col').sort_index(),
                   date_df.set_index('date_col', drop=False).sort_index(),
                   left_index=True, right_index=True, direction="forward")

df = df.reset_index(drop=True)

print("df:", df.shape)
print(df.head())

df = df.groupby("date_col").agg({"text" : " ".join, "tweet_count" : sum, "stock_price_col" : 'mean'})

print("df:", df.shape)

print("Get Tweets Tied to Trading Times")

print("Data Merged")

del date_df

filename = "twitter_data/twitter_data.parquet"
fastparquet.write(filename, df)

print("Full DataFrame Saved")

print("--End--")
