import fastparquet
import os
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
from google.cloud import storage
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

user_df = pd.read_parquet("user_list.parquet", engine="fastparquet")
date_df = pd.read_parquet("date_iex_data.parquet", engine="fastparquet")

df.loc[:,'date_col'] = df.created_at
df.date_col = df.date_col.map(lambda x: x.replace(second=0, microsecond=0))
df = pd.merge(df, date_df, on="date_col", how="left")
df = df.drop("date_col", 1)
df = pd.merge(df, user_df, on="id_str", how="left")
df = df.drop("id_str", 1)

del user_df, date_df

for i in tqdm(range(1, len(dates_list)-1)):
    prev_date = dates_list[i - 1]
    date = dates_list[i]
    temp_df = df.loc[(df["created_at"] > prev_date) & (df["created_at"] < date)]
    temp_df.to_csv("./day_data/data",date,".csv")

print("--End--")
