import fastparquet
import os
import glob
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import pandas as pd
import numpy as np
from dateutil.parser import *

print("--Start--")
print("spilt_data_by_day")

pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1,pat2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

def pre_processing(column):
    first_process = re.sub(combined_pat, '', column)
    second_process = re.sub(www_pat, '', first_process)
    third_process = second_process.lower()
    fourth_process = neg_pattern.sub(lambda x: negations_dic[x.group()], third_process)
    result = re.sub(r'[^A-Za-z ]','',fourth_process)
    return result.strip()

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
df.text = df.text.apply(pre_processing)
df.text = df.text.apply(nltk.word_tokenize)

print("Tweets Preprocessed")

# user_df = pd.read_parquet("user_list.parquet", engine="fastparquet")
date_df = pd.read_parquet("date_iex_data.parquet", engine="fastparquet")

print("Data Imported")

df.created_at = df.created_at.apply(parse)
df.created_at = df.created_at.map(lambda x: x.replace(tzinfo=None)).map(lambda x: x.replace(second=0, microsecond=0, tzinfo=None))
df.loc[:,'date_col'] = df.created_at

print("Changed DateTime to Minute By Minute")

df = df.set_index("created_at")

df = df.resample("1Min").sum()

tfid = TfidfVectorizer(max_features=150000,ngram_range=(1, 5))

df.text = tfid.fit(df.text)

print("Resampled To Get Tweet Text Per Minute")

df = df.merge(date_df, how="left", on="date_col")
df = df.drop("date_col", 1)

print("Data Merged")

del date_df

for i in tqdm(range(1, len(dates_list)-1)):
    prev_date = dates_list[i - 1]
    date = dates_list[i]
    filename = "day_data/data"+str(date)+".parquet"
    temp_df = df.loc[(df["created_at"] > prev_date) & (df["created_at"] < date)]
    fastparquet.write(filename, temp_df)

print("--End--")
