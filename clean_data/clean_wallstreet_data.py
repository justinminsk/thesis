import fastparquet
import os
import glob
import re
from scipy import stats
from tqdm import tqdm
import pandas as pd
import numpy as np
from dateutil.parser import parse
from sklearn.feature_extraction.text import TfidfVectorizer


print("--Start--")
print("clean_wallstreet_data")

df = pd.read_csv("wallstreet_data/WSJ.csv", encoding = "ISO-8859-1")

date_df = pd.read_parquet("iex_data/date_iex_data.parquet")

print("df:", df.shape)
print("Date_df:", date_df.shape)

print("Data Imported")

df.date = df.date.apply(parse).map(lambda x: x.replace(second=0, microsecond=0, tzinfo=None))

df = df.sort_values("date")

print("Resampled To Get Articles Text Per Minute")

# https://stackoverflow.com/questions/46656718/merge-2-dataframes-on-closest-past-date
df = pd.merge_asof(date_df.set_index('date_col', drop=False).sort_index(),
                   df.set_index('date', drop=False).sort_index(),
                   left_index=True, right_index=True, direction="backward")

df = df.reset_index(drop=True)

print("Got Articles Tied to Trading Times")

print("df:", df.shape)

df["time_since_col"] = df.date_col - df.date
df.time_since_col = df.time_since_col.dt.total_seconds() / 60 

print("Added Time Since Article Was Published")

df = df.drop(columns=["date"])

print("Dropped Date Col.")

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

def pre_processing(row):
    first_process = re.sub(combined_pat, '', row)
    second_process = re.sub(www_pat, '', first_process)
    third_process = second_process.lower()
    fourth_process = neg_pattern.sub(lambda x: negations_dic[x.group()], third_process)
    result = re.sub(r'[^A-Za-z ]','',fourth_process)
    return result.strip()

df.ws_content = df.ws_content.apply(pre_processing)

print("Preprocessed")
print(df.shape)

word_grams = TfidfVectorizer(analyzer = "word", ngram_range = (1, 5), stop_words="english")

word_vector = word_grams.fit_transform(df.ws_content)

word_df = pd.DataFrame()

for i, col in enumerate(word_grams.get_feature_names()):
    word_df[col] = pd.Series(word_vector[:, i].toarray().ravel())

df = pd.merge(df, word_df, left_index=True, right_index=True)

del word_df

print("TDIDF Completed")
print(df.shape)

print(" ")

print(df.head())

print(" ")

del date_df

filename = "wallstreet_data/wallstreet_data.parquet"
fastparquet.write(filename, df)

print("Full DataFrame Saved")

print("--End--")
