import nltk
import re
import fastparquet
import gc
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.cloud import storage
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.util import ngrams
from dateutil.parser import *


print("--Start--")

# need for preprocessing of text data
nltk.download('punkt')
nltk.download('stopwords')


# one hot the data
# https://stackoverflow.com/questions/45312377/how-to-one-hot-encode-from-a-pandas-column-containing-a-list
def pir_fast(df, column):
    v = df[column].values
    l = [len(x) for x in v.tolist()]
    f, u = pd.factorize(np.concatenate(v))
    n, m = len(v), u.size
    i = np.arange(n).repeat(l)

    dummies = pd.DataFrame(
        np.bincount(i * m + f, minlength=n * m).reshape(n, m),
        df.index, u
    )

    print(dummies.head())

    # Only get words and ids that have less then 4 tweets containing them
    dummies = dummies.drop([col for col, val in dummies.sum().iteritems() if val < 4], axis=1, inplace=True)

    return df.drop(column, 1).join(dummies)


# preprocess text data
def preprocess(x):
    x = re.sub(r"http\S+", "", x)
    x = tknzr.tokenize(x)
    x = [w for w in x if w not in set(nltk_stopwords)]
    x = [stemmer.stem(w) for w in x]
    bigrams = ngrams(x, 2)
    bigrams = ["_".join(list(gram)) for gram in bigrams]
    x = x + bigrams
    return x


# func to save to bucket
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))


# Get CLients
client = bigquery.Client()

# get minute by minute list 
start = parse("2018-12-12 09:29:00")
end = parse("2019-01-23 16:00:00")

dates_list = pd.date_range(start=start, end=end, freq="45min")

print("Going Minute by Minute")
for i in range(0, len(dates_list) - 1):
    print("Start " , dates_list[i] , " to " , dates_list[i+1])

    date_df = pd.read_pickle("./date_iex_data.pkl")
    print("Price Data is Loaded")
    
    query = "SELECT created_at, id_str, text, truncated, user.verified, user.followers_count, user.favourites_count, entities.urls" \
            " FROM `jminsk-thesis.twitter.tweets2`" \
            " WHERE lang='en' AND created_at BETWEEN '"+str(dates_list[i])+"' AND '"+str(dates_list[i+1])+"' "

    df = pd.io.gbq.read_gbq(query, project_id="jminsk-thesis", dialect="standard")

    print("BigQuery Data is Loaded")

    # get rid of date/times that are not part of the minute by minute price data
    if len(df.index) > 0: 
        df.loc[:,'date_col'] = df.created_at
        df.date_col = df.date_col.map(lambda x: x.replace(second=0, microsecond=0))
        df = pd.merge(df, date_df, on="date_col", how="left")
        df = df.drop("date_col", 1)
    else:
        continue 

    print("Data is Merged")

    # process urls to make space
    df.urls = df.urls.str.len()
    df = df.fillna(0)

    # start processing words
    nltk_stopwords = stopwords.words("english")
    tknzr = TweetTokenizer(preserve_case=False, reduce_len=True)
    stemmer = PorterStemmer()

    df.text = df.text.astype(str)

    df.text = df.text.apply(preprocess)

    print("Tweets are PreProcessed")

    # one hot the text
    df = pir_fast(df, "text")

    print("One Hot Done for Tweets")

    # replace true and false with 1 and 0
    df.truncated = df.truncated.astype(int)
    df.verified = df.verified.astype(int)

    # write to a new gbq
    print("Writing to Bucket")

    fastparquet.write("temp.parquet", df)

    upload_blob("jminsk_thesis", "./temp.parquet", "tweeterdata/data"+str(dates_list[i])+"to"+str(dates_list[i+1])+".parquet")

    # TODO: make a new table that will create a running count of each user this info will be added back later

    print(df.head())
    print("Shape: " + df.shape)
    gc.collect()

print("--End--")
