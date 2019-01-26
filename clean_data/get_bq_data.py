import nltk
import re
import fastparquet
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
def pir_fast(df):
    v = df.text.values
    l = [len(x) for x in v.tolist()]
    f, u = pd.factorize(np.concatenate(v))
    n, m = len(v), u.size
    i = np.arange(n).repeat(l)

    dummies = pd.SparseDataFrame(
        np.bincount(i * m + f, minlength=n * m).reshape(n, m),
        df.index, u
    )

    return df.drop('text', 1).join(dummies)


# preprocess text data
def preprocess(x):
    x = re.sub(r"http\S+", "", x)
    x = tknzr.tokenize(x)
    x = [w for w in x if w not in set(nltk_stopwords)]
    x = [stemmer.stem(w) for w in x]
    bigrams = ngrams(x, 2)
    bigrams = ["_".join(list(gram)) for gram in bigrams]
    trigrams = ngrams(x, 3)
    trigrams = ["_".join(list(gram)) for gram in trigrams]
    x = x + bigrams + trigrams
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

date_df = pd.read_pickle("./date_iex_data.pkl")
print("Price Data is Loaded")

print("Going Minute by Minute")
for i in range(0, len(dates_list) - 1):
    print("Start " , dates_list[i] , " to " , dates_list[i+1])
    
    query = "SELECT created_at, id_str, text, truncated, user.verified, user.followers_count, user.favourites_count, entities.urls" \
            " FROM `jminsk-thesis.twitter.tweets2`" \
            " WHERE lang='en' AND created_at BETWEEN '"+str(dates_list[i])+"' AND '"+str(dates_list[i+1])+"' "

    df = pd.io.gbq.read_gbq(query, project_id="jminsk-thesis", dialect="standard")

    print("BigQuery Data is Loaded")

    # get rid of date/times that are not part of the minute by minute price data
    if len(df.index) > 0: 
        df = df.rename(index=str, columns={"created_at": "date_col"})
        df.date_col = df.date_col.map(lambda x: x.replace(second=0, microsecond=0))
        df = pd.merge(df, date_df, on="date_col", how="left")
    else:
        continue 

    print("Data is Merged")

    # process urls to make space
    df.urls = df.urls.str.len().to_sparse(fill_value=0)
    df = df.fillna(0)

    # start processing words
    nltk_stopwords = stopwords.words("english")
    tknzr = TweetTokenizer(preserve_case=False)
    stemmer = PorterStemmer()

    df.text = df.text.astype(str)

    df.text = df.text.apply(preprocess)

    print("Tweets are PreProcessed")

    # one hot the text
    df = pir_fast(df)

    print("One Hot Done for Tweets")

    # replace true and false with 1 and 0
    df.truncated = df.truncated.astype(int).to_sparse(fill_value=0)
    df.verified = df.verified.astype(int).to_sparse(fill_value=0)

    # one hot the str ids
    df = pd.get_dummies(df, "str_id", sparse=True)

    print("IDs are One Hot")

    # write to a new gbq
    print("Writing to Bucket")

    fastparquet.write("temp.parquet", df)

    upload_blob("jminsk_thesis", "./temp.parquet", "tweeterdata/data"+str(dates_list[i])+"to"+str(dates_list[i+1])+".parquet")

    # df = pd.read_pickle("temp.pkl") 

    print(df.head())
    print(df.shape)
    print(df.stock_price_col.head())

print("--End--")
