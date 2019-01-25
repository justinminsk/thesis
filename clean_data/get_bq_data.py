import nltk
import re
import pandas as pd
import numpy as np
from google.cloud import bigquery
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.util import ngrams
from datetime import datetime

print("--Start--")

nltk.download('punkt')
nltk.download('stopwords')


# Get Tweets from bigquery
client = bigquery.Client()

start = datetime(2018, 12, 12)

end = datetime(2019, 1, 22)

dates_list = pd.date_range(start, end).tolist()

query = "SELECT created_at, id_str, text, truncated, user.verified, user.followers_count, user.favourites_count, entities.urls FROM `jminsk-thesis.twitter.tweets2` WHERE lang='en'"

df = pd.io.gbq.read_gbq(query, project_id="jminsk-thesis", dialect="standard")

print("BigQuery Data is Loaded")

# get rid of date/times that are not part of the minute by minute price data
date_df = pd.read_pickle("./date_iex_data.pkl")

print("Price Data is Loaded")

df.created_at = df.created_at.map(lambda x: x.replace(second=0))
df = df.rename(index=str, columns={"created_at": "date"})
date_df.date = date_df.date - pd.Timedelta(minutes=1)

df = pd.merge(df, date_df, on="date", how="outer")

print("Data is Merged")

del date_df

print("Price Data is Deleated ")

# process urls to make space
df.urls = df.urls.str.len()
df = df.fillna(0)

# start processing words
nltk_stopwords = stopwords.words("english")
tknzr = TweetTokenizer(preserve_case=False)
stemmer = PorterStemmer()

df.text = df.text.astype(str)

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

df.text = df.text.apply(preprocess)

print("Tweets are PreProcessed")

df.to_parquet

# one hot the data
# https://stackoverflow.com/questions/45312377/how-to-one-hot-encode-from-a-pandas-column-containing-a-list
def pir_fast(df):
    v = df.text.values
    l = [len(x) for x in v.tolist()]
    f, u = pd.factorize(np.concatenate(v))
    n, m = len(v), u.size
    i = np.arange(n).repeat(l)

    dummies = pd.DataFrame(
        np.bincount(i * m + f, minlength=n * m).reshape(n, m),
        df.index, u
    )

    return df.drop('text', 1).join(dummies)

# one hot the text
df = pir_fast(df)

print("One Hot Done for Tweets")

# replace true and false with 1 and 0
df.truncated = df.truncated.astype(int)
df.verified = df.verified.astype(int)

# one hot the str ids
df = pd.get_dummies(df, "str_id")

print("IDs are One Hot")

# write to a new gbq
print("Writing to BigQuery")

pd.io.gbq.to_gbq(df, "jminsk-thesis.twitter.clean_twitter")

print("Saved to BigQuery")
