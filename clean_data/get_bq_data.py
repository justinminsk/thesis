import nltk
import re
import pandas as pd
import numpy as np
from google.cloud import bigquery
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer


# nltk.download('punkt')
# nltk.download('stopwords')


# Get Tweets from bigquery
client = bigquery.Client()

query = "SELECT created_at, id_str, text, truncated, user.verified, user.followers_count, user.favourites_count, entities.urls FROM `jminsk-thesis.twitter.tweets2` WHERE lang='en' LIMIT 100"

df = pd.io.gbq.read_gbq(query, project_id="jminsk-thesis", dialect="standard")

# process non word dataframe
df.urls = df.urls.str.len()
df = df.fillna(0)

# print(df.ix[6])

# print(df.text[6])

# start processing words
nltk_stopwords = stopwords.words("english")
tknzr = TweetTokenizer(preserve_case=False)
stemmer = PorterStemmer()

def preprocess(x):
    x = re.sub(r"http\S+", "", x)
    x = tknzr.tokenize(x)
    x = [w for w in x if w not in set(nltk_stopwords)]
    x = [stemmer.stem(w) for w in x]
    return x

df.text = df.text.apply(preprocess)

# print(df)

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

# one hot the str ids
df = pd.get_dummies(df, "str_id")

# replace true and false with 1 and 0
df.truncated = df.truncated.astype(int)
df.verified = df.verified.astype(int)

# TODO: combine price to tweet dataframe, maybe sort tweets by vaild trade times
# TODO: add bi adn tri grams use nltk ngrams
