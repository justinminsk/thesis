import re
import pandas as pd 
import numpy as np 
from sklearn.externals import joblib
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import MinMaxScaler


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

df = pd.read_parquet("twitter_data.parquet", engine="fastparquet")

print("df:", df.shape)

df.text = df.text.apply(pre_processing)

print("preprocessed")

vectorizer = HashingVectorizer(stop_words="english", ngram_range=(1,5))

print("text hashed")

text_vector = vectorizer.fit_transform(df.text)

print("text vector:", text_vector.shape)

tfifd_vectorizer = TfidfTransformer()

tfifd_vector = tfifd_vectorizer.fit_transform(text_vector)

scaler = MinMaxScaler()

int_values = df[["tweet_count", "stock_price_col"]]

scaled_values = scaler.fit_transform(int_values)

print("scaled values:", scaled_values.shape)

joblib.dump(scaler, 'twitter_scaler.pkl') 

y_data = scaled_values[:,1]

np.save("y_twitter_data.pkl", y_data)

scaled_count = scaled_values[:,0].reshape(len(scaled_values, 1))

x_data = np.concatenate((tfifd_vector, scaled_count), axis=1)

np.save("x_twitter_data.pkl", x_data)
