import pandas as pd 
import numpy as np 
from sklearn.externals import joblib
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import MinMaxScaler


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

vectorizer = HashingVectorizer(stop_words="english", ngram_range=(1,5))

text_vector = vectorizer.fit_transform(df.text)

print("text vector:", text_vector.shape)

tfifd_vectorizer = TfidfTransformer()

tfifd_vector = tfifd_vectorizer.fit_transform(text_vector)

scaler = MinMaxScaler()

int_values = df[["tweet_count", "stock_price_col"]]

scaled_values = scaler.fit_transform(int_values)

joblib.dump(scaler, 'twitter_scaler.pkl') 

y_data = scaled_values[:,1]

np.save("y_twitter_data.pkl", y_data)

x_data = np.concatenate((tfifd_vector, scaled_values[:,0]), axis=1)

np.save("x_twitter_data.pkl", x_data)
