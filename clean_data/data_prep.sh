#!/bin/bash
mkdir iex_data
mkdir twitter_data
gsutil -m cp -r gs://jminsk_thesis/en_tweets .

python3 clean_iex_data.py
gsutil cp -r iex_data/ gs://jminsk_thesis/iex_data/

python3 clean_twitter_data.py
gsutil -m cp -r twitter_data/twitter_data.parquet gs://jminsk_thesis/twitter_data/twitter_data.parquet

python3 hash_tweets.py
gsutil -m cp twitter_data/twitter_scaler.pkl gs://jminsk_thesis/twitter_data/twitter_scaler.pkl
gsutil -m cp twitter_data/y_twitter_data.npy gs://jminsk_thesis/twitter_data/y_twitter_data.npy
gsutil -m cp twitter_data/x_twitter_data.npy gs://jminsk_thesis/twitter_data/x_twitter_data.npy
