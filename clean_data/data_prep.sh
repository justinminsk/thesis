#!/bin/bash
gsutil -m cp -r gs://jminsk_thesis/en_tweets .

python3 clean_iex_data.py
gsutil cp iex_clean.parquet gs://jminsk_thesis/iex_data/iex_clean.parquet
gsutil cp date_iex_data.parquet gs://jminsk_thesis/iex_data/date_iex_data.parquet 

python3 clean_twitter_data.py
gsutil -m cp -r day_data/ gs://jminsk_thesis/day_data
