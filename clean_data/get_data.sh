#!/bin/bash
gsutil -m cp -r gs://jminsk_thesis/en_tweets .

gsutil cp gs://jminsk_thesis/iex_data/iex_clean.parquet iex_data/iex_clean.parquet
gsutil cp gs://jminsk_thesis/iex_data/date_iex_data.parquet iex_data/date_iex_data.parquet

gsutil cp -r gs://jminsk_thesis/wallstreet_data .

gsutil -m cp -r gs://jminsk_thesis/twitter_data/twitter_data.parquet twitter_data/twitter_data.parquet
gsutil -m -r cp gs://jminsk_thesis/twitter_data/ .
