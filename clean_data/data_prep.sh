#!/bin/bash
gsutil -m cp -r gs://jminsk_thesis/en_tweets .

python3 clean_iex_data.py

python3 bq_to_user_list.py

gsutil cp iex_clean.parquet gs://jminsk_thesis/iex_data/iex_clean.parquet
gsutil cp date_iex_data.parquet gs://jminsk_thesis/iex_data/date_iex_data.parquet
gsutil cp user_list.parquet gs://jminsk_thesis/user_list/user_list.parquet

python3 split_data_by_day.py

gsutil -m cp -r ./day_data gs://jminsk_thesis/
