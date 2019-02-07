#!/bin/bash
gsutil -m cp -r gs://jminsk_thesis/en_tweets .

gsutil cp gs://jminsk_thesis/iex_data/iex_clean.parquet iex_clean.parquet
gsutil cp gs://jminsk_thesis/iex_data/date_iex_data.parquet date_iex_data.parquet

gsutil -m cp -r gs://jminsk_thesis/day_data .
