#!/bin/bash
gsutil cp gs://jminsk_thesis/iex_data/iex_clean.parquet iex_clean.parquet

gsutil -m cp -r gs://jminsk_thesis/twitter_data.parquet twitter_data.parquet
