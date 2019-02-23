#!/bin/bash
mkdir iex_data
mkdir twitter_data
mkdir wallstreet_data

gsutil cp -r gs://jminsk_thesis/iex_data .
gsutil -m cp -r gs://jminsk_thesis/wallstreet_data .
gsutil -m cp -r gs://jminsk_thesis/twitter_data .
