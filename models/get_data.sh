#!/bin/bash
mkdir iex_data
mkdir twitter_data

gsutil cp gs://jminsk_thesis/iex_data .

gsutil -m cp -r gs://jminsk_thesis/twitter_data .