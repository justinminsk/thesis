# Cleaning Data

## clean_iex_data.py

Will make a pkl with cleaned price data and a pkl with date and time as one column and average price in a second column

## get_bq_data

Will pull the data from bigquery and process the data by creating one hots and adding the price data to the column then saves in storage bucket

### notes on get_bq_data

Will not run on cloud shell

Used a n1-standard-8 (8 vCPUs, 30 GB memory) with  Allow full access to all Cloud APIs.

#### Run After VM is Made

sudo apt update

sudo apt install python3-pip

sudo pip3 install -U pandas matplotlib seaborn scikit-learn tweepy iexfinance pandas-gbq nltk google-cloud-storage fastparquet

sudo pip3 install --upgrade google-cloud-storage
