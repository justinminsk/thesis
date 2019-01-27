# Cleaning Data

## Order to run Files

1. clean_iex_data.py
1. bq_to_user_list.py
1. get_bq_data.py

## clean_iex_data.py

Will make a pkl with cleaned price data and a pkl with date and time as one column and average price in a second column

## bq_to_user_list.py

Gets a list with the count of how many times a user tweeted about Amazon

## get_bq_data

MOVING TO PYSPARK 

### notes on get_bq_data

Need to do Dimensionality Reduction (PCA, word2vec, count) prob per a time period (45 mins?) 

Need to create table with count and str_id to append on later

Will not run on cloud shell

Used a n1-standard-2 (2 vCPUs, 15 GB memory) with  Allow full access to all Cloud APIs.

#### Run After VM is Made

sudo apt update

sudo apt install python3-pip

sudo pip3 install -U pandas matplotlib seaborn scikit-learn tweepy iexfinance pandas-gbq nltk google-cloud-storage fastparquet tqdm

sudo pip3 install --upgrade google-cloud-storage
