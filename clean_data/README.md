# Cleaning Data

## Order to run Files

`bash data_prep.sh` 

OR

IF DATA IS IN GCS

`bash get_data.sh`

OR

1. `python3 clean_iex_data.py`
1. `python3 bq_to_user_list.py`
1. `python3 split_data_by_day.py.py`

## clean_iex_data.py

Makes the data used for one of the models and a second set of data with time and price

## bq_to_user_list.py

Gets a list with the count of how many times a user tweeted about Amazon

## split_data_by_day.py

splits data by day and appends price and user count

## clean_full_data.py

Uses spark to get final cleaned set of data


#### Notes for Possible Dockerfile

```sudo apt update;
sudo apt install python3-pip;
sudo pip3 install -U pandas matplotlib seaborn scikit-learn tweepy iexfinance pandas-gbq nltk google-cloud-storage fastparquet tqdm glob pyspark;
sudo pip3 install --upgrade google-cloud-storage colorama;```
