# Cleaning Data

## Order to run Files

`bash data_prep.sh` 

OR

IF DATA IS IN GCS

`bash get_data.sh`

OR

1. `python3 clean_iex_data.py`
1. `python3 clean_twitter_data.py`
1. `python3 clean_full_data.py`


#### Notes for Possible Dockerfile

```
sudo apt update;
sudo apt install python3-pip;
sudo pip3 install -U pandas matplotlib seaborn scikit-learn tweepy iexfinance pandas-gbq nltk google-cloud-storage fastparquet tqdm pyspark;
sudo pip3 install --upgrade google-cloud-storage colorama;
```
