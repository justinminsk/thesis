import pandas as pd
import numpy as np
from tqdm import tqdm
import fastparquet
from google.cloud import bigquery
from google.cloud import storage


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))

storage_client = storage.Client()
bucket = storage_client.get_bucket("jminsk_thesis")

blob_list = bucket.list_blobs()

final_df = pd.DataFrame()

for blob in tqdm(blob_list):
    if blob.name == "tweeterdata/":
        continue
    print(blob.name)
    download_blob("jminsk_thesis", blob.name, "./temp.parquet")
    df = pd.read_parquet('temp.parquet', engine='fastparquet')

    final_df = pd.concat([final_df, df]).fillna(0).to_sparse(fill_value=0)

final_df.to_gbq(project_id="jminsk-thesis", destination_table="twitter.clean_twitter_data", if_exists='replace')
    