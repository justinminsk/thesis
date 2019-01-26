import pandas as pd
import numpy as np
from tqdm import tqdm
import fastparquet
from google.cloud import bigquery
from google.cloud import storage


def list_blobs(bucket_name):
    """Lists all the blobs in the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    blobs = bucket.list_blobs()
    return blobs

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))

blob_list = list_blobs("jminsk-thesis/tweeterdata")

final_df = pd.DataFrame()

for blob in tqdm(blob_list):
    upload_blob("jminsk-thesis/tweeterdata", blob, "temp.parquet")
    pq = fastparquet.ParquetFile("temp.parquet")
    df = pq.to_pandas()

    final_df = pd.concat([final_df, df]).fillna(0).to_sparse(fill_value=0)

final_df.to_gbq(project_id="jminsk-thesis", destination_table="twitter.clean_twitter_data", if_exists='replace')
    