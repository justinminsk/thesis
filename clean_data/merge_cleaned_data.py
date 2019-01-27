import pandas as pd
import numpy as np
from tqdm import tqdm
import fastparquet
from google.cloud import bigquery
from google.cloud import storage


def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):
    """Lists all the blobs in the bucket that begin with the prefix.
    This can be used to list all blobs in a "folder", e.g. "public/".
    The delimiter argument can be used to restrict the results to only the
    "files" in the given "folder". Without the delimiter, the entire tree under
    the prefix is returned. For example, given these blobs:
        /a/1.txt
        /a/b/2.txt
    If you just specify prefix = '/a', you'll get back:
        /a/1.txt
        /a/b/2.txt
    However, if you specify prefix='/a' and delimiter='/', you'll get back:
        /a/1.txt
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix, delimiter=delimiter)

    return blobs

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

blob_list = list_blobs_with_prefix("jminsk-thesis", "/tweeterdata")

final_df = pd.DataFrame()

for blob in tqdm(blob_list):
    upload_blob("jminsk-thesis/tweeterdata", blob, "temp.parquet")
    pq = fastparquet.ParquetFile("temp.parquet")
    df = pq.to_pandas()

    final_df = pd.concat([final_df, df]).fillna(0).to_sparse(fill_value=0)

final_df.to_gbq(project_id="jminsk-thesis", destination_table="twitter.clean_twitter_data", if_exists='replace')
    