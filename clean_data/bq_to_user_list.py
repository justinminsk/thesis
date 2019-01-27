import fastparquet
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.cloud import storage


print("--Start--")
print("bq_to_user_list")
# Get CLients
client = bigquery.Client()

query = "SELECT id_str" \
        " FROM `jminsk-thesis.twitter.tweets2`" \
        " WHERE lang='en'"
    
df = pd.io.gbq.read_gbq(query, project_id="jminsk-thesis", dialect="standard")

print("BigQuery Data is Loaded")

print(df.shape)

df["id_count"] = 1

df = df.groupby(['id_str']).count()

df = df.reset_index()

print(df.head())
print(df.shape)

fastparquet.write("user_list.parquet", df)

print("--End--")
    