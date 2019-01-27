import fastparquet
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.cloud import storage


print("--Start--")
# Get CLients
client = bigquery.Client()

# get minute by minute list 
# start = parse("2018-12-18 19:59:00") # 2018-12-12 09:29:00
# end = parse("2019-01-23 16:00:00")

# dates_list = pd.date_range(start=start, end=end, freq="45min")

# for i in range(0, len(dates_list) - 1):
#    print("Start " , dates_list[i] , " to " , dates_list[i+1])

query = "SELECT id_str" \
        " FROM `jminsk-thesis.twitter.tweets2`" \
        " WHERE lang='en'"
    
df = pd.io.gbq.read_gbq(query, project_id="jminsk-thesis", dialect="standard")

print("BigQuery Data is Loaded")

print(df.shape)

df = df.id_str.value_counts()

print(df)
print(df.head())
print(df.shape)

fastparquet.write("user_list.parquet", df)
    