import fastparquet
# import gc
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.cloud import storage
# from dateutil.parser import *


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

df["count"] = 1

print(df.head())

df = df.set_index(["id_str"]).count(level="id_str")

print(df.head())

df = df.reset_index()

print(df.head())
    