import pandas as pd
from google.cloud import bigquery

client = bigquery.Client()

query = "SELECT created_at, id_str, text, truncated, user.verified, user.followers_count, user.favourites_count, entities.hashtags, lang FROM `jminsk-thesis.twitter.tweets2` LIMIT 10"

df = pd.io.gbq.read_gbq(query, project_id="jminsk-thesis", dialect="standard")
    
print(df.head())
