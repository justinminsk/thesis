import datetime
import time
import warnings
import fastparquet
import pandas as pd
import pyspark as ps
from datetime import date
from pyspark.sql import functions as f
from pyspark.sql.types import StringType, TimestampType


# gcloud dataproc clusters create twitter-spark --image-version 1.3 --master-machine-type n1-standard-8 --worker-machine-type n1-standard-8 --metadata 'PIP_PACKAGES=pandas==0.23.0 scipy==1.1.0 fastparquet' --initialization-actions gs://dataproc-initialization-actions/python/pip-install.sh 

inputdir = "gs://jminsk_thesis/"
outputdir= "gs://jminsk_thesis/"


if __name__ == "__main__":
    try:
        sc = ps.SparkContext()
        sc.setLogLevel("ERROR")
        sqlContext = ps.sql.SQLContext(sc)
        print('Created a SparkContext')
    except ValueError:
        warnings.warn('SparkContext already exists in this scope')
    print('Retrieving Data from {}'.format(inputdir))
    df = sqlContext.read.parquet(inputdir+"processed_twitter_pyspark")
    print(df.show(5))
    # https://stackoverflow.com/questions/31407461/datetime-range-filter-in-pyspark-sql
    start = date(2018, 12, 11)
    end =  date(2019, 1, 24)
    dates_list = pd.date_range(start=start, end=end, freq="D")
    for i in range(1, len(dates_list)-1):
        prev_date = dates_list[i - 1]
        date = dates_list[i]
        print("Create Pandas DF for "+str(date))
        dates = (prev_date,  date)
        date_from, date_to = [f.to_date(f.lit(s)).cast(TimestampType()) for s in dates]
        temp_df = df.where((df.date_col > date_from) & (df.date_col < date_to))
        temp_df = temp_df.toPandas()
        fastparquet.write(outputdir+"processed_twitter_data"+str(date), temp_df)
    sc.stop()
