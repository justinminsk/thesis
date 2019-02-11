import datetime
import time
import warnings
import pyspark as ps
from pyspark.sql import functions as f
from pyspark.sql.types import StringType, TimestampType


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
    print("Create Train DF 1")
    dates = ("2018-01-01",  "2018-12-26")
    date_from, date_to = [f.to_date(f.lit(s)).cast(TimestampType()) for s in dates]
    temp_df = df.where((df.date_col > date_from) & (df.date_col < date_to))
    temp_df = temp_df.toPandas()
    temp_df.to_parquet(outputdir+"processed_twitter_train_data_1")
    print("Create Train DF 2")
    dates = ("2018-12-25",  "2019-01-01")
    date_from, date_to = [f.to_date(f.lit(s)).cast(TimestampType()) for s in dates]
    temp_df = df.where((df.date_col > date_from) & (df.date_col < date_to))
    temp_df = temp_df.toPandas()
    temp_df.to_parquet(outputdir+"processed_twitter_train_data_2")
    print("Create Test DF 1")
    dates = ("2018-12-31",  "2019-01-13")
    date_from, date_to = [f.to_date(f.lit(s)).cast(TimestampType()) for s in dates]
    temp_df = df.where((df.date_col > date_from) & (df.date_col < date_to))
    temp_df = temp_df.toPandas()
    temp_df.to_parquet(outputdir+"processed_twitter_test_data_1")
    print("Create Test DF 1")
    dates = ("2019-01-12",  "2019-01-23")
    date_from, date_to = [f.to_date(f.lit(s)).cast(TimestampType()) for s in dates]
    temp_df = df.where((df.date_col > date_from) & (df.date_col < date_to))
    temp_df = temp_df.toPandas()
    temp_df.to_parquet(outputdir+"processed_twitter_test_data_2")
    print("Create Val DF")
    dates = ("2019-01-22",  "2019-01-25")
    date_from, date_to = [f.to_date(f.lit(s)).cast(TimestampType()) for s in dates]
    temp_df = df.where((df.date_col > date_from) & (df.date_col < date_to))
    temp_df = temp_df.toPandas()
    temp_df.to_parquet(outputdir+"processed_twitter_val_data")
    sc.stop()
