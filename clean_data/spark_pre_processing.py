import sys
import pyspark as ps
import warnings
import re
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType, IntegerType, StringType, BooleanType
from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.types import StringType
from pyspark.ml.feature import Tokenizer, NGram, CountVectorizer, IDF, VectorAssembler, Binarizer, OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml import PipelineModel


inputdir = "gs://jminsk_thesis/day_data/"
outputfile = "gs://jminsk_thesis/result.csv"
modeldir = "gs://jminsk_thesis/model"

pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1,pat2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
input_cols = ["created_at", "id_str_oh", '1_tfidf', "2_tfidf", "3_tfidf", "4_tfidf", "5_tfidf", "bi_truncated", "bi_verified", "followers_count", "favourites_count"]

def pre_processing(column):
    first_process = re.sub(combined_pat, '', column)
    second_process = re.sub(www_pat, '', first_process)
    third_process = second_process.lower()
    fourth_process = neg_pattern.sub(lambda x: negations_dic[x.group()], third_process)
    result = re.sub(r'[^A-Za-z ]','',fourth_process)
    return result.strip()


if __name__=="__main__":
	# create a SparkContext while checking if there is already SparkContext created
    try:
        sc = ps.SparkContext()
        sc.setLogLevel("ERROR")
        sqlContext = ps.sql.SQLContext(sc)
        print('Created a SparkContext')
    except ValueError:
        warnings.warn('SparkContext already exists in this scope')
    train_set = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(inputdir+'data2018-12-12 00:00:00.csv')
    train_set.limit(10).toPandas().head(10)
    print('preprocessing data...')
    reg_replaceUdf = f.udf(pre_processing, t.StringType())
    train_set = train_set.withColumn('text', reg_replaceUdf(f.col('text')))
    train_set.limit(10).toPandas().head(10)
    sc.stop()
