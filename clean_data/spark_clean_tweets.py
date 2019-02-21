import pyspark as ps
import warnings
import re
import datetime
import time
from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.types import StringType, TimestampType
from pyspark.ml.feature import Tokenizer, NGram, CountVectorizer, IDF, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel


"""
gcloud dataproc clusters create twitter-spark \
    --image-version 1.3 \
    --master-machine-type n1-standard-8 \
    --worker-machine-type n1-standard-8 \
    --metadata 'MINICONDA_VARIANT=3' \
    --metadata 'MINICONDA_VERSION=latest' \
    --metadata 'CONDA_PACKAGES=scipy=1.0.0 tensorflow=1.12.0' \
    --metadata 'PIP_PACKAGES=pandas==0.23.0 scipy==1.1.0 fastparquet==0.2.1' \
    --initialization-actions \
    gs://dataproc-initialization-actions/conda/bootstrap-conda.sh,gs://dataproc-initialization-actions/conda/install-conda-env.sh
"""

inputdir = "gs://jminsk_thesis/"
outputdir= "gs://jminsk_thesis/"

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

input_cols = ["1_tfidf", "2_tfidf", "3_tfidf", "4_tfidf", "5_tfidf", "ss_tweet_count"]

def pre_processing(column):
    first_process = re.sub(combined_pat, '', column)
    second_process = re.sub(www_pat, '', first_process)
    third_process = second_process.lower()
    fourth_process = neg_pattern.sub(lambda x: negations_dic[x.group()], third_process)
    result = re.sub(r'[^A-Za-z ]','',fourth_process)
    return result.strip()

def build_pipeline():
    tokenizer = [Tokenizer(inputCol='text',outputCol='words')]
    ngrams = [NGram(n=i, inputCol='words', outputCol='{0}_grams'.format(i)) for i in range(1,6)]
    cv = [CountVectorizer(vocabSize=100000, inputCol='{0}_grams'.format(i), outputCol='{0}_tf'.format(i)) for i in range(1,6)]
    idf = [IDF(inputCol='{0}_tf'.format(i), outputCol='{0}_tfidf'.format(i), minDocFreq=5) for i in range(1,6)]
    tweetvect = [VectorAssembler(inputCols=["tweet_count"], outputCol="vec_tweet_count")]
    ss = [StandardScaler(inputCol="vec_tweet_count", outputCol="ss_tweet_count")]
    assembler = [VectorAssembler(inputCols=input_cols, outputCol='features')]
    pipeline = Pipeline(stages=tokenizer+ngrams+cv+idf+tweetvect+ss+assembler)
    return pipeline

if __name__=="__main__":
	# create a SparkContext while checking if there is already SparkContext created
	try:
	    sc = ps.SparkContext()
	    sc.setLogLevel("ERROR")
	    sqlContext = ps.sql.SQLContext(sc)
	    print('Created a SparkContext')
	except ValueError:
	    warnings.warn('SparkContext already exists in this scope')
	print('Retrieving Data from {}'.format(inputdir+"twitter_data.parquet"))
	df = sqlContext.read.parquet(inputdir+"twitter_data.parquet")
	reg_replaceUdf = f.udf(pre_processing, t.StringType())
	df = df.withColumn('tweet', reg_replaceUdf(df.text))
	print('Get Feature Vectors')
	pipeline = build_pipeline()
	pipelineFit = pipeline.fit(df)
	df = pipelineFit.transform(df)
	select_list = ["date_col", "features", "stock_price_col"]
	df = df.select([column for column in df.columns if column in select_list])
	print("Write to Parquet")
	df.write.parquet(outputdir+"processed_twitter_pyspark")
	sc.stop()
