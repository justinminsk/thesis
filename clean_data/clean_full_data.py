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
"""
schema = StructType([
    StructField("A", IntegerType()),
    StructField("B", DoubleType()),
    StructField("C", StringType())
])
"""
# "created_at", "id_str_oh", "bi_truncated", "bi_verified", "followers_count", "favourites_count"
input_cols = ['1_tfidf', "2_tfidf", "3_tfidf", "4_tfidf", "5_tfidf"]

def pre_processing(column):
    first_process = re.sub(combined_pat, '', column)
    second_process = re.sub(www_pat, '', first_process)
    third_process = second_process.lower()
    fourth_process = neg_pattern.sub(lambda x: negations_dic[x.group()], third_process)
    result = re.sub(r'[^A-Za-z ]','',fourth_process)
    return result.strip()

def build_pipeline():
	tokenizer = [Tokenizer(inputCol='text',outputCol='words')]
	ngrams = [NGram(n=i, inputCol='words', outputCol='{0}_grams'.format(i)) for i in range(1,5)]
	cv = [CountVectorizer(vocabSize=100000, inputCol='{0}_grams'.format(i), outputCol='{0}_tf'.format(i)) for i in range(1,5)]
	idf = [IDF(inputCol='{0}_tf'.format(i), outputCol='{0}_tfidf'.format(i), minDocFreq=5) for i in range(1,5)]
	# binarizer1 = [Binarizer(threshold=1.0, inputCol="truncated", outputCol="bi_truncated")]
	# binarizer2 = [Binarizer(threshold=1.0, inputCol="verified", outputCol="bi_verified")]
	# stringind = [StringIndexer(inputCol="id_str", outputCol="id_str_idx")]
	# onehot = [OneHotEncoder(inputCol="id_str_idx", outputCol="id_str_oh")]
	assembler = [VectorAssembler(inputCols=input_cols, outputCol='features')]
	dt = [DecisionTreeRegressor(maxDepth=25, predictionCol="stock_price_col")]
	# binarizer1+binarizer2+stringind+onehot
	pipeline = Pipeline(stages=tokenizer+ngrams+cv+idf+assembler+dt)
	return pipeline

def main(sqlc,input_dir,loaded_model=None):
	print('retrieving data from {}'.format(input_dir))
	# TODO: Figure out how to train with a few days then test on a few
	# might need to replace csv with com.databricks.spark.csv
	if not loaded_model:
		train_set = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(input_dir+'data2018-12-12 00:00:00.csv')
	test_set = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(input_dir+'data2018-12-13 00:00:00.csv')
	print('preprocessing data...')
	reg_replaceUdf = f.udf(pre_processing, StringType())
	if not loaded_model:
		train_set = train_set.withColumn('text', reg_replaceUdf())
	test_set = test_set.withColumn('text', reg_replaceUdf())
	if not loaded_model:
		pipeline = build_pipeline()
		print('training...')
		model = pipeline.fit(train_set)
		model.show(10)
	else:
		model = loaded_model
	print('making predictions on test data...')
	predictions = model.transform(test_set)
	return model, predictions


if __name__=="__main__":
	# create a SparkContext while checking if there is already SparkContext created
	try:
	    sc = ps.SparkContext()
	    sc.setLogLevel("ERROR")
	    sqlContext = ps.sql.SQLContext(sc)
	    print('Created a SparkContext')
	except ValueError:
	    warnings.warn('SparkContext already exists in this scope')
	# build pipeline, fit the model and retrieve the outputs by running main() function
	pipelineFit, predictions = main(sqlContext,inputdir)
	print('predictions finished!')
	print('saving predictions to {}'.format(outputfile))
	predictions.select(predictions['stock_price_col'],predictions['text'],predictions['prediction']).coalesce(1).write.mode("overwrite").format("com.databricks.spark.csv").option("header", "true").csv(outputfile)
	# save the trained model to destination specified by the third command line argument
	print('saving model to {}'.format(modeldir))
	pipelineFit.save(modeldir)
	sc.stop()
