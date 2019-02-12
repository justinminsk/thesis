import pyspark as ps
import warnings


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