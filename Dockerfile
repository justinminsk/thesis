FROM tensorflow/tensorflow:latest-gpu-py3

RUN pip install --upgrade pip
RUN pip install tweepy iexfinance pandas matplotlib seaborn scikit-learn
RUN pip install --upgrade tensorflow
RUN pip install --upgrade google-api-python-client
RUN pip install python-dateutil
RUN pip install --upgrade oauth2client
