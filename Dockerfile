FROM python:3

RUN pip install --upgrade pip
RUN pip install tweepy iexfinance pandas matplotlib seaborn tensorflow scikit-learn tensorboard
RUN pip install --upgrade google-api-python-client
RUN pip install python-dateutil
RUN pip install --upgrade oauth2client
