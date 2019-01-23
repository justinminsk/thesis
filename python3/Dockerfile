FROM gcr.io/cloudshell-images/cloudshell:latest

# Add your content here

RUN sudo apt-get update
RUN sudo apt-get install python3.6
RUN sudo pip3 install -U pandas matplotlib seaborn scikit-learn tweepy iexfinance
RUN sudo pip3 install tensorflow

# To trigger a rebuild of your Cloud Shell image:

# 1. Commit your changes locally: git commit -a
# 2. Push your changes upstream: git push origin master

# This triggers a rebuild of your image hosted at https://gcr.io/jminsk-thesis/python3
# You can find the Cloud Source Repository hosting this file at https://source.developers.google.com/p/jminsk-thesis/r/python3