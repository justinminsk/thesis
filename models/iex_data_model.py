import logging
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, Embedding, LSTM
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

logging.basicConfig(filename='iex_model.text', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

logging.info("New Model")

logging.info("Getting Data")

df = pd.read_parquet("iex_clean.parquet")

# Add some data and clean 
df = df.drop(["label"], axis=1)
df = df.rename(index=str, columns={"average" : "target"})
df["day"] = df["date"].dt.dayofyear
df["hour"] = df["date"].dt.hour
df = df.set_index("date")

logging.info("Spliting List")

train_split = 0.8
num_train = int(train_split * len(df.target))

# Split data
train = df[0:num_train]
test = df[num_train:]

# Min Max Scale the y data
y_scaler = MinMaxScaler()
train[["target"]] = y_scaler.fit_transform(train[['target']])
test[["target"]] = y_scaler.transform(test[['target']])

# Split data shifted 5 steps
y_train = train.target.shift(-5).values[:-5]
x_train = train.drop(["target"], axis=1).values[0:-5]
y_test = test.target.shift(-5).values[:-5]
x_test = test.drop(["target"], axis=1).values[0:-5]

logging.info("Scaling Data")

# Min Max scale the train data
x_scaler = MinMaxScaler()
x_train = x_scaler.fit_transform(x_train)
x_test = x_scaler.transform(x_test)

logging.info(x_train.shape)
logging.info(x_test.shape)

# How many steps it should look back
look_back = 1

# reshape input to be [samples, time steps, features]
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, 19)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(x_train)
testPredict = model.predict(x_test)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
y_train = scaler.inverse_transform([y_train])
testPredict = scaler.inverse_transform(testPredict)
y_test = scaler.inverse_transform([y_test])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[0], trainPredict[:,0]))
logging.info('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[0], testPredict[:,0]))
logging.info('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.savefig("iex_pred.png")
