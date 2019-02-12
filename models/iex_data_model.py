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

df = df.drop(["label"], axis=1)
df = df.rename(index=str, columns={"average" : "target"})
df["day"] = df["date"].dt.dayofyear
df["hour"] = df["date"].dt.hour
df = df.set_index("date")

logging.info("Spliting List")

train_split = 0.8
num_train = int(train_split * len(df.target))

train = df[0:num_train]
test = df[num_train:]

y_scaler = MinMaxScaler()
train = y_scaler.fit_transform(train[['target']])
test = y_scaler.transform(test[['target']])

y_train = train.target.shift(-5).values[:-5]
x_train = train.drop(["target"], axis=1).values[0:-5]
y_test = test.target.shift(-5).values[:-5]
x_test = test.drop(["target"], axis=1).values[0:-5]

logging.info("Scaling Data")

x_scaler = MinMaxScaler()
x_train = x_scaler.fit_transform(x_train)
x_test = x_scaler.transform(x_test)

num_x_signals = x_train.shape[1]

logging.info(x_train.shape)
logging.info(x_test.shape)

logging.info("Starting Batch Gen")
# https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/23_Time-Series-Prediction.ipynb

def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train[idx:idx+sequence_length]
            y_batch[i] = y_train[idx:idx+sequence_length]
        
        yield (x_batch, y_batch)

batch_size = 100
sequence_length = 2 * 1440

generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)

x_batch, y_batch = next(generator)

print(x_batch.shape)
print(y_batch.shape)

validation_data = (np.expand_dims(x_test, axis=0),
                   np.expand_dims(y_test, axis=0))

logging.info("Start Training Model")

model = Sequential()

model.add(LSTM(units=512,
              return_sequences=True,
              input_shape=(None, num_x_signals,)))

model.add(Dense(1, activation='sigmoid'))

warmup_steps = 50

def loss_mse_warmup(y_true, y_pred):
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.
    
    y_true is the desired output.
    y_pred is the model's output.
    """

    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, warmup_steps:]
    y_pred_slice = y_pred[:, warmup_steps:]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculate the MSE loss for each value in these tensors.
    # This outputs a 3-rank tensor of the same shape.
    loss = tf.losses.mean_squared_error(labels=y_true_slice,
                                        predictions=y_pred_slice)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire tensor, we reduce it to a
    # single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean

optimizer = RMSprop(lr=1e-3)

model.compile(loss=loss_mse_warmup, optimizer=optimizer)

logging.info(model.summary())

path_checkpoint = 'iex_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)


callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)


callback_tensorboard = TensorBoard(log_dir='./iex_logs/',
                                   histogram_freq=0,
                                   write_graph=False)

callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)

callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]

model.fit_generator(generator=generator,
                    epochs=20,
                    steps_per_epoch=100,
                    validation_data=validation_data,
                    callbacks=callbacks)


try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

result = model.evaluate(x=np.expand_dims(x_test, axis=0),
                        y=np.expand_dims(y_test, axis=0))

log_text = "loss (test-set):", result
logging.info(log_text)

def plot_comparison(start_idx, length=100, train=True):
    """
    Plot the predicted and true output-signals.
    
    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """
    
    if train:
        # Use training-data.
        x = x_train
        y_true = y_train
        path = "train"
    else:
        # Use test-data.
        x = x_test
        y_true = y_test
        path = "test"
    
    # End-index for the sequences.
    end_idx = start_idx + length
    
    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]
    
    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)
    
    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])
    
    # For each output-signal.
    # Get the output-signal predicted by the model.
    signal_pred = y_pred_rescaled[:]
        
    # Get the true output-signal from the data-set.
    signal_true = y_true[:]

    # Make the plotting-canvas bigger.
    plt.figure(figsize=(15,5))
        
    # Plot and compare the two signals.
    plt.plot(signal_true, label='true')
    plt.plot(signal_pred, label='pred')
        
    # Plot grey box for warmup-period.
    p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)
        
    # Plot labels etc.
    plt.ylabel("Price")
    plt.legend()
    plt.savefig(path+".png")

plot_comparison(start_idx=50, length=1000, train=True)
plot_comparison(start_idx=50, length=1000, train=False)
