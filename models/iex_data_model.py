import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from tensorflow.python.keras.initializers import RandomUniform
from tensorflow.python.keras.models import Sequential, save_model
from tensorflow.python.keras.layers import Input, Dense, Dropout, Embedding, LSTM, GRU
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

print("New Model")

print("Getting Data")

df = pd.read_parquet("iex_data/iex_clean.parquet")

print('Cleaning Data')

# Add some data and clean 
df = df.drop(["label"], axis=1)
df = df.rename(index=str, columns={"average" : "target"})
df["day"] = df["date"].dt.dayofyear
df["hour"] = df["date"].dt.hour
df = df.set_index("date")

# Used to split later
train_split = 0.8
num_train = int(train_split * len(df.target))

# get targets
shift_steps = 1
df_targets = df.target

x_data = df.values
print("Shape x_data:", x_data.shape)

y_data = df_targets.values
y_data = y_data.reshape(y_data.shape[0], 1)
print("Shape y_data:", y_data.shape)

num_x_signals = x_data.shape[1]
num_y_signals = 1

print("Spliting List")

# Split data
x_train = x_data[0:num_train]
x_test = x_data[num_train:]

y_train = y_data[0:num_train]
y_test = y_data[num_train:]

print("Train Data Shape:", x_train.shape)
print("Test Data Shape:", x_test.shape)

print("Scaling Data")
x_scaler = MinMaxScaler(feature_range = (0, 1))
x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = x_scaler.transform(x_test)

y_scaler = MinMaxScaler(feature_range = (0, 1))
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# Save y scaler
joblib.dump(y_scaler, 'iex_model/iex_y_scaler.pkl') 

# y_train_scaled = y_train_scaled.reshape(y_train_scaled.shape[0],)
# y_test_scaled = y_test_scaled.reshape(y_test_scaled.shape[0],)

print("x Train:",x_train_scaled.shape)
print("y Trian", y_train_scaled.shape)

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
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx+sequence_length]
            y_batch[i] = y_train_scaled[idx:idx+sequence_length]
        
        yield (x_batch, y_batch)


batch_size = 100
sequence_length = 3200

generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)

x_batch, y_batch = next(generator)

print(x_batch.shape)
print(y_batch.shape)

validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))

print("Build Model")

warmup_steps = 20

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
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

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

# an LSTM to a GRU to a dense output
model = Sequential()
model.add(GRU(units=200, return_sequences=True, input_shape=(None, num_x_signals,)))
# model.add(Dropout(0.2))
model.add(GRU(100, return_sequences=True))
# model.add(Dropout(0.2))

init = RandomUniform(minval=-0.05, maxval=0.05)
model.add(Dense(num_y_signals, activation='linear', kernel_initializer=init)) #  activation='sigmoid'

optimizer = RMSprop(lr=1e-3)
model.compile(loss=loss_mse_warmup, optimizer=optimizer)

print(model.summary())


path_checkpoint = 'iex_model/iex_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)

callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=3, verbose=1)

callback_tensorboard = TensorBoard(log_dir='iex_model/iex_logs/',
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

print("Train Model")
model.fit_generator(generator=generator,
                    epochs=20,
                    steps_per_epoch=100,
                    validation_data=validation_data,
                    callbacks=callbacks)

result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))

save_model(model, "iex_model/model.h5")
print("loss (test-set):", result)                       
