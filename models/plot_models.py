import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from tensorflow.keras.models import load_model


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

def get_iex_data():
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
    return x_train_scaled, y_train, x_test_scaled, y_test, y_scaler

def plot_comparison(start_idx, length=100, train=True, model_type=""):
    """
    Plot the predicted and true output-signals.
    
    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """
    
    if train:
        # Use training-data.
        x = x_train_scaled
        y_true = y_train
    else:
        # Use test-data.
        x = x_test_scaled
        y_true = y_test
    
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
    
    # Get the output-signal predicted by the model.
    signal_pred = y_pred_rescaled[:, 0]
        
    # Get the true output-signal from the data-set.
    signal_true = y_true[:, 0]

    # Make the plotting-canvas bigger.
    plt.figure(figsize=(15,5))
        
    # Plot and compare the two signals.
    plt.plot(signal_true, label='true')
    plt.plot(signal_pred, label='pred')
        
    # Plot grey box for warmup-period.
    p = plt.axvspan(0, 20, facecolor='black', alpha=0.15)
        
    # Plot labels etc.
    plt.ylabel("Price")
    plt.legend()
    plt.savefig(model_type,"/graph.png")


if __name__ == "__main__":
    x_train_scaled, y_train, x_test_scaled, y_test, y_scaler = get_iex_data()
    model = load_model("iex_model/model.h5", custom_objects={"loss_mse_warmup": loss_mse_warmup})
    plot_comparison(start_idx=20, length=2000, train=False, model_type="iex_model")
