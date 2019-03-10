import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from tensorflow.keras.models import load_model
from scipy.sparse import load_npz


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
    return  x_test_scaled, y_test, y_scaler

def get_twitter_data():
    x_data = load_npz("twitter_data/x_twitter_data.npz")
    x_data = x_data.todense() # .tocsr()

    # Used to split later
    train_split = 0.8
    num_train = int(train_split * x_data.shape[0])

    # get targets
    shift_steps = 1

    print("Shape x_data:", x_data.shape)

    num_x_signals = x_data.shape[1]

    print("Spliting List")

    # Split data
    x_train = x_data[0:num_train]
    x_test = x_data[num_train:]

    return x_test

def get_wallstreet_data():
    x_data = load_npz("wallstreet_data/x_wallstreet_data.npz")
    x_data = x_data.todense() # .tocsr()

    # Used to split later
    train_split = 0.8
    num_train = int(train_split * x_data.shape[0])

    # get targets
    shift_steps = 1

    print("Shape x_data:", x_data.shape)

    num_x_signals = x_data.shape[1]

    print("Spliting List")

    # Split data
    x_train = x_data[0:num_train]
    x_test = x_data[num_train:]

    return x_test
    

if __name__ == "__main__":
    iex_x_test, y_test, y_scaler = get_iex_data()
    iex_model = load_model("models/iex_model/model.h5", custom_objects={"loss_mse_warmup": loss_mse_warmup})
    twitter_model = load_model("models/twitter_model/model.h5", custom_objects={"loss_mse_warmup": loss_mse_warmup})
    wallstreet_model = load_model("models/wallstreet_model/model.h5", custom_objects={"loss_mse_warmup": loss_mse_warmup})

    # plot data
    # get the true values
    y_true = y_test
    
    # End-index and start-index for the sequences.
    start_idx = 20
    length = 2000
    end_idx = start_idx + length
    
    # Select the sequences from the given start-index and
    # of the given length.
    y_true = y_true[start_idx:end_idx]
        
    # Get the true output-signal from the data-set.
    signal_true = y_true[:, 0]

    # Make the plotting-canvas bigger.
    plt.figure(figsize=(15,5))
        
    # Plot True Line
    plt.plot(signal_true, label='True Values')

    # Cycle Through Test Data for each model
    # Input-signals for the model.
    iex_x = iex_x_test
    iex_x = iex_x[start_idx:end_idx]
    iex_x = np.expand_dims(iex_x, axis=0)

    # Use the model to predict the output-signals.
    iex_y_pred = iex_model.predict(iex_x)
    
    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    iex_y_pred_rescaled = y_scaler.inverse_transform(iex_y_pred[0])
    
    # Get the output-signal predicted by the model.
    iex_signal_pred = iex_y_pred_rescaled[:, 0]

    plt.plot(iex_signal_pred, label='Iex Model')

    del iex_x, iex_y_pred, iex_y_pred_rescaled, iex_x_test

    # Get Twitter Data
    twitter_x = get_twitter_data()
    # Input-signals for the model.
    twitter_x = twitter_x[start_idx:end_idx]
    twitter_x = np.expand_dims(twitter_x, axis=0)

    # Use the model to predict the output-signals.
    twitter_y_pred = twitter_model.predict(twitter_x)
    
    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    zeros = np.zeros(2000)
    wallstreet_y_pred = np.hstack((zeros, twitter_y_pred))
    y_scaler = joblib.load("twitter_data/twitter_scaler.pkl")
    twitter_y_pred_rescaled = y_scaler.inverse_transform(twitter_y_pred)
    
    # Get the output-signal predicted by the model.
    np.delete(twitter_y_pred_rescaled, 0, axis=2)
    twitter_signal_pred = twitter_y_pred_rescaled[:, 0]

    plt.plot(twitter_signal_pred, label='Twitter Model')

    del twitter_x, twitter_y_pred, twitter_y_pred_rescaled

    # Get WAllstreet Data
    wallstreet_x = get_wallstreet_data()
    # Input-signals for the model.
    wallstreet_x = wallstreet_x[start_idx:end_idx]
    wallstreet_x = np.expand_dims(wallstreet_x, axis=0)

    # Use the model to predict the output-signals.
    wallstreet_y_pred = wallstreet_model.predict(wallstreet_x)
    
    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    zeros = np.zeros(2000)
    wallstreet_y_pred = np.hstack((zeros, wallstreet_y_pred))
    y_scaler = joblib.load("wallstreet_data/wallstree_scaler.pkl")
    wallstreet_y_pred_rescaled = y_scaler.inverse_transform(wallstreet_y_pred)
    
    # Get the output-signal predicted by the model.
    np.delete(wallstreet_y_pred_rescaled, 0, axis=2)
    wallstreet_signal_pred = wallstreet_y_pred_rescaled[:, 0]

    plt.plot(wallstreet_signal_pred, label='Wallstreet Model')

    del wallstreet_x, wallstreet_y_pred, wallstreet_y_pred_rescaled
        
    # Plot grey box for warmup-period.
    p = plt.axvspan(0, 20, facecolor='black', alpha=0.15)
        
    # Plot labels etc.
    plt.ylabel("Price")
    plt.xlabel("Minutes")
    plt.legend()
    save_path = "graph.png"
    plt.savefig(save_path)
