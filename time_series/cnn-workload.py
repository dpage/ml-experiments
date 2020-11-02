from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

DEBUG = True
keras = tf.keras


# Plot a simple graph
def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


# Create the dataset
def seq2seq_window_dataset(series, window_size, batch_size=32,
                           shuffle_buffer=1000):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)
  

# Use the model to perform a prediction
def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


# Load the data from activity.csv

# Create an array of timestamps, and an array of data
dates_in = []
series_in = []

csv = open("activity.csv", "r")
pm = False

for line in csv:
    values = line.strip().split(',')
    dates_in.append(np.datetime64(datetime.strptime(values[0], '%Y-%m-%d %H:%M:%S')))
    series_in.append(int(values[1]))

csv.close()

samples = len(dates_in)
dates = np.array(dates_in)
series = np.array(series_in)

# Split the data into training and validation data and plot both
train_samples = int(samples * 0.75)
valid_samples = samples - train_samples
dates_train = dates[:train_samples]
x_train = series[:train_samples]

if DEBUG:
    plt.figure(figsize=(10, 6))
    plot_series(dates_train, x_train, label="Training data")
    plt.show()

dates_valid = dates[train_samples:]
x_valid = series[train_samples:]

if DEBUG:
    plt.figure(figsize=(10, 6))
    plot_series(dates_valid, x_valid, label="Validation data")
    plt.show()


# Setup the keras session
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

# Create the training and validation data sets
window_size = 64
train_set = seq2seq_window_dataset(x_train, window_size, batch_size=128)
valid_set = seq2seq_window_dataset(x_valid, window_size, batch_size=128)

# Create a sequential model
model = keras.models.Sequential()

# We're using the WaveNet architecture, so...
# Input layer
model.add(keras.layers.InputLayer(input_shape=[None, 1]))

# Add multiple 1D convolutional layers with increasing dilation rates to
# allow each layer to detect patterns over longer time frequencies
for dilation_rate in (1, 2, 4, 8, 16, 32):
    model.add(
      keras.layers.Conv1D(filters=32,
                          kernel_size=2,
                          strides=1,
                          dilation_rate=dilation_rate,
                          padding="causal",
                          activation="relu")
    )

# Add one output layer, with 1 filter to give us one output per time step
model.add(keras.layers.Conv1D(filters=1, kernel_size=1))

# Setup the optimiser, with the learning rate cribbed from the tutorial for now
optimizer = keras.optimizers.Adam(lr=3e-4)

# Compile the model
model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
if DEBUG:
    print(model.summary())

# Save checkpoints when we get the best model
model_checkpoint = keras.callbacks.ModelCheckpoint(
    "checkpoint.h5", save_best_only=True)

# Use early stopping to prevent over fitting
epochs = 500
if DEBUG:
    epochs = 10

early_stopping = keras.callbacks.EarlyStopping(patience=50)
history = model.fit(train_set, epochs=epochs,
                    validation_data=valid_set,
                    callbacks=[early_stopping, model_checkpoint])


# Training is done, so load the best model from the last checkpoint
model = keras.models.load_model("checkpoint.h5")


cnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
cnn_forecast = cnn_forecast[train_samples - window_size:-1, -1, 0]

plt.figure(figsize=(10, 6))
plot_series(dates, np.concatenate([series[:train_samples], np.full(valid_samples, None, dtype=float)]), label="Training Data")
plot_series(dates, np.concatenate([np.full(train_samples, None, dtype=float), series[train_samples:]]), label="Validation Data")
plot_series(dates, np.concatenate([np.full(train_samples, None, dtype=float), cnn_forecast]), label="Forecast Data")
plt.show()


mae = keras.metrics.mean_absolute_error(x_valid, cnn_forecast).numpy()

print("MAE: {}".format(mae))