import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

DEBUG = False
keras = tf.keras


# Plot a simple graph
def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


# Generate a trend para. This will affect every element in an numpy array
def trend(time, slope=0):
    return slope * time
  

# Generate a seasonal pattern
def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


# Create a series following the seasonal pattern
def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)
  

# Generate some white noise
def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level
  

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


# Generate a time series of 4 span_years + 1 day
time = np.arange(4 * 365 + 1)

slope = 0.05
baseline = 10
amplitude = 40

# Generate the test data, adding together the baseline, trend and seasonality
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

# Now add some random noise
noise_level = 5
noise = white_noise(time, noise_level, seed=42)
series += noise

# Display the test data
if DEBUG:
    plt.figure(figsize=(10, 6))
    plot_series(time, series, label="Test data")
    plt.show()

# Split the data into training and validation data and plot both
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]

if DEBUG:
    plt.figure(figsize=(10, 6))
    plot_series(time_train, x_train, label="Training data")
    plt.show()

time_valid = time[split_time:]
x_valid = series[split_time:]

if DEBUG:
    plt.figure(figsize=(10, 6))
    plot_series(time_valid, x_valid, label="Validation data")
    plt.show()


# Setup the keras session
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

# Create the training and validation data sets
window_size = 64
train_set = seq2seq_window_dataset(x_train, window_size,
                                   batch_size=128)
valid_set = seq2seq_window_dataset(x_valid, window_size,
                                   batch_size=128)

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
cnn_forecast = cnn_forecast[split_time - window_size:-1, -1, 0]

plt.figure(figsize=(10, 6))
plot_series(time, np.concatenate([series[:1000], np.full(461, None, dtype=float)]), label="Training Data")
plot_series(time, np.concatenate([np.full(1000, None, dtype=float), series[1000:]]), label="Validation Data")
plot_series(time, np.concatenate([np.full(1000, None, dtype=float), cnn_forecast]), label="Forecast Data")
plt.show()


mae = keras.metrics.mean_absolute_error(x_valid, cnn_forecast).numpy()

print("MAE: {}".format(mae))