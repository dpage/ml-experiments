CREATE OR REPLACE FUNCTION public.tf_analyse(
	data_source_sql text,
	output_name text,
	output_path text)
    RETURNS void
    LANGUAGE 'plpython3u'
    COST 100
    VOLATILE PARALLEL UNSAFE
AS $BODY$
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

# Pandas print options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Setup the plot layout
plot_columns = 5
plot_rows = ceil(len(columns) / plot_columns)

# Create the data sets
rows = plpy.execute(data_source_sql)

# Check we have enough rows
if len(rows) < 2:
    plpy.error('At least 2 data rows must be available for analysis. {} rows retrieved.'.format(len(rows)))

columns = list(rows[0].keys())

# Check we have enough columns
if len(columns) < 2:
    plpy.error('At least 2 data columns must be available for analysis. {} columns retrieved.'.format(len(columns)))

# Create the dataframe
data = pd.DataFrame.from_records(rows, columns = columns)

# High level info
plpy.notice('{} Analysis\n         {}=========\n'.format(output_name.capitalize(), '=' * len(output_name)))
plpy.notice('Data\n         ----\n')
plpy.notice('Data shape: {}'.format(data.shape))
plpy.notice('Data sample:\n{}\n'.format(data.head()))

# Outliers
plpy.notice('Outliers\n         --------\n')

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
plpy.notice('Interquartile Range (IQR):\n{}\n'.format(IQR))
plpy.notice('Outliers detected using IQR:\n{}\n'.format((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))))

plt.cla()
fig, axs = plt.subplots(ncols=plot_columns, nrows=plot_rows, figsize=(20, 5 * plot_rows))
index = 0
axs = axs.flatten()
for k,v in data.items():
    sns.boxplot(y=k, data=data, ax=axs[index])
    index += 1
plt.tight_layout(pad=5, w_pad=0.5, h_pad=5.0)
plt.suptitle('{} Outliers'.format(output_name.capitalize()))
plt.savefig('{}/{}_outliers.png'.format(output_path, output_name))
plpy.notice('Created: {}/{}_outliers.png\n'.format(output_path, output_name))

# Distributions
plpy.notice('Distributions\n         -------------\n')
plpy.notice('Summary:\n{}\n'.format(data.describe()))

plt.cla()
fig, axs = plt.subplots(ncols=plot_columns, nrows=plot_rows, figsize=(20, 5 * plot_rows))
index = 0
axs = axs.flatten()
for k,v in data.items():
    sns.distplot(v, ax=axs[index])
    index += 1
plt.tight_layout(pad=5, w_pad=0.5, h_pad=5.0)
plt.suptitle('{} Distributions'.format(output_name.capitalize()))
plt.savefig('{}/{}_distributions.png'.format(output_path, output_name))
plpy.notice('Created: {}/{}_distributions.png\n'.format(output_path, output_name))

# Correlations
plpy.notice('Correlations\n         ------------\n')

corr = data.corr()
plpy.notice('Correlation data:\n{}\n'.format(corr))

plt.cla()
plt.figure(figsize=(20,20))
sns.heatmap(data.corr().abs(), annot=True, cmap='Blues')
plt.tight_layout(pad=5, w_pad=0.5, h_pad=5.0)
plt.suptitle('{} Correlations'.format(output_name.capitalize()))
plt.savefig('{}/{}_correlations.png'.format(output_path, output_name))
plpy.notice('Created: {}/{}_correlations.png\n'.format(output_path, output_name))
$BODY$;

ALTER FUNCTION public.tf_analyse(text, text, text)
    OWNER TO postgres;

COMMENT ON FUNCTION public.tf_analyse(text, text, text)
    IS 'Function to perform statistical analysis on an arbitrary data set.

Parameters:
  * data_source_sql: An SQL query returning at least 2 rows and 2 columns of numeric data to analyse.
  * output_name: The name of the output to use in titles etc.
  * output_path: The path of a directory under which to save generated graphs. Must be writeable by the database server''s service account (usually postgres).';

-- Function to build and train a model
CREATE OR REPLACE FUNCTION public.tf_model(
	data_source_sql text,
	structure integer[],
	output_name text,
	output_path text,
	epochs integer DEFAULT 5000,
	validation_pct integer DEFAULT 10,
	test_pct integer DEFAULT 10)
    RETURNS double precision
    LANGUAGE 'plpython3u'
    COST 100000
    VOLATILE PARALLEL UNSAFE
AS $BODY$
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from tensorflow.python.keras.callbacks import ModelCheckpoint, LambdaCallback, EarlyStopping

# Pandas print options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Reset everything
tf.keras.backend.clear_session()
tf.random.set_seed(42)

# Create the data sets
rows = plpy.execute(data_source_sql)

# Check we have enough rows
if len(rows) < 2:
    plpy.error('At least 5 data rows must be available for training. {} rows retrieved.'.format(len(rows)))

# Get a list of columns
columns = list(rows[0].keys())

# Check we have enough columns
if len(columns) < 2:
    plpy.error('At least 5 data columns must be available for training. {} columns retrieved.'.format(len(columns)))

plpy.notice('Total rows: {}'.format(len(rows)))

# Create the dataframe
data = pd.DataFrame.from_records(rows, columns = columns)

# Remove any rows with outliers
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
plpy.notice('Removing outliers...')
data = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]

# So how many rows remain?
actual_rows = len(data)

# Figure out how many rows to use for training, validation and test
test_rows = int((actual_rows/100) * test_pct)
validation_rows = int(((actual_rows)/100) * validation_pct)
training_rows = actual_rows - test_rows - validation_rows

# Split the data into input and output
input = data[columns[:-1]]
output = data[columns[-1:]]

# Split the input and output into training, validation and test sets
max_z = max(output[output.columns[0]])
training_input = input[:training_rows]
training_output = output[:training_rows]
validation_input = input[training_rows:training_rows+validation_rows]
validation_output = output[training_rows:training_rows+validation_rows]
test_input = input[training_rows+validation_rows:]
test_output = output[training_rows+validation_rows:]

plpy.notice('Rows: {}, training rows: {}, validation rows: {},  test rows: {}.'.format(actual_rows, len(training_input), len(validation_input), len(test_input)))

# Define the model
model = tf.keras.Sequential()
for units in structure:
    if len(model.layers) == 0:
        model.add(tf.keras.layers.Dense(units=units, input_shape=(len(columns) - 1,), activation = 'relu'))
    else:
        model.add(tf.keras.layers.Dense(units=units, activation = 'relu'))

model.add(tf.keras.layers.Dense(units=1, activation='linear'))

# Compile it
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer='adam')

summary = []
model.summary(print_fn=lambda x: summary.append(x))
plpy.notice('Model architecture:\n{}'.format('\n'.join(summary)))

# Save a checkpoint each time our loss metric improves.
checkpoint = ModelCheckpoint('{}/{}.h5'.format(output_path, output_name),
                             monitor='loss',
                             save_best_only=True,
                             mode='min')

# Use early stopping
early_stopping = EarlyStopping(patience=50)

# Display output
logger = LambdaCallback(
    on_epoch_end=lambda epoch,
    logs: plpy.notice(
        'epoch: {}, training RMSE: {} ({}%), validation RMSE: {} ({}%)'.format(
            epoch,
            sqrt(logs['loss']),
            round(100 / max_z * sqrt(logs['loss']), 5),
            sqrt(logs['val_loss']),
            round(100 / max_z * sqrt(logs['val_loss']), 5))))

# Train it!
history = model.fit(training_input,
                    training_output,
                    validation_data=(validation_input, validation_output),
                    epochs=epochs,
                    verbose=False,
                    batch_size=50,
                    callbacks=[logger, checkpoint, early_stopping])

# Graph the results
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs_range = range(len(history.history['loss']))

plt.figure(figsize=(12, 8))
plt.grid(True)
plt.plot(epochs_range, [x ** 0.5 for x in training_loss], label='Training')
plt.plot(epochs_range, [x ** 0.5 for x in validation_loss], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Root Mean Squared Error')
plt.legend(loc='upper right')
plt.title('Training and Validation Root Mean Squared Error')
plt.savefig('{}/{}_rmse.png'.format(output_path, output_name))
plpy.notice('Created: {}/{}_rmse.png\n'.format(output_path, output_name))

# Load the best model from the checkpoint
model = tf.keras.models.load_model('{}/{}.h5'.format(output_path, output_name))

# Dump the original test data, and test results for comparison
test_dump = test_input.copy()
test_dump['actual'] = test_output
test_dump['predicted'] = model.predict(test_input)[:,0]
test_dump['diff'] = abs(test_dump['predicted'] - test_dump['actual'])
test_dump['pc_diff'] = test_dump['diff'] / (test_dump['predicted'] + 1e-10) * 100

plpy.notice('Test data: \n{}\n'.format(test_dump))

# Test the model on the training and validation data to get the RMSE
evaluation = model.evaluate(training_input, training_output)
plpy.notice('Training RMSE:                  {}'.format(round(sqrt(evaluation), 5)))
if len(validation_input) > 0:
    evaluation = model.evaluate(validation_input, validation_output)
    plpy.notice('Validation RMSE:                {}'.format(round(sqrt(evaluation), 5)))

# Summarise the results from the test data set
plpy.notice('Test data mean absolute diff:   {}'.format(round(float(sum(test_dump['diff']) / len(test_dump)), 5)))
plpy.notice('Test data mean percentage diff: {}%'.format(round(float(sum(test_dump['pc_diff']) / len(test_dump)), 5)))

rmse = float(sqrt(abs(sum((test_dump['actual'] - test_dump['predicted']) ** 2) / len(test_dump))))
plpy.notice('Test data RMSE:                 {}'.format(round(rmse, 5)))

rmspe = float(sqrt(abs(sum((test_dump['actual'] - test_dump['predicted']) / (test_dump['actual']) + 1e-10))) / len(test_dump)) * 100
plpy.notice('Test data RMSPE:                {}%\n'.format(round(rmspe, 5)))

plpy.notice('Model saved to: {}/{}.h5'.format(output_path, output_name))

return rmspe
$BODY$;

ALTER FUNCTION public.tf_model(text, integer[], text, text, integer, integer, integer)
    OWNER TO postgres;

COMMENT ON FUNCTION public.tf_model(text, integer[], text, text, integer, integer, integer)
    IS 'Function to build and train a model to analyse an abitrary data set.

Parameters:
  * data_source_sql: An SQL query returning at least 5 rows and 3 columns of numeric data to analyse.
  * structure: An array of integers indicating the number of neurons in each of an arbitrary number of layer. A final output layer will be added with a single neuron.
  * output_name: The name of the output to use in titles etc.
  * output_path: The path of a directory under which to save generated graphs and the model. Must be writeable by the database server''s service account (usually postgres).
  * epochs: The maximum number of training epochs to run (default: 5000)
  * validation_pct: The percentage of the rows returned by the query specified in data_source_sql to use for model validation (default: 10).
  * test_pct: The percentage of the rows returned by the query specified in data_source_sql to use for model testing (default: 10).

Returns: The Root Mean Square Percentage Error calculated from the evaluation of the test data set.';

-- Function to make a prediction based on a model
CREATE OR REPLACE FUNCTION public.tf_predict(
	input_values double precision[],
	model_path text)
    RETURNS double precision[]
    LANGUAGE 'plpython3u'
    COST 100
    VOLATILE PARALLEL UNSAFE
AS $BODY$
import tensorflow as tf

# Reset everything
tf.keras.backend.clear_session()
tf.random.set_seed(42)

# Load the model
model = tf.keras.models.load_model(model_path)

# Are we dealing with a single prediction, or a list of them?
if not any(isinstance(sub, list) for sub in input_values):
    data = [input_values]
else:
    data = input_values

# Make the prediction(s)
result = model.predict([data])[0]
result = [ item for elem in result for item in elem]

return result
$BODY$;

ALTER FUNCTION public.tf_predict(double precision[], text)
    OWNER TO postgres;

COMMENT ON FUNCTION public.tf_predict(double precision[], text)
    IS 'Function to make predictions based on input values and a Tensorflow model.

Parameters:
  * input_values: An array of input values, or an array of arrays of input values, e.g. ''{2, 3}'' or ''{{2, 3}, {3, 4}}''.
  * model_path: The full path to a Tensorflow model saved in .h5 format. Must be writeable by the database server''s service account (usually postgres).

Returns: An array of predicted values.';
