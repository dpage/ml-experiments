-- Table to hold the training inputs and output
CREATE TABLE public.housing
(
    crim double precision NOT NULL,
    zn double precision NOT NULL,
    indus double precision NOT NULL,
    chas double precision NOT NULL,
    nox double precision NOT NULL,
    rm double precision NOT NULL,
    age double precision NOT NULL,
    dis double precision NOT NULL,
    rad double precision NOT NULL,
    tax double precision NOT NULL,
    ptratio double precision NOT NULL,
    b double precision NOT NULL,
    lstat double precision NOT NULL,
    medv double precision NOT NULL
)

TABLESPACE pg_default;

ALTER TABLE public.housing
    OWNER to postgres;

-- Create, train and test the model
CREATE OR REPLACE FUNCTION public.housing_v(
	crim double precision,
	zn double precision,
	indus double precision,
	chas double precision,
	nox double precision,
	rm double precision,
	age double precision,
	dis double precision,
	rad double precision,
	tax double precision,
	ptratio double precision,
	b double precision,
	lstat double precision)
    RETURNS double precision
    LANGUAGE 'plpython3u'
    COST 100
    VOLATILE PARALLEL UNSAFE
AS $BODY$
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt

from tensorflow.python.keras.callbacks import ModelCheckpoint, LambdaCallback, EarlyStopping, LearningRateScheduler

# Configurables
DEBUG = True
total_rows = 1000
validation_pct = 10
test_pct = 5
epochs = 5000

# Pandas print options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

# Create the data sets
rows = plpy.execute('SELECT crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat, medv FROM housing ORDER BY random() LIMIT {}'.format(total_rows))
data = pd.DataFrame.from_records(rows, columns = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'medv'])

# Remove any rows with outliers
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
plpy.notice('IQR:\n{}'.format(IQR))
plpy.notice('Outliers detected:\n{}'.format((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))))

if DEBUG:
    plt.cla()
    fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
    index = 0
    axs = axs.flatten()
    for k,v in data.items():
        sns.boxplot(y=k, data=data, ax=axs[index])
        index += 1
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
    plt.savefig("/Users/Shared/tf/housing_outliers.png")

    plt.cla()
    fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
    index = 0
    axs = axs.flatten()
    for k,v in data.items():
        sns.distplot(v, ax=axs[index])
        index += 1
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
    plt.savefig("/Users/Shared/tf/housing_distributions.png")

data = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]

# Look for data correlations
if DEBUG:
    corr = data.corr()
    plt.cla()
    plt.figure(figsize=(20,20))
    sns.heatmap(data.corr().abs(), annot=True).get_figure().savefig("/Users/Shared/tf/housing_correlation.png")

# So how many rows remain?
actual_rows = len(data)

# Check we have enough rows left
if actual_rows < 5:
    plpy.error('At least 5 data rows must be available for training. {} rows retrieved.'.format(actual_rows))

# Figure out how many rows to use for training, validation and test
test_rows = int((actual_rows/100) * test_pct)
validation_rows = int(((actual_rows)/100) * validation_pct)
training_rows = actual_rows - test_rows - validation_rows

# Split the data into input and output
input = data[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']]
output = data[['medv']]

max_z = max(output['medv'])
training_input = input[:training_rows]
training_output = output[:training_rows]
validation_input = input[training_rows:training_rows+validation_rows]
validation_output = output[training_rows:training_rows+validation_rows]
test_input = input[training_rows+validation_rows:]
test_output = output[training_rows+validation_rows:]


plpy.notice('Total rows: {}, training rows: {}, validation rows: {},  test rows: {}.'.format(actual_rows, len(training_input), len(validation_input), len(test_input)))

# Define the model
l1 = tf.keras.layers.Dense(units=32, input_shape=(13,), activation = 'relu')
l2 = tf.keras.layers.Dense(units=32, activation = 'relu')
l3 = tf.keras.layers.Dense(units=1, activation='linear')

model = tf.keras.Sequential([l1, l2, l3])

# Compile it
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer='adam')

if DEBUG:
	summary = []
	model.summary(print_fn=lambda x: summary.append(x))
	plpy.notice('Model architecture:\n{}'.format('\n'.join(summary)))

# Save a checkpoint each time our loss metric improves.
checkpoint = ModelCheckpoint('/Users/Shared/tf/housing.h5',
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
if DEBUG:
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

	plt.savefig('/Users/Shared/tf/housing.png')

# Load the best model from the checkpoint
model = tf.keras.models.load_model('/Users/Shared/tf/housing.h5')

if DEBUG:
    # Dump the original test data, and test results for comparison
    test_dump = test_input.copy()
    test_dump['actual'] = test_output
    test_dump['predicted'] = model.predict(test_input)[:,0]
    test_dump['diff'] = abs(test_dump['predicted'] - test_dump['actual'])
    test_dump['pc_diff'] = test_dump['diff'] / test_dump['predicted'] * 100

    plpy.notice('Test data: \n{}\n'.format(test_dump))

    plpy.notice('Test data mean absolute diff:   {}'.format(round(float((sum(test_dump['diff']) / len(test_dump))), 5)))
    plpy.notice('Test data mean percentage diff: {}%'.format(round(float((sum(test_dump['pc_diff']) / len(test_dump))), 5)))
    plpy.notice('Test data RMSE:                 {}\n'.format(round(float((sqrt(sum((test_dump['actual'] - test_dump['predicted']) ** 2) / len(test_dump)))), 5)))

# How good is it looking?
evaluation = model.evaluate(training_input, training_output)
plpy.notice('Training RMSE:                {}.'.format(round(sqrt(evaluation), 5)))

if len(validation_input) > 0:
    evaluation = model.evaluate(validation_input, validation_output)
    plpy.notice('Validation RMSE:              {}.'.format(round(sqrt(evaluation), 5)))

if len(test_input) > 0:
    evaluation = model.evaluate(test_input, test_output)
    plpy.notice('Test RMSE:                    {}.'.format(round(sqrt(evaluation), 5)))

# Get the result
result = model.predict([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])

return result[0][0]
$BODY$;

SELECT housing_v(0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.90, 4.98); -- Correct answer is 24.00
-- SELECT housing_v(0.26363, 0.00000, 8.56000, 0.00000, 0.52000, 6.22900, 91.20000, 2.54510, 5.00000, 384.00000, 20.90000, 391.23000, 15.55000); -- Correct answer is 19.4