-- Table to hold the training inputs and output
CREATE TABLE public.random1
(
    a double precision NOT NULL,
    b double precision NOT NULL,
    c double precision NOT NULL,
    d double precision NOT NULL,
    e double precision NOT NULL,
    z double precision NOT NULL,
    CONSTRAINT random1_pkey PRIMARY KEY (a, b, c, d, e)
)

TABLESPACE pg_default;

ALTER TABLE public.random1
    OWNER to postgres;

-- Function to populate the training data
CREATE OR REPLACE FUNCTION public.random1_generate(rows integer)
    RETURNS void
    LANGUAGE 'plpgsql'
    COST 100
    VOLATILE PARALLEL UNSAFE
AS $BODY$
DECLARE
  a float;
  b float;
  c float;
  d float;
  e float;
  z float;
BEGIN
    FOR l IN 1..rows LOOP
        SELECT round(random() * 10000 + 1) INTO a;
        SELECT round(random() * 10000 + 1) INTO b;
        SELECT round(random() * 10000 + 1) INTO c;
        SELECT round(random() * 10000 + 1) INTO d;
        SELECT round(random() * 10000 + 1) INTO e;
        z := cbrt((a * b) / (sin(c) * sqrt(d)) + (e * e * e));

        RAISE NOTICE 'a: %, b: %, c: %, d: %, e: %, z: %', a, b, c, d, e, z;
        BEGIN
            INSERT INTO random1 (a, b, c, d, e, z) VALUES (a, b, c, d, e, z);
        EXCEPTION WHEN unique_violation THEN
            l := l - 1;
        END;
    END LOOP;
END
$BODY$;

ALTER FUNCTION public.random1_generate(integer)
    OWNER TO postgres;

SELECT random1_generate(1000);

-- Create, train and test the model
CREATE OR REPLACE FUNCTION public.random1_v(
	a double precision,
	b double precision,
	c double precision,
	d double precision,
	e double precision)
    RETURNS double precision
    LANGUAGE 'plpython3u'
    COST 100
    VOLATILE PARALLEL UNSAFE
AS $BODY$
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

from tensorflow.python.keras.callbacks import ModelCheckpoint, LambdaCallback, EarlyStopping, LearningRateScheduler

tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

total_rows = 1000
validation_pct = 10
test_pct = 1
epochs = 1000

# Create the data sets
rows = plpy.execute('SELECT a, b, c, d, e, z FROM random1 ORDER BY random() LIMIT {}'.format(total_rows))
actual_rows = len(rows)

if actual_rows < 5:
    plpy.error('At least 5 data rows must be available for training. {} rows retrieved.'.format(actual_rows))

test_rows = int((actual_rows/100) * test_pct)
validation_rows = int(((actual_rows)/100) * validation_pct)
training_rows = actual_rows - test_rows - validation_rows

data = []
results = []

for row in rows:
    data.append([row['a'], row['b'], row['c'], row['d'], row['e']])
    results.append(row['z'])

max_z = max(results)
training_data = np.array(data[:training_rows], dtype=float)
training_results = np.array(results[:training_rows], dtype=float)
validation_data = np.array(data[training_rows:training_rows+validation_rows], dtype=float)
validation_results = np.array(results[training_rows:training_rows+validation_rows], dtype=float)
test_data = np.array(data[training_rows+validation_rows:], dtype=float)
test_results = np.array(results[training_rows+validation_rows:], dtype=float)

plpy.notice('Total rows: {}, training rows: {}, validation rows: {},  test rows: {}.'.format(actual_rows, len(training_data), len(validation_data), len(test_data)))

# Define the model
l1 = tf.keras.layers.Dense(units=16, input_shape=(5,), activation = 'relu')
l2 = tf.keras.layers.Dense(units=16, activation = 'relu')
l3 = tf.keras.layers.Dense(units=1, activation='linear')

model = tf.keras.Sequential([l1, l2, l3])

# Compile it
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer='adam')

summary = []
model.summary(print_fn=lambda x: summary.append(x))
plpy.notice('Model architecture:\n{}'.format('\n'.join(summary)))

# Save a checkpoint each time our loss metric improves.
checkpoint = ModelCheckpoint('/Users/Shared/tf/random1.h5',
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
            math.sqrt(logs['loss']),
            round(100 / max_z * math.sqrt(logs['loss']), 3),
            math.sqrt(logs['val_loss']),
            round(100 / max_z * math.sqrt(logs['val_loss']), 3))))

# Train it!
history = model.fit(training_data,
                    training_results,
                    validation_data=(validation_data, validation_results),
                    epochs=epochs,
                    verbose=False,
                    batch_size=50,
                    callbacks=[logger, checkpoint, early_stopping])

# Graph the results
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs_range = range(len(history.history['loss']))

plt.figure(figsize=(12, 8))
plt.plot(epochs_range, np.sqrt(training_loss), label='Training RMSE')
plt.plot(epochs_range, np.sqrt(validation_loss), label='Validation RMSE')
plt.legend(loc='upper right')
plt.title('Training and Validation RMSE')

plt.savefig('/Users/Shared/tf/random1.png')

# Load the best model from the checkpoint
model = tf.keras.models.load_model('/Users/Shared/tf/random1.h5')

# How good is it looking?
evaluation = model.evaluate(np.array(training_data), np.array(training_results))
plpy.notice('Training RMSE:   {} ({}%).'.format(math.sqrt(evaluation), round(100 / max_z * math.sqrt(evaluation), 3)))

if len(validation_data) > 0:
    evaluation = model.evaluate(np.array(validation_data), np.array(validation_results))
    plpy.notice('Validation RMSE: {} ({}%).'.format(math.sqrt(evaluation), round(100 / max_z * math.sqrt(evaluation), 3)))

if len(test_data) > 0:
    evaluation = model.evaluate(np.array(test_data), np.array(test_results))
    plpy.notice('Test RMSE:       {} ({}%).'.format(math.sqrt(evaluation), round(100 / max_z * math.sqrt(evaluation), 3)))

# Get the result
result = model.predict(np.array([[a, b, c, d, e]]))

return result[0][0]
$BODY$;

SELECT random1_v(5, 25, 67, 2, 29); -- Correct answer is 28.958992634781293