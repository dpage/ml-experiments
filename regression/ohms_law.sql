-- Table to hold the training inputs and output
CREATE TABLE public.ohms_law
(
    r double precision NOT NULL,
    i double precision NOT NULL,
    v double precision NOT NULL,
    CONSTRAINT ohms_law_pkey PRIMARY KEY (r, i)
)

TABLESPACE pg_default;

ALTER TABLE public.ohms_law
    OWNER to postgres;

-- Function to populate the training data
CREATE OR REPLACE FUNCTION public.ohms_law_generate(rows integer)
    RETURNS void
    LANGUAGE 'plpgsql'
    COST 100
    VOLATILE PARALLEL UNSAFE
AS $BODY$
DECLARE
  i float;
  r float;
  v float;
BEGIN
    FOR l IN 1..rows LOOP
        SELECT round(random() * 1000 + 1) INTO i;
        SELECT round(random() * 1000 + 1) INTO r;
        v := i * r;

        RAISE NOTICE 'i: %, r: %, v: %', i, r, v;
        BEGIN
            INSERT INTO ohms_law (r, i, v) VALUES (r, i, v);
        EXCEPTION WHEN unique_violation THEN
            l := l - 1;
        END;
    END LOOP;
END
$BODY$;

ALTER FUNCTION public.ohms_law_generate(integer)
    OWNER TO postgres;

SELECT ohms_law_generate(1000);

-- Create, train and test the model
CREATE OR REPLACE FUNCTION public.ohms_law_v(
	r double precision,
	i double precision)
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
validation_pct = 25
test_pct = 10
epochs = 1000

# Create the data sets
rows = plpy.execute('SELECT r, i, v FROM ohms_law ORDER BY random() LIMIT {}'.format(total_rows))
actual_rows = len(rows)

if actual_rows < 5:
    plpy.error('At least 5 data rows must be available for training. {} rows retrieved.'.format(actual_rows))

test_rows = int((actual_rows/100) * test_pct)
validation_rows = int(((actual_rows)/100) * validation_pct)
training_rows = actual_rows - test_rows - validation_rows

data = []
results = []

for row in rows:
    data.append([row['r'], row['i']])
    results.append(row['v'])

max_v = max(results)
training_data = np.array(data[:training_rows], dtype=float)
training_results = np.array(results[:training_rows], dtype=float)
validation_data = np.array(data[training_rows:training_rows+validation_rows], dtype=float)
validation_results = np.array(results[training_rows:training_rows+validation_rows], dtype=float)
test_data = np.array(data[training_rows+validation_rows:], dtype=float)
test_results = np.array(results[training_rows+validation_rows:], dtype=float)

plpy.notice('Total rows: {}, training rows: {}, validation rows: {},  test rows: {}.'.format(actual_rows, len(training_data), len(validation_data), len(test_data)))

# Define the model
l0 = tf.keras.layers.Input(shape=(2))
l1 = tf.keras.layers.Dense(units=16, activation = 'relu')
l2 = tf.keras.layers.Dense(units=1)

model = tf.keras.Sequential([l0, l1, l2])

# Compile it
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer='adam')

summary = []
model.summary(print_fn=lambda x: summary.append(x))
plpy.notice('Model architecture:\n{}'.format('\n'.join(summary)))

# Save a checkpoint each time our loss metric improves.
checkpoint = ModelCheckpoint('/Users/Shared/tf/ohms_law.h5',
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
            round(100 / max_v * math.sqrt(logs['loss']), 3),
            math.sqrt(logs['val_loss']),
            round(100 / max_v * math.sqrt(logs['val_loss']), 3))))

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

plt.savefig('/Users/Shared/tf/ohms_law.png')

# Load the best model from the checkpoint
model = tf.keras.models.load_model('/Users/Shared/tf/ohms_law.h5')

# How good is it looking?
evaluation = model.evaluate(np.array(training_data), np.array(training_results))
plpy.notice('Training RMSE:   {} ({}%).'.format(math.sqrt(evaluation), round(100 / max_v * math.sqrt(evaluation), 3)))

if len(validation_data) > 0:
    evaluation = model.evaluate(np.array(validation_data), np.array(validation_results))
    plpy.notice('Validation RMSE: {} ({}%).'.format(math.sqrt(evaluation), round(100 / max_v * math.sqrt(evaluation), 3)))

if len(test_data) > 0:
    evaluation = model.evaluate(np.array(test_data), np.array(test_results))
    plpy.notice('Test RMSE:       {} ({}%).'.format(math.sqrt(evaluation), round(100 / max_v * math.sqrt(evaluation), 3)))

# Get the result
result = model.predict(np.array([[r, i]]))

return result[0][0]
$BODY$;

ALTER FUNCTION public.ohms_law_v(double precision, double precision)
    OWNER TO postgres;

SELECT ohms_law_v(3, 4); -- Correct answer is 12