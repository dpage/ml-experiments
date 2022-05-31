-- This SQL script will create a set of objects for creating, updating Prophet
-- models in PostgreSQL, along with an additional function for running
-- predictions

CREATE OR REPLACE FUNCTION public.activate_python_venv(
	venv text)
    RETURNS void
    LANGUAGE 'plpython3u'
    COST 100
    VOLATILE PARALLEL UNSAFE
AS $BODY$
    import os
    import sys

    if sys.platform in ('win32', 'win64', 'cygwin'):
        activate_this = os.path.join(venv, 'Scripts', 'activate_this.py')
    else:
        activate_this = os.path.join(venv, 'bin', 'activate_this.py')

    exec(open(activate_this).read(), dict(__file__=activate_this))
$BODY$;

ALTER FUNCTION public.activate_python_venv(text)
    OWNER TO postgres;

COMMENT ON FUNCTION public.activate_python_venv(text)
    IS 'Activate a Python virtual environment in this database session.

Arguments:
    venv: The path to the virtual environment.
Returns:
    void';


CREATE OR REPLACE FUNCTION public.create_model(
	relation text,
	ts_column name,
	y_column name,
	model_name text,
	overwrite boolean DEFAULT false)
    RETURNS text
    LANGUAGE 'plpython3u'
    COST 100
    VOLATILE PARALLEL UNSAFE
AS $BODY$
import json
import os
import pathlib
import sys
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json

# Make sure we do not try to write outside of our directory
try:
    pathlib.Path(
        os.path.abspath(
            os.path.join('models', model_name + '.json')
        )
    ).relative_to(os.path.abspath('models'))
except ValueError:
    plpy.error('Invalid model name: {}'.format(model_name))

# Check for an existing model
model_file = os.path.abspath(os.path.join('models', model_name + '.json'))

if overwrite != True:
    if os.path.exists(model_file):
        plpy.error('Model {} already exists. Set the overwrite parameter to true to replace.'.format(model_file))

# Create the data set
rows = plpy.execute('SELECT {}::timestamp AS ds, {} AS y FROM {} ORDER BY {} ASC'.format(ts_column, y_column, relation, ts_column))

# Check we have enough rows
if len(rows) < 2:
    plpy.error('At least 5 data rows must be available for analysis. {} rows retrieved.'.format(len(rows)))

# Create the dataframe
columns = list(rows[0].keys())
data = pd.DataFrame.from_records(rows, columns = columns)

# Create the model
m = Prophet()
m.fit(data)

# Save the model
if not os.path.exists('models'):
    os.makedirs('models')

json_model = json.loads(model_to_json(m))
full_model = {'relation': relation,
              'ts_column': ts_column,
              'y_column': y_column,
              'model': json_model}

with open(model_file, 'w') as f:
    f.write(json.dumps(full_model))

return model_file
$BODY$;

ALTER FUNCTION public.create_model(text, name, name, text, boolean)
    OWNER TO postgres;

COMMENT ON FUNCTION public.create_model(text, name, name, text, boolean)
    IS 'Create a Prophet model for making predictions.

Arguments:
    relation: The name of the table from which observations should be loaded.
    ts_column: The name of the column containing the observation timestamp.
    y_column: The name of the column containing the observed value.
    model_name: The name for the model.
    overwrite: Overwrite an existing model of the same name if present (default: false)
Returns:
    text: The full path of the model file.';


CREATE OR REPLACE FUNCTION public.delete_model(
	model_name text)
    RETURNS void
    LANGUAGE 'plpython3u'
    COST 100
    VOLATILE PARALLEL UNSAFE
AS $BODY$
import os
import pathlib
import sys

# Make sure we do not try to write outside of our directory
try:
    pathlib.Path(
        os.path.abspath(
            os.path.join('models', model_name + '.json')
        )
    ).relative_to(os.path.abspath('models'))
except ValueError:
    plpy.error('Invalid model name: {}'.format(model_name))

# Check for an existing model
model_file = os.path.abspath(os.path.join('models', model_name + '.json'))

if not os.path.exists(model_file):
    plpy.error('Model {} does not exist.'.format(model_file))

os.remove(model_file)

return
$BODY$;

ALTER FUNCTION public.delete_model(text)
    OWNER TO postgres;

COMMENT ON FUNCTION public.delete_model(text)
    IS 'Delete an existing model.

Arguments:
    model_name: The name of the model to delete.
Returns:
    void';


CREATE OR REPLACE FUNCTION public.predict(
	model_name text,
	periods integer,
	frequency text,
	include_history boolean DEFAULT false,
	OUT ts timestamp without time zone,
	OUT y numeric,
	OUT y_lower numeric,
	OUT y_upper numeric)
    RETURNS SETOF record
    LANGUAGE 'plpython3u'
    COST 100
    VOLATILE PARALLEL UNSAFE
    ROWS 1000

AS $BODY$
import json
import os
import pathlib
import sys
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json

# Make sure we do not try to write outside of our directory
try:
    pathlib.Path(
        os.path.abspath(
            os.path.join('models', model_name + '.json')
        )
    ).relative_to(os.path.abspath('models'))
except ValueError:
    plpy.error('Invalid model name: {}'.format(model_name))

# Check for an existing model
model_file = os.path.abspath(os.path.join('models', model_name + '.json'))

if not os.path.exists(model_file):
    plpy.error('Model {} does not exist.'.format(model_file))

with open(model_file, 'r') as f:
    json_model = json.load(f)

m = model_from_json(json.dumps(json_model['model']))

# Forecast
future = m.make_future_dataframe(periods=periods,
                                 freq=frequency,
                                 include_history=include_history)

forecast = m.predict(future)

# Convert to the output
output = []
for d in range(0, len(forecast)):
    output.append((forecast['ds'][d], forecast['yhat'][d], forecast['yhat_lower'][d], forecast['yhat_upper'][d]))

return output
$BODY$;

ALTER FUNCTION public.predict(text, integer, text, boolean)
    OWNER TO postgres;

COMMENT ON FUNCTION public.predict(text, integer, text, boolean)
    IS 'Make a prediciton using a previously created model.

Arguments:
    model_name: The name of the model to use.
    periods: The number of periods to predict.
    frequency: The period length, expressed as a Pandas frequency string, e.g. "5T" for 5 minutes.
    include_history: Include historic predictions for existing data in the output (default: false).
Returns set of records:
    ts: The timestamp of the prediction.
    y: The predicted value.
    y_lower: The lower confidence bound of the predicted value.
    y_upper: The upper confidence bound of the predicted value.
    ';


CREATE OR REPLACE FUNCTION public.update_model(
	model_name text,
	warm_start boolean DEFAULT true)
    RETURNS text
    LANGUAGE 'plpython3u'
    COST 100
    VOLATILE PARALLEL UNSAFE
AS $BODY$
import json
import os
import pathlib
import sys
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json

def init_stan(m):
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        res[pname] = m.params[pname][0][0]
    for pname in ['delta', 'beta']:
        res[pname] = m.params[pname][0]
    return res


# Make sure we do not try to write outside of our directory
try:
    pathlib.Path(
        os.path.abspath(
            os.path.join('models', model_name + '.json')
        )
    ).relative_to(os.path.abspath('models'))
except ValueError:
    plpy.error('Invalid model name: {}'.format(model_name))

# Check for an existing model
model_file = os.path.abspath(os.path.join('models', model_name + '.json'))

if not os.path.exists(model_file):
    plpy.error('Model {} does not exist.'.format(model_file))

with open(model_file, 'r') as f:
    json_model = json.load(f)

# Get the meta data
relation = json_model['relation']
ts_column = json_model['ts_column']
y_column = json_model['y_column']

# Create the data set
rows = plpy.execute('SELECT {}::timestamp AS ds, {} AS y FROM {} ORDER BY {} ASC'.format(ts_column, y_column, relation, ts_column))

# Check we have enough rows
if len(rows) < 2:
    plpy.error('At least 5 data rows must be available for analysis. {} rows retrieved.'.format(len(rows)))

# Create the dataframe
columns = list(rows[0].keys())
data = pd.DataFrame.from_records(rows, columns = columns)

if warm_start:
    m = model_from_json(json.dumps(json_model['model']))
    m = Prophet().fit(data, init=init_stan(m))
else:
    m = Prophet()
    m.fit(data)

json_model = json.loads(model_to_json(m))
full_model = {'relation': relation,
              'ts_column': ts_column,
              'y_column': y_column,
              'model': json_model}

with open(model_file, 'w') as f:
    f.write(json.dumps(full_model))

return model_file
$BODY$;

ALTER FUNCTION public.update_model(text, boolean)
    OWNER TO postgres;

COMMENT ON FUNCTION public.update_model(text, boolean)
    IS 'Update an existing Prophet model.

Arguments:
    model_name: The name of the model to update.
    warm_start: Use the existing model to bootstrap the new one (default: true).
Returns:
    text: The full path of the model file.

Note that whilst enabling warm_start will significantly speed up refitting of the model, it should only be used after adding a small number of additional records; a cold start update should be performed when more significant numbers of records are added to ensure that changepoints in the data are properly recalculated. One strategy (dependent on observation frequency) might be to make a daily update using warm start, and a full update once a week.';
