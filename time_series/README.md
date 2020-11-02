# Time Series Prediction

Prediction of time series data is extremely useful in capacity management tools.
Linear Trend Analysis is a common option, but this only allows you to predict 
the general trend of the data. Using a WaveNet model architecture we can predict
data with trends and seasonality traits.

There are two scripts in this directory:

## cnn-demo.py

This script will generate a series of data with a general upward trend, noise,
and seasonality. It is largely based on the example at:

https://github.com/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l08c09_forecasting_with_cnn.ipynb

## cnn-workload.py

This is a minor variation to the *cnn-demo.py* script. Instead of generating its 
own test data, it will read a data set from *activity.csv* and learn and predict
based on that data. *activity.csv* was created by logging the number of rows in
*pg_stat_activity* on a PostgreSQL server every five minutes, whilst a workload
generator was running against the system, simulating user activity.