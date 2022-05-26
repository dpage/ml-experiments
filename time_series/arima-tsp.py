import pandas as pd
import matplotlib.pyplot as plt
import dateutil.parser
import pmdarima as pm
from pmdarima.arima.stationarity import ADFTest

import pickle

DATAFILE = 'activity.csv'

try:
    data = pd.read_csv(DATAFILE,
                       index_col=0,
                       parse_dates=[0],
                       date_parser=dateutil.parser.isoparse)
    is_dated = True
except:
    data = pd.read_csv(DATAFILE, index_col=0)
    is_dated = False

data = data[-4032:]
data = data.resample('60T').mean()

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10, 5), dpi=100, sharex=True)

# Usual Differencing
axes[0].plot(data[:], label='Original Series')
axes[0].plot(data[:].diff(1), label='Usual Differencing')
axes[0].set_title('Usual Differencing')
axes[0].legend(loc='upper left', fontsize=10)

# Seasonal Differencing
axes[1].plot(data[:], label='Original Series')
axes[1].plot(data[:].diff(12), label='Seasonal Differencing', color='green')
axes[1].set_title('Seasonal Differencing')
plt.legend(loc='upper left', fontsize=10)

plt.show()

pm.plot_acf(data)

try:
    # Try to load an existing model first
    with open(DATAFILE + '.pkl', 'rb') as pkl:
        smodel = pickle.load(pkl)
except:
    # Seasonal - fit stepwise auto-ARIMA
    smodel = pm.auto_arima(data, start_p=1, start_q=1,
                           test='adf',
                           max_p=3, max_q=3, m=168,
                           start_P=0, seasonal=True,
                           d=None, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)

    # Serialize with Pickle
    with open(DATAFILE + '.pkl', 'wb') as pkl:
        pickle.dump(smodel, pkl)

print(smodel.summary())

# Forecast
n_periods = int(len(data) * 0.25)
fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)

if is_dated:
    index_of_fc = pd.date_range(data.index[-1], periods=n_periods, freq='60min')
else:
    index_of_fc = pd.RangeIndex(data.index[-1], data.index[-1] + n_periods).to_series()

# make series for plotting purpose
fitted_series = pd.Series(fitted, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
fig, axes = plt.subplots( figsize=(10, 5), dpi=100, tight_layout=True)
plt.plot(data)
plt.plot(fitted_series, color='darkgreen')
plt.fill_between(lower_series.index,
                 lower_series,
                 upper_series,
                 color='k', alpha=.15)

plt.show()
