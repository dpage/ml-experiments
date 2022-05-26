import pandas as pd
import matplotlib.pyplot as plt
import dateutil.parser
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json

DATAFILE = 'data.csv'

data = pd.read_csv(DATAFILE,
                   parse_dates=[0],
                   date_parser=dateutil.parser.isoparse,
                   header=0,
                   names=['ds', 'y'])

try:
    # Try to load an existing model first
    with open(DATAFILE + '.json', 'r') as f:
        m = model_from_json(f.read())
except:
    m = Prophet()
    m.fit(data)

    with open(DATAFILE + '.json', 'w') as f:
        f.write(model_to_json(m))

# Forecast
n_periods = 288
future = m.make_future_dataframe(periods=n_periods,
                                 freq='5T',
                                 include_history=False)

forecast = m.predict(future)

# Plot
fig, axes = plt.subplots(figsize=(10, 5), dpi=100, tight_layout=True)
plt.plot(data['ds'][100000:], data['y'][100000:])
plt.plot(forecast['ds'], forecast['yhat'])
plt.fill_between(forecast['ds'],
                 forecast['yhat_lower'],
                 forecast['yhat_upper'],
                 color='k', alpha=.15)
plt.show()
