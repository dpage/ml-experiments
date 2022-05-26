import pandas as pd
import matplotlib.pyplot as plt
import dateutil.parser
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfd
from tensorflow_probability import sts

tf.enable_v2_behavior()

DATAFILE = 'activity.csv'

data = pd.read_csv(DATAFILE,
                   index_col=0,
                   parse_dates=[0],
                   date_parser=dateutil.parser.isoparse,
                   header=0,
                   names=['ds', 'y'])

data = tfp.sts.regularize_series(data[-4032:])

trend = sts.LocalLinearTrend(observed_time_series=data)
seasonal = tfp.sts.Seasonal(num_seasons=12, observed_time_series=data)
model = sts.Sum([trend, seasonal], observed_time_series=data)

variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=model)

num_variational_steps = 5

# Build and optimize the variational loss function.
elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn=model.joint_distribution(
        observed_time_series=data).log_prob,
    surrogate_posterior=variational_posteriors,
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    num_steps=num_variational_steps,
    jit_compile=True)

plt.plot(elbo_loss_curve)
plt.show()

# Draw samples from the variational posterior.
q_samples_ = variational_posteriors.sample(50)

# Forecast
n_periods = int(len(data) * 0.25)
forecast = tfp.sts.forecast(model,
                            observed_time_series=data,
                            parameter_samples=q_samples_,
                            num_steps_forecast=n_periods)

# Plot - FIXME!!
fig, axes = plt.subplots(figsize=(10, 5), dpi=100, tight_layout=True)
plt.plot(data)
plt.plot(forecast.sample(10).numpy()[..., 0])
plt.show()
