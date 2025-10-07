# Feature Engineering Usage Examples

This document provides practical examples of how to use the feature engineering module for different model types.

## Quick Start: Command Line Interface

The easiest way to generate features is using the CLI:

```bash
cd scripts

# Generate all feature sets (recommended)
python feature_engineering.py

# Generate specific feature set
python feature_engineering.py --type ml --drop-na
python feature_engineering.py --type prophet

# Custom input/output paths
python feature_engineering.py --input my_data.csv --output my_features.csv
```

## Using the Module in Python

### Setup

```python
import pandas as pd
import sys
sys.path.append('scripts')

from feature_engineering import (
    create_all_features,
    create_prophet_features,
    create_ml_features,
    get_feature_names
)
```

## Example 1: Prophet Model

Prophet needs data in a specific format with 'ds' (datestamp) and 'y' (target) columns.

```python
# Load data
df = pd.read_csv('data/processed/monthly_tourist_arrivals_filtered.csv')

# Create Prophet features
df_prophet = create_prophet_features(df)

# Columns: Date, Arrivals, year, month, quarter, easter_attacks, 
#          covid_period, economic_crisis, months_since_covid, 
#          recovery_index, ds, y

# Use intervention features as regressors in Prophet
from prophet import Prophet

model = Prophet()

# Add regressors
model.add_regressor('easter_attacks')
model.add_regressor('covid_period')
model.add_regressor('economic_crisis')
model.add_regressor('recovery_index')

# Train model
model.fit(df_prophet[['ds', 'y', 'easter_attacks', 'covid_period', 
                       'economic_crisis', 'recovery_index']])

# Make predictions (you need to provide future regressor values)
future = model.make_future_dataframe(periods=12, freq='MS')
# Add future regressor values here...
forecast = model.predict(future)
```

## Example 2: LSTM Model

LSTM needs scaled features with sequences.

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load data with ML features
df_ml = create_ml_features(
    df,
    include_lags=True,
    drop_na=True  # Remove rows with NaN from lags
)

# Select features for LSTM
feature_cols = ['month_sin', 'month_cos', 'easter_attacks', 
                'covid_period', 'economic_crisis', 'recovery_index',
                'Arrivals_lag_1', 'Arrivals_lag_3', 
                'Arrivals_lag_6', 'Arrivals_lag_12']

target_col = 'Arrivals'

# Split into train and test (chronological)
train_size = int(len(df_ml) * 0.8)
train_df = df_ml.iloc[:train_size]
test_df = df_ml.iloc[train_size:]

# Scale features
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_train = feature_scaler.fit_transform(train_df[feature_cols])
y_train = target_scaler.fit_transform(train_df[[target_col]])

X_test = feature_scaler.transform(test_df[feature_cols])
y_test = target_scaler.transform(test_df[[target_col]])

# Create sequences (optional, depending on your LSTM design)
def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(Xs), np.array(ys)

seq_length = 12
X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)

# Now train your LSTM model
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.LSTM(50, activation='relu', return_sequences=True,
                      input_shape=(seq_length, len(feature_cols))),
    keras.layers.LSTM(50, activation='relu'),
    keras.layers.Dense(25),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=8, 
          validation_split=0.2, verbose=1)

# Predict
predictions = model.predict(X_test_seq)
predictions_original = target_scaler.inverse_transform(predictions)
```

## Example 3: N-HiTS Model

Using NeuralForecast library with N-HiTS.

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MSE

# Prepare data in NeuralForecast format
df_nhits = df_ml.copy()
df_nhits['unique_id'] = 'SL_Tourism'  # Required by NeuralForecast
df_nhits = df_nhits.rename(columns={'Date': 'ds', 'Arrivals': 'y'})

# Select exogenous variables
exog_vars = ['month_sin', 'month_cos', 'easter_attacks', 
             'covid_period', 'economic_crisis', 'recovery_index']

# Initialize model
models = [
    NHITS(
        h=12,  # Forecast horizon
        input_size=24,  # Historical window
        loss=MSE(),
        max_steps=500,
        scaler_type='standard',
        futr_exog_list=exog_vars,  # Future exogenous variables
    )
]

# Train model
nf = NeuralForecast(models=models, freq='MS')
nf.fit(df=df_nhits)

# Forecast (you need to provide future exogenous values)
future_exog = create_future_exogenous(periods=12)  # Implement this
forecast = nf.predict(futr_df=future_exog)
```

## Example 4: Chronos (Zero-Shot)

Chronos is pre-trained and doesn't use engineered features.

```python
from chronos import ChronosPipeline
import torch

# Load original data (no features needed)
df = pd.read_csv('data/processed/monthly_tourist_arrivals_filtered.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Prepare time series
context = torch.tensor(df['Arrivals'].values)

# Load pre-trained Chronos model
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)

# Forecast
forecast = pipeline.predict(
    context=context,
    prediction_length=12,
    num_samples=100,  # For probabilistic forecasting
)

# Get point forecast (median) and prediction intervals
point_forecast = np.median(forecast[0].numpy(), axis=0)
lower_bound = np.percentile(forecast[0].numpy(), 10, axis=0)
upper_bound = np.percentile(forecast[0].numpy(), 90, axis=0)
```

## Example 5: Creating Features for Production Pipeline

For production forecasting, you need to create features for future dates.

```python
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Create future dates (next 12 months from last known date)
last_date = pd.to_datetime('2024-07-01')
future_dates = [last_date + relativedelta(months=i) for i in range(1, 13)]

# Create future dataframe
df_future = pd.DataFrame({
    'Date': future_dates,
    'Arrivals': [np.nan] * 12  # Unknown, will be predicted
})

# Create features (interventions are deterministically known)
from scripts.feature_engineering import create_ml_features

df_future_features = create_ml_features(
    df_future,
    include_lags=False,  # Can't use lags for pure future prediction
    include_rolling=False
)

# For features with historical dependencies (lags), you would:
# 1. Combine with historical data
# 2. Create features
# 3. Extract only the future rows

df_combined = pd.concat([df_historical, df_future], ignore_index=True)
df_all_features = create_ml_features(df_combined, include_lags=True)
df_future_with_lags = df_all_features.tail(12)
```

## Example 6: Custom Feature Selection

Get feature names programmatically.

```python
from scripts.feature_engineering import get_feature_names

# Get all feature categories
core_features = get_feature_names('core')
print(f"Core features: {core_features}")

intervention_features = get_feature_names('intervention')
print(f"Intervention features: {intervention_features}")

cyclical_features = get_feature_names('cyclical')
print(f"Cyclical features: {cyclical_features}")

lag_features = get_feature_names('lag')
print(f"Lag features: {lag_features}")

# Combine for custom feature set
my_features = core_features + cyclical_features + intervention_features
print(f"\nCustom feature set: {my_features}")
```

## Tips and Best Practices

1. **Always split data chronologically** - Never shuffle time series data
2. **Scale features for neural networks** - Use MinMaxScaler or StandardScaler
3. **Prophet handles seasonality internally** - Don't need lag features for Prophet
4. **Handle NaN values appropriately** - Lag features create NaN at the start
5. **Future regressors must be known** - Plan how to provide future intervention values
6. **Test on validation set** - Use 2023 as test set for final evaluation

## Quick Reference

| Model | Best Dataset | Key Features |
|-------|-------------|--------------|
| Prophet | `features_prophet.csv` | Interventions as regressors |
| LSTM | `features_ml_clean.csv` | All features, scaled |
| Chronos | Original CSV | No features (univariate) |
| N-HiTS | `features_ml_clean.csv` | Exogenous variables |

For more details, see `FEATURE_ENGINEERING.md`.
