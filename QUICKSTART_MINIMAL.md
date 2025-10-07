# Quick Start Guide - Minimal Feature Engineering

This guide shows how to quickly get started with the minimal feature engineering pipeline for Prophet, LSTM, and Chronos models.

## Prerequisites

```bash
pip install pandas numpy scikit-learn
# For specific models:
pip install prophet  # For Prophet
pip install tensorflow  # For LSTM
pip install chronos  # For Chronos
```

## Step 1: Generate Features

Run the feature generation script once:

```bash
python scripts/generate_minimal_features.py
```

This creates all necessary datasets in `data/processed/`.

## Step 2: Model-Specific Usage

### Prophet Model

```python
import pandas as pd
from prophet import Prophet

# Load Prophet data
df = pd.read_csv('data/processed/prophet_regressors.csv')
df['ds'] = pd.to_datetime(df['ds'])

# Split train/val/test
train = df[df['ds'] <= '2022-12-01']
val = df[(df['ds'] > '2022-12-01') & (df['ds'] <= '2023-12-01')]
test = df[df['ds'] > '2023-12-01']

# Initialize Prophet
model = Prophet()

# Add intervention regressors
model.add_regressor('easter_impact')
model.add_regressor('covid')
model.add_regressor('econ_crisis')
model.add_regressor('recovery')

# Train on train set
model.fit(train)

# Create future dataframe with regressors
future = val[['ds', 'easter_impact', 'covid', 'econ_crisis', 'recovery']]

# Predict
forecast = model.predict(future)

# Evaluate
from scripts.evaluation import calculate_metrics
metrics = calculate_metrics(val['y'].values, forecast['yhat'].values)
print(f"MAPE: {metrics['mape']:.2f}%")
print(f"RMSE: {metrics['rmse']:.0f}")
```

### LSTM Model

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scripts.evaluation import calculate_metrics

# Load pre-created sequences
train = np.load('data/processed/lstm_train.npz')
val = np.load('data/processed/lstm_val.npz')
test = np.load('data/processed/lstm_test.npz')

X_train, y_train = train['X'], train['y']
X_val, y_val = val['X'], val['y']
X_test, y_test = test['X'], test['y']

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(24, 9)),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=8,
    verbose=1
)

# Predict on test set
y_pred_scaled = model.predict(X_test)

# Inverse transform to original scale
import pickle
with open('data/processed/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Create dummy array for inverse transform (scaler expects 5 columns)
dummy = np.zeros((len(y_pred_scaled), 5))
dummy[:, 0] = y_pred_scaled.flatten()  # Arrivals is first column
y_pred = scaler.inverse_transform(dummy)[:, 0]

# Also inverse transform y_test
dummy_test = np.zeros((len(y_test), 5))
dummy_test[:, 0] = y_test
y_test_original = scaler.inverse_transform(dummy_test)[:, 0]

# Evaluate
metrics = calculate_metrics(y_test_original, y_pred)
print(f"MAPE: {metrics['mape']:.2f}%")
print(f"RMSE: {metrics['rmse']:.0f}")
```

### Chronos Model

```python
import numpy as np
import torch
from chronos import ChronosPipeline

# Load context and test data
context = np.load('data/processed/chronos_context.npy')
test_truth = np.load('data/processed/chronos_test.npy')

# Convert to tensor
context_tensor = torch.tensor(context, dtype=torch.float)

# Load pre-trained Chronos
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
    torch_dtype=torch.bfloat16
)

# Generate forecast (7 months)
forecast = pipeline.predict(
    context=context_tensor,
    prediction_length=7,
    num_samples=100
)

# Get point forecast (median)
point_forecast = np.median(forecast[0].numpy(), axis=0)

# Evaluate
from scripts.evaluation import calculate_metrics
metrics = calculate_metrics(test_truth, point_forecast)
print(f"MAPE: {metrics['mape']:.2f}%")
print(f"RMSE: {metrics['rmse']:.0f}")

# Get prediction intervals
lower = np.percentile(forecast[0].numpy(), 10, axis=0)
upper = np.percentile(forecast[0].numpy(), 90, axis=0)
```

## Step 3: Custom Feature Creation

If you need to create features programmatically:

```python
import pandas as pd
from scripts.feature_engineering import (
    create_minimal_features,
    create_prophet_data,
    create_lstm_data,
    split_train_val_test
)

# Load base data
df = pd.read_csv('data/processed/monthly_tourist_arrivals_filtered.csv')

# Create minimal features
df_features = create_minimal_features(df)
print(df_features.head())

# For Prophet
df_prophet = create_prophet_data(df)

# For LSTM
df_lstm = create_lstm_data(df, drop_na=True)

# Split data
train, val, test = split_train_val_test(df_lstm)
```

## Data Splits Reference

All models use the same chronological split:

```
Train:      2018-01 to 2022-12  (60 months)
Validation: 2023-01 to 2023-12  (12 months)  
Test:       2024-01 to 2024-07  (7 months)
```

## Feature Summary

| Model | Features Used | Input Format |
|-------|--------------|--------------|
| Prophet | 4 intervention flags | CSV with ds, y, regressors |
| LSTM | 9 features (arrivals, 2 lags, 2 cyclical, 4 interventions) | NPZ with sequences (24×9) |
| Chronos | Raw arrivals only | NumPy array |

## Troubleshooting

**Issue**: LSTM predictions are all similar values  
**Solution**: Check if scaler was fitted on training data only. Never fit scaler on test data.

**Issue**: Prophet gives warnings about regressor values  
**Solution**: Ensure future dataframe includes all regressor columns with appropriate values.

**Issue**: Chronos forecast length mismatch  
**Solution**: Set `prediction_length` parameter to match your desired forecast horizon.

## Next Steps

1. **Hyperparameter tuning**: Tune model-specific parameters
2. **Ensemble methods**: Combine predictions from multiple models
3. **Extended forecasts**: Forecast beyond 2024-07 (remember to set intervention flags appropriately)
4. **Visualization**: Plot predictions vs actuals with confidence intervals

See `MINIMAL_FEATURE_ENGINEERING.md` for detailed documentation.
