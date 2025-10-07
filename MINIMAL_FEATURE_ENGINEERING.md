# Minimal Feature Engineering Documentation

## Overview

This document describes the **simplified, minimal feature engineering** implementation for the Sri Lanka Tourism Forecasting project, focused exclusively on **Prophet, LSTM, and Chronos** models.

## Design Philosophy

### Core Principles

1. **Extreme Minimalism**: With only 91 monthly observations (2017-01 to 2024-07), we keep features to an absolute minimum to avoid overfitting
2. **No Optional Features**: Removed all "nice-to-have" features that don't provide clear signal
3. **Model-Focused**: Tailored specifically for Prophet, LSTM, and Chronos - no other models
4. **Zero Leakage**: All features are deterministically known at forecast time
5. **Consistent Splits**: Fixed chronological splits for reproducible evaluation

## What Changed from Previous Version

### ❌ Removed Features

- **Rolling statistics** (too few observations, overfitting risk)
- **recovery_index** (replaced by binary recovery flag)
- **months_since_covid** (replaced by binary recovery flag)
- **Lag features 3 and 6** (kept only 1 and 12)
- **Year, month, quarter** temporal features (Prophet handles internally, LSTM uses cyclical encoding)
- **All N-HiTS specific features** (not in scope)

### ✅ Kept/Added Features

- **Binary intervention flags**: easter_impact, covid, econ_crisis, recovery
- **Minimal lags**: lag_1 (short-term persistence), lag_12 (annual seasonality proxy)
- **Cyclical month encoding**: month_sin, month_cos (for LSTM only)
- **Raw arrivals**: For Chronos (zero-shot model)

## Feature Set Details

### Universal Features (All Models)

**Intervention Flags (Binary 0/1)**:
- `easter_impact`: 2019-04 to 2019-09 (6 months) - Easter attacks aftermath
- `covid`: 2020-03 to 2021-12 (22 months) - COVID-19 pandemic period
- `econ_crisis`: 2022-04 to 2022-09 (6 months) - Economic crisis peak
- `recovery`: 2022-11 onwards (21 months in dataset) - Post-crisis recovery

### Model-Specific Features

**Prophet**:
- Uses intervention flags as external regressors
- Prophet handles seasonality internally
- No lag features needed

**LSTM**:
- All 4 intervention flags
- `Arrivals_lag_1`, `Arrivals_lag_12`
- `month_sin`, `month_cos` (cyclical month encoding)
- All features scaled with MinMaxScaler

**Chronos**:
- Raw `Arrivals` series only (zero-shot pre-trained model)
- No feature engineering required

## Data Splits

### Fixed Chronological Split

```
Train:      2018-01 to 2022-12  (60 months)
Validation: 2023-01 to 2023-12  (12 months)
Test:       2024-01 to 2024-07  (7 months)
```

**Note**: Dataset starts from 2018-01 (not 2017-01) due to `lag_12` requiring 12 months of history.

**Rationale**:
- Train set includes full pre-crisis, crisis, and early recovery periods
- Validation is the pure recovery year (2023)
- Test is the latest available data for unbiased holdout evaluation

## Generated Datasets

### 1. `engineered_features.csv`

**Purpose**: Base feature set with all engineered features  
**Shape**: 79 rows × 10 columns  
**Date Range**: 2018-01 to 2024-07  

**Columns**:
```
Date, Arrivals, easter_impact, covid, econ_crisis, recovery,
month_sin, month_cos, Arrivals_lag_1, Arrivals_lag_12
```

### 2. `prophet_regressors.csv`

**Purpose**: Prophet-formatted data with regressors  
**Shape**: 91 rows × 6 columns  
**Date Range**: 2017-01 to 2024-07 (no lag-based removal)  

**Columns**:
```
ds, y, easter_impact, covid, econ_crisis, recovery
```

**Usage**:
```python
from prophet import Prophet

df = pd.read_csv('data/processed/prophet_regressors.csv')
df['ds'] = pd.to_datetime(df['ds'])

model = Prophet()
model.add_regressor('easter_impact')
model.add_regressor('covid')
model.add_regressor('econ_crisis')
model.add_regressor('recovery')

model.fit(df)
```

### 3. LSTM Sequence Files

**Files**: `lstm_train.npz`, `lstm_val.npz`, `lstm_test.npz`

Each contains:
- `X`: Input sequences (samples, 24 timesteps, 9 features)
- `y`: Target values (samples,)

**Shapes**:
- Train: X=(36, 24, 9), y=(36,)
- Val: X=(12, 24, 9), y=(12,)
- Test: X=(7, 24, 9), y=(7,)

**Feature Order** (9 features):
```
[Arrivals, Arrivals_lag_1, Arrivals_lag_12, month_sin, month_cos,
 easter_impact, covid, econ_crisis, recovery]
```

**Window Configuration**:
- Input window: 24 months (2 full annual cycles)
- Forecast horizon: 1 month ahead
- All features scaled with MinMaxScaler (fitted on train only)

**Usage**:
```python
import numpy as np

# Load sequences
train = np.load('data/processed/lstm_train.npz')
X_train, y_train = train['X'], train['y']

# Train LSTM
model = Sequential([
    LSTM(50, activation='relu', input_shape=(24, 9)),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=8)
```

### 4. `scaler.pkl`

**Purpose**: MinMaxScaler fitted on training data  
**Scaled Features**: Arrivals, Arrivals_lag_1, Arrivals_lag_12, month_sin, month_cos  
**Binary Features**: Kept as 0/1 (not scaled)  

**Usage**:
```python
import pickle

with open('data/processed/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Inverse transform predictions
predictions_scaled = model.predict(X_test)
predictions = scaler.inverse_transform(predictions_scaled)
```

### 5. Chronos Arrays

**Files**: `chronos_context.npy`, `chronos_test.npy`

- **Context**: 84 values (2017-01 to 2023-12) - training context
- **Test**: 7 values (2024-01 to 2024-07) - ground truth for evaluation

**Usage**:
```python
import numpy as np
import torch
from chronos import ChronosPipeline

context = np.load('data/processed/chronos_context.npy')
test_truth = np.load('data/processed/chronos_test.npy')

# Convert to tensor
context_tensor = torch.tensor(context, dtype=torch.float)

# Load Chronos model
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
    torch_dtype=torch.bfloat16
)

# Forecast
forecast = pipeline.predict(
    context=context_tensor,
    prediction_length=7,
    num_samples=100
)

# Evaluate against test_truth
point_forecast = np.median(forecast[0].numpy(), axis=0)
```

## Generating the Features

To regenerate all feature datasets:

```bash
python scripts/generate_minimal_features.py
```

This will create:
1. `engineered_features.csv`
2. `prophet_regressors.csv`
3. `lstm_train.npz`, `lstm_val.npz`, `lstm_test.npz`
4. `scaler.pkl`
5. `chronos_context.npy`, `chronos_test.npy`

## Validation

Run the test suite to validate all features:

```bash
python test_minimal_features.py
```

Tests validate:
- ✓ Minimal feature structure (10 columns)
- ✓ New intervention periods are correct
- ✓ Only lag_1 and lag_12 exist
- ✓ No rolling statistics
- ✓ Prophet format is correct
- ✓ LSTM sequences have correct shape
- ✓ Data splits match specification
- ✓ All output files exist

## API Reference

### Feature Engineering Functions

```python
from scripts.feature_engineering import (
    create_minimal_features,
    create_prophet_data,
    create_lstm_data,
    split_train_val_test,
    create_lstm_sequences,
    get_feature_columns
)
```

#### `create_minimal_features(df, date_column='Date', target_column='Arrivals')`

Creates minimal feature set with interventions, cyclical encoding, and 2 lag features.

**Returns**: DataFrame with 10 columns

#### `create_prophet_data(df, date_column='Date', target_column='Arrivals')`

Creates Prophet-formatted data (ds, y, regressors).

**Returns**: DataFrame with 6 columns

#### `create_lstm_data(df, date_column='Date', target_column='Arrivals', drop_na=True)`

Creates LSTM-ready features, optionally dropping NaN from lags.

**Returns**: DataFrame (79 rows if drop_na=True, 91 if False)

#### `split_train_val_test(df, date_column='Date', train_end='2022-12-01', val_end='2023-12-01')`

Splits data chronologically into train/val/test.

**Returns**: Tuple of 3 DataFrames

#### `create_lstm_sequences(df, feature_columns, target_column, window_size=24, forecast_horizon=1)`

Creates sliding window sequences for LSTM.

**Returns**: Tuple of (X, y) numpy arrays

#### `get_feature_columns()`

Returns dictionary of feature column lists for different purposes.

**Returns**: Dict with keys: 'interventions', 'cyclical', 'lags', 'lstm_features', 'prophet_regressors'

## Metrics and Evaluation

Use the standard evaluation metrics from `scripts/evaluation.py`:

```python
from scripts.evaluation import calculate_metrics

metrics = calculate_metrics(y_true, y_pred)
# Returns: MSE, RMSE, MAE, R², MAPE
```

## Future Extensions

If extending forecast beyond 2024-07:

1. **Intervention flags for future months**:
   - `easter_impact`, `covid`, `econ_crisis`: Set to 0 (events ended)
   - `recovery`: Set to 1 (assume recovery continues)

2. **Lag features**: Will be available from previous predictions

3. **Cyclical features**: Generated from target month

## Summary

This minimal feature engineering approach provides:

✓ **Simplicity**: Only essential features, easy to understand and maintain  
✓ **Robustness**: Low risk of overfitting with small dataset  
✓ **Focused**: Optimized for Prophet, LSTM, Chronos only  
✓ **Reproducible**: Fixed splits and deterministic features  
✓ **Production-Ready**: Clean API and reusable functions  

---

**Last Updated**: 2024  
**Dataset Range**: 2017-01 to 2024-07 (91 months)  
**Effective Range** (after lag_12): 2018-01 to 2024-07 (79 months)
