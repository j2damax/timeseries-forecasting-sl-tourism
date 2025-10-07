# Feature Engineering Documentation

## Overview

This document describes the feature engineering implementation for the Sri Lanka Tourism Forecasting project.

## Design Principles

The feature engineering follows these key principles based on project constraints:

1. **Minimal Features**: With only 91 observations (Jan 2017 - Jul 2024), we avoid overfitting by keeping features minimal and interpretable.

2. **No Data Leakage**: All features are created ensuring:
   - Rolling statistics use only past data
   - Intervention features are deterministically known at forecast time
   - No future information used in past predictions

3. **Parity Layer**: A shared core feature set that all models can access, with model-specific additions.

4. **Reusability**: All feature engineering code is modular and can be reused in production pipelines.

## Feature Categories

### 1. Core Temporal Features (Parity Layer)

Available to all models:

- **year**: Calendar year (2017-2024)
- **month**: Month number (1-12)
- **quarter**: Quarter (1-4)

These provide basic time context for all models.

### 2. Cyclical Encoding (Neural Networks)

For deep learning models (LSTM, N-HiTS):

- **month_sin**: Sine transformation of month (preserves cyclical nature)
- **month_cos**: Cosine transformation of month (preserves cyclical nature)

Formula: `sin(2π × month / 12)` and `cos(2π × month / 12)`

This encoding helps neural networks understand that December (12) is close to January (1).

### 3. Intervention Features

Binary flags for known structural shocks:

- **easter_attacks**: 1 for April 2019 (Easter Sunday attacks), 0 otherwise
- **covid_period**: 1 for March 2020 - December 2021, 0 otherwise
- **economic_crisis**: 1 for January 2022 onwards, 0 otherwise

These are deterministically known at forecast time - we know future dates will have specific values for these flags.

### 4. Recovery Index

Continuous measure of post-shock recovery:

- **months_since_covid**: Months elapsed since March 2020 (clipped to 0 for earlier dates)
- **recovery_index**: Normalized recovery measure (0-1 scale, capped at 60 months)

Formula: `recovery_index = min(months_since_covid / 60, 1.0)`

This provides a smooth, continuous measure of the recovery trajectory.

### 5. Lag Features (ML/DL Models)

Lagged values of tourist arrivals:

- **Arrivals_lag_1**: Previous month's arrivals
- **Arrivals_lag_3**: Arrivals 3 months ago
- **Arrivals_lag_6**: Arrivals 6 months ago  
- **Arrivals_lag_12**: Arrivals 12 months ago (year-over-year)

**Important**: Lag features create NaN values for the first 12 months of the dataset.

### 6. Rolling Window Statistics (Optional)

Moving averages and volatility:

- **Arrivals_rolling_mean_3**: 3-month rolling average
- **Arrivals_rolling_std_3**: 3-month rolling standard deviation
- **Arrivals_rolling_mean_12**: 12-month rolling average
- **Arrivals_rolling_std_12**: 12-month rolling standard deviation

**Note**: These use only past data (no future leakage) and create NaN for initial windows.

## Generated Datasets

The feature engineering process creates 4 dataset variants:

### 1. Full Features (`monthly_tourist_arrivals_features_full.csv`)

- **Shape**: 91 rows × 20 columns
- **Purpose**: Complete feature set for exploration
- **Contains**: All features including lags and rolling statistics
- **Missing Values**: 48 (from lag features)
- **Use Case**: Analysis and feature selection

### 2. Prophet Features (`monthly_tourist_arrivals_features_prophet.csv`)

- **Shape**: 91 rows × 9 columns
- **Purpose**: Optimized for Facebook Prophet
- **Contains**: 
  - Date (as 'ds')
  - Arrivals (as 'y')
  - Core temporal features
  - Intervention features
  - Recovery index
- **Missing Values**: 0
- **Use Case**: Direct input to Prophet model (interventions as regressors)

**Columns**: Date, Arrivals, year, month, quarter, easter_attacks, covid_period, economic_crisis, recovery_index

### 3. ML Features with NaN (`monthly_tourist_arrivals_features_ml.csv`)

- **Shape**: 91 rows × 16 columns
- **Purpose**: ML/DL models that can handle missing values
- **Contains**: Core + cyclical + interventions + lags (no rolling)
- **Missing Values**: 22 (from lag features)
- **Use Case**: Models with custom NaN handling

### 4. ML Features Clean (`monthly_tourist_arrivals_features_ml_clean.csv`)

- **Shape**: 79 rows × 16 columns
- **Purpose**: ML/DL models requiring complete data
- **Contains**: Same as ML features, but with first 12 rows dropped
- **Missing Values**: 0
- **Date Range**: Jan 2018 - Jul 2024 (79 months)
- **Use Case**: LSTM, N-HiTS, other neural networks

**Columns**: Date, Arrivals, year, month, quarter, month_sin, month_cos, easter_attacks, covid_period, economic_crisis, months_since_covid, recovery_index, Arrivals_lag_1, Arrivals_lag_3, Arrivals_lag_6, Arrivals_lag_12

## Usage

### Command Line Interface (CLI)

Generate feature sets directly from the command line:

```bash
# Generate all feature sets (default)
cd scripts
python feature_engineering.py

# Generate specific feature set
python feature_engineering.py --type ml --drop-na
python feature_engineering.py --type prophet

# Custom input/output
python feature_engineering.py --input my_data.csv --output my_features.csv

# ML features with options
python feature_engineering.py --type ml --no-lags --with-rolling
```

### In Notebooks

```python
import pandas as pd
import sys
sys.path.append('../scripts')
from feature_engineering import create_all_features, create_prophet_features, create_ml_features

# Load raw data
df = pd.read_csv('../data/processed/monthly_tourist_arrivals_filtered.csv')

# Create features
df_all = create_all_features(df)
df_prophet = create_prophet_features(df)
df_ml = create_ml_features(df, include_lags=True, drop_na=True)
```

### In Production/Prediction Pipeline

```python
from scripts.feature_engineering import create_ml_features

# For forecasting with new data
df_new = pd.read_csv('new_arrivals_data.csv')
df_features = create_ml_features(df_new, include_lags=True, drop_na=False)

# Apply your trained model
predictions = model.predict(df_features)
```

## Model-Specific Recommendations

### Prophet

- **Use**: `monthly_tourist_arrivals_features_prophet.csv`
- **Features**: Prophet handles seasonality internally
- **Regressors**: Use intervention features and recovery_index as external regressors
- **No lags needed**: Prophet uses its own autoregressive component

### LSTM

- **Use**: `monthly_tourist_arrivals_features_ml_clean.csv`
- **Features**: All temporal + cyclical + interventions + lags
- **Scaling**: Apply MinMaxScaler before training
- **Sequence**: Create sequences from lag features or time windows

### Chronos (Transformer)

- **Use**: Original `monthly_tourist_arrivals_filtered.csv` (univariate)
- **Features**: Chronos is zero-shot, uses only the target series
- **No feature engineering**: Pre-trained on diverse time series

### N-HiTS

- **Use**: `monthly_tourist_arrivals_features_ml_clean.csv` (can use univariate or multivariate)
- **Features**: Can use exogenous variables (interventions, cyclical encoding)
- **Scaling**: Apply MinMaxScaler
- **Format**: Check N-HiTS/NeuralForecast specific format requirements

## Future Considerations

When deploying these models for forecasting future months:

1. **Temporal Features**: Can be computed for any future date
2. **Cyclical Features**: Can be computed for any future date
3. **Intervention Features**: Must be specified for future (e.g., will there be a crisis?)
4. **Recovery Index**: Can be computed deterministically for future dates
5. **Lag Features**: Use most recent actual values for lags
6. **Rolling Features**: Compute using actual historical data only

## Validation

All features have been validated to ensure:

- ✅ No data leakage
- ✅ No future information in historical predictions
- ✅ All intervention features are deterministic and known at forecast time
- ✅ Minimal feature count to avoid overfitting
- ✅ Compatible with small dataset size (91 observations)

## References

- Notebook: `notebooks/02_Feature_Engineering.ipynb`
- Functions: `scripts/preprocessing.py`
- Pipeline: `scripts/feature_engineering.py`
- Original data: `data/processed/monthly_tourist_arrivals_filtered.csv`
