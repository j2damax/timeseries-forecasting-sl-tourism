# Minimal Feature Engineering Pipeline

## Overview

This project uses a **minimal feature engineering approach** designed for small datasets (91 observations). The philosophy is to create only the most essential features that:

1. **Avoid overfitting** - Limited to 16-20 features maximum
2. **Prevent data leakage** - All features use only past data
3. **Are deterministic** - All exogenous features can be defined for future forecasts
4. **Are interpretable** - Every feature has clear business meaning

## Key Principle: Simplicity

With only 91 monthly observations (Jan 2017 - Jul 2024), we prioritize:
- ✅ Core temporal features (year, month, quarter)
- ✅ Known interventions (Easter attacks, COVID-19, economic crisis)
- ✅ Essential lag features (1, 3, 6, 12 months)
- ✅ Cyclical encoding for neural networks (month_sin, month_cos)
- ❌ Hundreds of engineered features (risk of overfitting)
- ❌ Complex feature interactions (not enough data)

## The Pipeline

### Single Entry Point: `scripts/feature_engineering.py`

This is the **only** feature engineering module needed. It provides:

1. **Command Line Interface** - Generate features instantly
2. **Python API** - Use functions in notebooks/scripts
3. **Multiple Output Formats** - Optimized for different models

### Usage

#### Option 1: CLI (Recommended)

```bash
cd scripts

# Generate all feature sets
python feature_engineering.py

# Generate specific set
python feature_engineering.py --type ml --drop-na
python feature_engineering.py --type prophet
```

#### Option 2: Python Functions

```python
from scripts.feature_engineering import (
    create_all_features,
    create_prophet_features,
    create_ml_features
)

df = pd.read_csv('data/processed/monthly_tourist_arrivals_filtered.csv')

# Generate features
df_ml = create_ml_features(df, include_lags=True, drop_na=True)
df_prophet = create_prophet_features(df)
```

## Feature Categories

### 1. Core Temporal (3 features)
- `year`, `month`, `quarter`
- Available to all models
- Basic time context

### 2. Cyclical Encoding (2 features)
- `month_sin`, `month_cos`
- For neural networks
- Preserves cyclical nature

### 3. Interventions (5 features)
- `easter_attacks` - April 2019 flag
- `covid_period` - Mar 2020 - Dec 2021 flag
- `economic_crisis` - Jan 2022+ flag
- `months_since_covid` - Continuous recovery metric
- `recovery_index` - Normalized recovery (0-1)

### 4. Lag Features (4 features, optional)
- `Arrivals_lag_1`, `Arrivals_lag_3`, `Arrivals_lag_6`, `Arrivals_lag_12`
- Captures temporal dependencies
- Only for ML/DL models

### 5. Rolling Statistics (4 features, optional)
- `Arrivals_rolling_mean_3`, `Arrivals_rolling_mean_12`
- `Arrivals_rolling_std_3`, `Arrivals_rolling_std_12`
- Uses only past data
- Rarely used (optional)

## Output Datasets

| File | Rows | Cols | Purpose |
|------|------|------|---------|
| `features_full.csv` | 91 | 20 | Complete set (exploration) |
| `features_prophet.csv` | 91 | 9 | Prophet-optimized |
| `features_ml.csv` | 91 | 16 | ML/DL with NaN |
| `features_ml_clean.csv` | 79 | 16 | ML/DL without NaN |

## Model-Specific Recommendations

### Facebook Prophet
- **Dataset**: `features_prophet.csv`
- **Features**: Interventions as regressors
- **Why**: Prophet handles seasonality internally

### LSTM
- **Dataset**: `features_ml_clean.csv`
- **Features**: All (temporal + cyclical + interventions + lags)
- **Why**: Needs complete data, benefits from engineered features

### Chronos (Zero-Shot)
- **Dataset**: Original CSV (univariate)
- **Features**: None
- **Why**: Pre-trained on diverse time series

### N-HiTS
- **Dataset**: `features_ml_clean.csv` or original
- **Features**: Can use exogenous variables or univariate
- **Why**: Flexible architecture

## Testing

All features are validated with comprehensive tests:

```bash
python test_feature_engineering.py
```

Tests verify:
1. ✓ Correct shapes and columns
2. ✓ No data leakage
3. ✓ Intervention periods are accurate
4. ✓ Cyclical encoding forms unit circle
5. ✓ All output files exist

## Why "Minimal"?

Traditional feature engineering might create:
- Polynomial features
- Interaction terms
- Many lag combinations
- Multiple rolling windows
- Fourier features
- Exponential smoothing

**We avoid these because:**
- 91 observations is too small
- Risk of overfitting is very high
- Models perform better with simple, interpretable features
- Easier to maintain and debug

## References

- **Code**: `scripts/feature_engineering.py`
- **Tests**: `test_feature_engineering.py`
- **Notebook**: `notebooks/02_Feature_Engineering.ipynb`
- **Docs**: `FEATURE_ENGINEERING.md`, `USAGE_EXAMPLES.md`
