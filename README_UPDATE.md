# Update to README.md - Minimal Feature Engineering

## Suggested Addition to README.md

Add the following section to the README.md file to reflect the new minimal feature engineering approach:

---

## Feature Engineering (Updated)

This project now uses a **streamlined, minimal feature engineering** approach focused exclusively on **Prophet, LSTM, and Chronos** models.

### Quick Start

Generate all feature datasets:

```bash
python scripts/generate_minimal_features.py
```

This creates:
- `engineered_features.csv` - Base minimal features (79 rows × 10 columns)
- `prophet_regressors.csv` - Prophet-formatted data (91 rows × 6 columns)
- `lstm_train/val/test.npz` - LSTM sequences with 24-month windows
- `chronos_context/test.npy` - Raw arrivals for Chronos model
- `scaler.pkl` - MinMaxScaler for LSTM predictions

### Minimal Feature Set

**Only 10 essential features**:
1. `Date` - Timestamp
2. `Arrivals` - Target variable
3. `easter_impact` - Easter attacks period (Apr-Sep 2019)
4. `covid` - COVID-19 pandemic (Mar 2020-Dec 2021)
5. `econ_crisis` - Economic crisis peak (Apr-Sep 2022)
6. `recovery` - Recovery period (Nov 2022 onwards)
7. `month_sin` - Cyclical month encoding (sine)
8. `month_cos` - Cyclical month encoding (cosine)
9. `Arrivals_lag_1` - 1-month lag
10. `Arrivals_lag_12` - 12-month lag (annual seasonality)

**What was removed**:
- Rolling statistics (4 features)
- Extra lag features (lag_3, lag_6)
- Recovery index (replaced by binary recovery flag)
- Temporal features (year, month, quarter)

### Data Splits

Fixed chronological split for consistent evaluation:

```
Train:      2018-01 to 2022-12  (60 months)
Validation: 2023-01 to 2023-12  (12 months)
Test:       2024-01 to 2024-07  (7 months)
```

### Model-Specific Usage

**Prophet**:
```python
df = pd.read_csv('data/processed/prophet_regressors.csv')
# Use easter_impact, covid, econ_crisis, recovery as regressors
```

**LSTM**:
```python
train = np.load('data/processed/lstm_train.npz')
X_train, y_train = train['X'], train['y']
# Shape: X=(36, 24, 9), y=(36,)
```

**Chronos**:
```python
context = np.load('data/processed/chronos_context.npy')
# 84 values for zero-shot forecasting
```

### Documentation

- **MINIMAL_FEATURE_ENGINEERING.md** - Complete technical documentation
- **QUICKSTART_MINIMAL.md** - Quick start guide with examples
- **IMPLEMENTATION_SUMMARY.md** - What changed and why

### Testing

Validate the feature engineering:

```bash
python test_minimal_features.py
```

All tests should pass ✓

---

## Alternative: Replace Existing Feature Engineering Section

If there's an existing "Feature Engineering" section in README.md, replace it with the above content to reflect the new minimal approach.
