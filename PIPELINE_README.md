# End-to-End Training & Evaluation Pipeline

This directory contains the complete end-to-end training and evaluation pipeline for the Sri Lanka Tourism Forecasting project.

## Overview

The pipeline implements a comprehensive workflow for:
- Generating minimal feature sets optimized for each model type
- Splitting data chronologically with fixed date boundaries
- Training Prophet, LSTM, and Chronos models
- Evaluating and comparing model performance
- Validating predictions with automated tests
- Generating reports and visualizations

## Quick Start

### Run the Complete Pipeline

```bash
# Run all models
python scripts/run_pipeline.py

# Skip specific models
python scripts/run_pipeline.py --skip-prophet
python scripts/run_pipeline.py --skip-lstm
python scripts/run_pipeline.py --skip-chronos

# Force retrain even if artifacts exist
python scripts/run_pipeline.py --rebuild
```

### Run Individual Components

```bash
# 1. Generate minimal features
python scripts/generate_minimal_features.py

# 2. Train Prophet model
python scripts/train_prophet.py

# 3. Train LSTM model
python scripts/train_lstm.py

# 4. Run Chronos inference
python scripts/train_chronos.py

# 5. Run prediction tests
python scripts/prediction_tests.py
```

## Data Splits

The pipeline uses **fixed chronological date boundaries**:

- **Train:** 2017-01-01 to 2022-12-01 (72 samples)
- **Validation:** 2023-01-01 to 2023-12-01 (12 samples)
- **Test:** 2024-01-01 to 2024-07-01 (7 samples)

**Rationale:** Keeps recent 18+ months for forward-looking performance checks while leaving final 7 months as pure holdout.

## Input Files

The pipeline requires these minimal feature files (auto-generated):

1. **`data/processed/minimal_features_prophet.csv`**
   - Columns: `ds`, `y`, `easter_attacks`, `covid_period`, `economic_crisis`
   - 91 rows (full time range)

2. **`data/processed/minimal_features_lstm.csv`**
   - Columns: `Date`, `Arrivals`, interventions, `month_sin`, `month_cos`, `Arrivals_lag_1`, `Arrivals_lag_12`
   - 79 rows (reduced due to lag features)

3. **`data/processed/chronos_series.csv`**
   - Columns: `Date`, `Arrivals`
   - 91 rows (univariate series)

## Model Implementations

### Prophet

- **Hyperparameters:** Grid search over `changepoint_prior_scale` and `seasonality_prior_scale`
- **Validation:** Rolling-origin validation (12 steps)
- **Regressors:** Intervention flags as external regressors
- **Output:** `models/prophet/final_model.pkl`

### LSTM

- **Architecture:** 2 LSTM layers (64→32) + Dense layers
- **Window Size:** 24 months
- **Horizon:** 1 month ahead
- **Scaling:** MinMaxScaler (fitted on train only)
- **Forecasting:** Both teacher forcing and autoregressive modes
- **Output:** `models/lstm/best_weights.h5`, `artifacts/lstm_scaler.pkl`

### Chronos

- **Mode:** Zero-shot (no fine-tuning)
- **Context:** Full train+validation series
- **Outputs:** Point forecast (mean) + prediction intervals (P10, P50, P90)
- **Note:** Current implementation uses stub for demonstration

## Evaluation Metrics

For each model, the pipeline reports:

- **RMSE** - Root Mean Squared Error
- **MAE** - Mean Absolute Error
- **MAPE** - Mean Absolute Percentage Error (primary ranking metric)
- **R²** - Coefficient of Determination

## Prediction Tests

Automated tests validate:

1. **No Future Leakage** - Train < Val < Test chronologically
2. **Prophet Regressor Alignment** - All required columns present
3. **LSTM Window Integrity** - Windows are chronological
4. **Forecast Length Consistency** - 7 months for test period
5. **Metric Reproducibility** - Stored vs. recomputed metrics match
6. **Non-negative Predictions** - No negative tourist arrivals
7. **Chronos Interval Coherency** - P10 ≤ P50 ≤ P90
8. **Autoregressive Drift** - Teacher vs. autoreg predictions reasonable

## Directory Structure

```
.
├── data/
│   └── processed/
│       ├── minimal_features_prophet.csv
│       ├── minimal_features_lstm.csv
│       ├── chronos_series.csv
│       └── splits.json
├── models/
│   ├── prophet/
│   │   └── final_model.pkl
│   ├── lstm/
│   │   └── best_weights.h5
│   └── chronos/
├── artifacts/
│   └── lstm_scaler.pkl
├── forecasts/
│   ├── prophet_test_forecast.csv
│   ├── prophet_test_plot.png
│   ├── lstm_test_forecast.csv
│   ├── lstm_test_plot.png
│   ├── chronos_test_forecast.csv
│   └── chronos_test_intervals.png
├── reports/
│   ├── prophet_metrics.json
│   ├── lstm_metrics.json
│   ├── chronos_metrics.json
│   ├── model_comparison.csv
│   └── summary.md
└── logs/
    └── pipeline.log
```

## Generated Outputs

### Forecast Files

All forecast files include actual values and predictions for the test period.

- **Prophet:** `ds`, `y_true`, `y_pred`, `residual`
- **LSTM:** `Date`, `y_true`, `y_pred_teacher`, `y_pred_autoreg`, `residual_teacher`, `residual_autoreg`
- **Chronos:** `Date`, `y_true`, `y_pred_mean`, `p10`, `p50`, `p90`

### Metrics Files

JSON files containing:
- Hyperparameters (if applicable)
- Test metrics (RMSE, MAE, MAPE, R², MSE)
- Validation metrics (Prophet, LSTM)

### Visualizations

- Line plots showing actual vs. predicted values
- Intervention periods shaded (Prophet)
- Prediction intervals (Chronos)

### Reports

- **`model_comparison.csv`** - Side-by-side metric comparison
- **`summary.md`** - Comprehensive summary with justifications

## Development

### Adding New Models

1. Create training script in `scripts/train_<model>.py`
2. Follow the pattern:
   - Load minimal features
   - Split chronologically
   - Train/tune model
   - Generate forecasts
   - Save artifacts and metrics
3. Update `run_pipeline.py` to include new model
4. Add tests to `prediction_tests.py`

### Debugging

- Check `logs/pipeline.log` for detailed execution logs
- Run individual scripts for isolated testing
- Use `--skip-*` flags to bypass slow models during development

## Notes

### Chronos Implementation

The current Chronos implementation is a **stub** for demonstration purposes. For production:

```bash
pip install chronos-forecasting
```

Then replace the stub in `train_chronos.py` with:

```python
from chronos import ChronosPipeline
pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-small")
forecast = pipeline.predict(context, prediction_length, num_samples)
```

### Performance

- Prophet: ~5-10 minutes (with rolling-origin validation)
- LSTM: ~10-20 minutes (depends on early stopping)
- Chronos: <1 minute (stub), ~5 minutes (full)

### Reproducibility

- Fixed random seeds: 42 (numpy, tensorflow)
- No shuffling in time series splits
- All hyperparameters saved in metrics files
- Environment details in `reports/environment.txt` (optional)

## References

- **Prophet:** [facebook/prophet](https://github.com/facebook/prophet)
- **LSTM:** TensorFlow/Keras
- **Chronos:** [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)

## License

See project LICENSE file.
