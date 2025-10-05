# Copilot Instructions for Sri Lanka Tourism Forecasting Project

## Project Overview

This is a time series forecasting project for monthly tourist arrivals to Sri Lanka, featuring a complete ML pipeline from web scraping to model comparison. The project implements multiple forecasting models (Prophet, LSTM, Chronos, N-HiTS) to predict tourism trends.

## Key Architecture Patterns

### Data Pipeline Structure

- **Data Flow**: Raw PDFs → `data/raw/` → Extracted CSV → `data/processed/monthly_tourist_arrivals.csv`
- **Entry Point**: Always run `python scripts/ingest_tourism_data.py` first to generate the base dataset
- **Data Format**: Standardized to `Date` (datetime) and `Arrivals` (int) columns

### Notebook Workflow (Sequential Processing)

```
01_Data_Ingestion_and_EDA.ipynb      # Load and explore data
02_Feature_Engineering.ipynb         # Create time-based and lag features
03_Model_Development_Prophet.ipynb   # Facebook Prophet implementation
04_Model_Development_LSTM.ipynb      # Deep learning with TensorFlow
05_Model_Development_Chronos.ipynb   # Amazon's Chronos transformer
06_Model_Development_Novel.ipynb     # Custom N-HiTS model
07_Model_Comparison_and_Analysis.ipynb # Evaluate all models
```

### Module Organization

- `scripts/data_loader.py`: Use `load_csv_data()` with `date_column='Date'` parameter
- `scripts/preprocessing.py`: Feature engineering functions (time features, lags, scaling)
- `scripts/evaluation.py`: Standard metrics (`calculate_metrics()` returns MSE, RMSE, MAE, R2, MAPE)

## Critical Developer Workflows

### Data Ingestion (Required First Step)

```bash
python scripts/ingest_tourism_data.py
```

This scrapes SLTDA website, downloads PDFs, extracts data via pdfplumber, and creates the master CSV. The script handles various PDF formats and gracefully skips corrupted files.

### Model Development Pattern

1. Load data using `scripts.data_loader.load_csv_data()`
2. Apply preprocessing from `scripts.preprocessing` module
3. Split data chronologically (not randomly - this is time series!)
4. Train model with consistent hyperparameter tuning
5. Evaluate using `scripts.evaluation.calculate_metrics()`

### Dependencies Management

- Core stack: pandas, numpy, matplotlib, scikit-learn
- Time series: prophet, statsmodels
- Deep learning: tensorflow, torch
- Web scraping: requests, beautifulsoup4, pdfplumber

## Project-Specific Conventions

### Data Handling

- **Always preserve chronological order** - never shuffle time series data
- **Standard column names**: `Date` (datetime), `Arrivals` (target variable)
- **Missing values**: Use forward fill (`method='ffill'`) as default for tourism data continuity

### Model Implementation

- Each model gets its own notebook (03-06) following the same structure
- Use `scripts/evaluation.py` for consistent metrics across all models
- Save model artifacts to avoid re-training during comparison phase

### Feature Engineering Patterns

- Time features: Extract year, month, quarter, day_of_week from `Date` column
- Lag features: Create 1, 3, 6, 12 month lags for tourism seasonality
- Scaling: Use MinMaxScaler for neural networks, keep raw values for Prophet/statistical models

## Integration Points

### External Data Sources

- **SLTDA Website**: `https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports`
- **PDF Processing**: Uses pdfplumber with regex patterns to extract month/year/arrivals data
- **Error Handling**: Gracefully handles various PDF formats and corrupted files

### Model Comparison Framework

- All models output predictions in same format for `07_Model_Comparison_and_Analysis.ipynb`
- Use consistent train/validation/test splits across models
- Generate comparable visualizations and metrics tables

### Development Environment

- Notebooks are designed for interactive development and experimentation
- Scripts provide reusable functions for production-like workflows
- Use `%matplotlib inline` in notebooks for consistent plotting

## Common Pitfalls to Avoid

- Don't randomly split time series data - always split chronologically
- Don't forget to run data ingestion script first - processed CSV is not in git
- Don't mix scaling approaches between statistical and neural network models
- Don't ignore seasonality - tourism data has strong monthly/yearly patterns
