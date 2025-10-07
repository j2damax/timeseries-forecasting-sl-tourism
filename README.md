# timeseries-forecasting-sl-tourism

Time series forecasting for Sri Lanka tourism data.

## Overview

This project provides a complete pipeline for analyzing and forecasting monthly tourist arrivals to Sri Lanka using data from the Sri Lanka Tourism Development Authority (SLTDA).

## Data Ingestion

The project includes an automated data ingestion pipeline that:
- Scrapes the SLTDA website for monthly tourist arrival reports
- Downloads PDF reports automatically
- Extracts tourist arrival statistics from PDFs
- Compiles data into a clean, time-series ready CSV format

### Running the Data Ingestion Pipeline

```bash
python scripts/ingest_tourism_data.py
```

This will:
1. Download all monthly reports from https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports
2. Extract month, year, and total arrivals from each PDF
3. Save the consolidated dataset to `data/processed/monthly_tourist_arrivals.csv`

For more details, see [scripts/README.md](scripts/README.md)

## Feature Engineering

The project uses a **minimal feature engineering approach** optimized for small datasets (91 observations):

- **Parity Layer**: Shared core features (year, month, quarter) for all models
- **Intervention Features**: Binary flags for structural shocks (Easter attacks 2019, COVID-19, Economic crisis)
- **Cyclical Encoding**: Sine/cosine transformations for neural networks
- **Lag Features**: 1, 3, 6, 12 month lags for temporal dependencies
- **Recovery Index**: Smooth continuous measure of post-COVID recovery

### Generated Datasets

Four dataset variants optimized for different model types:

1. `monthly_tourist_arrivals_features_full.csv` (91×20) - Complete feature set
2. `monthly_tourist_arrivals_features_prophet.csv` (91×9) - Prophet-optimized
3. `monthly_tourist_arrivals_features_ml.csv` (91×16) - ML/DL with NaN
4. `monthly_tourist_arrivals_features_ml_clean.csv` (79×16) - ML/DL without NaN

### Running Feature Engineering

```bash
# Generate all feature sets using the CLI
cd scripts
python feature_engineering.py

# Generate specific feature set
python feature_engineering.py --type ml --drop-na
python feature_engineering.py --type prophet

# Or run the feature engineering notebook
jupyter notebook notebooks/02_Feature_Engineering.ipynb

# Validate the implementation
python test_feature_engineering.py
```

For comprehensive documentation:
- **[MINIMAL_FEATURE_ENGINEERING.md](MINIMAL_FEATURE_ENGINEERING.md)** - Overview of minimal approach
- **[FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md)** - Complete feature documentation
- **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** - Model-specific usage examples
- **[FEATURE_ENGINEERING_SUMMARY.md](FEATURE_ENGINEERING_SUMMARY.md)** - Implementation summary

## Project Structure

```
.
├── data/
│   ├── raw/              # Downloaded PDF files (not tracked in git)
│   └── processed/        # Cleaned CSV datasets and engineered features
├── scripts/
│   ├── ingest_tourism_data.py   # Data ingestion pipeline
│   ├── data_loader.py           # Data loading utilities
│   ├── preprocessing.py         # Data preprocessing and feature functions
│   ├── feature_engineering.py   # Reusable feature engineering pipeline
│   └── evaluation.py            # Model evaluation utilities
├── notebooks/
│   ├── 01_Data_Ingestion_and_EDA.ipynb        # Data exploration
│   ├── 02_Feature_Engineering.ipynb           # Feature creation
│   ├── 03_Model_Development_Prophet.ipynb     # Prophet model
│   ├── 04_Model_Development_LSTM.ipynb        # LSTM model
│   ├── 05_Model_Development_Chronos.ipynb     # Chronos model
│   ├── 06_Model_Development_Novel.ipynb       # N-HiTS model
│   └── 07_Model_Comparison_and_Analysis.ipynb # Model comparison
├── test_feature_engineering.py  # Feature engineering validation tests
├── FEATURE_ENGINEERING.md       # Feature documentation
├── USAGE_EXAMPLES.md            # Model-specific examples
└── requirements.txt             # Python dependencies

```

## Installation

```bash
pip install -r requirements.txt
```

## Dependencies

- pandas - Data manipulation
- numpy - Numerical computing
- requests - HTTP requests for web scraping
- beautifulsoup4 - HTML parsing
- pdfplumber - PDF text extraction
- matplotlib, seaborn, plotly - Visualization
- scikit-learn - Machine learning utilities
- statsmodels, prophet - Time series forecasting
- tensorflow, pytorch - Deep learning models
