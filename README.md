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

## Project Structure

```
.
├── data/
│   ├── raw/              # Downloaded PDF files (not tracked in git)
│   └── processed/        # Cleaned CSV datasets (not tracked in git)
├── scripts/
│   ├── ingest_tourism_data.py  # Data ingestion pipeline
│   ├── data_loader.py          # Data loading utilities
│   ├── preprocessing.py        # Data preprocessing functions
│   └── evaluation.py           # Model evaluation utilities
├── notebooks/            # Jupyter notebooks for analysis
└── requirements.txt      # Python dependencies

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
