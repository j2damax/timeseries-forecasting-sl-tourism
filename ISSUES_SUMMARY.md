# GitHub Issues Summary for Tourism Forecasting Project

This document provides a quick reference for the 12 GitHub issues to be created for the Sri Lanka Tourism Forecasting project. For detailed implementation tasks, see `GITHUB_ISSUES_TEMPLATE.md`.

## Issue Creation Priority

### High Priority (Core Implementation)
1. **Issue 1**: Data Ingestion Pipeline Validation and Enhancement
2. **Issue 2**: Exploratory Data Analysis (EDA) Implementation  
3. **Issue 3**: Feature Engineering Implementation
4. **Issue 4**: Facebook Prophet Model Development
5. **Issue 5**: LSTM Model Development
6. **Issue 7**: N-HiTS Model Development (Novel Approach)
7. **Issue 8**: Model Comparison and Analysis

### Medium Priority (Supporting Features)
8. **Issue 6**: Amazon Chronos Model Development
9. **Issue 9**: Preprocessing Module Enhancement
10. **Issue 10**: Evaluation Module Enhancement
11. **Issue 11**: Documentation and README Updates

### Low Priority (Quality Assurance)
12. **Issue 12**: Testing and Validation Framework

---

## Quick Issue Reference

### Issue 1: Data Ingestion Pipeline Validation
- **Focus**: Validate and enhance `scripts/ingest_tourism_data.py`
- **Key Tasks**: Run pipeline, verify data quality, error handling
- **Deliverable**: Clean dataset from 2000-2024 in `data/processed/monthly_tourist_arrivals.csv`

### Issue 2: EDA Implementation  
- **Focus**: Complete `01_Data_Ingestion_and_EDA.ipynb`
- **Key Tasks**: Visualize time series, identify seasonality, analyze structural breaks
- **Deliverable**: Comprehensive understanding of data characteristics

### Issue 3: Feature Engineering
- **Focus**: Complete `02_Feature_Engineering.ipynb`
- **Key Tasks**: Time features, lag features, rolling stats, event indicators
- **Deliverable**: Processed datasets ready for model training

### Issue 4: Prophet Model
- **Focus**: Complete `03_Model_Development_Prophet.ipynb`
- **Key Tasks**: Configure Prophet, add custom events, generate 12-month forecasts
- **Deliverable**: Prophet model with 2023 forecasts and evaluation metrics

### Issue 5: LSTM Model
- **Focus**: Complete `04_Model_Development_LSTM.ipynb`
- **Key Tasks**: Design LSTM architecture, sequence generation, training, forecasting
- **Deliverable**: LSTM model with 2023 forecasts and evaluation metrics

### Issue 6: Chronos Model
- **Focus**: Complete `05_Model_Development_Chronos.ipynb`
- **Key Tasks**: Load pre-trained model, zero-shot forecasting, probabilistic predictions
- **Deliverable**: Chronos forecasts with prediction intervals

### Issue 7: N-HiTS Model
- **Focus**: Complete `06_Model_Development_Novel.ipynb`
- **Key Tasks**: Configure hierarchical architecture, train, generate probabilistic forecasts
- **Deliverable**: State-of-the-art N-HiTS forecasts with uncertainty quantification

### Issue 8: Model Comparison
- **Focus**: Complete `07_Model_Comparison_and_Analysis.ipynb`
- **Key Tasks**: Aggregate results, create comparison table, visualize, analyze
- **Deliverable**: Comprehensive model comparison answering research question

### Issue 9: Preprocessing Module
- **Focus**: Enhance `scripts/preprocessing.py`
- **Key Tasks**: Add feature functions, scaling utilities, validation functions
- **Deliverable**: Reusable preprocessing toolkit

### Issue 10: Evaluation Module
- **Focus**: Enhance `scripts/evaluation.py`
- **Key Tasks**: Metrics functions, visualization functions, comparison utilities
- **Deliverable**: Standardized evaluation toolkit

### Issue 11: Documentation Updates
- **Focus**: Update all README files and documentation
- **Key Tasks**: Document workflow, update dependencies, add results summary
- **Deliverable**: Complete, reproducible documentation

### Issue 12: Testing Framework
- **Focus**: Create basic tests for utility functions
- **Key Tasks**: Test data loader, preprocessing, evaluation functions
- **Deliverable**: Test suite preventing regressions

---

## Implementation Workflow

### Phase 1: Data Foundation (Week 1)
- Issue 1: Data Ingestion
- Issue 2: EDA
- Issue 3: Feature Engineering

### Phase 2: Model Development (Weeks 2-4)
- Issue 4: Prophet (Week 2)
- Issue 5: LSTM (Week 2-3)
- Issue 6: Chronos (Week 3)
- Issue 7: N-HiTS (Week 3-4)

### Phase 3: Analysis and Infrastructure (Week 5)
- Issue 8: Model Comparison
- Issue 9: Preprocessing Module
- Issue 10: Evaluation Module

### Phase 4: Documentation and Quality (Week 6)
- Issue 11: Documentation Updates
- Issue 12: Testing Framework

---

## Key Project Constraints

### Data Split
- **Training**: January 2000 - December 31, 2022
- **Test**: January 1, 2023 - December 31, 2023
- **Horizon**: 12 months (full annual cycle)

### Evaluation Metrics
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)

### Critical Conventions
- ✅ Chronological splits (never shuffle time series)
- ✅ Forward fill for missing values
- ✅ MinMaxScaler for neural networks
- ✅ Raw values for Prophet/statistical models
- ✅ Standard columns: `Date` (datetime), `Arrivals` (int)

### Research Question
**Can advanced deep learning architectures (LSTM, Chronos, N-HiTS) outperform statistical methods (Prophet) for Sri Lankan tourism forecasting, given strong seasonality but extreme shocks?**

---

## Dependencies Summary

### Core Stack
- pandas, numpy, matplotlib, seaborn, plotly
- scikit-learn

### Time Series
- prophet
- statsmodels

### Deep Learning
- tensorflow (LSTM)
- torch (Chronos, N-HiTS)
- neuralforecast (N-HiTS)
- chronos-forecasting (Chronos)

### Web Scraping
- requests
- beautifulsoup4
- pdfplumber

### Development
- jupyter
- tqdm
- pytest (optional, for testing)

---

## Success Criteria

A successful implementation will:

1. ✅ **Technical Proficiency**: Correct implementation of all models
2. ✅ **Rigorous Evaluation**: Comprehensive metrics and visualizations
3. ✅ **Actionable Insights**: Clear recommendations based on analysis
4. ✅ **Reproducibility**: Well-documented, clean code
5. ✅ **Research Contribution**: Answer whether DL outperforms statistical methods
6. ✅ **Practical Value**: Forecasts usable for real-world decisions

---

## How to Use This Document

1. **Create GitHub Issues**: Use `GITHUB_ISSUES_TEMPLATE.md` for detailed task lists
2. **Follow Priority Order**: Start with high-priority issues
3. **Track Progress**: Use issue checkboxes to monitor completion
4. **Reference Documentation**: Link to `.github/agents/tourism-forecasting.md` and other docs
5. **Maintain Conventions**: Follow project-specific patterns and standards

---

## Notes

- Each issue is designed to be self-contained with clear acceptance criteria
- Issues reference project documentation for context
- Implementation tasks include detailed checklists
- Issues cover technical, infrastructure, and documentation aspects
- Priority reflects project workflow and dependencies

For complete details, see `GITHUB_ISSUES_TEMPLATE.md`.
