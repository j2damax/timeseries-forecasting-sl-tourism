# Tourism Forecasting Agent Instructions

## Project Overview

This project focuses on **forecasting international tourist arrivals in Sri Lanka** through a comparative analysis of statistical, machine learning, and deep learning models. The work addresses a critical economic need while providing a rigorous test of various forecasting methodologies.

## Context and Motivation

### Problem Statement
The tourism sector is a cornerstone of the Sri Lankan economy, serving as:
- A critical source of foreign exchange
- A significant generator of employment
- A catalyst for regional development

Accurate forecasting of tourist arrivals is essential for:
- Government bodies formulating national economic policy
- Private enterprises making investment and resource allocation decisions
- Strategic planning and budgeting within the tourism industry

### Key Challenges
The Sri Lankan tourism time series presents unique analytical challenges:

1. **Strong Seasonality**: Dictated by climate and cultural festivals
2. **Profound Volatility**: High sensitivity to domestic and international shocks
3. **Structural Breaks**: Multiple regime changes over the past two decades:
   - End of civil conflict (pre-2009)
   - Post-war tourism boom (mid-2009 onwards)
   - Easter Sunday terrorist attacks (2019)
   - COVID-19 pandemic collapse (2020-2021)
   - Economic crisis and volatile recovery (2022-)

These characteristics make historical patterns potentially unreliable guides for future trends.

## Data Specifications

### Data Source
- **Official source**: Sri Lanka Tourism Development Authority (SLTDA)
- **Time range**: January 2000 to most recent available month in 2024
- **Frequency**: Monthly observations
- **Variable**: Total number of international tourist arrivals per month

### Forecasting Specifications
- **Target Variable**: `Total Tourist Arrivals`
- **Forecasting Horizon**: 12 months (one full annual cycle)
- **Training/Test Split**: 
  - Training: Historical data up to December 31, 2022
  - Test: January 1, 2023 to December 31, 2023

### Rationale for Data Selection
The extended time horizon (2000-2024) is critical as it:
- Captures multiple distinct economic and political regimes
- Includes periods of extreme volatility
- Provides a rigorous test for model robustness and adaptability
- Encompasses the full spectrum of structural changes affecting tourism

## Model Portfolio

Based on literature review, implement and evaluate the following models:

### 1. Statistical Models

#### SARIMA (Seasonal ARIMA)
- **Justification**: Consistently successful in literature for Sri Lankan tourism data
- **Key strength**: Explicit mechanism for handling strong annual seasonality
- **Benchmark status**: Any new model must be competitive with well-tuned SARIMA
- **Reference configurations** from literature:
  - SARIMA (1,1,11)(2,1,3) [Basnayake & Chandrasekara, 2022]
  - SARIMA (1,0,16)(36,0,24)12 [Peiris, 2016]

#### Holt-Winters Exponential Smoothing
- **Justification**: Proven effective for handling trend and seasonality
- **Strength**: Good performance for Sri Lanka's top source markets

### 2. Modern Machine Learning Models

#### Facebook Prophet
- **Strengths**:
  - High interpretability and ease of use
  - Explicit incorporation of custom events and holidays
  - Particularly valuable for series influenced by external shocks
- **Limitations**:
  - Rigid trend component (piecewise linear or logistic functions)
  - May struggle with complex, non-linear growth patterns
  - Could underperform during rapid recovery phases

### 3. Deep Learning Models

#### LSTM (Long Short-Term Memory Networks)
- **Strengths**:
  - Captures complex, non-linear temporal dependencies
  - No strong prior assumptions required
  - General-purpose sequence modeling capability
- **Limitations**:
  - "Black box" nature (reduced interpretability)
  - Requires substantial training data
  - May be less efficient at capturing crisp, multi-frequency seasonality

#### Amazon Chronos
- **Strengths**:
  - Zero-shot forecasting capability
  - Pre-trained on vast array of time series
  - Strong baseline with no training required
- **Limitations**:
  - Univariate nature (cannot use engineered features)
  - Limited interpretability

#### N-HiTS (Novel Model)
- **Strengths**:
  - State-of-the-art architecture for time series
  - Designed to handle multiple underlying frequencies (trend + seasonality)
  - Hierarchical structure for efficient signal decomposition
  - Can produce probabilistic forecasts with quantile loss
- **Limitations**:
  - High complexity
  - Computationally expensive hyperparameter tuning
  - Univariate focus in standard implementations

## Implementation Guidelines

### Data Preparation and Feature Engineering

1. **Temporal Features**:
   - Day, month, year
   - Seasonality indicators
   - Lag features
   - Rolling window statistics

2. **Event-Based Features**:
   - Dummy variables for major shocks (2019 attacks, COVID-19, economic crisis)
   - Holiday indicators
   - Special event markers

3. **Data Preprocessing**:
   - Handle missing values appropriately
   - Apply feature scaling/transformation where needed
   - Ensure proper train/test split respecting temporal order

### Model Development Workflow

For each model:

1. **Data Loading and Validation**
   - Use utilities in `scripts/data_loader.py`
   - Validate required columns and data quality

2. **Feature Engineering**
   - Implement in dedicated notebook (02_Feature_Engineering.ipynb)
   - Document all transformations

3. **Model Training**
   - Create dedicated notebook for each model:
     - 03_Model_Development_Prophet.ipynb
     - 04_Model_Development_LSTM.ipynb
     - 05_Model_Development_Chronos.ipynb
     - 06_Model_Development_Novel.ipynb (N-HiTS)
   - Implement proper cross-validation for hyperparameter tuning
   - Use time series cross-validation (not random splits)

4. **Model Evaluation**
   - Use utilities in `scripts/evaluation.py`
   - Calculate comprehensive metrics:
     - MSE (Mean Squared Error)
     - RMSE (Root Mean Squared Error)
     - MAE (Mean Absolute Error)
     - R² Score
     - MAPE (Mean Absolute Percentage Error)

### Evaluation and Comparison

#### Quantitative Metrics
Create a comparison table showing all metrics across models to identify:
- Best overall performer
- Model-specific strengths and weaknesses
- Trade-offs between different approaches

#### Visual Analysis
Generate comprehensive visualizations:

1. **Forecast Plots**:
   - Overlay actual vs. predicted for all models
   - Include 95% prediction intervals for probabilistic models (Chronos, N-HiTS)
   - Show full test period (12 months)

2. **Residual Analysis**:
   - Time series plot of residuals (error = actual - forecast)
   - Identify systematic biases
   - Check for white noise behavior

3. **Error Distribution**:
   - Histograms of residuals for each model
   - Verify approximate normality with zero mean
   - Identify skewness or multimodality

#### Error Analysis
Conduct qualitative examination:
- Identify periods with highest prediction errors
- Correlate errors with real-world events
- Analyze model behavior during post-COVID recovery
- Compare model adaptability to non-linear dynamics

### Code Quality and Documentation

1. **Use Existing Utilities**:
   - `scripts/data_loader.py` for data loading
   - `scripts/evaluation.py` for metrics and visualization
   - Extend these as needed rather than duplicating code

2. **Notebook Structure**:
   - Start each notebook with clear markdown explaining objectives
   - Include data exploration and visualization
   - Document model configuration choices
   - Present results with visualizations
   - Summarize findings and next steps

3. **Code Style**:
   - Follow Python best practices
   - Use type hints where appropriate
   - Include docstrings for functions
   - Maintain consistency with existing code

## Research Context

### Literature Findings

The literature review establishes that:

1. **SARIMA is the benchmark**: Multiple independent studies confirm its excellence for Sri Lankan tourism forecasting
2. **Machine learning shows promise**: MLR and XGBR demonstrated high accuracy in recent studies
3. **Deep learning gap**: Limited application of LSTM to Sri Lankan context; no documented use of TFT or N-HiTS

### Central Research Question

**Can advanced deep learning architectures, designed to capture complex non-linear patterns, outperform the established statistical champion (SARIMA) on a time series characterized by strong, regular seasonality but punctuated by extreme, irregular shocks?**

## Critical Considerations

### Model Strengths vs. Data Characteristics

The unique challenge of this dataset is the tension between:
- **Regular patterns**: Strong annual seasonality (favors SARIMA, Holt-Winters)
- **Irregular shocks**: Extreme volatility and structural breaks (favors flexible ML/DL models)

The most successful model will likely be one that:
- Captures regular seasonal patterns efficiently
- Adapts quickly to structural changes
- Provides robust forecasts during volatile periods

### Data Limitations

**Current limitation**: Univariate time series only

**Potential enhancements** for future work:
- Currency exchange rates
- Domestic inflation indicators
- International flight prices
- Google search query volumes for Sri Lankan travel
- Other exogenous economic indicators

### Ethical Considerations

- Forecasts should be rigorous and data-driven
- Avoid politically motivated optimism
- Recognize impact on policy and investment decisions
- Maintain transparency in methodology and limitations

## Future Extensions

### Recommended Next Steps

1. **SARIMAX Implementation**:
   - Direct comparison with literature benchmarks
   - Formal statistical testing of engineered event variables
   - Validation of feature significance

2. **Temporal Fusion Transformer (TFT)**:
   - Multi-horizon, multivariate forecasting capability
   - Can incorporate static covariates, known future inputs, and observed past inputs
   - Built-in interpretability mechanisms (variable selection, attention weights)
   - Would enable integration of exogenous variables

3. **Ensemble Methods**:
   - Combine predictions from best-performing models
   - Weighted averaging (e.g., N-HiTS + Prophet)
   - Often more robust than individual models

## Deliverables

### Required Outputs

1. **Jupyter Notebooks**:
   - 01_Data_Ingestion_and_EDA.ipynb (completed)
   - 02_Feature_Engineering.ipynb (completed)
   - 03_Model_Development_Prophet.ipynb (to be completed)
   - 04_Model_Development_LSTM.ipynb (to be completed)
   - 05_Model_Development_Chronos.ipynb (to be completed)
   - 06_Model_Development_Novel.ipynb (N-HiTS, to be completed)

2. **Evaluation Report**:
   - Comparison table of all metrics
   - Visualization suite (forecasts, residuals, distributions)
   - Error analysis and interpretation
   - Model recommendations

3. **Documentation**:
   - Updated README with project overview
   - Clear instructions for reproducing results
   - Dependencies and environment setup
   - Data sources and references

## Success Criteria

A successful implementation will:

1. **Demonstrate technical proficiency**: Correct implementation of all models
2. **Provide rigorous evaluation**: Comprehensive metrics and visualizations
3. **Generate actionable insights**: Clear recommendations based on comparative analysis
4. **Maintain reproducibility**: Well-documented, clean code that others can run
5. **Address research question**: Determine whether deep learning outperforms statistical methods for this specific problem
6. **Show practical value**: Produce forecasts that could support real-world decision-making in Sri Lankan tourism sector

## Key Principles

1. **Benchmark-driven**: SARIMA performance is the standard to meet or exceed
2. **Rigorous evaluation**: Use multiple metrics and validation approaches
3. **Transparent methodology**: Document all assumptions and limitations
4. **Practical focus**: Solutions should be usable by stakeholders
5. **Research contribution**: Fill identified gaps in literature on Sri Lankan tourism forecasting
6. **Ethical responsibility**: Recognize impact of forecasts on economic policy and investment decisions
