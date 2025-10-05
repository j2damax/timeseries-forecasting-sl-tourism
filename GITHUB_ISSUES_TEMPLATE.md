# GitHub Issues for Tourism Forecasting Project

This document contains detailed GitHub issues to be created for the Sri Lanka Tourism Forecasting project. Each issue is structured with comprehensive implementation tasks based on the project documentation.

---

## Issue 1: Data Ingestion Pipeline Validation and Enhancement

**Labels**: `enhancement`, `data-pipeline`, `high-priority`

**Description**:

Validate and enhance the existing data ingestion pipeline for Sri Lanka Tourism Development Authority (SLTDA) reports.

**Context**:

The project has an automated data ingestion script (`scripts/ingest_tourism_data.py`) that scrapes SLTDA website, downloads PDF reports, and extracts tourist arrival data. This issue focuses on ensuring the pipeline is robust and complete.

**References**:
- `.github/copilot-instructions.md` - Data Pipeline Structure
- `README.md` - Data Ingestion section
- `scripts/README.md` - Script documentation
- `.github/agents/tourism-forecasting.md` - Data Specifications

**Implementation Tasks**:

- [ ] **Phase 1: Validate Current Implementation**
  - [ ] Run `python scripts/ingest_tourism_data.py` to test the data ingestion pipeline
  - [ ] Verify that PDFs are downloaded to `data/raw/` directory
  - [ ] Confirm output CSV is created at `data/processed/monthly_tourist_arrivals.csv`
  - [ ] Check data quality: Date column (datetime), Arrivals column (int)
  - [ ] Validate data range: January 2000 to most recent available month in 2024

- [ ] **Phase 2: Error Handling and Robustness**
  - [ ] Test error handling for corrupted PDFs
  - [ ] Verify graceful handling of various PDF formats
  - [ ] Add logging for PDFs that fail to extract
  - [ ] Implement retry mechanism for failed downloads (if not already present)
  - [ ] Add progress indicators for long-running operations

- [ ] **Phase 3: Data Quality Checks**
  - [ ] Implement validation for duplicate entries
  - [ ] Check for missing months in the time series
  - [ ] Add data range validation (reasonable min/max values)
  - [ ] Verify chronological ordering
  - [ ] Document any data anomalies found

- [ ] **Phase 4: Documentation Updates**
  - [ ] Update `scripts/README.md` with any changes
  - [ ] Document known limitations or PDF format variations
  - [ ] Add troubleshooting guide for common issues
  - [ ] Include sample output statistics

**Acceptance Criteria**:
- Data ingestion script runs successfully from start to finish
- Output CSV contains data from January 2000 to latest available month
- No duplicate entries in final dataset
- All errors are logged appropriately
- Documentation is up-to-date and comprehensive

**Expected Data Format**:
```
Date,Arrivals
2000-01-01,58000
2000-02-01,61000
...
2024-XX-01,XXXXXX
```

---

## Issue 2: Exploratory Data Analysis (EDA) Implementation

**Labels**: `notebook`, `data-analysis`, `high-priority`

**Description**:

Complete the exploratory data analysis notebook (`01_Data_Ingestion_and_EDA.ipynb`) to understand the characteristics of Sri Lankan tourism data.

**Context**:

Understanding the data is critical before model development. The tourism time series has unique characteristics including strong seasonality, structural breaks, and extreme volatility from various shocks.

**References**:
- `.github/agents/tourism-forecasting.md` - Context and Motivation, Key Challenges
- `.github/copilot-instructions.md` - Data Handling conventions
- `README.md` - Project overview

**Implementation Tasks**:

- [ ] **Phase 1: Data Loading**
  - [ ] Use `scripts.data_loader.load_csv_data()` with `date_column='Date'`
  - [ ] Validate required columns: Date, Arrivals
  - [ ] Check for missing values
  - [ ] Display basic statistics (count, mean, std, min, max)

- [ ] **Phase 2: Time Series Visualization**
  - [ ] Create line plot of tourist arrivals over time (2000-2024)
  - [ ] Highlight major events:
    - End of civil conflict (2009)
    - Post-war tourism boom (2009-2019)
    - Easter Sunday attacks (April 2019)
    - COVID-19 pandemic (2020-2021)
    - Economic crisis (2022-)
  - [ ] Add annotations for significant events
  - [ ] Use `%matplotlib inline` for consistent plotting

- [ ] **Phase 3: Seasonality Analysis**
  - [ ] Create monthly boxplots to identify seasonal patterns
  - [ ] Generate seasonal decomposition plots (trend, seasonal, residual)
  - [ ] Calculate and visualize year-over-year growth rates
  - [ ] Identify peak and low tourism months

- [ ] **Phase 4: Statistical Analysis**
  - [ ] Test for stationarity (ADF test)
  - [ ] Calculate autocorrelation and partial autocorrelation
  - [ ] Identify structural breaks using statistical tests
  - [ ] Analyze volatility patterns over different periods

- [ ] **Phase 5: Distribution Analysis**
  - [ ] Plot histogram of arrivals
  - [ ] Check for normality (Q-Q plot)
  - [ ] Identify outliers
  - [ ] Calculate coefficient of variation

- [ ] **Phase 6: Key Insights Documentation**
  - [ ] Document strong annual seasonality patterns
  - [ ] Identify periods of high volatility
  - [ ] Note structural breaks and their implications
  - [ ] Summarize findings for model selection

**Acceptance Criteria**:
- Notebook runs from start to finish without errors
- All visualizations are clear and properly labeled
- Statistical tests are interpreted correctly
- Key insights are documented with markdown cells
- Findings align with known tourism patterns (seasonality, shocks)

**Key Challenges to Highlight**:
1. Strong seasonality dictated by climate and festivals
2. High sensitivity to domestic and international shocks
3. Multiple structural breaks over 24-year period
4. Profound volatility especially post-2019

---

## Issue 3: Feature Engineering Implementation

**Labels**: `notebook`, `feature-engineering`, `high-priority`

**Description**:

Implement comprehensive feature engineering in `02_Feature_Engineering.ipynb` to prepare data for model training.

**Context**:

Feature engineering is critical for machine learning and deep learning models. The features should capture temporal patterns, seasonality, and domain-specific characteristics of tourism data.

**References**:
- `.github/copilot-instructions.md` - Feature Engineering Patterns
- `.github/agents/tourism-forecasting.md` - Data Preparation and Feature Engineering
- `scripts/preprocessing.py` - Existing preprocessing functions

**Implementation Tasks**:

- [ ] **Phase 1: Temporal Features**
  - [ ] Extract year from Date column
  - [ ] Extract month from Date column
  - [ ] Extract quarter from Date column
  - [ ] Extract day_of_week from Date column
  - [ ] Create month name for interpretability
  - [ ] Create is_peak_season binary feature (based on EDA findings)

- [ ] **Phase 2: Lag Features**
  - [ ] Create 1-month lag of arrivals
  - [ ] Create 3-month lag of arrivals
  - [ ] Create 6-month lag of arrivals
  - [ ] Create 12-month lag of arrivals (same month previous year)
  - [ ] Document rationale: tourism seasonality patterns

- [ ] **Phase 3: Rolling Window Statistics**
  - [ ] 3-month rolling mean
  - [ ] 6-month rolling mean
  - [ ] 12-month rolling mean
  - [ ] 3-month rolling standard deviation (volatility measure)
  - [ ] Handle edge cases at the beginning of series

- [ ] **Phase 4: Event-Based Features**
  - [ ] Create dummy variable for Easter Sunday attacks (April 2019)
  - [ ] Create dummy variable for COVID-19 period (March 2020 - December 2021)
  - [ ] Create dummy variable for economic crisis (2022 onwards)
  - [ ] Create dummy variable for post-war boom (2009-2019)
  - [ ] Document all event dates and rationale

- [ ] **Phase 5: Growth Rate Features**
  - [ ] Month-over-month growth rate
  - [ ] Year-over-year growth rate
  - [ ] 3-month average growth rate
  - [ ] Handle division by zero and missing values

- [ ] **Phase 6: Feature Scaling**
  - [ ] Document when to use MinMaxScaler (for neural networks)
  - [ ] Document when to keep raw values (for Prophet/statistical models)
  - [ ] Create utility functions in `scripts/preprocessing.py` if needed
  - [ ] Save scaled and unscaled versions

- [ ] **Phase 7: Data Validation**
  - [ ] Check for any introduced NaN values
  - [ ] Validate feature distributions
  - [ ] Document feature importance considerations
  - [ ] Save processed datasets for model training

**Acceptance Criteria**:
- All temporal features correctly extracted
- Lag features properly aligned (no data leakage)
- Rolling statistics handle edge cases correctly
- Event features accurately mark known shocks
- Missing values handled appropriately (forward fill for tourism data)
- Processed data saved and ready for model training
- Documentation explains each feature and its purpose

**Important Conventions**:
- Always preserve chronological order (never shuffle time series data)
- Use forward fill (`method='ffill'`) for missing values
- Standard column names: `Date` (datetime), `Arrivals` (target)

---

## Issue 4: Facebook Prophet Model Development

**Labels**: `notebook`, `model-development`, `prophet`, `high-priority`

**Description**:

Implement Facebook Prophet forecasting model in `03_Model_Development_Prophet.ipynb`.

**Context**:

Prophet is well-suited for time series with strong seasonal patterns and the ability to incorporate custom events. It provides interpretable forecasts and is particularly valuable for series influenced by external shocks.

**References**:
- `.github/agents/tourism-forecasting.md` - Model Portfolio: Facebook Prophet
- `.github/copilot-instructions.md` - Model Development Pattern
- Data Specifications: Training (up to Dec 2022), Test (Jan-Dec 2023)

**Implementation Tasks**:

- [ ] **Phase 1: Data Preparation**
  - [ ] Load processed data from feature engineering
  - [ ] Split data chronologically:
    - Training: January 2000 - December 31, 2022
    - Test: January 1, 2023 - December 31, 2023
  - [ ] Prepare data in Prophet format (columns: `ds`, `y`)
  - [ ] Verify no data leakage in split

- [ ] **Phase 2: Prophet Configuration**
  - [ ] Initialize Prophet model with appropriate parameters
  - [ ] Configure seasonality:
    - Yearly seasonality (enabled)
    - Monthly seasonality (if beneficial)
  - [ ] Set growth parameter (linear or logistic)
  - [ ] Configure changepoint settings for structural breaks

- [ ] **Phase 3: Custom Events and Holidays**
  - [ ] Add Easter Sunday attacks (April 2019) as special event
  - [ ] Add COVID-19 pandemic period (2020-2021) as lockdown event
  - [ ] Add economic crisis (2022) as special event
  - [ ] Configure Sri Lankan holidays if relevant
  - [ ] Document event windows and parameters

- [ ] **Phase 4: Model Training**
  - [ ] Fit Prophet model on training data
  - [ ] Monitor training process
  - [ ] Extract and visualize components:
    - Trend component
    - Yearly seasonality
    - Event effects
  - [ ] Validate model convergence

- [ ] **Phase 5: Forecasting**
  - [ ] Generate forecasts for test period (12 months)
  - [ ] Extract point forecasts and prediction intervals
  - [ ] Create forecast dataframe with dates and predictions
  - [ ] Save model artifacts for comparison phase

- [ ] **Phase 6: Evaluation**
  - [ ] Use `scripts.evaluation.calculate_metrics()` for:
    - MSE (Mean Squared Error)
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - R² Score
    - MAPE (Mean Absolute Percentage Error)
  - [ ] Compare predictions vs actual for 2023
  - [ ] Identify periods with highest errors

- [ ] **Phase 7: Visualization**
  - [ ] Plot forecast vs actual with prediction intervals
  - [ ] Visualize trend and seasonality components
  - [ ] Create residual plot (error = actual - forecast)
  - [ ] Generate error distribution histogram
  - [ ] Add proper labels, titles, and legends

- [ ] **Phase 8: Interpretation and Documentation**
  - [ ] Document model strengths for this dataset
  - [ ] Note any limitations or challenges observed
  - [ ] Summarize key findings
  - [ ] Save results for model comparison

**Acceptance Criteria**:
- Model trained on correct data split (train: 2000-2022, test: 2023)
- Custom events properly incorporated
- 12-month forecasts generated for 2023
- All evaluation metrics calculated using standard utilities
- Comprehensive visualizations created
- Model artifacts saved for comparison
- Documentation explains model choices and results

**Prophet Strengths**:
- High interpretability
- Explicit incorporation of custom events
- Good for series with strong seasonality
- Robust to missing data and outliers

**Prophet Limitations**:
- Rigid trend component (piecewise linear/logistic)
- May struggle with complex non-linear patterns
- Could underperform during rapid recovery phases

---

## Issue 5: LSTM Model Development

**Labels**: `notebook`, `model-development`, `deep-learning`, `lstm`, `high-priority`

**Description**:

Implement Long Short-Term Memory (LSTM) neural network in `04_Model_Development_LSTM.ipynb`.

**Context**:

LSTM networks can capture complex, non-linear temporal dependencies without strong prior assumptions. They're particularly suitable for sequences with long-term dependencies and volatile patterns.

**References**:
- `.github/agents/tourism-forecasting.md` - Model Portfolio: LSTM
- `.github/copilot-instructions.md` - Model Development Pattern, Feature Engineering
- Dependencies: tensorflow (deep learning framework)

**Implementation Tasks**:

- [ ] **Phase 1: Data Preparation**
  - [ ] Load processed data with features from feature engineering
  - [ ] Split data chronologically (train: 2000-2022, test: 2023)
  - [ ] Apply MinMaxScaler to normalize features (required for neural networks)
  - [ ] Create sequences for LSTM input (e.g., 12-month windows)
  - [ ] Prepare X (features) and y (target) arrays

- [ ] **Phase 2: Sequence Generation**
  - [ ] Define lookback window (e.g., 12 months)
  - [ ] Create sliding windows for training data
  - [ ] Reshape data for LSTM format: (samples, timesteps, features)
  - [ ] Generate corresponding target values
  - [ ] Validate sequence shapes

- [ ] **Phase 3: LSTM Architecture Design**
  - [ ] Design LSTM architecture (number of layers, units)
  - [ ] Add LSTM layers with appropriate units (e.g., 50-100)
  - [ ] Consider stacking multiple LSTM layers
  - [ ] Add Dropout layers for regularization (prevent overfitting)
  - [ ] Add Dense output layer for predictions
  - [ ] Document architecture choices

- [ ] **Phase 4: Model Compilation**
  - [ ] Choose optimizer (e.g., Adam)
  - [ ] Select loss function (e.g., MSE for regression)
  - [ ] Add metrics (MAE for monitoring)
  - [ ] Configure learning rate if needed

- [ ] **Phase 5: Model Training**
  - [ ] Set up early stopping callback
  - [ ] Configure model checkpoint to save best weights
  - [ ] Define batch size and epochs
  - [ ] Use validation split (e.g., 20% of training data)
  - [ ] Train model and monitor training/validation loss
  - [ ] Visualize training history

- [ ] **Phase 6: Forecasting**
  - [ ] Generate predictions for test period (2023)
  - [ ] Inverse transform predictions (scale back to original values)
  - [ ] Create forecast dataframe with dates
  - [ ] Save model and predictions

- [ ] **Phase 7: Evaluation**
  - [ ] Use `scripts.evaluation.calculate_metrics()` for:
    - MSE, RMSE, MAE, R², MAPE
  - [ ] Compare with Prophet results
  - [ ] Analyze error patterns

- [ ] **Phase 8: Visualization**
  - [ ] Plot training history (loss curves)
  - [ ] Visualize forecast vs actual
  - [ ] Create residual analysis plots
  - [ ] Generate error distribution histogram
  - [ ] Compare with Prophet visually

- [ ] **Phase 9: Hyperparameter Tuning (Optional)**
  - [ ] Experiment with different architectures
  - [ ] Try different lookback windows
  - [ ] Test various LSTM units
  - [ ] Adjust dropout rates
  - [ ] Document best configuration

- [ ] **Phase 10: Documentation**
  - [ ] Explain LSTM architecture choices
  - [ ] Document hyperparameters
  - [ ] Note training time and computational requirements
  - [ ] Summarize strengths and limitations for this dataset
  - [ ] Save all results for comparison

**Acceptance Criteria**:
- LSTM model properly configured for time series forecasting
- Data correctly shaped and scaled for LSTM input
- Model trained with appropriate early stopping
- 12-month forecasts generated for 2023
- All metrics calculated and compared with Prophet
- Comprehensive visualizations created
- Model artifacts saved for comparison phase

**LSTM Strengths**:
- Captures complex, non-linear temporal dependencies
- No strong prior assumptions required
- Can learn from long sequences

**LSTM Limitations**:
- Black box nature (reduced interpretability)
- Requires substantial training data
- Computationally expensive
- May be less efficient at capturing crisp seasonality

**Important Notes**:
- Use MinMaxScaler for neural networks (not for Prophet)
- Preserve chronological order in train/test split
- Use time series cross-validation if tuning hyperparameters

---

## Issue 6: Amazon Chronos Model Development

**Labels**: `notebook`, `model-development`, `deep-learning`, `chronos`, `medium-priority`

**Description**:

Implement Amazon Chronos zero-shot forecasting model in `05_Model_Development_Chronos.ipynb`.

**Context**:

Chronos is a pre-trained foundation model for time series forecasting. It provides zero-shot forecasting capability (no training required) and can serve as a strong baseline with minimal effort.

**References**:
- `.github/agents/tourism-forecasting.md` - Model Portfolio: Amazon Chronos
- `.github/copilot-instructions.md` - Model Development Pattern
- Chronos documentation: https://github.com/amazon-science/chronos-forecasting

**Implementation Tasks**:

- [ ] **Phase 1: Environment Setup**
  - [ ] Install Chronos package (add to requirements.txt if needed)
  - [ ] Import required libraries
  - [ ] Verify PyTorch installation (dependency)
  - [ ] Test Chronos model loading

- [ ] **Phase 2: Data Preparation**
  - [ ] Load processed data from feature engineering
  - [ ] Split data chronologically (train: 2000-2022, test: 2023)
  - [ ] Prepare univariate time series (Arrivals only)
  - [ ] Note: Chronos is univariate, cannot use engineered features
  - [ ] Convert data to appropriate format for Chronos

- [ ] **Phase 3: Model Selection**
  - [ ] Choose Chronos model size (tiny, small, base, large)
  - [ ] Load pre-trained model from HuggingFace
  - [ ] Document model variant used
  - [ ] Note: No training required (zero-shot)

- [ ] **Phase 4: Forecasting Configuration**
  - [ ] Set prediction length (12 months for 2023)
  - [ ] Configure number of samples for probabilistic forecast
  - [ ] Set context length (historical data to use)
  - [ ] Document all parameters

- [ ] **Phase 5: Generate Forecasts**
  - [ ] Use historical data up to Dec 2022 as context
  - [ ] Generate 12-month ahead forecasts
  - [ ] Extract point forecasts (median)
  - [ ] Extract prediction intervals (quantiles)
  - [ ] Create forecast dataframe

- [ ] **Phase 6: Evaluation**
  - [ ] Use `scripts.evaluation.calculate_metrics()` for:
    - MSE, RMSE, MAE, R², MAPE
  - [ ] Compare with Prophet and LSTM
  - [ ] Analyze prediction intervals (coverage)
  - [ ] Identify forecast strengths and weaknesses

- [ ] **Phase 7: Visualization**
  - [ ] Plot forecast vs actual with prediction intervals
  - [ ] Show uncertainty quantiles (95%, 80%)
  - [ ] Create residual plots
  - [ ] Generate error distribution histogram
  - [ ] Compare visually with other models

- [ ] **Phase 8: Analysis and Documentation**
  - [ ] Evaluate zero-shot performance
  - [ ] Compare computational efficiency vs LSTM
  - [ ] Note advantages of pre-trained model
  - [ ] Document limitations (univariate, no custom events)
  - [ ] Summarize findings
  - [ ] Save results for comparison

**Acceptance Criteria**:
- Chronos model successfully loaded and run
- 12-month forecasts generated with prediction intervals
- All metrics calculated and compared
- Visualizations show forecast uncertainty
- Zero-shot performance analyzed
- Documentation explains model choice and results

**Chronos Strengths**:
- Zero-shot forecasting (no training needed)
- Pre-trained on vast array of time series
- Strong baseline with minimal effort
- Provides probabilistic forecasts

**Chronos Limitations**:
- Univariate nature (cannot use engineered features)
- Limited interpretability
- No customization for domain-specific events
- Requires PyTorch (additional dependency)

**Key Considerations**:
- Compare zero-shot performance vs trained models
- Evaluate if pre-training captures tourism patterns
- Assess prediction interval quality

---

## Issue 7: N-HiTS Model Development (Novel Approach)

**Labels**: `notebook`, `model-development`, `deep-learning`, `n-hits`, `high-priority`

**Description**:

Implement N-HiTS (Neural Hierarchical Interpolation for Time Series) in `06_Model_Development_Novel.ipynb`.

**Context**:

N-HiTS is a state-of-the-art neural architecture specifically designed for time series forecasting. It uses hierarchical structure to efficiently decompose signals into multiple underlying frequencies (trend + seasonality), making it well-suited for data with strong seasonal patterns.

**References**:
- `.github/agents/tourism-forecasting.md` - Model Portfolio: N-HiTS
- `.github/copilot-instructions.md` - Model Development Pattern
- N-HiTS paper: https://arxiv.org/abs/2201.12886
- NeuralForecast library: https://github.com/Nixtla/neuralforecast

**Implementation Tasks**:

- [ ] **Phase 1: Environment Setup**
  - [ ] Install NeuralForecast library (or alternative N-HiTS implementation)
  - [ ] Add to requirements.txt if needed
  - [ ] Import required libraries (PyTorch-based)
  - [ ] Verify installation and dependencies

- [ ] **Phase 2: Data Preparation**
  - [ ] Load processed data from feature engineering
  - [ ] Split data chronologically (train: 2000-2022, test: 2023)
  - [ ] Format data for N-HiTS (typically requires specific structure)
  - [ ] Note: Can use univariate or include exogenous features
  - [ ] Apply scaling if required

- [ ] **Phase 3: Model Configuration**
  - [ ] Configure N-HiTS architecture:
    - Number of stacks (for hierarchical decomposition)
    - Number of blocks per stack
    - Hidden layer units
    - Pooling kernel sizes
  - [ ] Set input window size (historical context)
  - [ ] Set output window size (12 months forecast)
  - [ ] Configure loss function (MSE or MAE)
  - [ ] Document all hyperparameters

- [ ] **Phase 4: Probabilistic Forecasting Setup**
  - [ ] Enable quantile loss for probabilistic forecasts
  - [ ] Configure prediction intervals (e.g., 80%, 95%)
  - [ ] Set up quantile regression if using
  - [ ] Document probabilistic configuration

- [ ] **Phase 5: Model Training**
  - [ ] Configure optimizer and learning rate
  - [ ] Set batch size and number of epochs
  - [ ] Implement validation split
  - [ ] Add early stopping callback
  - [ ] Train model and monitor loss
  - [ ] Visualize training progress

- [ ] **Phase 6: Hyperparameter Tuning**
  - [ ] Experiment with different stack configurations
  - [ ] Try various input/output window sizes
  - [ ] Test different learning rates
  - [ ] Validate on validation set
  - [ ] Select best configuration
  - [ ] Document tuning process and results

- [ ] **Phase 7: Forecasting**
  - [ ] Generate 12-month forecasts for 2023
  - [ ] Extract point forecasts
  - [ ] Extract prediction intervals/quantiles
  - [ ] Inverse transform if scaling was applied
  - [ ] Create forecast dataframe

- [ ] **Phase 8: Evaluation**
  - [ ] Use `scripts.evaluation.calculate_metrics()` for all metrics
  - [ ] Compare with all previous models (Prophet, LSTM, Chronos)
  - [ ] Analyze prediction interval coverage
  - [ ] Evaluate forecast accuracy across different periods
  - [ ] Identify where N-HiTS performs best/worst

- [ ] **Phase 9: Interpretability Analysis**
  - [ ] Visualize hierarchical decomposition if possible
  - [ ] Analyze trend vs seasonal components
  - [ ] Examine attention weights or feature importance
  - [ ] Document interpretability features

- [ ] **Phase 10: Visualization**
  - [ ] Plot forecast vs actual with prediction intervals
  - [ ] Show hierarchical decomposition
  - [ ] Create residual analysis plots
  - [ ] Generate error distribution histogram
  - [ ] Create comparison plots with all other models

- [ ] **Phase 11: Comprehensive Documentation**
  - [ ] Explain N-HiTS architecture and why it's suitable
  - [ ] Document hyperparameter choices
  - [ ] Analyze computational requirements
  - [ ] Compare with traditional and other DL approaches
  - [ ] Summarize strengths for tourism forecasting
  - [ ] Note limitations and challenges
  - [ ] Save all results for final comparison

**Acceptance Criteria**:
- N-HiTS model properly configured and trained
- Hierarchical structure leverages trend and seasonality
- 12-month probabilistic forecasts generated
- All metrics calculated and compared with other models
- Interpretability analysis completed
- Comprehensive documentation of novel approach
- Model artifacts saved for comparison

**N-HiTS Strengths**:
- State-of-the-art architecture for time series
- Designed for multiple underlying frequencies
- Hierarchical structure for efficient signal decomposition
- Can produce probabilistic forecasts
- Often outperforms LSTM and traditional methods

**N-HiTS Limitations**:
- High complexity
- Computationally expensive hyperparameter tuning
- Requires careful configuration
- Less interpretable than Prophet
- Univariate focus in standard implementations

**Research Contribution**:
- First documented application of N-HiTS to Sri Lankan tourism data
- Compare state-of-the-art DL with established statistical methods
- Evaluate if hierarchical decomposition benefits seasonal tourism data

---

## Issue 8: Model Comparison and Analysis

**Labels**: `notebook`, `evaluation`, `analysis`, `high-priority`

**Description**:

Complete comprehensive model comparison and analysis in `07_Model_Comparison_and_Analysis.ipynb`.

**Context**:

This is the final deliverable that synthesizes all model results, provides rigorous comparative evaluation, and answers the central research question: Can advanced deep learning outperform statistical methods for Sri Lankan tourism forecasting?

**References**:
- `.github/agents/tourism-forecasting.md` - Evaluation and Comparison, Success Criteria
- `.github/copilot-instructions.md` - Model Comparison Framework
- All previous model development notebooks (03-06)

**Implementation Tasks**:

- [ ] **Phase 1: Results Aggregation**
  - [ ] Load saved results from all models:
    - Prophet (statistical/ML)
    - LSTM (deep learning)
    - Chronos (pre-trained DL)
    - N-HiTS (novel DL)
  - [ ] Verify all models have 12-month forecasts for 2023
  - [ ] Compile predictions into single dataframe
  - [ ] Load actual values for 2023

- [ ] **Phase 2: Quantitative Comparison**
  - [ ] Create comprehensive metrics table:
    - Rows: Models (Prophet, LSTM, Chronos, N-HiTS)
    - Columns: MSE, RMSE, MAE, R², MAPE
  - [ ] Calculate metrics for all models using `scripts.evaluation.calculate_metrics()`
  - [ ] Rank models by each metric
  - [ ] Identify best overall performer
  - [ ] Identify model-specific strengths (which metrics favor which models)

- [ ] **Phase 3: Visual Comparison**
  - [ ] **Forecast Overlay Plot**:
    - Single plot with actual 2023 values
    - Overlay all model predictions
    - Include prediction intervals for Chronos and N-HiTS
    - Use distinct colors and line styles
    - Add legend and proper labels
  - [ ] **Individual Model Plots**:
    - Separate subplot for each model vs actual
    - Show prediction intervals where available
    - Highlight periods of high/low accuracy
  - [ ] **Metrics Bar Chart**:
    - Visual comparison of RMSE and MAPE across models
    - Make it easy to see relative performance

- [ ] **Phase 4: Residual Analysis**
  - [ ] For each model, create:
    - Time series plot of residuals (error = actual - forecast)
    - Identify systematic biases (over/under prediction)
    - Check for patterns in residuals
  - [ ] **Comparative Residual Plot**:
    - All models' residuals on same plot
    - Identify which models have consistent errors
  - [ ] **Residual Statistics**:
    - Mean error (bias check)
    - Standard deviation of errors
    - Maximum and minimum errors

- [ ] **Phase 5: Error Distribution Analysis**
  - [ ] Create histograms of residuals for each model
  - [ ] Check for approximate normality
  - [ ] Identify skewness (tendency to over/under predict)
  - [ ] Look for multimodality
  - [ ] Statistical tests for normality (if appropriate)

- [ ] **Phase 6: Temporal Error Analysis**
  - [ ] Identify months with highest prediction errors
  - [ ] Analyze errors by season (peak vs low season)
  - [ ] Correlate errors with real-world events:
    - Economic recovery patterns
    - Seasonal variations
    - Any unexpected shocks in 2023
  - [ ] Compare model adaptability to changes

- [ ] **Phase 7: Model Characteristics Comparison**
  - [ ] Create comparison table of model attributes:
    - Interpretability (High/Medium/Low)
    - Training time
    - Inference speed
    - Handles seasonality (Explicit/Implicit)
    - Handles events (Yes/No/Partial)
    - Probabilistic forecasts (Yes/No)
    - Computational requirements
  - [ ] Document trade-offs between models

- [ ] **Phase 8: Research Question Analysis**
  - [ ] **Central Question**: Do DL models (LSTM, Chronos, N-HiTS) outperform statistical/ML (Prophet)?
  - [ ] Analyze results in context of:
    - Strong annual seasonality (favors statistical methods?)
    - Structural breaks and shocks (favors flexible DL?)
    - Data volume (24 years sufficient for DL?)
  - [ ] Discuss which model characteristics mattered most
  - [ ] Evaluate if complexity of DL was justified

- [ ] **Phase 9: Practical Recommendations**
  - [ ] **Best Model for Production**:
    - Consider accuracy, interpretability, and maintainability
    - Recommend for different use cases
  - [ ] **Model Ensemble Consideration**:
    - Evaluate if combining models could improve results
    - Suggest weighted averaging approach
  - [ ] **Limitations and Caveats**:
    - Document what models cannot predict
    - Note data limitations (univariate)
    - Discuss forecast horizon constraints

- [ ] **Phase 10: Future Work Recommendations**
  - [ ] Suggest improvements:
    - SARIMAX with exogenous variables
    - Temporal Fusion Transformer (multivariate)
    - Ensemble methods
  - [ ] Identify additional data sources:
    - Exchange rates
    - Flight prices
    - Google search trends
    - Economic indicators
  - [ ] Propose extended forecasting horizons

- [ ] **Phase 11: Executive Summary**
  - [ ] Write clear, non-technical summary:
    - Best performing model
    - Key findings
    - Practical recommendations
    - Forecast for next 12 months (if appropriate)
  - [ ] Make results accessible to stakeholders

- [ ] **Phase 12: Documentation and Reproducibility**
  - [ ] Ensure all visualizations have clear titles and labels
  - [ ] Document all assumptions
  - [ ] Verify notebook runs from start to finish
  - [ ] Save final comparison plots
  - [ ] Create summary report (optional PDF export)

**Acceptance Criteria**:
- All models compared using consistent metrics
- Comprehensive quantitative comparison table created
- Multiple visualization types showing different aspects
- Residual and error analysis completed
- Research question addressed with evidence
- Practical recommendations provided
- Documentation is clear and thorough
- Results reproducible from saved model outputs

**Key Deliverables**:
1. Metrics comparison table
2. Forecast overlay visualization
3. Residual analysis plots
4. Error distribution analysis
5. Model characteristics comparison
6. Research findings summary
7. Practical recommendations

**Success Criteria**:
- Determine best performing model for Sri Lankan tourism
- Understand trade-offs between statistical and DL approaches
- Provide actionable insights for tourism forecasting
- Contribute to literature on tourism forecasting methodologies

---

## Issue 9: Preprocessing Module Enhancement

**Labels**: `enhancement`, `utility-scripts`, `medium-priority`

**Description**:

Enhance `scripts/preprocessing.py` with comprehensive preprocessing functions for time series forecasting.

**Context**:

The preprocessing module should provide reusable functions for feature engineering, scaling, and data transformations used across all model notebooks.

**References**:
- `.github/copilot-instructions.md` - Module Organization, Feature Engineering Patterns
- `.github/agents/tourism-forecasting.md` - Data Preparation and Feature Engineering

**Implementation Tasks**:

- [ ] **Phase 1: Time Feature Functions**
  - [ ] `extract_time_features(df, date_column='Date')`:
    - Extract year, month, quarter, day_of_week
    - Return dataframe with added columns
  - [ ] Add docstrings with parameter types and descriptions
  - [ ] Include example usage in docstring

- [ ] **Phase 2: Lag Feature Functions**
  - [ ] `create_lag_features(df, target_column='Arrivals', lags=[1,3,6,12])`:
    - Create specified lag features
    - Handle edge cases at series start
    - Return dataframe with lag columns
  - [ ] Document rationale for default lags (tourism seasonality)

- [ ] **Phase 3: Rolling Statistics Functions**
  - [ ] `create_rolling_features(df, target_column='Arrivals', windows=[3,6,12])`:
    - Rolling mean for each window
    - Rolling std for volatility
    - Handle edge cases
    - Return dataframe with rolling features
  - [ ] Include min_periods parameter

- [ ] **Phase 4: Event Feature Functions**
  - [ ] `create_event_features(df, date_column='Date')`:
    - Add binary indicators for major events:
      - Easter Sunday attacks (April 2019)
      - COVID-19 period (March 2020 - Dec 2021)
      - Economic crisis (2022 onwards)
      - Post-war boom (2009-2019)
    - Return dataframe with event columns
    - Make events configurable via parameters

- [ ] **Phase 5: Scaling Functions**
  - [ ] `scale_features(df, scaler_type='minmax', columns=None)`:
    - Apply MinMaxScaler or StandardScaler
    - Fit on specified columns or all numeric
    - Return scaled dataframe and fitted scaler
  - [ ] `inverse_scale_features(df, scaler, columns=None)`:
    - Reverse scaling transformation
    - Use fitted scaler from scale_features

- [ ] **Phase 6: Train/Test Split Function**
  - [ ] `chronological_train_test_split(df, test_start_date)`:
    - Split data chronologically (never shuffle)
    - Return train_df, test_df
    - Validate that split preserves order
  - [ ] Add validation to prevent data leakage

- [ ] **Phase 7: Missing Value Handling**
  - [ ] `handle_missing_values(df, method='ffill')`:
    - Forward fill for tourism data continuity
    - Support multiple methods (ffill, bfill, interpolate)
    - Return dataframe with handled missing values
  - [ ] Document when to use each method

- [ ] **Phase 8: Growth Rate Functions**
  - [ ] `calculate_growth_rates(df, target_column='Arrivals')`:
    - Month-over-month growth rate
    - Year-over-year growth rate
    - Handle division by zero
    - Return dataframe with growth columns

- [ ] **Phase 9: Data Validation Functions**
  - [ ] `validate_no_data_leakage(train_df, test_df, date_column='Date')`:
    - Verify train dates < test dates
    - Check for overlap
    - Raise error if leakage detected
  - [ ] `validate_time_series_continuity(df, date_column='Date')`:
    - Check for gaps in time series
    - Report missing months
    - Return validation report

- [ ] **Phase 10: Testing and Documentation**
  - [ ] Add comprehensive docstrings to all functions
  - [ ] Include type hints for parameters and returns
  - [ ] Create example usage in module docstring
  - [ ] Add inline comments for complex logic
  - [ ] Update `scripts/README.md` with preprocessing functions
  - [ ] Consider adding unit tests (optional)

**Acceptance Criteria**:
- All functions implemented with clear interfaces
- Comprehensive docstrings with examples
- Type hints for all parameters
- Functions handle edge cases gracefully
- No data leakage introduced by any function
- Module well-documented and easy to use
- Functions used successfully in model notebooks

**Key Principles**:
- Preserve chronological order (never shuffle)
- Forward fill for missing values (tourism data continuity)
- Prevent data leakage in all transformations
- Make functions reusable across all models

---

## Issue 10: Evaluation Module Enhancement

**Labels**: `enhancement`, `utility-scripts`, `medium-priority`

**Description**:

Enhance `scripts/evaluation.py` with comprehensive evaluation and visualization functions for model comparison.

**Context**:

The evaluation module should provide standardized metrics and visualization functions used across all model notebooks and the final comparison notebook.

**References**:
- `.github/copilot-instructions.md` - Module Organization, evaluation.py
- `.github/agents/tourism-forecasting.md` - Evaluation and Comparison

**Implementation Tasks**:

- [ ] **Phase 1: Core Metrics Function**
  - [ ] Verify `calculate_metrics(y_true, y_pred)` returns:
    - MSE (Mean Squared Error)
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - R² Score
    - MAPE (Mean Absolute Percentage Error)
  - [ ] Return as dictionary with clear keys
  - [ ] Add input validation (same length, no NaNs)
  - [ ] Include docstring with metric interpretations

- [ ] **Phase 2: Additional Metrics Functions**
  - [ ] `calculate_directional_accuracy(y_true, y_pred)`:
    - Measure if model predicts correct direction of change
    - Useful for volatile series
  - [ ] `calculate_prediction_interval_coverage(y_true, y_pred_lower, y_pred_upper)`:
    - Calculate % of actuals within prediction intervals
    - Target: ~95% for 95% intervals
  - [ ] `calculate_residual_statistics(y_true, y_pred)`:
    - Mean error (bias)
    - Std of errors
    - Max/min errors
    - Skewness

- [ ] **Phase 3: Forecast Visualization Functions**
  - [ ] `plot_forecast_vs_actual(dates, y_true, y_pred, model_name, save_path=None)`:
    - Line plot with actual vs predicted
    - Proper labels, title, legend
    - Optional save to file
    - Return matplotlib figure
  - [ ] `plot_forecast_with_intervals(dates, y_true, y_pred, lower, upper, model_name)`:
    - Forecast plot with prediction intervals
    - Shaded regions for uncertainty
    - Clear visualization of confidence

- [ ] **Phase 4: Residual Visualization Functions**
  - [ ] `plot_residuals(dates, y_true, y_pred, model_name)`:
    - Time series plot of errors
    - Add zero line
    - Identify systematic biases
  - [ ] `plot_residual_distribution(y_true, y_pred, model_name)`:
    - Histogram of residuals
    - Add normal curve overlay
    - Show mean and std

- [ ] **Phase 5: Model Comparison Visualizations**
  - [ ] `plot_multiple_forecasts(dates, y_true, predictions_dict)`:
    - predictions_dict: {model_name: y_pred}
    - Overlay all models on one plot
    - Distinct colors/styles for each model
    - Clear legend
  - [ ] `plot_metrics_comparison(metrics_dict)`:
    - Bar chart comparing metrics across models
    - metrics_dict: {model_name: {metric: value}}
    - Separate subplots for each metric

- [ ] **Phase 6: Error Analysis Functions**
  - [ ] `identify_high_error_periods(dates, y_true, y_pred, threshold_percentile=90)`:
    - Find periods with errors above threshold
    - Return dataframe with dates and errors
    - Useful for understanding model failures
  - [ ] `analyze_errors_by_season(dates, y_true, y_pred)`:
    - Group errors by month
    - Identify seasonal patterns in errors
    - Return statistics by season

- [ ] **Phase 7: Model Summary Functions**
  - [ ] `create_metrics_table(metrics_dict)`:
    - Create pandas DataFrame from metrics_dict
    - Format numbers appropriately
    - Add ranking column for each metric
    - Return styled dataframe
  - [ ] `print_model_summary(model_name, metrics, training_time=None)`:
    - Print formatted summary of model performance
    - Include all metrics
    - Add training time if provided

- [ ] **Phase 8: Visualization Utilities**
  - [ ] `set_plot_style()`:
    - Configure matplotlib style for consistency
    - Set figure size defaults
    - Configure font sizes
  - [ ] `save_all_plots(figures_dict, output_dir)`:
    - Save multiple figures to directory
    - figures_dict: {filename: figure}
    - Create directory if needed

- [ ] **Phase 9: Export Functions**
  - [ ] `export_results_to_csv(dates, y_true, predictions_dict, output_path)`:
    - Save all predictions to CSV
    - Columns: Date, Actual, Model1_Pred, Model2_Pred, ...
  - [ ] `export_metrics_to_csv(metrics_dict, output_path)`:
    - Save metrics comparison table to CSV

- [ ] **Phase 10: Testing and Documentation**
  - [ ] Add comprehensive docstrings to all functions
  - [ ] Include type hints
  - [ ] Add example usage in module docstring
  - [ ] Update `scripts/README.md`
  - [ ] Test all functions with sample data
  - [ ] Ensure consistency with `%matplotlib inline` in notebooks

**Acceptance Criteria**:
- All metrics functions return standardized outputs
- Visualization functions create publication-quality plots
- Comparison functions handle multiple models elegantly
- All functions well-documented with examples
- Module provides complete evaluation toolkit
- Functions used successfully across all notebooks

**Key Outputs**:
- Standardized metrics dictionary
- Consistent visualization style
- Easy-to-use comparison functions
- Publication-ready plots

---

## Issue 11: Documentation and README Updates

**Labels**: `documentation`, `medium-priority`

**Description**:

Update all documentation to reflect the complete project implementation, including data ingestion, model development, and evaluation workflows.

**Context**:

Comprehensive documentation is essential for reproducibility and understanding the project. All README files and documentation should accurately describe the implemented pipeline.

**References**:
- All project documentation files
- Completed notebooks and scripts
- `.github/agents/tourism-forecasting.md` - Deliverables section

**Implementation Tasks**:

- [ ] **Phase 1: Main README.md Updates**
  - [ ] Update project overview with complete description
  - [ ] Add section on forecasting models implemented:
    - Prophet
    - LSTM
    - Chronos
    - N-HiTS
  - [ ] Document notebook workflow (01-07)
  - [ ] Add results summary (key findings)
  - [ ] Include visualization examples
  - [ ] Update installation instructions
  - [ ] Verify all dependencies listed

- [ ] **Phase 2: scripts/README.md Updates**
  - [ ] Document all scripts:
    - ingest_tourism_data.py
    - data_loader.py
    - preprocessing.py
    - evaluation.py
  - [ ] Add usage examples for each script
  - [ ] Document function APIs
  - [ ] Include troubleshooting guide

- [ ] **Phase 3: Notebook Documentation**
  - [ ] Ensure each notebook has:
    - Clear markdown introduction
    - Section headers
    - Explanation of methods
    - Interpretation of results
    - Summary of findings
  - [ ] Verify notebooks run from top to bottom
  - [ ] Add markdown cells between code cells
  - [ ] Document key parameters and choices

- [ ] **Phase 4: Data Documentation**
  - [ ] Create `data/README.md`:
    - Explain data structure
    - Document data sources
    - Describe data cleaning steps
    - Note any data quality issues
  - [ ] Document expected data format
  - [ ] Include sample data statistics

- [ ] **Phase 5: Results Documentation**
  - [ ] Create `RESULTS.md` (optional):
    - Summary of all model performance
    - Metrics comparison table
    - Key visualizations
    - Recommendations
    - Future work suggestions
  - [ ] Make findings accessible to non-technical stakeholders

- [ ] **Phase 6: Requirements and Dependencies**
  - [ ] Verify `requirements.txt` is complete:
    - pandas, numpy
    - matplotlib, seaborn, plotly
    - scikit-learn
    - prophet
    - tensorflow
    - torch (for Chronos)
    - neuralforecast (for N-HiTS)
    - requests, beautifulsoup4, pdfplumber
  - [ ] Add version constraints if needed
  - [ ] Test installation on clean environment

- [ ] **Phase 7: Reproducibility Instructions**
  - [ ] Document complete workflow:
    1. Install dependencies
    2. Run data ingestion
    3. Execute notebooks in order (01-07)
  - [ ] Add expected runtime estimates
  - [ ] Document system requirements
  - [ ] Include troubleshooting for common issues

- [ ] **Phase 8: Citations and References**
  - [ ] Add references section to README:
    - SLTDA data source
    - Model papers (Prophet, LSTM, Chronos, N-HiTS)
    - Sri Lankan tourism literature
  - [ ] Include relevant links
  - [ ] Acknowledge data sources

- [ ] **Phase 9: License and Attribution**
  - [ ] Verify LICENSE file is present (MIT)
  - [ ] Add attribution for data sources
  - [ ] Include acknowledgments if appropriate

- [ ] **Phase 10: Final Review**
  - [ ] Proofread all documentation
  - [ ] Check for broken links
  - [ ] Verify code examples work
  - [ ] Ensure consistency across documents
  - [ ] Test instructions on fresh clone

**Acceptance Criteria**:
- All README files are comprehensive and accurate
- Installation and usage instructions are clear
- Notebooks are well-documented with markdown
- Dependencies are complete and tested
- Project is fully reproducible from documentation
- Documentation accessible to technical and non-technical audiences

**Key Documentation Files**:
- README.md (main)
- scripts/README.md
- data/README.md (to be created)
- RESULTS.md (optional)
- All notebook markdown cells

---

## Issue 12: Testing and Validation Framework

**Labels**: `testing`, `quality-assurance`, `low-priority`

**Description**:

Implement basic testing and validation framework for critical components of the forecasting pipeline.

**Context**:

While comprehensive unit tests may not be required for research notebooks, critical utility functions should have basic tests to ensure correctness and prevent regressions.

**References**:
- `scripts/` modules (data_loader, preprocessing, evaluation)
- `.github/copilot-instructions.md` - Project conventions

**Implementation Tasks**:

- [ ] **Phase 1: Test Setup**
  - [ ] Create `tests/` directory
  - [ ] Add `tests/__init__.py`
  - [ ] Choose testing framework (pytest recommended)
  - [ ] Add pytest to requirements.txt (or requirements-dev.txt)
  - [ ] Create sample test data fixtures

- [ ] **Phase 2: Data Loader Tests**
  - [ ] Create `tests/test_data_loader.py`:
    - Test `load_csv_data()` with valid file
    - Test with date parsing
    - Test with missing file (should raise error)
    - Test `validate_data()` with required columns
    - Test validation with missing columns
  - [ ] Use sample CSV files in `tests/fixtures/`

- [ ] **Phase 3: Preprocessing Tests**
  - [ ] Create `tests/test_preprocessing.py`:
    - Test time feature extraction
    - Test lag feature creation
    - Test rolling features
    - Test chronological split (no data leakage)
    - Test scaling and inverse scaling
  - [ ] Verify no data leakage in any function
  - [ ] Test edge cases (start of series, missing values)

- [ ] **Phase 4: Evaluation Tests**
  - [ ] Create `tests/test_evaluation.py`:
    - Test `calculate_metrics()` with known values
    - Verify metric calculations are correct
    - Test with edge cases (perfect prediction, all zeros)
    - Test visualization functions don't crash
  - [ ] Use simple synthetic data

- [ ] **Phase 5: Data Ingestion Tests (Optional)**
  - [ ] Create `tests/test_ingestion.py`:
    - Test PDF extraction functions with sample PDFs
    - Test month/year extraction
    - Test arrivals extraction
    - Mock HTTP requests for scraping tests
  - [ ] Use small sample PDFs in fixtures

- [ ] **Phase 6: Integration Tests**
  - [ ] Test complete preprocessing pipeline:
    - Load data → Extract features → Split → Scale
    - Verify output format is correct
  - [ ] Test complete evaluation pipeline:
    - Predictions → Metrics → Visualizations
    - Verify all outputs are generated

- [ ] **Phase 7: Validation Scripts**
  - [ ] Create `validate_data.py` script:
    - Check processed data quality
    - Verify date ranges
    - Check for missing values
    - Validate data types
    - Report statistics
  - [ ] Run as part of workflow validation

- [ ] **Phase 8: CI/CD Setup (Optional)**
  - [ ] Create `.github/workflows/tests.yml`:
    - Run tests on push
    - Test on multiple Python versions
    - Check code style (optional: black, flake8)
  - [ ] Add status badge to README

- [ ] **Phase 9: Test Documentation**
  - [ ] Create `tests/README.md`:
    - Explain test structure
    - Document how to run tests
    - List what is tested
  - [ ] Add testing section to main README
  - [ ] Document test coverage

- [ ] **Phase 10: Continuous Validation**
  - [ ] Document validation checklist:
    - Data ingestion produces expected output
    - Train/test split has no leakage
    - All models produce 12-month forecasts
    - Metrics are calculated consistently
  - [ ] Create validation notebook (optional)

**Acceptance Criteria**:
- Basic test suite for utility functions
- All tests pass
- Critical functions have coverage (data loading, preprocessing, metrics)
- Tests prevent common errors (data leakage, incorrect splits)
- Testing documentation is clear
- Tests can be run easily (`pytest tests/`)

**Priority**:
This is lower priority than model implementation but important for:
- Ensuring utility functions work correctly
- Preventing regressions during updates
- Documenting expected behavior

**Note**: 
Research notebooks themselves typically don't need unit tests, but reusable utility functions should have basic coverage.

---

## Summary

These 12 GitHub issues provide a comprehensive implementation roadmap for the Sri Lanka Tourism Forecasting project. The issues are structured to:

1. **Data Pipeline** (Issues 1-3): Ingest, explore, and prepare data
2. **Model Development** (Issues 4-7): Implement four forecasting models
3. **Evaluation** (Issue 8): Compare models and draw conclusions
4. **Infrastructure** (Issues 9-10): Build reusable utilities
5. **Documentation** (Issue 11): Ensure reproducibility
6. **Quality Assurance** (Issue 12): Testing framework

Each issue includes:
- Clear context and motivation
- References to project documentation
- Detailed implementation tasks with checkboxes
- Acceptance criteria
- Model-specific strengths and limitations
- Key technical considerations

The issues follow the project's conventions:
- Chronological data splits (no shuffling)
- Forward fill for missing values
- MinMaxScaler for neural networks
- Consistent evaluation metrics
- Comprehensive documentation

This structure ensures the project can be implemented systematically, with clear milestones and deliverables at each stage.
