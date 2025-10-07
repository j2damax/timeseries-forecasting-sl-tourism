"""
Feature Engineering Pipeline for Sri Lanka Tourism Forecasting.

This module provides a reusable pipeline for creating features from
the raw tourism data. It can be used both in notebooks and in production
prediction pipelines.

Design Principles:
- Minimal features to avoid overfitting (small dataset: 91 observations)
- No data leakage (rolling stats use only past data)
- All features are deterministically known at forecast time
- Separate feature sets for different model types

Usage:
    from feature_engineering import create_all_features, create_prophet_features, create_ml_features
    
    # For all features
    df_full = create_all_features(df, date_column='Date', target_column='Arrivals')
    
    # For Prophet
    df_prophet = create_prophet_features(df, date_column='Date', target_column='Arrivals')
    
    # For ML/DL models
    df_ml = create_ml_features(df, date_column='Date', target_column='Arrivals', include_lags=True)
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from preprocessing import (
    create_time_features,
    create_cyclical_features,
    create_lag_features,
    create_rolling_features,
    create_intervention_features,
    create_recovery_index
)


def create_all_features(
    df: pd.DataFrame,
    date_column: str = 'Date',
    target_column: str = 'Arrivals',
    lag_periods: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Create all features for tourism forecasting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with date and target columns
    date_column : str
        Name of the date column
    target_column : str
        Name of the target variable column
    lag_periods : Optional[List[int]]
        List of lag periods (default: [1, 3, 6, 12])
    rolling_windows : Optional[List[int]]
        List of rolling window sizes (default: [3, 12])
        
    Returns
    -------
    pd.DataFrame
        DataFrame with all engineered features
    """
    if lag_periods is None:
        lag_periods = [1, 3, 6, 12]
    
    if rolling_windows is None:
        rolling_windows = [3, 12]
    
    # Make a copy to avoid modifying original
    df_result = df.copy()
    
    # Ensure date is datetime
    df_result[date_column] = pd.to_datetime(df_result[date_column])
    
    # Sort by date
    df_result = df_result.sort_values(date_column).reset_index(drop=True)
    
    # 1. Core temporal features
    df_result = create_time_features(df_result, date_column)
    
    # 2. Cyclical encoding
    df_result = create_cyclical_features(df_result, date_column)
    
    # 3. Intervention features
    df_result = create_intervention_features(df_result, date_column)
    
    # 4. Recovery index
    df_result = create_recovery_index(df_result, date_column)
    
    # 5. Lag features
    df_result = create_lag_features(df_result, target_column, lag_periods)
    
    # 6. Rolling features (optional)
    if rolling_windows:
        df_result = create_rolling_features(df_result, target_column, rolling_windows)
    
    return df_result


def create_prophet_features(
    df: pd.DataFrame,
    date_column: str = 'Date',
    target_column: str = 'Arrivals'
) -> pd.DataFrame:
    """
    Create features specifically for Prophet model.
    
    Prophet handles seasonality internally, so we only need:
    - Date (as 'ds')
    - Target (as 'y')
    - Intervention features (as regressors)
    - Recovery index (as regressor)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    date_column : str
        Name of the date column
    target_column : str
        Name of the target variable column
        
    Returns
    -------
    pd.DataFrame
        DataFrame formatted for Prophet with regressors
    """
    df_result = df.copy()
    
    # Ensure date is datetime
    df_result[date_column] = pd.to_datetime(df_result[date_column])
    
    # Sort by date
    df_result = df_result.sort_values(date_column).reset_index(drop=True)
    
    # Core temporal features (for reference, Prophet won't use these directly)
    df_result = create_time_features(df_result, date_column)
    
    # Intervention features (will be used as regressors)
    df_result = create_intervention_features(df_result, date_column)
    
    # Recovery index (will be used as regressor)
    df_result = create_recovery_index(df_result, date_column)
    
    # Prophet needs 'ds' for date and 'y' for target
    df_result['ds'] = df_result[date_column]
    df_result['y'] = df_result[target_column]
    
    return df_result


def create_ml_features(
    df: pd.DataFrame,
    date_column: str = 'Date',
    target_column: str = 'Arrivals',
    include_lags: bool = True,
    include_rolling: bool = False,
    lag_periods: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    drop_na: bool = False
) -> pd.DataFrame:
    """
    Create features for ML/DL models (LSTM, N-HiTS).
    
    Includes:
    - Core temporal features
    - Cyclical encoding
    - Intervention features
    - Recovery index
    - Optional lag features
    - Optional rolling features
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    date_column : str
        Name of the date column
    target_column : str
        Name of the target variable column
    include_lags : bool
        Whether to include lag features
    include_rolling : bool
        Whether to include rolling window features
    lag_periods : Optional[List[int]]
        List of lag periods (default: [1, 3, 6, 12])
    rolling_windows : Optional[List[int]]
        List of rolling window sizes (default: [3, 12])
    drop_na : bool
        Whether to drop rows with NaN values
        
    Returns
    -------
    pd.DataFrame
        DataFrame with ML/DL features
    """
    if lag_periods is None:
        lag_periods = [1, 3, 6, 12]
    
    if rolling_windows is None:
        rolling_windows = [3, 12]
    
    df_result = df.copy()
    
    # Ensure date is datetime
    df_result[date_column] = pd.to_datetime(df_result[date_column])
    
    # Sort by date
    df_result = df_result.sort_values(date_column).reset_index(drop=True)
    
    # 1. Core temporal features
    df_result = create_time_features(df_result, date_column)
    
    # 2. Cyclical encoding (important for neural networks)
    df_result = create_cyclical_features(df_result, date_column)
    
    # 3. Intervention features
    df_result = create_intervention_features(df_result, date_column)
    
    # 4. Recovery index
    df_result = create_recovery_index(df_result, date_column)
    
    # 5. Lag features (optional)
    if include_lags:
        df_result = create_lag_features(df_result, target_column, lag_periods)
    
    # 6. Rolling features (optional)
    if include_rolling:
        df_result = create_rolling_features(df_result, target_column, rolling_windows)
    
    # 7. Drop NaN if requested
    if drop_na:
        df_result = df_result.dropna().reset_index(drop=True)
    
    return df_result


def get_feature_names(feature_type: str = 'all') -> List[str]:
    """
    Get list of feature names for different model types.
    
    Parameters
    ----------
    feature_type : str
        Type of features: 'core', 'intervention', 'cyclical', 'lag', 'rolling', 'all'
        
    Returns
    -------
    List[str]
        List of feature column names
    """
    feature_dict = {
        'core': ['year', 'month', 'quarter'],
        'intervention': ['easter_attacks', 'covid_period', 'economic_crisis', 'recovery_index', 'months_since_covid'],
        'cyclical': ['month_sin', 'month_cos'],
        'lag': ['Arrivals_lag_1', 'Arrivals_lag_3', 'Arrivals_lag_6', 'Arrivals_lag_12'],
        'rolling': ['Arrivals_rolling_mean_3', 'Arrivals_rolling_mean_12', 
                   'Arrivals_rolling_std_3', 'Arrivals_rolling_std_12']
    }
    
    if feature_type == 'all':
        all_features = []
        for features in feature_dict.values():
            all_features.extend(features)
        return all_features
    
    return feature_dict.get(feature_type, [])


if __name__ == '__main__':
    # Example usage
    print("Feature Engineering Module for Sri Lanka Tourism Forecasting")
    print("\nAvailable functions:")
    print("  - create_all_features(): Create all features")
    print("  - create_prophet_features(): Create features for Prophet model")
    print("  - create_ml_features(): Create features for ML/DL models")
    print("  - get_feature_names(): Get feature name lists")
    
    print("\nExample usage:")
    print("  from feature_engineering import create_ml_features")
    print("  df = pd.read_csv('data/processed/monthly_tourist_arrivals_filtered.csv')")
    print("  df_features = create_ml_features(df, include_lags=True, drop_na=True)")
