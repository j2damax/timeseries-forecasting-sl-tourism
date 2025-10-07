"""
Minimal Feature Engineering Pipeline for Sri Lanka Tourism Forecasting.

This module provides a streamlined pipeline focused on Prophet, LSTM, and Chronos models only.

Design Principles:
- Minimal features to avoid overfitting (dataset: 91 observations)
- No data leakage (lags use only past data)
- All features are deterministically known at forecast time
- Simplified feature sets for Prophet, LSTM, and Chronos models

Key Changes from Previous Version:
- Only 2 lag features: lag_1 and lag_12 (short-term + annual seasonality)
- Removed rolling statistics (too few observations)
- Simplified intervention flags: easter_impact, covid, econ_crisis, recovery
- Removed recovery_index (replaced by binary recovery flag)
- Removed optional features and complexity

Usage:
    from feature_engineering import create_minimal_features, create_prophet_data, create_lstm_data
    
    # For minimal base features
    df_features = create_minimal_features(df)
    
    # For Prophet
    df_prophet = create_prophet_data(df)
    
    # For LSTM
    df_lstm = create_lstm_data(df)
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from preprocessing import (
    create_cyclical_features,
    create_lag_features,
    create_intervention_features
)


def create_minimal_features(
    df: pd.DataFrame,
    date_column: str = 'Date',
    target_column: str = 'Arrivals'
) -> pd.DataFrame:
    """
    Create minimal feature set for tourism forecasting (Prophet, LSTM, Chronos).
    
    Features created:
    - Intervention flags: easter_impact, covid, econ_crisis, recovery
    - Cyclical month encoding: month_sin, month_cos
    - Lag features: lag_1, lag_12
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with date and target columns
    date_column : str
        Name of the date column
    target_column : str
        Name of the target variable column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with minimal engineered features
    """
    df_result = df.copy()
    
    # Ensure date is datetime and set as index
    df_result[date_column] = pd.to_datetime(df_result[date_column])
    df_result = df_result.sort_values(date_column).reset_index(drop=True)
    
    # 1. Intervention features (binary flags)
    df_result = create_intervention_features(df_result, date_column)
    
    # 2. Cyclical month encoding (for LSTM)
    df_result = create_cyclical_features(df_result, date_column)
    
    # 3. Minimal lag features (1 and 12 months only)
    df_result = create_lag_features(df_result, target_column, [1, 12])
    
    return df_result


def create_prophet_data(
    df: pd.DataFrame,
    date_column: str = 'Date',
    target_column: str = 'Arrivals'
) -> pd.DataFrame:
    """
    Create data specifically formatted for Prophet model.
    
    Prophet handles seasonality internally, so we only include:
    - Date (as 'ds')
    - Target (as 'y')
    - Intervention features as regressors
    
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
        DataFrame formatted for Prophet (ds, y, and regressor columns)
    """
    df_result = df.copy()
    
    # Ensure date is datetime
    df_result[date_column] = pd.to_datetime(df_result[date_column])
    df_result = df_result.sort_values(date_column).reset_index(drop=True)
    
    # Add intervention features only (Prophet handles seasonality)
    df_result = create_intervention_features(df_result, date_column)
    
    # Rename for Prophet format
    df_result = df_result.rename(columns={
        date_column: 'ds',
        target_column: 'y'
    })
    
    # Keep only necessary columns
    prophet_cols = ['ds', 'y', 'easter_impact', 'covid', 'econ_crisis', 'recovery']
    df_result = df_result[prophet_cols]
    
    return df_result


def create_lstm_data(
    df: pd.DataFrame,
    date_column: str = 'Date',
    target_column: str = 'Arrivals',
    drop_na: bool = True
) -> pd.DataFrame:
    """
    Create data for LSTM model with all features.
    
    Includes:
    - Target variable
    - Lag features (lag_1, lag_12)
    - Cyclical month encoding (month_sin, month_cos)
    - Intervention features
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    date_column : str
        Name of the date column
    target_column : str
        Name of the target variable column
    drop_na : bool
        Whether to drop rows with NaN from lag features
        
    Returns
    -------
    pd.DataFrame
        DataFrame with features for LSTM
    """
    df_result = create_minimal_features(df, date_column, target_column)
    
    if drop_na:
        df_result = df_result.dropna().reset_index(drop=True)
    
    return df_result


def split_train_val_test(
    df: pd.DataFrame,
    date_column: str = 'Date',
    train_end: str = '2022-12-01',
    val_end: str = '2023-12-01'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets chronologically.
    
    Default split:
    - Train: 2017-01 to 2022-12 (72 months)
    - Validation: 2023-01 to 2023-12 (12 months)
    - Test: 2024-01 to 2024-07 (7 months)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with date column
    date_column : str
        Name of the date column
    train_end : str
        Last date for training set (inclusive)
    val_end : str
        Last date for validation set (inclusive)
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Train, validation, and test DataFrames
    """
    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    df_copy = df_copy.sort_values(date_column).reset_index(drop=True)
    
    train_df = df_copy[df_copy[date_column] <= train_end].copy()
    val_df = df_copy[(df_copy[date_column] > train_end) & 
                     (df_copy[date_column] <= val_end)].copy()
    test_df = df_copy[df_copy[date_column] > val_end].copy()
    
    return train_df, val_df, test_df


def create_lstm_sequences(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    window_size: int = 24,
    forecast_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for LSTM training.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with features
    feature_columns : List[str]
        List of feature column names to use as inputs
    target_column : str
        Name of target column
    window_size : int
        Number of past time steps to use as input (default: 24 months)
    forecast_horizon : int
        Number of steps ahead to forecast (default: 1 month)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X (sequences) and y (targets) arrays
    """
    X, y = [], []
    
    feature_data = df[feature_columns].values
    target_data = df[target_column].values
    
    for i in range(len(df) - window_size - forecast_horizon + 1):
        X.append(feature_data[i:i + window_size])
        y.append(target_data[i + window_size + forecast_horizon - 1])
    
    return np.array(X), np.array(y)


def get_feature_columns() -> dict:
    """
    Get lists of feature columns for different purposes.
    
    Returns
    -------
    dict
        Dictionary with feature column lists
    """
    return {
        'interventions': ['easter_impact', 'covid', 'econ_crisis', 'recovery'],
        'cyclical': ['month_sin', 'month_cos'],
        'lags': ['Arrivals_lag_1', 'Arrivals_lag_12'],
        'lstm_features': ['Arrivals', 'Arrivals_lag_1', 'Arrivals_lag_12', 'month_sin', 'month_cos', 
                         'easter_impact', 'covid', 'econ_crisis', 'recovery'],
        'prophet_regressors': ['easter_impact', 'covid', 'econ_crisis', 'recovery']
    }



if __name__ == '__main__':
    # Example usage
    print("Minimal Feature Engineering Module for Sri Lanka Tourism Forecasting")
    print("\nFocused on Prophet, LSTM, and Chronos models only")
    print("\nAvailable functions:")
    print("  - create_minimal_features(): Create minimal feature set (all models)")
    print("  - create_prophet_data(): Create Prophet-formatted data")
    print("  - create_lstm_data(): Create LSTM features")
    print("  - split_train_val_test(): Split data chronologically")
    print("  - create_lstm_sequences(): Create sliding window sequences")
    print("  - get_feature_columns(): Get feature column lists")
    
    print("\nExample usage:")
    print("  from feature_engineering import create_lstm_data, split_train_val_test")
    print("  df = pd.read_csv('data/processed/monthly_tourist_arrivals_filtered.csv')")
    print("  df_lstm = create_lstm_data(df, drop_na=True)")
    print("  train, val, test = split_train_val_test(df_lstm)")

