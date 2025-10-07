"""
Preprocessing utilities for time series data.

This module provides functions for data cleaning, transformation,
and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional


def handle_missing_values(df: pd.DataFrame, method: str = "forward_fill") -> pd.DataFrame:
    """
    Handle missing values in the time series data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    method : str
        Method to handle missing values: 'forward_fill', 'backward_fill', or 'interpolate'
        
    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled
    """
    df_copy = df.copy()
    
    if method == "forward_fill":
        df_copy = df_copy.fillna(method='ffill')
    elif method == "backward_fill":
        df_copy = df_copy.fillna(method='bfill')
    elif method == "interpolate":
        df_copy = df_copy.interpolate(method='linear')
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return df_copy


def create_time_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Create time-based features from a date column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_column : str
        Name of the date column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional time features
    """
    df_copy = df.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    
    # Extract time features
    df_copy['year'] = df_copy[date_column].dt.year
    df_copy['month'] = df_copy[date_column].dt.month
    df_copy['quarter'] = df_copy[date_column].dt.quarter
    
    return df_copy


def create_lag_features(df: pd.DataFrame, column: str, lags: list) -> pd.DataFrame:
    """
    Create lagged features for a given column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column name to create lags for
    lags : list
        List of lag periods
        
    Returns
    -------
    pd.DataFrame
        DataFrame with lag features
    """
    df_copy = df.copy()
    
    for lag in lags:
        df_copy[f'{column}_lag_{lag}'] = df_copy[column].shift(lag)
    
    return df_copy


def create_rolling_features(df: pd.DataFrame, column: str, windows: list, min_periods: Optional[int] = None) -> pd.DataFrame:
    """
    Create rolling window statistics for a given column.
    Uses only past data to avoid leakage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column name to create rolling features for
    windows : list
        List of window sizes
    min_periods : Optional[int]
        Minimum number of observations required (defaults to window size)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with rolling features
    """
    df_copy = df.copy()
    
    for window in windows:
        # Set min_periods to window size to avoid partial windows at the start
        periods = min_periods if min_periods is not None else window
        
        df_copy[f'{column}_rolling_mean_{window}'] = df_copy[column].rolling(
            window=window, min_periods=periods).mean()
        df_copy[f'{column}_rolling_std_{window}'] = df_copy[column].rolling(
            window=window, min_periods=periods).std()
    
    return df_copy


def scale_features(df: pd.DataFrame, columns: list, method: str = "standard") -> Tuple[pd.DataFrame, object]:
    """
    Scale features using StandardScaler or MinMaxScaler.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : list
        List of columns to scale
    method : str
        Scaling method: 'standard' or 'minmax'
        
    Returns
    -------
    Tuple[pd.DataFrame, object]
        Scaled DataFrame and the fitted scaler object
    """
    df_copy = df.copy()
    
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    df_copy[columns] = scaler.fit_transform(df_copy[columns])
    
    return df_copy, scaler


def train_test_split_timeseries(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train and test sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    test_size : float
        Proportion of data to use for testing
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Train and test DataFrames
    """
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    
    return train_df, test_df


def create_cyclical_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Create cyclical encoding of month for neural networks.
    Uses sine and cosine transformations to preserve cyclical nature.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_column : str
        Name of the date column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with cyclical month features
    """
    df_copy = df.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    
    # Cyclical encoding for month (12 months)
    df_copy['month_sin'] = np.sin(2 * np.pi * df_copy[date_column].dt.month / 12)
    df_copy['month_cos'] = np.cos(2 * np.pi * df_copy[date_column].dt.month / 12)
    
    return df_copy


def create_intervention_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Create intervention/event features for known structural shocks.
    All features are deterministically known at forecast time.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_column : str
        Name of the date column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with intervention features
    """
    df_copy = df.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    
    # Easter Sunday attacks (April 2019)
    df_copy['easter_attacks'] = ((df_copy[date_column].dt.year == 2019) & 
                                  (df_copy[date_column].dt.month == 4)).astype(int)
    
    # COVID-19 pandemic period (March 2020 - December 2021)
    df_copy['covid_period'] = ((df_copy[date_column] >= '2020-03-01') & 
                                (df_copy[date_column] <= '2021-12-31')).astype(int)
    
    # Economic crisis (2022 onwards)
    df_copy['economic_crisis'] = (df_copy[date_column] >= '2022-01-01').astype(int)
    
    return df_copy


def create_recovery_index(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Create smooth recovery index for post-shock periods.
    This provides a continuous measure of time since major shocks.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_column : str
        Name of the date column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with recovery index
    """
    df_copy = df.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    
    # Recovery index: months since COVID-19 started (for post-COVID recovery tracking)
    covid_start = pd.to_datetime('2020-03-01')
    df_copy['months_since_covid'] = ((df_copy[date_column] - covid_start).dt.days / 30.44).clip(lower=0)
    
    # Normalize to 0-1 range (cap at 60 months for smoother scaling)
    df_copy['recovery_index'] = (df_copy['months_since_covid'] / 60).clip(upper=1.0)
    
    return df_copy
