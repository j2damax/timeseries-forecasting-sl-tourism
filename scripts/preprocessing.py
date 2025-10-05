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
    df_copy['day'] = df_copy[date_column].dt.day
    df_copy['day_of_week'] = df_copy[date_column].dt.dayofweek
    df_copy['day_of_year'] = df_copy[date_column].dt.dayofyear
    df_copy['quarter'] = df_copy[date_column].dt.quarter
    df_copy['is_weekend'] = df_copy['day_of_week'].isin([5, 6]).astype(int)
    
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


def create_rolling_features(df: pd.DataFrame, column: str, windows: list) -> pd.DataFrame:
    """
    Create rolling window statistics for a given column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column name to create rolling features for
    windows : list
        List of window sizes
        
    Returns
    -------
    pd.DataFrame
        DataFrame with rolling features
    """
    df_copy = df.copy()
    
    for window in windows:
        df_copy[f'{column}_rolling_mean_{window}'] = df_copy[column].rolling(window=window).mean()
        df_copy[f'{column}_rolling_std_{window}'] = df_copy[column].rolling(window=window).std()
        df_copy[f'{column}_rolling_min_{window}'] = df_copy[column].rolling(window=window).min()
        df_copy[f'{column}_rolling_max_{window}'] = df_copy[column].rolling(window=window).max()
    
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
