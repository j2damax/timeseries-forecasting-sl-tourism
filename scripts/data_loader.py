"""
Data loading utilities for time series forecasting.

This module provides functions to load and validate tourism data.
"""

import pandas as pd
import os
from typing import Optional, Union


def load_csv_data(file_path: str, date_column: Optional[str] = None) -> pd.DataFrame:
    """
    Load time series data from a CSV file.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file
    date_column : str, optional
        Name of the date column to parse as datetime
        
    Returns
    -------
    pd.DataFrame
        Loaded data as a pandas DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    if date_column:
        df = pd.read_csv(file_path, parse_dates=[date_column])
    else:
        df = pd.read_csv(file_path)
    
    return df


def validate_data(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that the DataFrame contains required columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : list
        List of required column names
        
    Returns
    -------
    bool
        True if all required columns are present
        
    Raises
    ------
    ValueError
        If any required columns are missing
    """
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True


def load_tourism_data(data_dir: str = "../data") -> pd.DataFrame:
    """
    Load tourism data from the data directory.
    
    Parameters
    ----------
    data_dir : str
        Path to the data directory
        
    Returns
    -------
    pd.DataFrame
        Tourism data as a pandas DataFrame
    """
    # This is a placeholder - update with actual file path when data is available
    data_path = os.path.join(data_dir, "tourism_data.csv")
    return load_csv_data(data_path, date_column="date")
