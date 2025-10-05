"""
Model evaluation utilities for time series forecasting.

This module provides functions to evaluate and compare forecasting models.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate common forecasting evaluation metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing various metrics
    """
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred)
    }
    
    return metrics


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    float
        MAPE value
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Compare multiple models based on their evaluation metrics.
    
    Parameters
    ----------
    results : Dict[str, Dict[str, float]]
        Dictionary with model names as keys and metrics dictionaries as values
        
    Returns
    -------
    pd.DataFrame
        DataFrame comparing all models
    """
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values('RMSE')
    
    return comparison_df


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                     dates: pd.DatetimeIndex = None,
                     title: str = "Actual vs Predicted") -> plt.Figure:
    """
    Plot actual vs predicted values.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    dates : pd.DatetimeIndex, optional
        Date index for x-axis
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if dates is not None:
        ax.plot(dates, y_true, label='Actual', color='blue', linewidth=2)
        ax.plot(dates, y_pred, label='Predicted', color='red', linewidth=2, alpha=0.7)
    else:
        ax.plot(y_true, label='Actual', color='blue', linewidth=2)
        ax.plot(y_pred, label='Predicted', color='red', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                   dates: pd.DatetimeIndex = None) -> plt.Figure:
    """
    Plot residuals (errors) over time.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    dates : pd.DatetimeIndex, optional
        Date index for x-axis
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Residuals over time
    if dates is not None:
        ax1.plot(dates, residuals, color='purple', linewidth=1)
    else:
        ax1.plot(residuals, color='purple', linewidth=1)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Residuals histogram
    ax2.hist(residuals, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Residual Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residuals Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def cross_validate_timeseries(df: pd.DataFrame, model_func, n_splits: int = 5,
                               test_size: int = 30) -> List[Dict[str, float]]:
    """
    Perform time series cross-validation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input time series data
    model_func : callable
        Function that trains and returns predictions
    n_splits : int
        Number of cross-validation splits
    test_size : int
        Size of test set for each split
        
    Returns
    -------
    List[Dict[str, float]]
        List of metric dictionaries for each fold
    """
    results = []
    total_size = len(df)
    
    for i in range(n_splits):
        # Calculate split indices
        test_end = total_size - (n_splits - i - 1) * test_size
        test_start = test_end - test_size
        train_end = test_start
        
        if train_end <= 0:
            continue
        
        # Split data
        train_data = df.iloc[:train_end]
        test_data = df.iloc[test_start:test_end]
        
        # Train and predict
        y_pred = model_func(train_data, test_data)
        y_true = test_data.iloc[:, -1].values  # Assuming target is last column
        
        # Calculate metrics
        fold_metrics = calculate_metrics(y_true, y_pred)
        results.append(fold_metrics)
    
    return results
