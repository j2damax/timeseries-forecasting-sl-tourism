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
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description='Feature Engineering Pipeline for Sri Lanka Tourism Forecasting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate all feature sets (default behavior)
  python feature_engineering.py
  
  # Generate specific feature set
  python feature_engineering.py --type ml
  python feature_engineering.py --type prophet
  python feature_engineering.py --type all
  
  # Use custom input/output paths
  python feature_engineering.py --input data/my_data.csv --output data/my_features.csv
  
  # ML features with custom options
  python feature_engineering.py --type ml --no-lags --with-rolling --drop-na
        '''
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='../data/processed/monthly_tourist_arrivals_filtered.csv',
        help='Input CSV file path (default: ../data/processed/monthly_tourist_arrivals_filtered.csv)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../data/processed',
        help='Output directory for generated feature files (default: ../data/processed)'
    )
    
    parser.add_argument(
        '--type',
        type=str,
        choices=['all', 'prophet', 'ml', 'full'],
        default='full',
        help='Type of features to generate: all=all types, prophet=Prophet model, ml=ML/DL models, full=complete feature set (default: full)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Custom output file path (overrides --output-dir and default naming)'
    )
    
    # ML-specific options
    parser.add_argument(
        '--no-lags',
        action='store_true',
        help='Exclude lag features (ML type only)'
    )
    
    parser.add_argument(
        '--with-rolling',
        action='store_true',
        help='Include rolling window features (ML type only)'
    )
    
    parser.add_argument(
        '--drop-na',
        action='store_true',
        help='Drop rows with NaN values (ML type only)'
    )
    
    args = parser.parse_args()
    
    # Load input data
    try:
        print(f"Loading data from: {args.input}")
        df = pd.read_csv(args.input)
        print(f"✓ Loaded {len(df)} rows")
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Generate features based on type
    from pathlib import Path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.type == 'full' or args.type == 'all':
        # Generate all feature sets
        print("\n" + "="*60)
        print("Generating all feature sets...")
        print("="*60)
        
        # 1. Full features
        df_full = create_all_features(df)
        output_file = output_dir / 'monthly_tourist_arrivals_features_full.csv'
        df_full.to_csv(output_file, index=False)
        print(f"✓ Full features: {output_file} ({df_full.shape})")
        
        # 2. Prophet features
        df_prophet = create_prophet_features(df)
        # Save without ds/y columns for storage, keep minimal
        df_prophet_save = df_prophet[['Date', 'Arrivals', 'year', 'month', 'quarter',
                                      'easter_attacks', 'covid_period', 'economic_crisis',
                                      'recovery_index']]
        output_file = output_dir / 'monthly_tourist_arrivals_features_prophet.csv'
        df_prophet_save.to_csv(output_file, index=False)
        print(f"✓ Prophet features: {output_file} ({df_prophet_save.shape})")
        
        # 3. ML features (with NaN)
        df_ml = create_ml_features(df, include_lags=True, drop_na=False)
        output_file = output_dir / 'monthly_tourist_arrivals_features_ml.csv'
        df_ml.to_csv(output_file, index=False)
        print(f"✓ ML features (with NaN): {output_file} ({df_ml.shape})")
        
        # 4. ML features (clean)
        df_ml_clean = create_ml_features(df, include_lags=True, drop_na=True)
        output_file = output_dir / 'monthly_tourist_arrivals_features_ml_clean.csv'
        df_ml_clean.to_csv(output_file, index=False)
        print(f"✓ ML features (clean): {output_file} ({df_ml_clean.shape})")
        
        print("\n" + "="*60)
        print("Feature generation complete!")
        print("="*60)
        
    elif args.type == 'prophet':
        print("\nGenerating Prophet features...")
        df_features = create_prophet_features(df)
        
        if args.output:
            output_file = Path(args.output)
        else:
            output_file = output_dir / 'features_prophet.csv'
        
        df_features.to_csv(output_file, index=False)
        print(f"✓ Saved: {output_file} ({df_features.shape})")
        
    elif args.type == 'ml':
        print("\nGenerating ML/DL features...")
        df_features = create_ml_features(
            df,
            include_lags=not args.no_lags,
            include_rolling=args.with_rolling,
            drop_na=args.drop_na
        )
        
        if args.output:
            output_file = Path(args.output)
        else:
            suffix = '_clean' if args.drop_na else ''
            output_file = output_dir / f'features_ml{suffix}.csv'
        
        df_features.to_csv(output_file, index=False)
        print(f"✓ Saved: {output_file} ({df_features.shape})")
    
    print("\nFeature Engineering Complete ✓")
