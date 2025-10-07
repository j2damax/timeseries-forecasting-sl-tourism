"""
Chronological data splitting utilities.

This module provides functions to split time series data chronologically
according to fixed date boundaries defined in the problem statement.

Split boundaries:
- Train: 2017-01-01 to 2022-12-01 (inclusive)
- Validation: 2023-01-01 to 2023-12-01
- Test: 2024-01-01 to 2024-07-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Tuple, Dict
from datetime import datetime


# Fixed date boundaries per problem statement
TRAIN_START = pd.Timestamp('2017-01-01')
TRAIN_END = pd.Timestamp('2022-12-01')
VAL_START = pd.Timestamp('2023-01-01')
VAL_END = pd.Timestamp('2023-12-01')
TEST_START = pd.Timestamp('2024-01-01')
TEST_END = pd.Timestamp('2024-07-01')


def validate_chronological_split(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                                  test_df: pd.DataFrame, date_col: str = 'Date') -> bool:
    """
    Validate that splits are chronological with no overlap.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    val_df : pd.DataFrame
        Validation data
    test_df : pd.DataFrame
        Test data
    date_col : str
        Name of the date column
        
    Returns
    -------
    bool
        True if splits are valid, raises ValueError otherwise
    """
    train_max = train_df[date_col].max()
    val_min = val_df[date_col].min()
    val_max = val_df[date_col].max()
    test_min = test_df[date_col].min()
    
    if train_max >= val_min:
        raise ValueError(f"Train max ({train_max}) >= Val min ({val_min}). Data leakage detected!")
    
    if val_max >= test_min:
        raise ValueError(f"Val max ({val_max}) >= Test min ({test_min}). Data leakage detected!")
    
    print("✓ Chronological split validation passed: No future leakage detected")
    return True


def split_data_chronologically(df: pd.DataFrame, date_col: str = 'Date',
                                train_start: pd.Timestamp = TRAIN_START,
                                train_end: pd.Timestamp = TRAIN_END,
                                val_start: pd.Timestamp = VAL_START,
                                val_end: pd.Timestamp = VAL_END,
                                test_start: pd.Timestamp = TEST_START,
                                test_end: pd.Timestamp = TEST_END) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically using fixed date boundaries.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with date column
    date_col : str
        Name of the date column
    train_start, train_end, val_start, val_end, test_start, test_end : pd.Timestamp
        Date boundaries for splits
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Sort by date to ensure chronological order
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Create boolean masks for each split
    train_mask = (df[date_col] >= train_start) & (df[date_col] <= train_end)
    val_mask = (df[date_col] >= val_start) & (df[date_col] <= val_end)
    test_mask = (df[date_col] >= test_start) & (df[date_col] <= test_end)
    
    # Split data
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    # Validate splits
    validate_chronological_split(train_df, val_df, test_df, date_col)
    
    return train_df, val_df, test_df


def get_split_info(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                   test_df: pd.DataFrame, date_col: str = 'Date') -> Dict:
    """
    Get information about data splits.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    val_df : pd.DataFrame
        Validation data
    test_df : pd.DataFrame
        Test data
    date_col : str
        Name of the date column
        
    Returns
    -------
    Dict
        Dictionary containing split information
    """
    info = {
        'train': {
            'start': str(train_df[date_col].min()),
            'end': str(train_df[date_col].max()),
            'n_samples': len(train_df)
        },
        'validation': {
            'start': str(val_df[date_col].min()),
            'end': str(val_df[date_col].max()),
            'n_samples': len(val_df)
        },
        'test': {
            'start': str(test_df[date_col].min()),
            'end': str(test_df[date_col].max()),
            'n_samples': len(test_df)
        },
        'total_samples': len(train_df) + len(val_df) + len(test_df)
    }
    
    return info


def save_split_info(split_info: Dict, output_file: str):
    """
    Save split information to JSON file.
    
    Parameters
    ----------
    split_info : Dict
        Split information dictionary
    output_file : str
        Path to output JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"✓ Split info saved to: {output_file}")


def print_split_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                        test_df: pd.DataFrame, date_col: str = 'Date'):
    """
    Print a summary of the data splits.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    val_df : pd.DataFrame
        Validation data
    test_df : pd.DataFrame
        Test data
    date_col : str
        Name of the date column
    """
    print("\n" + "="*60)
    print("DATA SPLIT SUMMARY")
    print("="*60)
    
    print(f"\nTrain:")
    print(f"  Period: {train_df[date_col].min()} to {train_df[date_col].max()}")
    print(f"  Samples: {len(train_df)}")
    
    print(f"\nValidation:")
    print(f"  Period: {val_df[date_col].min()} to {val_df[date_col].max()}")
    print(f"  Samples: {len(val_df)}")
    
    print(f"\nTest:")
    print(f"  Period: {test_df[date_col].min()} to {test_df[date_col].max()}")
    print(f"  Samples: {len(test_df)}")
    
    print(f"\nTotal samples: {len(train_df) + len(val_df) + len(test_df)}")
    print("="*60)


def main():
    """Demonstration of data splitting."""
    # Load minimal features for demonstration
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'processed'
    
    # Example with Prophet features
    prophet_file = data_dir / 'minimal_features_prophet.csv'
    df = pd.read_csv(prophet_file, parse_dates=['ds'])
    
    # Split data
    train_df, val_df, test_df = split_data_chronologically(df, date_col='ds')
    
    # Print summary
    print_split_summary(train_df, val_df, test_df, date_col='ds')
    
    # Get and save split info
    split_info = get_split_info(train_df, val_df, test_df, date_col='ds')
    output_file = data_dir / 'splits.json'
    save_split_info(split_info, str(output_file))


if __name__ == "__main__":
    main()
