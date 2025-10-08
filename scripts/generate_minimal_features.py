"""
Generate minimal feature files for Prophet, LSTM, and Chronos models.

This script creates the three required input files:
- data/processed/minimal_features_prophet.csv
- data/processed/minimal_features_lstm.csv  
- data/processed/chronos_series.csv

Based on the problem statement requirements.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import create_time_features


def generate_minimal_prophet_features(input_file: str, output_file: str):
    """
    Generate minimal features for Prophet model.
    
    Expected columns: ds, y, easter_attacks, covid_period, economic_crisis
    """
    print(f"\n{'='*60}")
    print("Generating minimal features for Prophet...")
    print(f"{'='*60}")
    
    # Load the existing prophet features file
    df = pd.read_csv(input_file, parse_dates=['Date'])
    
    # Select only the required columns for Prophet
    df_minimal = df[['Date', 'Arrivals', 'easter_attacks', 'covid_period', 'economic_crisis']].copy()
    
    # Rename columns to Prophet format
    df_minimal.columns = ['ds', 'y', 'easter_attacks', 'covid_period', 'economic_crisis']
    
    # Save
    df_minimal.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")
    print(f"  Shape: {df_minimal.shape}")
    print(f"  Columns: {list(df_minimal.columns)}")
    print(f"  Date range: {df_minimal['ds'].min()} to {df_minimal['ds'].max()}")
    
    return df_minimal


def generate_minimal_lstm_features(input_file: str, output_file: str):
    """
    Generate minimal features for LSTM model.
    
    Expected columns: Date, Arrivals, intervention flags, month_sin, month_cos, 
                     Arrivals_lag_1, Arrivals_lag_12
    """
    print(f"\n{'='*60}")
    print("Generating minimal features for LSTM...")
    print(f"{'='*60}")
    
    # Load the existing ML features file (clean version)
    df = pd.read_csv(input_file, parse_dates=['Date'])
    
    # Select minimal columns for LSTM
    # Include Date, target, interventions, cyclical encodings, and key lags
    required_cols = [
        'Date', 'Arrivals', 
        'easter_attacks', 'covid_period', 'economic_crisis',
        'month_sin', 'month_cos',
        'Arrivals_lag_1', 'Arrivals_lag_12'
    ]
    
    df_minimal = df[required_cols].copy()
    
    # Save
    df_minimal.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")
    print(f"  Shape: {df_minimal.shape}")
    print(f"  Columns: {list(df_minimal.columns)}")
    print(f"  Date range: {df_minimal['Date'].min()} to {df_minimal['Date'].max()}")
    
    return df_minimal


def generate_chronos_series(input_file: str, output_file: str):
    """
    Generate univariate series for Chronos model.
    
    Expected columns: Date, Arrivals
    """
    print(f"\n{'='*60}")
    print("Generating series for Chronos...")
    print(f"{'='*60}")
    
    # Load the filtered data (original univariate series)
    df = pd.read_csv(input_file, parse_dates=['Date'])
    
    # Keep only Date and Arrivals
    df_series = df[['Date', 'Arrivals']].copy()
    
    # Save
    df_series.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")
    print(f"  Shape: {df_series.shape}")
    print(f"  Columns: {list(df_series.columns)}")
    print(f"  Date range: {df_series['Date'].min()} to {df_series['Date'].max()}")
    
    return df_series


def main():
    """Main execution function."""
    # Set paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'processed'
    
    # Input files
    prophet_input = data_dir / 'monthly_tourist_arrivals_features_prophet.csv'
    lstm_input = data_dir / 'monthly_tourist_arrivals_features_ml_clean.csv'
    chronos_input = data_dir / 'monthly_tourist_arrivals_filtered.csv'
    
    # Output files
    prophet_output = data_dir / 'minimal_features_prophet.csv'
    lstm_output = data_dir / 'minimal_features_lstm.csv'
    chronos_output = data_dir / 'chronos_series.csv'
    
    print("\n" + "="*60)
    print("MINIMAL FEATURE GENERATION")
    print("="*60)
    
    # Check if input files exist
    for file, name in [(prophet_input, 'Prophet'), (lstm_input, 'LSTM'), (chronos_input, 'Chronos')]:
        if not file.exists():
            print(f"✗ Error: {name} input file not found: {file}")
            print(f"  Please run feature engineering first: python scripts/feature_engineering.py")
            return 1
    
    # Generate minimal features
    try:
        generate_minimal_prophet_features(str(prophet_input), str(prophet_output))
        generate_minimal_lstm_features(str(lstm_input), str(lstm_output))
        generate_chronos_series(str(chronos_input), str(chronos_output))
        
        print("\n" + "="*60)
        print("✓ MINIMAL FEATURE GENERATION COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print(f"  1. {prophet_output.name}")
        print(f"  2. {lstm_output.name}")
        print(f"  3. {chronos_output.name}")
        print("\nThese files are ready for model training.")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during feature generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
