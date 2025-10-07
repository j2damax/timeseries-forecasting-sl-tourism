"""
Generate minimal feature datasets for Prophet, LSTM, and Chronos models.

This script creates the streamlined feature sets according to the minimal
feature engineering plan:
- Only lag_1 and lag_12
- Binary intervention flags (easter_impact, covid, econ_crisis, recovery)
- Cyclical month encoding
- No rolling statistics
- No recovery_index (replaced by recovery flag)

Output files:
1. engineered_features.csv - Base features with all columns
2. prophet_regressors.csv - Prophet-specific format (ds, y, regressors)
3. lstm_train.npz, lstm_val.npz, lstm_test.npz - LSTM sequences
4. scaler.pkl - MinMaxScaler for LSTM
5. chronos_series.npy - Raw arrivals tensor for Chronos

Data splits:
- Train: 2017-01 to 2022-12
- Validation: 2023-01 to 2023-12
- Test: 2024-01 to 2024-07
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from feature_engineering import (
    create_minimal_features,
    create_prophet_data,
    create_lstm_data,
    split_train_val_test,
    create_lstm_sequences,
    get_feature_columns
)

from sklearn.preprocessing import MinMaxScaler


def main():
    """Generate all minimal feature datasets."""
    print("=" * 70)
    print("MINIMAL FEATURE ENGINEERING FOR PROPHET, LSTM, CHRONOS")
    print("=" * 70)
    print()
    
    # Load base data
    print("1. Loading base dataset...")
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'monthly_tourist_arrivals_filtered.csv'
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"   Loaded {len(df)} rows from {df['Date'].min()} to {df['Date'].max()}")
    print()
    
    # Generate minimal features
    print("2. Creating minimal features...")
    df_features = create_minimal_features(df)
    print(f"   Features: {list(df_features.columns)}")
    print(f"   Shape: {df_features.shape}")
    
    # Drop rows with NaN from lag_12
    df_clean = df_features.dropna().reset_index(drop=True)
    print(f"   After dropping NaN: {df_clean.shape} (starts from {df_clean['Date'].min()})")
    print()
    
    # Save engineered features
    output_dir = Path(__file__).parent.parent / 'data' / 'processed'
    features_path = output_dir / 'engineered_features.csv'
    df_clean.to_csv(features_path, index=False)
    print(f"3. Saved: {features_path}")
    print()
    
    # Create Prophet data
    print("4. Creating Prophet dataset...")
    df_prophet = create_prophet_data(df)
    prophet_path = output_dir / 'prophet_regressors.csv'
    df_prophet.to_csv(prophet_path, index=False)
    print(f"   Columns: {list(df_prophet.columns)}")
    print(f"   Shape: {df_prophet.shape}")
    print(f"   Saved: {prophet_path}")
    print()
    
    # Create LSTM data with train/val/test splits
    print("5. Creating LSTM datasets with splits...")
    df_lstm = create_lstm_data(df, drop_na=True)
    
    # Split data
    train_df, val_df, test_df = split_train_val_test(
        df_lstm,
        train_end='2022-12-01',
        val_end='2023-12-01'
    )
    
    print(f"   Train: {len(train_df)} rows ({train_df['Date'].min()} to {train_df['Date'].max()})")
    print(f"   Val:   {len(val_df)} rows ({val_df['Date'].min()} to {val_df['Date'].max()})")
    print(f"   Test:  {len(test_df)} rows ({test_df['Date'].min()} to {test_df['Date'].max()})")
    print()
    
    # Get feature columns for LSTM
    feature_cols_dict = get_feature_columns()
    lstm_feature_cols = feature_cols_dict['lstm_features']
    
    # Scale features (fit on train only)
    print("6. Scaling features for LSTM...")
    scaler = MinMaxScaler()
    
    # Separate numeric columns (exclude binary interventions for now, scale them too for consistency)
    numeric_cols = ['Arrivals', 'Arrivals_lag_1', 'Arrivals_lag_12', 'month_sin', 'month_cos']
    binary_cols = ['easter_impact', 'covid', 'econ_crisis', 'recovery']
    
    # Fit scaler on train data (all numeric features)
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()
    
    # Scale numeric features
    train_scaled[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    val_scaled[numeric_cols] = scaler.transform(val_df[numeric_cols])
    test_scaled[numeric_cols] = scaler.transform(test_df[numeric_cols])
    
    # Keep binary features as-is (0/1)
    # They're already in train_scaled, val_scaled, test_scaled
    
    print(f"   Scaled features: {numeric_cols}")
    print(f"   Binary features (kept 0/1): {binary_cols}")
    print()
    
    # Save scaler
    scaler_path = output_dir / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   Saved scaler: {scaler_path}")
    print()
    
    # Create LSTM sequences (24-month window -> 1-month ahead)
    print("7. Creating LSTM sequences (window=24, horizon=1)...")
    window_size = 24
    
    # Feature columns for sequences (exclude Date)
    seq_feature_cols = [col for col in lstm_feature_cols if col != 'Date']
    
    # Combine train + val + test for creating sequences, then split by target date
    # This ensures we have enough history for validation and test predictions
    all_data_scaled = pd.concat([train_scaled, val_scaled, test_scaled]).reset_index(drop=True)
    
    # Create all sequences
    X_all, y_all = create_lstm_sequences(
        all_data_scaled, seq_feature_cols, 'Arrivals', window_size=window_size
    )
    
    # Get corresponding dates for each sequence (date of the target)
    dates_all = all_data_scaled['Date'].iloc[window_size:].reset_index(drop=True)
    
    # Split sequences based on target date
    train_mask = dates_all <= '2022-12-01'
    val_mask = (dates_all > '2022-12-01') & (dates_all <= '2023-12-01')
    test_mask = dates_all > '2023-12-01'
    
    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    
    print(f"   Train sequences: X={X_train.shape}, y={y_train.shape}")
    print(f"   Val sequences:   X={X_val.shape}, y={y_val.shape}")
    print(f"   Test sequences:  X={X_test.shape}, y={y_test.shape}")
    print()
    
    # Save sequences
    np.savez(output_dir / 'lstm_train.npz', X=X_train, y=y_train)
    np.savez(output_dir / 'lstm_val.npz', X=X_val, y=y_val)
    np.savez(output_dir / 'lstm_test.npz', X=X_test, y=y_test)
    
    print(f"   Saved: lstm_train.npz, lstm_val.npz, lstm_test.npz")
    print()
    
    # Create Chronos data (raw arrivals series)
    print("8. Creating Chronos dataset...")
    # Context: train + val (2017-01 to 2023-12)
    context_df = df[df['Date'] <= '2023-12-01'].copy()
    test_df_chronos = df[df['Date'] > '2023-12-01'].copy()
    
    chronos_context = context_df['Arrivals'].values
    chronos_test = test_df_chronos['Arrivals'].values
    
    print(f"   Context series: {len(chronos_context)} months ({context_df['Date'].min()} to {context_df['Date'].max()})")
    print(f"   Test series: {len(chronos_test)} months ({test_df_chronos['Date'].min()} to {test_df_chronos['Date'].max()})")
    
    # Save as numpy arrays
    np.save(output_dir / 'chronos_context.npy', chronos_context)
    np.save(output_dir / 'chronos_test.npy', chronos_test)
    
    print(f"   Saved: chronos_context.npy, chronos_test.npy")
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY - Generated Files:")
    print("=" * 70)
    print(f"1. engineered_features.csv       - Base features ({df_clean.shape})")
    print(f"2. prophet_regressors.csv        - Prophet format ({df_prophet.shape})")
    print(f"3. lstm_train.npz                - LSTM train sequences")
    print(f"4. lstm_val.npz                  - LSTM validation sequences")
    print(f"5. lstm_test.npz                 - LSTM test sequences")
    print(f"6. scaler.pkl                    - MinMaxScaler for LSTM")
    print(f"7. chronos_context.npy           - Chronos context ({len(chronos_context)} values)")
    print(f"8. chronos_test.npy              - Chronos test targets ({len(chronos_test)} values)")
    print()
    print("Feature Engineering Complete! ✓")
    print("=" * 70)


if __name__ == '__main__':
    main()
