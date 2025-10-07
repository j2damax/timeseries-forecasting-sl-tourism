"""
Test script for minimal feature engineering validation.

This script validates the simplified feature engineering focused on
Prophet, LSTM, and Chronos models only.

Tests:
1. Minimal features have correct structure
2. New intervention features (easter_impact, covid, econ_crisis, recovery)
3. Only lag_1 and lag_12 exist
4. No rolling statistics
5. Prophet data format is correct
6. LSTM sequences are properly created
7. Chronos data arrays exist
8. Data splits are correct (train/val/test)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from feature_engineering import (
    create_minimal_features,
    create_prophet_data,
    create_lstm_data,
    split_train_val_test,
    create_lstm_sequences,
    get_feature_columns
)


def test_data_loading():
    """Test that base data loads correctly."""
    print("=" * 70)
    print("TEST 1: Data Loading")
    print("=" * 70)
    
    df = pd.read_csv('data/processed/monthly_tourist_arrivals_filtered.csv')
    
    assert df.shape == (91, 2), f"Expected (91, 2), got {df.shape}"
    assert 'Date' in df.columns, "Missing 'Date' column"
    assert 'Arrivals' in df.columns, "Missing 'Arrivals' column"
    
    print(f"✓ Data loaded: {df.shape}")
    print(f"✓ Date range: {df['Date'].min()} to {df['Date'].max()}")
    print()
    return df


def test_minimal_features(df):
    """Test create_minimal_features function."""
    print("=" * 70)
    print("TEST 2: Minimal Features Creation")
    print("=" * 70)
    
    df_features = create_minimal_features(df)
    
    # Expected columns
    expected_cols = ['Date', 'Arrivals', 'easter_impact', 'covid', 'econ_crisis', 
                     'recovery', 'month_sin', 'month_cos', 'Arrivals_lag_1', 'Arrivals_lag_12']
    
    assert list(df_features.columns) == expected_cols, \
        f"Columns mismatch. Expected {expected_cols}, got {list(df_features.columns)}"
    
    # Check shape
    assert df_features.shape == (91, 10), f"Expected (91, 10), got {df_features.shape}"
    
    # Check no rolling statistics
    rolling_cols = [col for col in df_features.columns if 'rolling' in col]
    assert len(rolling_cols) == 0, f"Found rolling statistics: {rolling_cols}"
    
    # Check no recovery_index
    assert 'recovery_index' not in df_features.columns, "recovery_index should not exist"
    assert 'months_since_covid' not in df_features.columns, "months_since_covid should not exist"
    
    # Check only 2 lags exist
    lag_cols = [col for col in df_features.columns if 'lag' in col]
    assert len(lag_cols) == 2, f"Expected 2 lag columns, got {len(lag_cols)}: {lag_cols}"
    assert 'Arrivals_lag_1' in lag_cols, "Missing Arrivals_lag_1"
    assert 'Arrivals_lag_12' in lag_cols, "Missing Arrivals_lag_12"
    
    print(f"✓ Minimal features created: {df_features.shape}")
    print(f"✓ Columns: {list(df_features.columns)}")
    print(f"✓ No rolling statistics")
    print(f"✓ Only 2 lag features: lag_1, lag_12")
    print()
    return df_features


def test_new_intervention_features(df):
    """Test that new intervention features are correctly defined."""
    print("=" * 70)
    print("TEST 3: New Intervention Features")
    print("=" * 70)
    
    df_features = create_minimal_features(df)
    
    # Easter impact: April 2019 - September 2019 (6 months)
    easter_rows = df_features[df_features['easter_impact'] == 1]
    assert len(easter_rows) == 6, f"Easter impact should be 6 months, got {len(easter_rows)}"
    easter_dates = pd.to_datetime(easter_rows['Date'])
    assert easter_dates.min() == pd.Timestamp('2019-04-01'), \
        f"Easter impact start wrong: {easter_dates.min()}"
    assert easter_dates.max() == pd.Timestamp('2019-09-01'), \
        f"Easter impact end wrong: {easter_dates.max()}"
    
    # COVID: March 2020 - December 2021 (22 months)
    covid_rows = df_features[df_features['covid'] == 1]
    assert len(covid_rows) == 22, f"COVID should be 22 months, got {len(covid_rows)}"
    covid_dates = pd.to_datetime(covid_rows['Date'])
    assert covid_dates.min() == pd.Timestamp('2020-03-01'), \
        f"COVID start wrong: {covid_dates.min()}"
    assert covid_dates.max() == pd.Timestamp('2021-12-01'), \
        f"COVID end wrong: {covid_dates.max()}"
    
    # Economic crisis: April 2022 - September 2022 (6 months)
    crisis_rows = df_features[df_features['econ_crisis'] == 1]
    assert len(crisis_rows) == 6, f"Econ crisis should be 6 months, got {len(crisis_rows)}"
    crisis_dates = pd.to_datetime(crisis_rows['Date'])
    assert crisis_dates.min() == pd.Timestamp('2022-04-01'), \
        f"Econ crisis start wrong: {crisis_dates.min()}"
    assert crisis_dates.max() == pd.Timestamp('2022-09-01'), \
        f"Econ crisis end wrong: {crisis_dates.max()}"
    
    # Recovery: November 2022 onwards (21 months through July 2024)
    recovery_rows = df_features[df_features['recovery'] == 1]
    expected_recovery = 21  # Nov 2022 to Jul 2024
    assert len(recovery_rows) == expected_recovery, \
        f"Recovery should be {expected_recovery} months, got {len(recovery_rows)}"
    recovery_dates = pd.to_datetime(recovery_rows['Date'])
    assert recovery_dates.min() == pd.Timestamp('2022-11-01'), \
        f"Recovery start wrong: {recovery_dates.min()}"
    assert recovery_dates.max() == pd.Timestamp('2024-07-01'), \
        f"Recovery end wrong: {recovery_dates.max()}"
    
    print(f"✓ Easter impact: 6 months (Apr 2019 - Sep 2019)")
    print(f"✓ COVID: 22 months (Mar 2020 - Dec 2021)")
    print(f"✓ Economic crisis: 6 months (Apr 2022 - Sep 2022)")
    print(f"✓ Recovery: {expected_recovery} months (Nov 2022 - Jul 2024)")
    print()


def test_prophet_data(df):
    """Test Prophet data format."""
    print("=" * 70)
    print("TEST 4: Prophet Data Format")
    print("=" * 70)
    
    df_prophet = create_prophet_data(df)
    
    # Must have ds and y columns
    assert 'ds' in df_prophet.columns, "Missing 'ds' column for Prophet"
    assert 'y' in df_prophet.columns, "Missing 'y' column for Prophet"
    
    # Must have only intervention features as regressors
    expected_cols = ['ds', 'y', 'easter_impact', 'covid', 'econ_crisis', 'recovery']
    assert list(df_prophet.columns) == expected_cols, \
        f"Prophet columns mismatch. Expected {expected_cols}, got {list(df_prophet.columns)}"
    
    # Should have no missing values
    assert df_prophet.isnull().sum().sum() == 0, "Prophet data has NaN values"
    
    # Should have all 91 rows (no lag-based NaN removal)
    assert len(df_prophet) == 91, f"Expected 91 rows, got {len(df_prophet)}"
    
    print(f"✓ Prophet data created: {df_prophet.shape}")
    print(f"✓ Columns: {list(df_prophet.columns)}")
    print(f"✓ No missing values")
    print(f"✓ All 91 rows preserved")
    print()
    return df_prophet


def test_lstm_data(df):
    """Test LSTM data creation."""
    print("=" * 70)
    print("TEST 5: LSTM Data")
    print("=" * 70)
    
    # Test with NaN
    df_lstm = create_lstm_data(df, drop_na=False)
    assert len(df_lstm) == 91, f"Expected 91 rows with NaN, got {len(df_lstm)}"
    
    # Test without NaN
    df_lstm_clean = create_lstm_data(df, drop_na=True)
    expected_rows = 91 - 12  # Remove 12 months for lag_12
    assert len(df_lstm_clean) == expected_rows, \
        f"Expected {expected_rows} rows without NaN, got {len(df_lstm_clean)}"
    
    # Check has all required features
    required_features = ['Arrivals', 'Arrivals_lag_1', 'Arrivals_lag_12', 
                        'month_sin', 'month_cos', 'easter_impact', 
                        'covid', 'econ_crisis', 'recovery']
    for feat in required_features:
        assert feat in df_lstm_clean.columns, f"Missing feature: {feat}"
    
    # No NaN in clean version
    assert df_lstm_clean.isnull().sum().sum() == 0, "Clean LSTM data has NaN"
    
    print(f"✓ LSTM data (with NaN): {df_lstm.shape}")
    print(f"✓ LSTM data (clean): {df_lstm_clean.shape}")
    print(f"✓ All required features present")
    print()
    return df_lstm_clean


def test_data_splits(df):
    """Test train/val/test splits."""
    print("=" * 70)
    print("TEST 6: Data Splits")
    print("=" * 70)
    
    df_lstm = create_lstm_data(df, drop_na=True)
    train_df, val_df, test_df = split_train_val_test(df_lstm)
    
    # Check date ranges
    train_dates = pd.to_datetime(train_df['Date'])
    val_dates = pd.to_datetime(val_df['Date'])
    test_dates = pd.to_datetime(test_df['Date'])
    
    # Train: 2018-01 to 2022-12 (60 months, starts from 2018-01 due to lag_12)
    assert train_dates.min() == pd.Timestamp('2018-01-01'), \
        f"Train start wrong: {train_dates.min()}"
    assert train_dates.max() == pd.Timestamp('2022-12-01'), \
        f"Train end wrong: {train_dates.max()}"
    assert len(train_df) == 60, f"Expected 60 train rows, got {len(train_df)}"
    
    # Val: 2023-01 to 2023-12 (12 months)
    assert val_dates.min() == pd.Timestamp('2023-01-01'), \
        f"Val start wrong: {val_dates.min()}"
    assert val_dates.max() == pd.Timestamp('2023-12-01'), \
        f"Val end wrong: {val_dates.max()}"
    assert len(val_df) == 12, f"Expected 12 val rows, got {len(val_df)}"
    
    # Test: 2024-01 to 2024-07 (7 months)
    assert test_dates.min() == pd.Timestamp('2024-01-01'), \
        f"Test start wrong: {test_dates.min()}"
    assert test_dates.max() == pd.Timestamp('2024-07-01'), \
        f"Test end wrong: {test_dates.max()}"
    assert len(test_df) == 7, f"Expected 7 test rows, got {len(test_df)}"
    
    print(f"✓ Train: {len(train_df)} rows (2018-01 to 2022-12)")
    print(f"✓ Val:   {len(val_df)} rows (2023-01 to 2023-12)")
    print(f"✓ Test:  {len(test_df)} rows (2024-01 to 2024-07)")
    print()


def test_output_files():
    """Test that all output files exist and have correct structure."""
    print("=" * 70)
    print("TEST 7: Output Files")
    print("=" * 70)
    
    output_dir = Path('data/processed')
    
    # Check CSV files
    files = {
        'engineered_features.csv': (79, 10),
        'prophet_regressors.csv': (91, 6),
    }
    
    for filename, expected_shape in files.items():
        filepath = output_dir / filename
        assert filepath.exists(), f"File not found: {filepath}"
        df = pd.read_csv(filepath)
        assert df.shape == expected_shape, \
            f"{filename}: Expected {expected_shape}, got {df.shape}"
        print(f"✓ {filename}: {df.shape}")
    
    # Check numpy files
    numpy_files = ['chronos_context.npy', 'chronos_test.npy']
    for filename in numpy_files:
        filepath = output_dir / filename
        assert filepath.exists(), f"File not found: {filepath}"
        data = np.load(filepath)
        print(f"✓ {filename}: {data.shape}")
    
    # Check npz files
    npz_files = ['lstm_train.npz', 'lstm_val.npz', 'lstm_test.npz']
    for filename in npz_files:
        filepath = output_dir / filename
        assert filepath.exists(), f"File not found: {filepath}"
        data = np.load(filepath)
        print(f"✓ {filename}: X={data['X'].shape}, y={data['y'].shape}")
    
    # Check scaler
    scaler_path = output_dir / 'scaler.pkl'
    assert scaler_path.exists(), f"Scaler not found: {scaler_path}"
    print(f"✓ scaler.pkl exists")
    
    print()


def test_lstm_sequences():
    """Test LSTM sequence creation."""
    print("=" * 70)
    print("TEST 8: LSTM Sequences")
    print("=" * 70)
    
    # Load sequences
    output_dir = Path('data/processed')
    train_data = np.load(output_dir / 'lstm_train.npz')
    val_data = np.load(output_dir / 'lstm_val.npz')
    test_data = np.load(output_dir / 'lstm_test.npz')
    
    X_train, y_train = train_data['X'], train_data['y']
    X_val, y_val = val_data['X'], val_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    
    # Check shapes
    assert X_train.ndim == 3, "X_train should be 3D (samples, timesteps, features)"
    assert X_train.shape[1] == 24, f"Window size should be 24, got {X_train.shape[1]}"
    assert X_train.shape[2] == 9, f"Expected 9 features, got {X_train.shape[2]}"
    
    # Check y shapes match
    assert len(y_train) == len(X_train), "y_train length mismatch"
    assert len(y_val) == len(X_val), "y_val length mismatch"
    assert len(y_test) == len(X_test), "y_test length mismatch"
    
    # Check values are scaled (should be between 0 and 1 for most values)
    # Note: some features might be outside [0,1] due to scaling
    print(f"✓ Train sequences: X={X_train.shape}, y={y_train.shape}")
    print(f"✓ Val sequences:   X={X_val.shape}, y={y_val.shape}")
    print(f"✓ Test sequences:  X={X_test.shape}, y={y_test.shape}")
    print(f"✓ Window size: 24 months")
    print(f"✓ Features: 9 (Arrivals + 2 lags + 2 cyclical + 4 interventions)")
    print()


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("MINIMAL FEATURE ENGINEERING VALIDATION TESTS")
    print("=" * 70 + "\n")
    
    try:
        df = test_data_loading()
        df_features = test_minimal_features(df)
        test_new_intervention_features(df)
        df_prophet = test_prophet_data(df)
        df_lstm = test_lstm_data(df)
        test_data_splits(df)
        test_output_files()
        test_lstm_sequences()
        
        print("=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print("\nMinimal feature engineering is valid and ready for modeling!")
        print("\nKey changes from previous version:")
        print("  ✓ Only 2 lag features (lag_1, lag_12)")
        print("  ✓ New intervention flags (easter_impact, covid, econ_crisis, recovery)")
        print("  ✓ No rolling statistics")
        print("  ✓ No recovery_index")
        print("  ✓ Focused on Prophet, LSTM, Chronos only")
        
        return True
        
    except AssertionError as e:
        print("\n" + "=" * 70)
        print("TEST FAILED ✗")
        print("=" * 70)
        print(f"\nError: {e}")
        return False
    except Exception as e:
        print("\n" + "=" * 70)
        print("TEST ERROR ✗")
        print("=" * 70)
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
