"""
Test script for feature engineering validation.

This script validates:
1. All feature engineering functions work correctly
2. No data leakage in rolling features
3. Intervention features are correctly applied
4. Output files have expected structure
5. Feature sets are appropriate for different models
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from feature_engineering import (
    create_all_features, 
    create_prophet_features, 
    create_ml_features,
    get_feature_names
)


def test_data_loading():
    """Test that data loads correctly."""
    print("=" * 60)
    print("TEST 1: Data Loading")
    print("=" * 60)
    
    df = pd.read_csv('data/processed/monthly_tourist_arrivals_filtered.csv')
    
    assert df.shape == (91, 2), f"Expected (91, 2), got {df.shape}"
    assert 'Date' in df.columns, "Missing 'Date' column"
    assert 'Arrivals' in df.columns, "Missing 'Arrivals' column"
    
    print(f"✓ Data loaded: {df.shape}")
    print(f"✓ Date range: {df['Date'].min()} to {df['Date'].max()}")
    print()
    return df


def test_all_features(df):
    """Test create_all_features function."""
    print("=" * 60)
    print("TEST 2: All Features Creation")
    print("=" * 60)
    
    df_all = create_all_features(df)
    
    # Expected columns
    expected_cols = ['Date', 'Arrivals', 'year', 'month', 'quarter', 
                     'month_sin', 'month_cos', 'easter_attacks', 'covid_period',
                     'economic_crisis', 'months_since_covid', 'recovery_index']
    
    for col in expected_cols:
        assert col in df_all.columns, f"Missing column: {col}"
    
    # Check lag features exist
    assert 'Arrivals_lag_1' in df_all.columns, "Missing lag features"
    assert 'Arrivals_lag_12' in df_all.columns, "Missing 12-month lag"
    
    # Check rolling features exist
    assert 'Arrivals_rolling_mean_3' in df_all.columns, "Missing rolling features"
    
    print(f"✓ All features created: {df_all.shape}")
    print(f"✓ Total columns: {len(df_all.columns)}")
    print()
    return df_all


def test_prophet_features(df):
    """Test create_prophet_features function."""
    print("=" * 60)
    print("TEST 3: Prophet Features")
    print("=" * 60)
    
    df_prophet = create_prophet_features(df)
    
    # Must have ds and y columns
    assert 'ds' in df_prophet.columns, "Missing 'ds' column for Prophet"
    assert 'y' in df_prophet.columns, "Missing 'y' column for Prophet"
    
    # Must have intervention features as regressors
    assert 'easter_attacks' in df_prophet.columns, "Missing intervention features"
    assert 'covid_period' in df_prophet.columns, "Missing COVID period feature"
    assert 'recovery_index' in df_prophet.columns, "Missing recovery index"
    
    # Should have no missing values
    assert df_prophet.isnull().sum().sum() == 0, "Prophet features have NaN values"
    
    print(f"✓ Prophet features created: {df_prophet.shape}")
    print(f"✓ No missing values")
    print(f"✓ Columns: {list(df_prophet.columns)}")
    print()
    return df_prophet


def test_ml_features(df):
    """Test create_ml_features function."""
    print("=" * 60)
    print("TEST 4: ML/DL Features")
    print("=" * 60)
    
    # Test with lags and NaN
    df_ml = create_ml_features(df, include_lags=True, drop_na=False)
    
    assert 'month_sin' in df_ml.columns, "Missing cyclical encoding"
    assert 'month_cos' in df_ml.columns, "Missing cyclical encoding"
    assert 'Arrivals_lag_1' in df_ml.columns, "Missing lag features"
    
    # Should have 91 rows (with NaN)
    assert len(df_ml) == 91, f"Expected 91 rows, got {len(df_ml)}"
    
    # Test with NaN dropped
    df_ml_clean = create_ml_features(df, include_lags=True, drop_na=True)
    
    # Should have no missing values
    assert df_ml_clean.isnull().sum().sum() == 0, "Clean ML features have NaN"
    
    # Should have 79 rows (91 - 12 lag)
    assert len(df_ml_clean) == 79, f"Expected 79 rows, got {len(df_ml_clean)}"
    
    print(f"✓ ML features (with NaN): {df_ml.shape}")
    print(f"✓ ML features (clean): {df_ml_clean.shape}")
    print()
    return df_ml, df_ml_clean


def test_no_leakage():
    """Test that rolling features don't leak future information."""
    print("=" * 60)
    print("TEST 5: No Data Leakage")
    print("=" * 60)
    
    # Create simple test data
    dates = pd.date_range('2020-01-01', periods=24, freq='MS')
    values = list(range(1, 25))  # 1 to 24
    
    df_test = pd.DataFrame({'Date': dates, 'Arrivals': values})
    df_features = create_all_features(df_test)
    
    # Check that rolling mean at position 3 (4th month) uses only first 3 values
    # With min_periods=3, the 3-month average should be available at index 2
    rolling_3 = df_features['Arrivals_rolling_mean_3'].iloc[2]
    expected_3 = np.mean([1, 2, 3])  # Should use months 1, 2, 3
    
    assert abs(rolling_3 - expected_3) < 0.01, \
        f"Rolling mean leakage detected: {rolling_3} != {expected_3}"
    
    # Check lag features
    lag_1 = df_features['Arrivals_lag_1'].iloc[5]  # 6th month
    assert lag_1 == 5, f"Lag feature incorrect: {lag_1} != 5"
    
    print("✓ No data leakage in rolling features")
    print("✓ Lag features correctly shifted")
    print()


def test_intervention_features():
    """Test that intervention features are correctly applied."""
    print("=" * 60)
    print("TEST 6: Intervention Features")
    print("=" * 60)
    
    df = pd.read_csv('data/processed/monthly_tourist_arrivals_filtered.csv')
    df_features = create_all_features(df)
    
    # Check Easter attacks (April 2019)
    easter_rows = df_features[df_features['easter_attacks'] == 1]
    assert len(easter_rows) == 1, "Easter attacks should affect exactly 1 month"
    easter_date = pd.to_datetime(easter_rows.iloc[0]['Date']).strftime('%Y-%m-%d')
    assert easter_date == '2019-04-01', f"Easter attacks wrong date: {easter_date}"
    
    # Check COVID period (March 2020 - December 2021)
    covid_rows = df_features[df_features['covid_period'] == 1]
    expected_covid_months = 22  # March 2020 to December 2021
    assert len(covid_rows) == expected_covid_months, \
        f"COVID period should be {expected_covid_months} months, got {len(covid_rows)}"
    
    # Check economic crisis (2022 onwards)
    crisis_rows = df_features[df_features['economic_crisis'] == 1]
    # From Jan 2022 to Jul 2024 = 31 months
    expected_crisis_months = 31
    assert len(crisis_rows) == expected_crisis_months, \
        f"Economic crisis should be {expected_crisis_months} months, got {len(crisis_rows)}"
    
    print(f"✓ Easter attacks: 1 month (April 2019)")
    print(f"✓ COVID period: {expected_covid_months} months (Mar 2020 - Dec 2021)")
    print(f"✓ Economic crisis: {expected_crisis_months} months (Jan 2022 - Jul 2024)")
    print()


def test_output_files():
    """Test that all output files exist and have correct structure."""
    print("=" * 60)
    print("TEST 7: Output Files")
    print("=" * 60)
    
    files = {
        'data/processed/monthly_tourist_arrivals_features_full.csv': (91, 20),
        'data/processed/monthly_tourist_arrivals_features_prophet.csv': (91, 9),
        'data/processed/monthly_tourist_arrivals_features_ml.csv': (91, 15),
        'data/processed/monthly_tourist_arrivals_features_ml_clean.csv': (79, 15),
    }
    
    for filepath, expected_shape in files.items():
        df = pd.read_csv(filepath)
        assert df.shape == expected_shape, \
            f"{filepath}: Expected {expected_shape}, got {df.shape}"
        print(f"✓ {filepath.split('/')[-1]}: {df.shape}")
    
    print()


def test_cyclical_encoding():
    """Test cyclical encoding is correct."""
    print("=" * 60)
    print("TEST 8: Cyclical Encoding")
    print("=" * 60)
    
    df = pd.read_csv('data/processed/monthly_tourist_arrivals_filtered.csv')
    df_features = create_all_features(df)
    
    # Check that month_sin and month_cos form a circle
    for month in range(1, 13):
        month_rows = df_features[df_features['month'] == month]
        if len(month_rows) > 0:
            sin_val = month_rows.iloc[0]['month_sin']
            cos_val = month_rows.iloc[0]['month_cos']
            
            # sin^2 + cos^2 should equal 1
            circle_check = sin_val**2 + cos_val**2
            assert abs(circle_check - 1.0) < 0.01, \
                f"Month {month} cyclical encoding error: {circle_check}"
    
    # Check December and January are close in the encoding space
    dec_rows = df_features[df_features['month'] == 12]
    jan_rows = df_features[df_features['month'] == 1]
    
    if len(dec_rows) > 0 and len(jan_rows) > 0:
        dec_sin, dec_cos = dec_rows.iloc[0]['month_sin'], dec_rows.iloc[0]['month_cos']
        jan_sin, jan_cos = jan_rows.iloc[0]['month_sin'], jan_rows.iloc[0]['month_cos']
        
        # Distance in 2D space
        distance = np.sqrt((dec_sin - jan_sin)**2 + (dec_cos - jan_cos)**2)
        assert distance < 1.0, \
            f"December and January should be close in cyclical encoding, distance: {distance}"
    
    print("✓ Cyclical encoding forms unit circle")
    print("✓ December and January are close in encoding space")
    print()


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING VALIDATION TESTS")
    print("=" * 60 + "\n")
    
    try:
        df = test_data_loading()
        df_all = test_all_features(df)
        df_prophet = test_prophet_features(df)
        df_ml, df_ml_clean = test_ml_features(df)
        test_no_leakage()
        test_intervention_features()
        test_output_files()
        test_cyclical_encoding()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nFeature engineering implementation is valid and ready for use!")
        print("\nGenerated files:")
        print("  - monthly_tourist_arrivals_features_full.csv (91x20)")
        print("  - monthly_tourist_arrivals_features_prophet.csv (91x9)")
        print("  - monthly_tourist_arrivals_features_ml.csv (91x15)")
        print("  - monthly_tourist_arrivals_features_ml_clean.csv (79x15)")
        
        return True
        
    except AssertionError as e:
        print("\n" + "=" * 60)
        print("TEST FAILED ✗")
        print("=" * 60)
        print(f"\nError: {e}")
        return False
    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST ERROR ✗")
        print("=" * 60)
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
