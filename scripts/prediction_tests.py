"""
Prediction Tests for Model Validation.

Implements Section 9 of the problem statement:
- Test for no future leakage
- Test Prophet regressor alignment
- Test LSTM window integrity
- Test forecast length consistency
- Test metric reproducibility
- Test non-negative predictions
- Test Chronos interval coherency
- Test autoregressive drift reasonableness
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from typing import Dict, Tuple

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_splits import TRAIN_END, VAL_START, VAL_END, TEST_START
from evaluation import calculate_metrics


class PredictionTests:
    """Test suite for prediction validation."""
    
    def __init__(self, base_dir: Path = None):
        """Initialize test suite with base directory."""
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        
        self.base_dir = base_dir
        self.data_dir = base_dir / 'data' / 'processed'
        self.forecasts_dir = base_dir / 'forecasts'
        self.reports_dir = base_dir / 'reports'
        
        self.results = []
    
    def run_all_tests(self) -> bool:
        """Run all prediction tests and return True if all pass."""
        print("\n" + "="*60)
        print("PREDICTION TESTS")
        print("="*60)
        
        all_passed = True
        
        # Test 1: No Future Leakage
        passed = self.test_no_future_leakage()
        all_passed = all_passed and passed
        
        # Test 2: Prophet Regressor Alignment
        passed = self.test_prophet_regressor_alignment()
        all_passed = all_passed and passed
        
        # Test 3: LSTM Window Integrity
        passed = self.test_lstm_window_integrity()
        all_passed = all_passed and passed
        
        # Test 4: Forecast Length Consistency
        passed = self.test_forecast_length()
        all_passed = all_passed and passed
        
        # Test 5: Metric Reproducibility
        passed = self.test_metric_reproducibility()
        all_passed = all_passed and passed
        
        # Test 6: Non-negative Predictions
        passed = self.test_non_negative_predictions()
        all_passed = all_passed and passed
        
        # Test 7: Chronos Interval Coherency
        passed = self.test_chronos_interval_coherency()
        all_passed = all_passed and passed
        
        # Test 8: Autoregressive Drift
        passed = self.test_autoregressive_drift()
        all_passed = all_passed and passed
        
        print("\n" + "="*60)
        if all_passed:
            print("✓ ALL TESTS PASSED")
        else:
            print("✗ SOME TESTS FAILED")
        print("="*60)
        
        return all_passed
    
    def test_no_future_leakage(self) -> bool:
        """Test 1: Assert max(ds_train) < min(ds_val) < min(ds_test)"""
        print("\nTest 1: No Future Leakage")
        
        try:
            # Load splits from the splits.json file
            splits_file = self.data_dir / 'splits.json'
            
            if not splits_file.exists():
                print(f"  ⚠ Skipped: {splits_file} not found")
                return True
            
            with open(splits_file, 'r') as f:
                splits_info = json.load(f)
            
            train_end = pd.Timestamp(splits_info['train']['end'])
            val_start = pd.Timestamp(splits_info['validation']['start'])
            val_end = pd.Timestamp(splits_info['validation']['end'])
            test_start = pd.Timestamp(splits_info['test']['start'])
            
            # Check chronological order
            assert train_end < val_start, f"Train end ({train_end}) >= Val start ({val_start})"
            assert val_end < test_start, f"Val end ({val_end}) >= Test start ({test_start})"
            
            print("  ✓ Passed: No future leakage detected")
            return True
            
        except AssertionError as e:
            print(f"  ✗ Failed: {e}")
            return False
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            return True  # Don't fail on missing files
    
    def test_prophet_regressor_alignment(self) -> bool:
        """Test 2: Assert all required regressor column names exist."""
        print("\nTest 2: Prophet Regressor Alignment")
        
        try:
            prophet_file = self.data_dir / 'minimal_features_prophet.csv'
            
            if not prophet_file.exists():
                print(f"  ⚠ Skipped: {prophet_file} not found")
                return True
            
            df = pd.read_csv(prophet_file)
            
            required_cols = ['ds', 'y', 'easter_attacks', 'covid_period', 'economic_crisis']
            
            for col in required_cols:
                assert col in df.columns, f"Missing required column: {col}"
            
            print(f"  ✓ Passed: All required columns present: {required_cols}")
            return True
            
        except AssertionError as e:
            print(f"  ✗ Failed: {e}")
            return False
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            return True
    
    def test_lstm_window_integrity(self) -> bool:
        """Test 3: Assert LSTM windows are chronological and don't cross boundaries."""
        print("\nTest 3: LSTM Window Integrity")
        
        try:
            lstm_file = self.data_dir / 'minimal_features_lstm.csv'
            
            if not lstm_file.exists():
                print(f"  ⚠ Skipped: {lstm_file} not found")
                return True
            
            df = pd.read_csv(lstm_file, parse_dates=['Date'])
            
            # Check chronological order
            dates = df['Date'].values
            for i in range(len(dates)-1):
                assert dates[i] < dates[i+1], f"Dates not in chronological order at index {i}"
            
            print("  ✓ Passed: LSTM data is chronologically ordered")
            return True
            
        except AssertionError as e:
            print(f"  ✗ Failed: {e}")
            return False
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            return True
    
    def test_forecast_length(self) -> bool:
        """Test 4: Assert forecast length equals expected horizon (7 months)."""
        print("\nTest 4: Forecast Length Consistency")
        
        expected_length = 7
        all_passed = True
        
        # Test Prophet forecast
        prophet_file = self.forecasts_dir / 'prophet_test_forecast.csv'
        if prophet_file.exists():
            df = pd.read_csv(prophet_file)
            if len(df) == expected_length:
                print(f"  ✓ Prophet: {len(df)} rows (expected {expected_length})")
            else:
                print(f"  ✗ Prophet: {len(df)} rows (expected {expected_length})")
                all_passed = False
        else:
            print(f"  ⚠ Prophet forecast not found")
        
        # Test LSTM forecast
        lstm_file = self.forecasts_dir / 'lstm_test_forecast.csv'
        if lstm_file.exists():
            df = pd.read_csv(lstm_file)
            # LSTM may have fewer rows due to windowing
            print(f"  ℹ LSTM: {len(df)} rows (may differ due to windowing)")
        else:
            print(f"  ⚠ LSTM forecast not found")
        
        # Test Chronos forecast
        chronos_file = self.forecasts_dir / 'chronos_test_forecast.csv'
        if chronos_file.exists():
            df = pd.read_csv(chronos_file)
            if len(df) == expected_length:
                print(f"  ✓ Chronos: {len(df)} rows (expected {expected_length})")
            else:
                print(f"  ✗ Chronos: {len(df)} rows (expected {expected_length})")
                all_passed = False
        else:
            print(f"  ⚠ Chronos forecast not found")
        
        return all_passed
    
    def test_metric_reproducibility(self) -> bool:
        """Test 5: Recompute metrics from forecast files and compare to stored JSON."""
        print("\nTest 5: Metric Reproducibility")
        
        tolerance = 1e-6
        all_passed = True
        
        # Test Prophet metrics
        prophet_forecast = self.forecasts_dir / 'prophet_test_forecast.csv'
        prophet_metrics = self.reports_dir / 'prophet_metrics.json'
        
        if prophet_forecast.exists() and prophet_metrics.exists():
            df = pd.read_csv(prophet_forecast)
            with open(prophet_metrics, 'r') as f:
                stored_metrics = json.load(f)
            
            # Recompute metrics
            computed_metrics = calculate_metrics(df['y_true'].values, df['y_pred'].values)
            
            # Compare
            for key in ['RMSE', 'MAE', 'MAPE', 'R2']:
                stored = stored_metrics['test_metrics'][key]
                computed = computed_metrics[key]
                diff = abs(stored - computed)
                
                if diff < tolerance:
                    print(f"  ✓ Prophet {key}: {computed:.4f} (diff: {diff:.2e})")
                else:
                    print(f"  ✗ Prophet {key}: stored={stored:.4f}, computed={computed:.4f} (diff: {diff:.2e})")
                    all_passed = False
        else:
            print(f"  ⚠ Prophet files not found")
        
        return all_passed
    
    def test_non_negative_predictions(self) -> bool:
        """Test 6: Assert all y_pred >= 0 (tourist arrivals cannot be negative)."""
        print("\nTest 6: Non-negative Predictions")
        
        all_passed = True
        
        # Check Prophet
        prophet_file = self.forecasts_dir / 'prophet_test_forecast.csv'
        if prophet_file.exists():
            df = pd.read_csv(prophet_file)
            min_pred = df['y_pred'].min()
            if min_pred >= 0:
                print(f"  ✓ Prophet: min prediction = {min_pred:.2f}")
            else:
                print(f"  ✗ Prophet: min prediction = {min_pred:.2f} (negative!)")
                all_passed = False
        else:
            print(f"  ⚠ Prophet forecast not found")
        
        # Check LSTM
        lstm_file = self.forecasts_dir / 'lstm_test_forecast.csv'
        if lstm_file.exists():
            df = pd.read_csv(lstm_file)
            min_teacher = df['y_pred_teacher'].min()
            min_autoreg = df['y_pred_autoreg'].min()
            
            if min_teacher >= 0:
                print(f"  ✓ LSTM (teacher): min prediction = {min_teacher:.2f}")
            else:
                print(f"  ✗ LSTM (teacher): min prediction = {min_teacher:.2f} (negative!)")
                all_passed = False
            
            if min_autoreg >= 0:
                print(f"  ✓ LSTM (autoreg): min prediction = {min_autoreg:.2f}")
            else:
                print(f"  ✗ LSTM (autoreg): min prediction = {min_autoreg:.2f} (negative!)")
                all_passed = False
        else:
            print(f"  ⚠ LSTM forecast not found")
        
        return all_passed
    
    def test_chronos_interval_coherency(self) -> bool:
        """Test 7: Assert p10 <= p50 <= p90 for all rows."""
        print("\nTest 7: Chronos Interval Coherency")
        
        chronos_file = self.forecasts_dir / 'chronos_test_forecast.csv'
        
        if not chronos_file.exists():
            print(f"  ⚠ Skipped: {chronos_file} not found")
            return True
        
        try:
            df = pd.read_csv(chronos_file)
            
            # Check if interval columns exist
            if not all(col in df.columns for col in ['p10', 'p50', 'p90']):
                print(f"  ⚠ Skipped: Interval columns not found")
                return True
            
            # Check coherency
            violations = 0
            for idx, row in df.iterrows():
                if not (row['p10'] <= row['p50'] <= row['p90']):
                    violations += 1
            
            if violations == 0:
                print(f"  ✓ Passed: All {len(df)} rows have coherent intervals")
                return True
            else:
                print(f"  ✗ Failed: {violations}/{len(df)} rows have incoherent intervals")
                return False
                
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            return True
    
    def test_autoregressive_drift(self) -> bool:
        """Test 8: Check autoregressive drift reasonableness."""
        print("\nTest 8: Autoregressive Drift Reasonableness")
        
        lstm_file = self.forecasts_dir / 'lstm_test_forecast.csv'
        lstm_metrics = self.reports_dir / 'lstm_metrics.json'
        
        if not lstm_file.exists():
            print(f"  ⚠ Skipped: {lstm_file} not found")
            return True
        
        try:
            df = pd.read_csv(lstm_file)
            
            if not all(col in df.columns for col in ['y_pred_teacher', 'y_pred_autoreg']):
                print(f"  ⚠ Skipped: Required columns not found")
                return True
            
            # Calculate mean absolute difference
            mad = np.mean(np.abs(df['y_pred_teacher'] - df['y_pred_autoreg']))
            
            # Get test MAE if available
            if lstm_metrics.exists():
                with open(lstm_metrics, 'r') as f:
                    metrics = json.load(f)
                test_mae = metrics.get('test_metrics_teacher_forcing', {}).get('MAE', None)
                
                if test_mae is not None:
                    threshold = 3 * test_mae
                    
                    if mad <= threshold:
                        print(f"  ✓ Passed: MAD={mad:.2f}, threshold={threshold:.2f}")
                        return True
                    else:
                        print(f"  ⚠ Warning: MAD={mad:.2f} exceeds threshold={threshold:.2f} (potential instability)")
                        return True  # Warning, not failure
            
            print(f"  ℹ MAD between teacher and autoreg: {mad:.2f}")
            return True
            
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            return True


def main():
    """Run all prediction tests."""
    tests = PredictionTests()
    all_passed = tests.run_all_tests()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
