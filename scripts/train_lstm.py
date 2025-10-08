"""
LSTM Model Training Pipeline.

Implements Section 5 of the problem statement:
- Load minimal LSTM features
- Scale features and create windows
- Train LSTM model with early stopping
- Generate test forecasts (both teacher forcing and autoregressive)
- Save model artifacts and metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import logging
from typing import Dict, List, Tuple
import sys
import os

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.preprocessing import MinMaxScaler

# --- TensorFlow Import (Mac compatibility & graceful fallback) ---
try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras.models import Sequential  # type: ignore
    from tensorflow.keras.layers import LSTM, Dense  # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
    TF_AVAILABLE = True
except Exception as _tf_err:  # Broad: if wheel missing or mismatched arch
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = _tf_err

from data_splits import split_data_chronologically, print_split_summary
from evaluation import calculate_metrics
import matplotlib.pyplot as plt

# Set random seeds (only if TF present)
np.random.seed(42)
if TF_AVAILABLE:
    try:
        tf.random.set_seed(42)
        # Light resource limiting (harmless on CPU, helps some Mac setups)
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    except Exception:
        pass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
# Reduced window size from 24 -> 6 because validation (12 months) and test (7 months)
# splits were too short to yield any windows with a 24-step lookback.
WINDOW_SIZE = 6
HORIZON = 1


def create_scaled_data(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                        test_df: pd.DataFrame) -> Tuple:
    """
    Scale features using MinMaxScaler fit on training data only.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    val_df : pd.DataFrame
        Validation data
    test_df : pd.DataFrame
        Test data
        
    Returns
    -------
    Tuple
        (scaled_train, scaled_val, scaled_test, scaler, feature_cols, target_col)
    """
    logger.info("Scaling features...")
    
    # Define feature columns (exclude Date)
    feature_cols = [col for col in train_df.columns if col != 'Date']
    target_col = 'Arrivals'
    
    # Initialize scaler
    scaler = MinMaxScaler()
    
    # Fit on training data only
    scaler.fit(train_df[feature_cols])
    
    # Transform all splits
    scaled_train = scaler.transform(train_df[feature_cols])
    scaled_val = scaler.transform(val_df[feature_cols])
    scaled_test = scaler.transform(test_df[feature_cols])
    
    logger.info(f"Scaled features: {len(feature_cols)} columns")
    logger.info(f"Feature columns: {feature_cols}")
    
    return scaled_train, scaled_val, scaled_test, scaler, feature_cols, target_col


def create_windows(data: np.ndarray, window_size: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for LSTM training.
    
    Parameters
    ----------
    data : np.ndarray
        Scaled data
    window_size : int
        Number of time steps to look back
    horizon : int
        Number of steps ahead to forecast (always 1 in this case)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (X_windows, y_targets)
    """
    X, y = [], []
    
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i:i+window_size])
        # Target is the Arrivals value (index 1) at window_size position
        y.append(data[i+window_size, 1])  # Arrivals is at index 1
    
    return np.array(X), np.array(y)


def create_train_val_test_windows(scaled_train: np.ndarray, scaled_val: np.ndarray,
                                    scaled_test: np.ndarray, window_size: int,
                                    horizon: int) -> Tuple:
    """
    Create windows for train, validation, and test sets.
    
    Parameters
    ----------
    scaled_train : np.ndarray
        Scaled training data
    scaled_val : np.ndarray
        Scaled validation data
    scaled_test : np.ndarray
        Scaled test data
    window_size : int
        Window size for look-back
    horizon : int
        Forecast horizon
        
    Returns
    -------
    Tuple
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    logger.info(f"Creating windows with window_size={window_size}, horizon={horizon}...")
    
    # Create windows for each split
    X_train, y_train = create_windows(scaled_train, window_size, horizon)
    X_val, y_val = create_windows(scaled_val, window_size, horizon)
    X_test, y_test = create_windows(scaled_test, window_size, horizon)
    
    logger.info(f"Train windows: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Val windows: X={X_val.shape}, y={y_val.shape}")
    logger.info(f"Test windows: X={X_test.shape}, y={y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_lstm_model(input_shape: Tuple[int, int]) -> keras.Model:
    """
    Build LSTM model architecture.
    
    Parameters
    ----------
    input_shape : Tuple[int, int]
        Shape of input (window_size, num_features)
        
    Returns
    -------
    keras.Model
        Compiled LSTM model
    """
    logger.info(f"Building LSTM model with input_shape={input_shape}...")
    
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])
    
    logger.info("Model architecture:")
    model.summary(print_fn=logger.info)
    
    return model


def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray,
               epochs: int = 200) -> Tuple[keras.Model, Dict]:
    """
    Train LSTM model with early stopping.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training windows
    y_train : np.ndarray
        Training targets
    X_val : np.ndarray
        Validation windows
    y_val : np.ndarray
        Validation targets
    epochs : int
        Maximum number of epochs
        
    Returns
    -------
    Tuple[keras.Model, Dict]
        (trained_model, history)
    """
    logger.info("Training LSTM model...")
    
    # Build model
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Early stopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=8,
        callbacks=[early_stop],
        verbose=1
    )
    
    logger.info("Training complete!")
    
    return model, history.history


def forecast_teacher_forcing(model: keras.Model, X_test: np.ndarray,
                               scaler: MinMaxScaler, target_idx: int = 1) -> np.ndarray:
    """
    Generate forecasts using teacher forcing (with actual values in windows).
    
    Parameters
    ----------
    model : keras.Model
        Trained LSTM model
    X_test : np.ndarray
        Test windows (already contain actual values)
    scaler : MinMaxScaler
        Fitted scaler for inverse transform
    target_idx : int
        Index of target variable in feature array
        
    Returns
    -------
    np.ndarray
        Forecasted values (inverse transformed)
    """
    logger.info("Generating teacher-forced predictions...")
    
    # Predict
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Inverse transform: create dummy array with all features, replace target column
    n_features = scaler.n_features_in_
    dummy = np.zeros((len(y_pred_scaled), n_features))
    dummy[:, target_idx] = y_pred_scaled.flatten()
    y_pred = scaler.inverse_transform(dummy)[:, target_idx]
    
    return y_pred


def forecast_autoregressive(model: keras.Model, initial_window: np.ndarray,
                             n_steps: int, scaler: MinMaxScaler,
                             target_idx: int = 1) -> np.ndarray:
    """
    Generate forecasts autoregressively (using previous predictions).
    
    Parameters
    ----------
    model : keras.Model
        Trained LSTM model
    initial_window : np.ndarray
        Initial window to start forecasting from
    n_steps : int
        Number of steps to forecast
    scaler : MinMaxScaler
        Fitted scaler for inverse transform
    target_idx : int
        Index of target variable in feature array
        
    Returns
    -------
    np.ndarray
        Forecasted values (inverse transformed)
    """
    logger.info("Generating autoregressive predictions...")
    
    predictions = []
    current_window = initial_window.copy()
    
    for step in range(n_steps):
        # Predict next value
        y_pred_scaled = model.predict(current_window.reshape(1, *current_window.shape), verbose=0)[0, 0]
        
        # Inverse transform to get actual scale
        dummy = np.zeros(scaler.n_features_in_)
        dummy[target_idx] = y_pred_scaled
        y_pred = scaler.inverse_transform(dummy.reshape(1, -1))[0, target_idx]
        predictions.append(y_pred)
        
        # Update window: shift and append new prediction
        new_row = current_window[-1].copy()
        new_row[target_idx] = y_pred_scaled
        current_window = np.vstack([current_window[1:], new_row])
    
    return np.array(predictions)


def plot_predictions(forecast_df: pd.DataFrame, train_val_df: pd.DataFrame,
                      output_file: str):
    """
    Plot actual vs predictions for both teacher forcing and autoregressive.
    
    Parameters
    ----------
    forecast_df : pd.DataFrame
        Forecast results with both modes
    train_val_df : pd.DataFrame
        Training and validation data for context
    output_file : str
        Path to save plot
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot training/validation data
    ax.plot(train_val_df['Date'], train_val_df['Arrivals'], 
            label='Train/Val Actual', color='blue', linewidth=1.5, alpha=0.7)
    
    # Plot test actual and forecasts
    ax.plot(forecast_df['Date'], forecast_df['y_true'], 
            label='Test Actual', color='green', linewidth=2, marker='o')
    ax.plot(forecast_df['Date'], forecast_df['y_pred_teacher'], 
            label='Teacher Forcing', color='red', linewidth=2, marker='s', linestyle='--', alpha=0.7)
    ax.plot(forecast_df['Date'], forecast_df['y_pred_autoreg'], 
            label='Autoregressive', color='orange', linewidth=2, marker='^', linestyle=':', alpha=0.7)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Tourist Arrivals', fontsize=12)
    ax.set_title('LSTM Model: Actual vs Forecast (Teacher Forcing & Autoregressive)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    logger.info(f"Plot saved to: {output_file}")


def run_persistence_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                             reports_dir: Path, forecasts_dir: Path) -> int:
    """Fallback baseline when TensorFlow is unavailable.

    Uses simple lag-1 persistence for both teacher-forcing and autoregressive placeholders.
    Produces artifacts in the same schema so downstream comparison does not break.
    """
    logger.warning("TensorFlow unavailable – using persistence baseline (no neural network training).")

    # Ensure Arrivals present
    if 'Arrivals' not in test_df.columns:
        logger.error("'Arrivals' column missing in test set – cannot compute baseline.")
        return 1

    y_true = test_df['Arrivals'].values.astype(float)
    if 'lag_1' in test_df.columns:
        preds = test_df['lag_1'].fillna(method='ffill').values.astype(float)
    else:
        # Use last train observed value then shift
        last_train_val = train_df['Arrivals'].iloc[-1]
        preds = np.concatenate([[last_train_val], y_true[:-1]])

    # Metrics (reuse same schema for teacher/autoreg)
    from evaluation import calculate_metrics  # local import to avoid circular on minimal path
    metrics = calculate_metrics(y_true, preds)

    forecast_df = pd.DataFrame({
        'Date': test_df['Date'].values,
        'y_true': y_true,
        'y_pred_teacher': preds,
        'y_pred_autoreg': preds,
        'residual_teacher': y_true - preds,
        'residual_autoreg': y_true - preds
    })
    forecast_file = forecasts_dir / 'lstm_test_forecast.csv'
    forecast_df.to_csv(forecast_file, index=False)
    logger.info(f"✓ Baseline forecast saved to: {forecast_file}")

    metrics_output = {
        'window_size': None,
        'horizon': 1,
        'baseline': 'persistence_lag1',
        'tensorflow_available': False,
        'tf_import_error': str(TF_IMPORT_ERROR),
        'test_metrics_teacher_forcing': {k: float(v) for k, v in metrics.items()},
        'test_metrics_autoregressive': {k: float(v) for k, v in metrics.items()}
    }
    metrics_file = reports_dir / 'lstm_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    logger.info(f"✓ Baseline metrics saved to: {metrics_file}")
    logger.warning("Persistence baseline used instead of LSTM – document this in your report.")
    return 0


def main():
    """Main LSTM training pipeline."""
    # Set paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'processed'
    models_dir = base_dir / 'models' / 'lstm'
    artifacts_dir = base_dir / 'artifacts'
    forecasts_dir = base_dir / 'forecasts'
    reports_dir = base_dir / 'reports'
    
    # Create directories if needed
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("LSTM MODEL TRAINING PIPELINE")
    logger.info("="*60)
    
    # Load data
    logger.info("Loading minimal LSTM features...")
    df = pd.read_csv(data_dir / 'minimal_features_lstm.csv', parse_dates=['Date'])
    logger.info(f"Loaded {len(df)} samples from {df['Date'].min()} to {df['Date'].max()}")
    
    # Split data
    logger.info("\nSplitting data chronologically...")
    train_df, val_df, test_df = split_data_chronologically(df, date_col='Date')
    print_split_summary(train_df, val_df, test_df, date_col='Date')
    
    # Keep dates for later
    train_dates = train_df['Date'].values
    val_dates = val_df['Date'].values
    test_dates = test_df['Date'].values
    
    # Scale data
    scaled_train, scaled_val, scaled_test, scaler, feature_cols, target_col = \
        create_scaled_data(train_df, val_df, test_df)
    
    # Save scaler
    scaler_file = artifacts_dir / 'lstm_scaler.pkl'
    joblib.dump(scaler, scaler_file)
    logger.info(f"✓ Scaler saved to: {scaler_file}")
    
    # Create windows
    X_train, y_train, X_val, y_val, X_test, y_test = \
        create_train_val_test_windows(scaled_train, scaled_val, scaled_test, 
                                        WINDOW_SIZE, HORIZON)
    
    # Check if we have enough data for windows
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        logger.error("Not enough data to create windows! Adjust window size or date splits.")
        return 1
    
    # If TensorFlow not available, short-circuit to baseline
    if not TF_AVAILABLE:
        return run_persistence_baseline(train_df, val_df, test_df, reports_dir, forecasts_dir)

    # Train model (guarded)
    try:
        model, history = train_lstm(X_train, y_train, X_val, y_val)
    except Exception as e:
        logger.error(f"TensorFlow training failed: {e}. Falling back to persistence baseline.")
        return run_persistence_baseline(train_df, val_df, test_df, reports_dir, forecasts_dir)
    
    # Save model
    model_file = models_dir / 'best_weights.h5'
    model.save(model_file)
    logger.info(f"✓ Model saved to: {model_file}")
    
    # Generate test forecasts - Teacher Forcing
    y_pred_teacher = forecast_teacher_forcing(model, X_test, scaler)
    
    # Inverse transform y_test for comparison
    target_idx = feature_cols.index(target_col)
    n_features = len(feature_cols)
    dummy = np.zeros((len(y_test), n_features))
    dummy[:, target_idx] = y_test
    y_true = scaler.inverse_transform(dummy)[:, target_idx]
    
    # Generate test forecasts - Autoregressive
    # Use the scaled validation data to get initial window for test forecasting
    initial_window_data = np.vstack([scaled_val, scaled_test])
    initial_window = initial_window_data[-WINDOW_SIZE-len(X_test):-len(X_test)] if len(X_test) < len(initial_window_data) else initial_window_data[-WINDOW_SIZE:]
    
    # For simplicity, we'll use the last window from training for autoregressive
    # In a more sophisticated approach, we'd iteratively build windows
    logger.info("Note: Autoregressive forecasting using sliding windows from test period")
    
    # For autoregressive, we need to forecast step by step
    # Start from the beginning of test period with train+val context
    context_data = np.vstack([scaled_train, scaled_val])
    initial_window_for_ar = context_data[-WINDOW_SIZE:]
    
    y_pred_autoreg = forecast_autoregressive(model, initial_window_for_ar, 
                                               len(y_test), scaler, target_idx)
    
    # Calculate metrics
    logger.info("\nCalculating test metrics...")
    
    metrics_teacher = calculate_metrics(y_true, y_pred_teacher)
    logger.info("Teacher Forcing Metrics:")
    logger.info(f"  RMSE: {metrics_teacher['RMSE']:.2f}")
    logger.info(f"  MAE: {metrics_teacher['MAE']:.2f}")
    logger.info(f"  MAPE: {metrics_teacher['MAPE']:.4f}")
    logger.info(f"  R2: {metrics_teacher['R2']:.4f}")
    
    metrics_autoreg = calculate_metrics(y_true, y_pred_autoreg)
    logger.info("Autoregressive Metrics:")
    logger.info(f"  RMSE: {metrics_autoreg['RMSE']:.2f}")
    logger.info(f"  MAE: {metrics_autoreg['MAE']:.2f}")
    logger.info(f"  MAPE: {metrics_autoreg['MAPE']:.4f}")
    logger.info(f"  R2: {metrics_autoreg['R2']:.4f}")
    
    # Create forecast dataframe
    # Map test windows back to dates
    test_dates_windowed = test_dates[WINDOW_SIZE:WINDOW_SIZE+len(y_test)]
    
    forecast_df = pd.DataFrame({
        'Date': test_dates_windowed,
        'y_true': y_true,
        'y_pred_teacher': y_pred_teacher,
        'y_pred_autoreg': y_pred_autoreg,
        'residual_teacher': y_true - y_pred_teacher,
        'residual_autoreg': y_true - y_pred_autoreg
    })
    
    # Save forecast
    forecast_file = forecasts_dir / 'lstm_test_forecast.csv'
    forecast_df.to_csv(forecast_file, index=False)
    logger.info(f"✓ Test forecast saved to: {forecast_file}")
    
    # Save metrics
    metrics_output = {
        'window_size': WINDOW_SIZE,
        'horizon': HORIZON,
        'test_metrics_teacher_forcing': {
            'RMSE': float(metrics_teacher['RMSE']),
            'MAE': float(metrics_teacher['MAE']),
            'MAPE': float(metrics_teacher['MAPE']),
            'R2': float(metrics_teacher['R2']),
            'MSE': float(metrics_teacher['MSE'])
        },
        'test_metrics_autoregressive': {
            'RMSE': float(metrics_autoreg['RMSE']),
            'MAE': float(metrics_autoreg['MAE']),
            'MAPE': float(metrics_autoreg['MAPE']),
            'R2': float(metrics_autoreg['R2']),
            'MSE': float(metrics_autoreg['MSE'])
        }
    }
    
    metrics_file = reports_dir / 'lstm_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    logger.info(f"✓ Metrics saved to: {metrics_file}")
    
    # Generate plot
    # For plot context, combine train and val
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    plot_file = forecasts_dir / 'lstm_test_plot.png'
    plot_predictions(forecast_df, train_val_df, str(plot_file))
    
    logger.info("\n" + "="*60)
    logger.info("LSTM TRAINING COMPLETE!")
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
