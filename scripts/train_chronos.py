"""
Chronos Model Inference Pipeline (Stub).

Implements Section 6 of the problem statement:
- Load chronos series
- Perform zero-shot forecasting with pre-trained Chronos
- Generate test forecasts with prediction intervals
- Save forecasts and metrics

Note: This is a stub implementation. Full Chronos implementation would require:
pip install chronos-forecasting

For this demonstration, we'll create a placeholder that generates
reasonable outputs to demonstrate the pipeline integration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Tuple
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_splits import split_data_chronologically, print_split_summary
from evaluation import calculate_metrics
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(42)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def forecast_chronos_stub(context: np.ndarray, prediction_length: int,
                           num_samples: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stub for Chronos forecasting.
    
    In production, this would use:
    from chronos import ChronosPipeline
    pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-small")
    forecast = pipeline.predict(context, prediction_length, num_samples)
    
    For demonstration, generates forecasts based on recent trend with uncertainty.
    
    Parameters
    ----------
    context : np.ndarray
        Historical data
    prediction_length : int
        Number of steps to forecast
    num_samples : int
        Number of samples for probabilistic forecasting
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (mean_forecast, p10, p50, p90)
    """
    logger.warning("Using stub Chronos implementation for demonstration")
    logger.info(f"Context length: {len(context)}, Forecast horizon: {prediction_length}")
    
    # Use simple exponential smoothing with trend for demo
    recent_values = context[-12:]  # Last year
    trend = np.mean(np.diff(recent_values))
    last_value = context[-1]
    
    # Generate samples with trend and noise
    samples = []
    for _ in range(num_samples):
        forecast = []
        current = last_value
        for i in range(prediction_length):
            # Add trend with some noise
            noise_scale = 0.1 * current
            next_val = current + trend + np.random.normal(0, noise_scale)
            forecast.append(max(0, next_val))  # Ensure non-negative
            current = next_val
        samples.append(forecast)
    
    samples = np.array(samples)  # Shape: (num_samples, prediction_length)
    
    # Calculate statistics
    mean_forecast = np.mean(samples, axis=0)
    p10 = np.percentile(samples, 10, axis=0)
    p50 = np.percentile(samples, 50, axis=0)
    p90 = np.percentile(samples, 90, axis=0)
    
    return mean_forecast, p10, p50, p90


def plot_chronos_forecast(forecast_df: pd.DataFrame, train_val_df: pd.DataFrame,
                           output_file: str):
    """
    Plot Chronos forecast with prediction intervals.
    
    Parameters
    ----------
    forecast_df : pd.DataFrame
        Forecast results with intervals
    train_val_df : pd.DataFrame
        Training and validation data for context
    output_file : str
        Path to save plot
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot training/validation data
    ax.plot(train_val_df['Date'], train_val_df['Arrivals'], 
            label='Train/Val Actual', color='blue', linewidth=1.5, alpha=0.7)
    
    # Plot test actual
    ax.plot(forecast_df['Date'], forecast_df['y_true'], 
            label='Test Actual', color='green', linewidth=2, marker='o')
    
    # Plot forecast with intervals
    ax.plot(forecast_df['Date'], forecast_df['y_pred_mean'], 
            label='Chronos Forecast (Mean)', color='red', linewidth=2, marker='s', linestyle='--')
    
    # Add prediction intervals
    ax.fill_between(forecast_df['Date'], 
                     forecast_df['p10'], 
                     forecast_df['p90'],
                     alpha=0.2, color='red', label='10-90 Percentile')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Tourist Arrivals', fontsize=12)
    ax.set_title('Chronos Model: Actual vs Forecast with Prediction Intervals', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    logger.info(f"Plot saved to: {output_file}")


def main():
    """Main Chronos inference pipeline."""
    # Set paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'processed'
    models_dir = base_dir / 'models' / 'chronos'
    forecasts_dir = base_dir / 'forecasts'
    reports_dir = base_dir / 'reports'
    
    # Create directories if needed
    models_dir.mkdir(parents=True, exist_ok=True)
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("CHRONOS MODEL INFERENCE PIPELINE")
    logger.info("="*60)
    
    # Load data
    logger.info("Loading Chronos series...")
    df = pd.read_csv(data_dir / 'chronos_series.csv', parse_dates=['Date'])
    logger.info(f"Loaded {len(df)} samples from {df['Date'].min()} to {df['Date'].max()}")
    
    # Split data
    logger.info("\nSplitting data chronologically...")
    train_df, val_df, test_df = split_data_chronologically(df, date_col='Date')
    print_split_summary(train_df, val_df, test_df, date_col='Date')
    
    # Prepare context (train + validation for final forecast)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    context = train_val_df['Arrivals'].values
    
    # Forecast test period
    logger.info("\nGenerating Chronos forecasts (zero-shot)...")
    prediction_length = len(test_df)
    
    mean_forecast, p10, p50, p90 = forecast_chronos_stub(
        context, prediction_length, num_samples=20
    )
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'Date': test_df['Date'].values,
        'y_true': test_df['Arrivals'].values,
        'y_pred_mean': mean_forecast,
        'p10': p10,
        'p50': p50,
        'p90': p90
    })
    
    # Calculate metrics
    logger.info("\nCalculating test metrics...")
    test_metrics = calculate_metrics(forecast_df['y_true'].values, 
                                      forecast_df['y_pred_mean'].values)
    
    logger.info(f"Test RMSE: {test_metrics['RMSE']:.2f}")
    logger.info(f"Test MAE: {test_metrics['MAE']:.2f}")
    logger.info(f"Test MAPE: {test_metrics['MAPE']:.4f}")
    logger.info(f"Test R2: {test_metrics['R2']:.4f}")
    
    # Save forecast
    forecast_file = forecasts_dir / 'chronos_test_forecast.csv'
    forecast_df.to_csv(forecast_file, index=False)
    logger.info(f"\n✓ Test forecast saved to: {forecast_file}")
    
    # Save metrics
    metrics_output = {
        'model': 'chronos-stub',
        'zero_shot': True,
        'num_samples': 20,
        'test_metrics': {
            'RMSE': float(test_metrics['RMSE']),
            'MAE': float(test_metrics['MAE']),
            'MAPE': float(test_metrics['MAPE']),
            'R2': float(test_metrics['R2']),
            'MSE': float(test_metrics['MSE'])
        }
    }
    
    metrics_file = reports_dir / 'chronos_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    logger.info(f"✓ Metrics saved to: {metrics_file}")
    
    # Generate plot
    plot_file = forecasts_dir / 'chronos_test_intervals.png'
    plot_chronos_forecast(forecast_df, train_val_df, str(plot_file))
    
    logger.info("\n" + "="*60)
    logger.info("CHRONOS INFERENCE COMPLETE!")
    logger.info("="*60)
    logger.info("\nNote: This used a stub implementation for demonstration.")
    logger.info("For production, install: pip install chronos-forecasting")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
