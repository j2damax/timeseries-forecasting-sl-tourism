"""
Prophet Model Training Pipeline.

Implements Section 4 of the problem statement:
- Load minimal Prophet features
- Hyperparameter tuning with rolling-origin validation
- Train final model on train+validation
- Generate test forecasts
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

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from prophet import Prophet
from data_splits import split_data_chronologically, get_split_info, print_split_summary
from evaluation import calculate_metrics
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(42)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_future_dataframe_with_regressors(model: Prophet, train_df: pd.DataFrame, 
                                             periods: int, freq: str = 'MS') -> pd.DataFrame:
    """
    Create future dataframe with regressor values.
    
    For test period, we assume intervention flags are known (realistic for forecasting).
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    
    # Get regressor columns (excluding ds and y)
    regressors = [col for col in train_df.columns if col not in ['ds', 'y']]
    
    # For simplicity, we'll set future regressors to 0 (no interventions expected)
    # In production, these would be based on domain knowledge
    for reg in regressors:
        if reg in train_df.columns:
            # Use the last known value for continuity (or 0 for interventions)
            future[reg] = 0
    
    return future


def rolling_origin_validation(train_df: pd.DataFrame, val_df: pd.DataFrame,
                                changepoint_prior_scale: float,
                                seasonality_prior_scale: float) -> Dict[str, float]:
    """
    Perform rolling-origin validation for Prophet.
    
    For each validation month t (Jan→Dec 2023):
    - Fit model on Train plus all validation months < t
    - Forecast exactly 1 step ahead (month t)
    - Aggregate errors across 12 validation steps
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    val_df : pd.DataFrame
        Validation data
    changepoint_prior_scale : float
        Changepoint prior scale hyperparameter
    seasonality_prior_scale : float
        Seasonality prior scale hyperparameter
        
    Returns
    -------
    Dict[str, float]
        Aggregated validation metrics
    """
    errors = []
    
    for i in range(len(val_df)):
        # Create training data up to current validation point
        train_up_to = pd.concat([train_df, val_df.iloc[:i]], ignore_index=True)
        
        # Get target validation point
        val_point = val_df.iloc[i:i+1]
        
        # Initialize and fit model
        model = Prophet(
            yearly_seasonality='auto',
            weekly_seasonality=False,
            daily_seasonality=False,
            growth='linear',
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale
        )
        
        # Add regressors
        for reg in ['easter_attacks', 'covid_period', 'economic_crisis']:
            model.add_regressor(reg)
        
        # Fit model (suppress output)
        with np.errstate(all='ignore'):
            model.fit(train_up_to, algorithm='LBFGS')
        
        # Create future dataframe for 1-step ahead
        future = pd.DataFrame({
            'ds': val_point['ds'].values,
            'easter_attacks': val_point['easter_attacks'].values,
            'covid_period': val_point['covid_period'].values,
            'economic_crisis': val_point['economic_crisis'].values
        })
        
        # Forecast
        forecast = model.predict(future)
        
        # Calculate error
        y_true = val_point['y'].values[0]
        y_pred = forecast['yhat'].values[0]
        
        errors.append({
            'y_true': y_true,
            'y_pred': y_pred,
            'error': y_true - y_pred
        })
    
    # Aggregate errors
    y_true_arr = np.array([e['y_true'] for e in errors])
    y_pred_arr = np.array([e['y_pred'] for e in errors])
    
    metrics = calculate_metrics(y_true_arr, y_pred_arr)
    
    return metrics


def tune_hyperparameters(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Tune Prophet hyperparameters using rolling-origin validation.
    
    Grid search over:
    - changepoint_prior_scale: [0.05, 0.1, 0.2]
    - seasonality_prior_scale: [5, 10]
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    val_df : pd.DataFrame
        Validation data
        
    Returns
    -------
    Tuple[Dict, Dict]
        (best_params, all_results)
    """
    logger.info("Starting hyperparameter tuning...")
    
    # Define grid
    changepoint_scales = [0.05, 0.1, 0.2]
    seasonality_scales = [5, 10]
    
    results = []
    
    for cp_scale in changepoint_scales:
        for s_scale in seasonality_scales:
            logger.info(f"Testing cp_scale={cp_scale}, s_scale={s_scale}...")
            
            # Perform rolling-origin validation
            metrics = rolling_origin_validation(train_df, val_df, cp_scale, s_scale)
            
            result = {
                'changepoint_prior_scale': cp_scale,
                'seasonality_prior_scale': s_scale,
                'val_mape': metrics['MAPE'],
                'val_rmse': metrics['RMSE'],
                'val_mae': metrics['MAE'],
                'val_r2': metrics['R2']
            }
            results.append(result)
            
            logger.info(f"  MAPE: {metrics['MAPE']:.4f}, RMSE: {metrics['RMSE']:.2f}")
    
    # Find best parameters (minimize MAPE, tie-breaker: RMSE)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['val_mape', 'val_rmse'])
    best_result = results_df.iloc[0].to_dict()
    
    best_params = {
        'changepoint_prior_scale': best_result['changepoint_prior_scale'],
        'seasonality_prior_scale': best_result['seasonality_prior_scale']
    }
    
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best validation MAPE: {best_result['val_mape']:.4f}")
    
    return best_params, results


def train_final_model(train_val_df: pd.DataFrame, best_params: Dict) -> Prophet:
    """
    Train final Prophet model on combined train+validation data.
    
    Parameters
    ----------
    train_val_df : pd.DataFrame
        Combined training and validation data
    best_params : Dict
        Best hyperparameters from tuning
        
    Returns
    -------
    Prophet
        Trained Prophet model
    """
    logger.info("Training final model on train+validation data...")
    
    model = Prophet(
        yearly_seasonality='auto',
        weekly_seasonality=False,
        daily_seasonality=False,
        growth='linear',
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        seasonality_prior_scale=best_params['seasonality_prior_scale']
    )
    
    # Add regressors
    for reg in ['easter_attacks', 'covid_period', 'economic_crisis']:
        model.add_regressor(reg)
    
    # Fit model
    with np.errstate(all='ignore'):
        model.fit(train_val_df, algorithm='LBFGS')
    
    logger.info("Final model training complete!")
    
    return model


def generate_test_forecast(model: Prophet, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate forecasts for test period.
    
    Parameters
    ----------
    model : Prophet
        Trained Prophet model
    test_df : pd.DataFrame
        Test data with regressor values
        
    Returns
    -------
    pd.DataFrame
        Forecast dataframe with actual values
    """
    logger.info("Generating test forecasts...")

    # Prophet expects the future dataframe to contain all regressors used in training
    required_cols = ['ds', 'easter_attacks', 'covid_period', 'economic_crisis']
    missing = [c for c in required_cols if c not in test_df.columns]
    if missing:
        raise ValueError(f"Test dataframe missing required regressor columns: {missing}")

    # Prepare future (exactly the test horizon only)
    future = test_df[required_cols].copy()
    logger.info(f"Test horizon length: {len(future)}")

    # Run prediction
    forecast_full = model.predict(future)

    # Align forecast rows strictly by date to avoid any accidental reindexing
    forecast = forecast_full[['ds', 'yhat']].copy()

    # Sanity checks
    if len(forecast) != len(test_df):
        logger.warning(
            f"Forecast rows ({len(forecast)}) != test rows ({len(test_df)}). Attempting inner merge alignment.")
        merged = pd.merge(test_df[['ds', 'y']], forecast, on='ds', how='inner')
        if len(merged) != len(test_df):
            raise ValueError(
                f"After alignment merge, forecast rows {len(merged)} still != test rows {len(test_df)}")
        aligned = merged
    else:
        # Direct alignment
        aligned = test_df[['ds', 'y']].copy()
        aligned['yhat'] = forecast['yhat'].values

    # Build result dataframe
    result = aligned.rename(columns={'y': 'y_true', 'yhat': 'y_pred'})
    result['residual'] = result['y_true'] - result['y_pred']

    # Final check
    assert len(result) == len(test_df), "Result length must equal test dataframe length"
    return result


def plot_forecast(forecast_df: pd.DataFrame, train_val_df: pd.DataFrame, 
                  output_file: str):
    """
    Plot actual vs forecast with intervention periods shaded.
    
    Parameters
    ----------
    forecast_df : pd.DataFrame
        Test forecast results
    train_val_df : pd.DataFrame
        Training and validation data for context
    output_file : str
        Path to save plot
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot training/validation data
    ax.plot(train_val_df['ds'], train_val_df['y'], 
            label='Train/Val Actual', color='blue', linewidth=1.5)
    
    # Plot test actual and forecast
    ax.plot(forecast_df['ds'], forecast_df['y_true'], 
            label='Test Actual', color='green', linewidth=2, marker='o')
    ax.plot(forecast_df['ds'], forecast_df['y_pred'], 
            label='Test Forecast', color='red', linewidth=2, marker='s', linestyle='--')
    
    # Shade intervention periods (approximate)
    # Easter attacks: April 2019
    ax.axvspan(pd.Timestamp('2019-04-01'), pd.Timestamp('2019-06-01'), 
               alpha=0.2, color='orange', label='Easter Attacks')
    # COVID: 2020-2021
    ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-12-31'), 
               alpha=0.2, color='red', label='COVID-19')
    # Economic crisis: 2022+
    ax.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2023-12-31'), 
               alpha=0.2, color='yellow', label='Economic Crisis')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Tourist Arrivals', fontsize=12)
    ax.set_title('Prophet Model: Actual vs Forecast', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    logger.info(f"Plot saved to: {output_file}")


def main():
    """Main Prophet training pipeline."""
    # Set paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'processed'
    models_dir = base_dir / 'models' / 'prophet'
    forecasts_dir = base_dir / 'forecasts'
    reports_dir = base_dir / 'reports'
    
    # Create directories if needed
    models_dir.mkdir(parents=True, exist_ok=True)
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("PROPHET MODEL TRAINING PIPELINE")
    logger.info("="*60)
    
    # Load data
    logger.info("Loading minimal Prophet features...")
    df = pd.read_csv(data_dir / 'minimal_features_prophet.csv', parse_dates=['ds'])
    logger.info(f"Loaded {len(df)} samples from {df['ds'].min()} to {df['ds'].max()}")
    
    # Split data
    logger.info("\nSplitting data chronologically...")
    train_df, val_df, test_df = split_data_chronologically(df, date_col='ds')
    print_split_summary(train_df, val_df, test_df, date_col='ds')
    
    # Tune hyperparameters
    logger.info("\nTuning hyperparameters with rolling-origin validation...")
    best_params, tuning_results = tune_hyperparameters(train_df, val_df)
    
    # Combine train and validation for final training
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Train final model
    final_model = train_final_model(train_val_df, best_params)
    
    # Generate test forecast
    test_forecast = generate_test_forecast(final_model, test_df)
    
    # Calculate test metrics
    logger.info("\nCalculating test metrics...")
    test_metrics = calculate_metrics(test_forecast['y_true'].values, 
                                      test_forecast['y_pred'].values)
    
    logger.info(f"Test RMSE: {test_metrics['RMSE']:.2f}")
    logger.info(f"Test MAE: {test_metrics['MAE']:.2f}")
    logger.info(f"Test MAPE: {test_metrics['MAPE']:.4f}")
    logger.info(f"Test R2: {test_metrics['R2']:.4f}")
    
    # Save model
    model_file = models_dir / 'final_model.pkl'
    joblib.dump(final_model, model_file)
    logger.info(f"\n✓ Model saved to: {model_file}")
    
    # Save test forecast
    forecast_file = forecasts_dir / 'prophet_test_forecast.csv'
    test_forecast.to_csv(forecast_file, index=False)
    logger.info(f"✓ Test forecast saved to: {forecast_file}")
    
    # Save metrics
    metrics_output = {
        'best_hyperparameters': best_params,
        'hyperparameter_tuning_results': tuning_results,
        'test_metrics': {
            'RMSE': float(test_metrics['RMSE']),
            'MAE': float(test_metrics['MAE']),
            'MAPE': float(test_metrics['MAPE']),
            'R2': float(test_metrics['R2']),
            'MSE': float(test_metrics['MSE'])
        }
    }
    
    metrics_file = reports_dir / 'prophet_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    logger.info(f"✓ Metrics saved to: {metrics_file}")
    
    # Generate plot
    plot_file = forecasts_dir / 'prophet_test_plot.png'
    plot_forecast(test_forecast, train_val_df, str(plot_file))
    
    logger.info("\n" + "="*60)
    logger.info("PROPHET TRAINING COMPLETE!")
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
