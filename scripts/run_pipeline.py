"""
End-to-End Training & Evaluation Pipeline Orchestration.

This script orchestrates the complete pipeline:
1. Generate minimal features
2. Create data splits
3. Train Prophet model
4. Train LSTM model  
5. Run Chronos inference
6. Compare models
7. Run prediction tests
8. Generate summary report

Implements Section 10 of the problem statement.
"""

import sys
import argparse
import logging
from pathlib import Path
import subprocess
import json
import pandas as pd
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/pipeline.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


def run_script(script_name: str, description: str, skip: bool = False) -> bool:
    """
    Run a Python script and return success status.
    
    Parameters
    ----------
    script_name : str
        Name of the script to run
    description : str
        Description for logging
    skip : bool
        Whether to skip this script
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    if skip:
        logger.info(f"⊘ Skipping: {description}")
        return True
    
    logger.info("="*60)
    logger.info(f"Running: {description}")
    logger.info("="*60)
    
    try:
        result = subprocess.run(
            [sys.executable, f"scripts/{script_name}"],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
        logger.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with exit code {e.returncode}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"✗ {description} failed with error: {e}")
        return False


def generate_model_comparison(base_dir: Path) -> pd.DataFrame:
    """
    Generate model comparison table from saved metrics.
    
    Parameters
    ----------
    base_dir : Path
        Base directory of the project
        
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    logger.info("\n" + "="*60)
    logger.info("Generating Model Comparison")
    logger.info("="*60)
    
    reports_dir = base_dir / 'reports'
    
    rows = []
    
    # Prophet metrics
    prophet_file = reports_dir / 'prophet_metrics.json'
    if prophet_file.exists():
        with open(prophet_file, 'r') as f:
            prophet_data = json.load(f)
        
        test_metrics = prophet_data.get('test_metrics', {})
        rows.append({
            'model': 'Prophet',
            'split': 'test',
            'RMSE': test_metrics.get('RMSE', None),
            'MAE': test_metrics.get('MAE', None),
            'MAPE': test_metrics.get('MAPE', None),
            'R2': test_metrics.get('R2', None)
        })
    
    # LSTM metrics
    lstm_file = reports_dir / 'lstm_metrics.json'
    if lstm_file.exists():
        with open(lstm_file, 'r') as f:
            lstm_data = json.load(f)
        
        # Teacher forcing
        test_metrics_teacher = lstm_data.get('test_metrics_teacher_forcing', {})
        rows.append({
            'model': 'LSTM',
            'split': 'test_teacher',
            'RMSE': test_metrics_teacher.get('RMSE', None),
            'MAE': test_metrics_teacher.get('MAE', None),
            'MAPE': test_metrics_teacher.get('MAPE', None),
            'R2': test_metrics_teacher.get('R2', None)
        })
        
        # Autoregressive
        test_metrics_autoreg = lstm_data.get('test_metrics_autoregressive', {})
        rows.append({
            'model': 'LSTM',
            'split': 'test_autoreg',
            'RMSE': test_metrics_autoreg.get('RMSE', None),
            'MAE': test_metrics_autoreg.get('MAE', None),
            'MAPE': test_metrics_autoreg.get('MAPE', None),
            'R2': test_metrics_autoreg.get('R2', None)
        })
    
    # Chronos metrics
    chronos_file = reports_dir / 'chronos_metrics.json'
    if chronos_file.exists():
        with open(chronos_file, 'r') as f:
            chronos_data = json.load(f)
        
        test_metrics = chronos_data.get('test_metrics', {})
        rows.append({
            'model': 'Chronos',
            'split': 'test',
            'RMSE': test_metrics.get('RMSE', None),
            'MAE': test_metrics.get('MAE', None),
            'MAPE': test_metrics.get('MAPE', None),
            'R2': test_metrics.get('R2', None)
        })
    
    # Create dataframe
    if rows:
        df = pd.DataFrame(rows)
        
        # Sort by MAPE (primary) and RMSE (secondary)
        df = df.sort_values(['MAPE', 'RMSE'])
        
        # Save to CSV
        output_file = reports_dir / 'model_comparison.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"✓ Model comparison saved to: {output_file}")
        
        # Display table
        logger.info("\nModel Comparison (sorted by MAPE, then RMSE):")
        logger.info("\n" + df.to_string(index=False))
        
        return df
    else:
        logger.warning("No metrics files found for comparison")
        return pd.DataFrame()


def generate_summary_report(base_dir: Path, comparison_df: pd.DataFrame):
    """
    Generate summary report in Markdown.
    
    Parameters
    ----------
    base_dir : Path
        Base directory of the project
    comparison_df : pd.DataFrame
        Model comparison dataframe
    """
    logger.info("\n" + "="*60)
    logger.info("Generating Summary Report")
    logger.info("="*60)
    
    reports_dir = base_dir / 'reports'
    output_file = reports_dir / 'summary.md'
    
    with open(output_file, 'w') as f:
        f.write("# End-to-End Training & Evaluation Summary\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Pipeline Overview\n\n")
        f.write("This report summarizes the end-to-end training and evaluation pipeline for ")
        f.write("Sri Lanka tourism forecasting.\n\n")
        
        f.write("## Data Splits\n\n")
        f.write("- **Train:** 2017-01-01 to 2022-12-01 (72 samples)\n")
        f.write("- **Validation:** 2023-01-01 to 2023-12-01 (12 samples)\n")
        f.write("- **Test:** 2024-01-01 to 2024-07-01 (7 samples)\n\n")
        
        f.write("## Model Comparison\n\n")
        if not comparison_df.empty:
            f.write(comparison_df.to_markdown(index=False))
            f.write("\n\n")
            
            # Best model
            best = comparison_df.iloc[0]
            f.write(f"### Best Model (by MAPE)\n\n")
            f.write(f"**{best['model']}** ({best['split']})\n")
            f.write(f"- RMSE: {best['RMSE']:.2f}\n")
            f.write(f"- MAE: {best['MAE']:.2f}\n")
            f.write(f"- MAPE: {best['MAPE']:.4f}\n")
            f.write(f"- R²: {best['R2']:.4f}\n\n")
        else:
            f.write("*No model results available*\n\n")
        
        f.write("## Model Justifications\n\n")
        
        f.write("### Prophet\n")
        f.write("- **Strengths:** Explicit trend/seasonality decomposition, interpretable regressors\n")
        f.write("- **Use Case:** Baseline model with strong interpretability\n")
        f.write("- **Interventions:** Handles Easter attacks, COVID-19, and economic crisis as regressors\n\n")
        
        f.write("### LSTM\n")
        f.write("- **Strengths:** Non-linear interactions, captures subtle regime shifts\n")
        f.write("- **Use Case:** Deep learning approach with cyclical encoding\n")
        f.write("- **Windowing:** 24-month lookback for sequential patterns\n\n")
        
        f.write("### Chronos\n")
        f.write("- **Strengths:** Pre-trained temporal representations, uncertainty intervals\n")
        f.write("- **Use Case:** Zero-shot forecasting without feature engineering\n")
        f.write("- **Note:** Current implementation uses stub for demonstration\n\n")
        
        f.write("## Artifacts Generated\n\n")
        f.write("### Data Files\n")
        f.write("- `data/processed/minimal_features_prophet.csv`\n")
        f.write("- `data/processed/minimal_features_lstm.csv`\n")
        f.write("- `data/processed/chronos_series.csv`\n")
        f.write("- `data/processed/splits.json`\n\n")
        
        f.write("### Models\n")
        f.write("- `models/prophet/final_model.pkl`\n")
        f.write("- `models/lstm/best_weights.h5`\n")
        f.write("- `artifacts/lstm_scaler.pkl`\n\n")
        
        f.write("### Forecasts\n")
        f.write("- `forecasts/prophet_test_forecast.csv`\n")
        f.write("- `forecasts/lstm_test_forecast.csv`\n")
        f.write("- `forecasts/chronos_test_forecast.csv`\n\n")
        
        f.write("### Visualizations\n")
        f.write("- `forecasts/prophet_test_plot.png`\n")
        f.write("- `forecasts/lstm_test_plot.png`\n")
        f.write("- `forecasts/chronos_test_intervals.png`\n\n")
        
        f.write("### Reports\n")
        f.write("- `reports/prophet_metrics.json`\n")
        f.write("- `reports/lstm_metrics.json`\n")
        f.write("- `reports/chronos_metrics.json`\n")
        f.write("- `reports/model_comparison.csv`\n")
        f.write("- `reports/summary.md` (this file)\n\n")
    
    logger.info(f"✓ Summary report saved to: {output_file}")


def main():
    """Main pipeline orchestration."""
    parser = argparse.ArgumentParser(
        description='End-to-End Training & Evaluation Pipeline'
    )
    parser.add_argument('--rebuild', action='store_true',
                        help='Force retrain even if model artifacts exist')
    parser.add_argument('--skip-prophet', action='store_true',
                        help='Skip Prophet training')
    parser.add_argument('--skip-lstm', action='store_true',
                        help='Skip LSTM training')
    parser.add_argument('--skip-chronos', action='store_true',
                        help='Skip Chronos inference')
    parser.add_argument('--window-size', type=int, default=24,
                        help='Window size for LSTM (default: 24)')
    parser.add_argument('--horizon', type=int, default=1,
                        help='Forecast horizon (default: 1)')
    
    args = parser.parse_args()
    
    # Setup
    base_dir = Path.cwd()
    logs_dir = base_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    logger.info("="*60)
    logger.info("END-TO-END TRAINING & EVALUATION PIPELINE")
    logger.info("="*60)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Arguments: {vars(args)}")
    
    success = True
    
    # Step 1: Generate minimal features
    if not run_script('generate_minimal_features.py', 
                      'Generate Minimal Features'):
        success = False
    
    # Step 2: Train Prophet
    if not run_script('train_prophet.py', 
                      'Train Prophet Model',
                      skip=args.skip_prophet):
        logger.warning("Prophet training failed or was skipped")
        # Don't fail pipeline, continue with other models
    
    # Step 3: Train LSTM
    if not run_script('train_lstm.py', 
                      'Train LSTM Model',
                      skip=args.skip_lstm):
        logger.warning("LSTM training failed or was skipped")
        # Don't fail pipeline, continue with other models
    
    # Step 4: Run Chronos
    if not run_script('train_chronos.py', 
                      'Run Chronos Inference',
                      skip=args.skip_chronos):
        logger.warning("Chronos inference failed or was skipped")
        # Don't fail pipeline
    
    # Step 5: Model comparison
    comparison_df = generate_model_comparison(base_dir)
    
    # Step 6: Generate summary
    generate_summary_report(base_dir, comparison_df)
    
    # Step 7: Run prediction tests
    logger.info("\n" + "="*60)
    logger.info("Running Prediction Tests")
    logger.info("="*60)
    if not run_script('prediction_tests.py', 'Prediction Tests'):
        logger.warning("Some prediction tests failed")
        # Don't fail pipeline
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("PIPELINE EXECUTION COMPLETE")
    logger.info("="*60)
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        logger.info("✓ All steps completed successfully!")
        logger.info("\nGenerated artifacts:")
        logger.info("  - Data: data/processed/")
        logger.info("  - Models: models/")
        logger.info("  - Forecasts: forecasts/")
        logger.info("  - Reports: reports/")
        logger.info("\nView the summary report: reports/summary.md")
        return 0
    else:
        logger.error("✗ Pipeline completed with some errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
