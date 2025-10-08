"""Baseline forecasting scripts: naive and seasonal naive.
Saves forecasts and metrics for comparison.
"""
from __future__ import annotations
import os
import json
import argparse
import pandas as pd
import numpy as np
from typing import Dict

from data_loader import load_csv_data
from evaluation import calculate_metrics
# NOTE: Baseline forecasts (naive & seasonal naive) do not need engineered features.
# We keep imports minimal. If future extension requires features, use:
# from preprocessing import create_time_features, create_lag_features

OUTPUT_DIR = "forecasts"
REPORT_DIR = "reports"


def naive_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    last_val = series.iloc[-1]
    return np.repeat(last_val, horizon)


def seasonal_naive_forecast(series: pd.Series, horizon: int, season_length: int = 12) -> np.ndarray:
    if len(series) < season_length:
        # Fallback to naive if not enough history
        return naive_forecast(series, horizon)
    seasonal_slice = series.iloc[-season_length:]
    reps = int(np.ceil(horizon / season_length))
    return np.tile(seasonal_slice.values, reps)[:horizon]


def compute_and_save(df: pd.DataFrame, horizon: int = 7) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    df = df.sort_values("Date")
    train_val = df.iloc[:-horizon]
    test = df.iloc[-horizon:]

    y_train_val = train_val["Arrivals"].astype(float)
    y_test = test["Arrivals"].astype(float)

    forecasts: Dict[str, np.ndarray] = {
        "naive": naive_forecast(y_train_val, horizon),
        "seasonal_naive": seasonal_naive_forecast(y_train_val, horizon, 12),
    }

    rows = []
    for name, preds in forecasts.items():
        metrics = calculate_metrics(y_test.values, preds)
        metrics_row = {"model": name, "split": "test", **metrics}
        rows.append(metrics_row)

        # save forecast csv
        fc_df = test[["Date", "Arrivals"]].copy()
        fc_df["Forecast"] = preds
        out_path = os.path.join(OUTPUT_DIR, f"{name}_test_forecast.csv")
        fc_df.to_csv(out_path, index=False)

        # save metrics json
        metrics_path = os.path.join(REPORT_DIR, f"{name}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_row, f, indent=2)

    comp_df = pd.DataFrame(rows)
    comp_path = os.path.join(REPORT_DIR, "baseline_model_comparison.csv")
    comp_df.to_csv(comp_path, index=False)

    print("\nBaseline Models (test horizon)")
    print(comp_df.sort_values("RMSE"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/monthly_tourist_arrivals.csv")
    parser.add_argument("--horizon", type=int, default=7)
    args = parser.parse_args()

    df = load_csv_data(args.data, date_column="Date")
    # Feature engineering intentionally omitted (not required for baselines)

    compute_and_save(df, horizon=args.horizon)


if __name__ == "__main__":
    main()
