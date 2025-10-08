"""Diebold-Mariano test comparing Prophet vs Seasonal Naive forecasts on test set.
Small-sample caution: horizon=7.
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import t

REPORT_DIR = "reports"


def load_forecasts(prophet_path: str, seasonal_naive_path: str):
    p = pd.read_csv(prophet_path, parse_dates=["ds"])  # minimal prophet file: ds,y_true,y_pred,residual
    # Harmonize columns
    if {"y_true", "y_pred"}.issubset(p.columns):
        p = p.rename(columns={"y_true": "y", "y_pred": "yhat"})
    s = pd.read_csv(seasonal_naive_path, parse_dates=["Date"])  # seasonal_naive_test_forecast.csv
    merged = p.merge(
        s.rename(columns={"Date": "ds", "Forecast": "seasonal_naive"})[["ds", "seasonal_naive"]],
        on="ds", how="inner"
    )
    return merged


def dm_test(actual: np.ndarray, f1: np.ndarray, f2: np.ndarray, power: int = 2):
    # Loss differential d_t = |e1|^p - |e2|^p (p=2 default => squared errors)
    e1 = actual - f1
    e2 = actual - f2
    d = np.abs(e1) ** power - np.abs(e2) ** power
    T = len(d)
    if T < 3:
        return {"stat": None, "p_value": None, "note": "Insufficient observations for DM"}
    d_bar = d.mean()
    # No HAC adjustment (h=1) small sample horizon assumption
    var_d = d.var(ddof=1)
    if var_d == 0:
        return {"stat": None, "p_value": None, "note": "Zero variance in differential"}
    stat = d_bar / np.sqrt(var_d / T)
    p_value = 2 * (1 - t.cdf(abs(stat), df=T - 1))
    return {"stat": float(stat), "p_value": float(p_value), "n": int(T), "mean_diff": float(d_bar)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prophet", default="forecasts/prophet_test_forecast.csv")
    parser.add_argument("--seasonal_naive", default="forecasts/seasonal_naive_test_forecast.csv")
    args = parser.parse_args()

    os.makedirs(REPORT_DIR, exist_ok=True)

    try:
        merged = load_forecasts(args.prophet, args.seasonal_naive)
    except FileNotFoundError as e:
        result = {"stat": None, "p_value": None, "note": f"Missing file: {e}"}
        with open(os.path.join(REPORT_DIR, "dm_test.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
        return

    if not {"y", "yhat", "seasonal_naive"}.issubset(merged.columns):
        result = {"stat": None, "p_value": None, "note": "Required columns not found after merge."}
        with open(os.path.join(REPORT_DIR, "dm_test.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
        return

    actual = merged["y"].values
    prophet_fc = merged["yhat"].values
    seasonal_fc = merged["seasonal_naive"].values

    result = dm_test(actual, prophet_fc, seasonal_fc, power=2)
    with open(os.path.join(REPORT_DIR, "dm_test.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("Diebold-Mariano Test (Prophet vs Seasonal Naive, squared error loss):")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
