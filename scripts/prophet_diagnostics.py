"""Prophet residual diagnostics, interval coverage, and basic statistical tests.
Outputs:
 - reports/prophet_residuals.csv
 - reports/prophet_diagnostics.json (ACF stats, coverage)
 - forecasts/prophet_residual_plot.png
 - forecasts/prophet_residual_acf.png
"""
from __future__ import annotations
import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf, q_stat
from scipy.stats import probplot

REPORT_DIR = "reports"
FORECAST_DIR = "forecasts"
MODEL_DIR = "models/prophet"


def load_prophet_artifacts(forecast_path: str, data_path: str):
    fc = pd.read_csv(forecast_path, parse_dates=["ds"])  # prophet_test_forecast.csv expected columns
    df = pd.read_csv(data_path, parse_dates=["Date"])  # full data
    return df, fc


def compute_residuals(df: pd.DataFrame, fc: pd.DataFrame):
    """Unify schema and compute residuals.

    Supports two prophet forecast formats:
    1. Extended Prophet output with columns: ds, yhat, yhat_lower, yhat_upper, y (after merge)
    2. Minimal custom file with columns: ds, y_true, y_pred, residual (already provided)
    """
    # Detect minimal custom schema
    if {"y_true", "y_pred"}.issubset(fc.columns):
        df_res = fc.rename(columns={"y_true": "y", "y_pred": "yhat"}).copy()
        # Interval columns may not exist
        if "residual" not in df_res.columns:
            df_res["residual"] = df_res["y"] - df_res["yhat"]
        return df_res

    # Fallback: assume standard prophet yhat schema and merge actuals
    merged = fc.merge(
        df.rename(columns={"Date": "ds", "Arrivals": "y"})[["ds", "y"]],
        on="ds", how="left"
    )
    merged["residual"] = merged["y"] - merged["yhat"]
    return merged


def interval_coverage(df_res: pd.DataFrame):
    if {"yhat_lower", "yhat_upper"}.issubset(df_res.columns):
        cov = ((df_res["y"] >= df_res["yhat_lower"]) & (df_res["y"] <= df_res["yhat_upper"]))
        return {
            "interval_coverage": float(cov.mean()),
            "count": int(cov.sum()),
            "n": int(len(df_res))
        }
    return {"interval_coverage": None, "count": None, "n": int(len(df_res)), "note": "No interval columns present"}


def ljung_box(residuals: np.ndarray, lags: int = 12):
    # q_stat returns (Q, p-values)
    acf_vals = acf(residuals, fft=False, nlags=lags)
    Q, p = q_stat(acf_vals[1:], len(residuals))
    return {"lb_lags": lags, "lb_Q": float(Q[-1]), "lb_p": float(p[-1])}


def plot_residuals(df_res: pd.DataFrame, out_path: str):
    plt.figure(figsize=(8,4))
    plt.plot(df_res["ds"], df_res["residual"], marker="o")
    plt.axhline(0, color='black', linewidth=1)
    plt.title("Prophet Residuals (Test Horizon)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_acf(residuals: np.ndarray, out_path: str):
    from statsmodels.graphics.tsaplots import plot_acf as pacf_plot
    fig, ax = plt.subplots(figsize=(6,4))
    pacf_plot(residuals, ax=ax, lags=min(10, len(residuals)-1))
    ax.set_title("Prophet Residual ACF (Test)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_qq(residuals: np.ndarray, out_path: str):
    plt.figure(figsize=(5,5))
    probplot(residuals, dist="norm", plot=plt)
    plt.title("Prophet Residuals Q-Q")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast", default="forecasts/prophet_test_forecast.csv")
    parser.add_argument("--data", default="data/processed/monthly_tourist_arrivals.csv")
    args = parser.parse_args()

    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(FORECAST_DIR, exist_ok=True)

    df, fc = load_prophet_artifacts(args.forecast, args.data)
    df_res = compute_residuals(df, fc)
    df_res.to_csv(os.path.join(REPORT_DIR, "prophet_residuals.csv"), index=False)

    coverage = interval_coverage(df_res)
    residuals = df_res["residual"].values

    if len(residuals) >= 5:
        lb = ljung_box(residuals, lags=min(12, len(residuals)-1))
    else:
        lb = {"lb_lags": None, "lb_Q": None, "lb_p": None, "note": "Insufficient residual count"}

    # Plots
    plot_residuals(df_res, os.path.join(FORECAST_DIR, "prophet_residual_plot.png"))
    plot_acf(residuals, os.path.join(FORECAST_DIR, "prophet_residual_acf.png"))
    plot_qq(residuals, os.path.join(FORECAST_DIR, "prophet_residual_qq.png"))

    diagnostics = {"coverage": coverage, "ljung_box": lb, "n_residuals": int(len(residuals))}
    with open(os.path.join(REPORT_DIR, "prophet_diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)

    print("Prophet Diagnostics:")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
