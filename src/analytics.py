"""
analytics.py
------------
Reusable, unit-tested analytics functions for UPI transaction data.
All functions accept a pd.DataFrame and return scalar values or DataFrames.
No side effects; no printing; no global state.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# KPI Calculations
# ---------------------------------------------------------------------------

def total_transaction_volume(df: pd.DataFrame) -> float:
    """
    Sum of transaction_volume (in crore) for the given dataframe.
    Returns 0.0 if the dataframe is empty.
    """
    if df.empty or "transaction_volume" not in df.columns:
        return 0.0
    return float(df["transaction_volume"].sum())


def total_transaction_value(df: pd.DataFrame) -> float:
    """
    Sum of transaction_value (in ₹ lakh crore) for the given dataframe.
    Returns 0.0 if the dataframe is empty.
    """
    if df.empty or "transaction_value" not in df.columns:
        return 0.0
    return float(df["transaction_value"].sum())


def average_transaction_value(df: pd.DataFrame) -> float:
    """
    Average value per UPI transaction in ₹.
    Calculated as: sum(value_lakh_crore * 1e12) / sum(volume_crore * 1e7)
    Returns np.nan if volume is zero or data is empty.
    """
    if df.empty:
        return np.nan
    total_vol = df["transaction_volume"].sum()
    total_val = df["transaction_value"].sum()
    if total_vol <= 0:
        return np.nan
    # value is in lakh crore (1e12 rupees), volume is in crore (1e7 txns)
    return float((total_val * 1e12) / (total_vol * 1e7))


def period_over_period_growth(
    df: pd.DataFrame, metric: str = "transaction_volume", periods: int = 1
) -> float:
    """
    Percentage change from (last - periods) to last row for a given metric.
    Returns np.nan if insufficient data.
    """
    if df.empty or metric not in df.columns or len(df) < periods + 1:
        return np.nan
    prev = df[metric].iloc[-(periods + 1)]
    curr = df[metric].iloc[-1]
    if prev == 0 or pd.isna(prev) or pd.isna(curr):
        return np.nan
    return float(((curr - prev) / prev) * 100)


def latest_period(df: pd.DataFrame) -> str:
    """Return the most recent reporting period label."""
    if df.empty or "period" not in df.columns:
        return "N/A"
    return str(df["period"].iloc[-1])


def highest_volume_period(df: pd.DataFrame) -> dict:
    """Return the period with the highest transaction volume."""
    if df.empty:
        return {"period": "N/A", "value": np.nan}
    idx = df["transaction_volume"].idxmax()
    return {
        "period": df.loc[idx, "period"],
        "value": df.loc[idx, "transaction_volume"],
    }


def lowest_volume_period(df: pd.DataFrame) -> dict:
    """Return the period with the lowest transaction volume."""
    if df.empty:
        return {"period": "N/A", "value": np.nan}
    idx = df["transaction_volume"].idxmin()
    return {
        "period": df.loc[idx, "period"],
        "value": df.loc[idx, "transaction_volume"],
    }


def monthly_growth_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the dataframe with recalculated month-over-month
    growth rates for both volume and value.
    """
    if df.empty:
        return df
    result = df.copy()
    result["volume_mom_growth"] = result["transaction_volume"].pct_change() * 100
    result["value_mom_growth"] = result["transaction_value"].pct_change() * 100
    return result


def yoy_growth_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the dataframe with year-over-year growth rates
    for both volume and value (12-period lag).
    Only meaningful when at least 13 rows exist.
    """
    if df.empty or len(df) < 13:
        return df
    result = df.copy()
    result["volume_yoy_growth"] = result["transaction_volume"].pct_change(12) * 100
    result["value_yoy_growth"] = result["transaction_value"].pct_change(12) * 100
    return result


# ---------------------------------------------------------------------------
# Anomaly / Trend Detection
# ---------------------------------------------------------------------------

def detect_anomalies(
    df: pd.DataFrame,
    metric: str = "transaction_volume",
    z_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Flag rows where the month-over-month change in `metric` exceeds
    `z_threshold` standard deviations from the mean of all MoM changes.

    Returns the anomalous rows with an added 'anomaly_reason' column.
    Statistical method is clearly labelled.
    """
    if df.empty or metric not in df.columns or len(df) < 3:
        return pd.DataFrame()

    changes = df[metric].pct_change() * 100
    mean_chg = changes.mean()
    std_chg = changes.std()

    if std_chg == 0 or pd.isna(std_chg):
        return pd.DataFrame()

    z_scores = (changes - mean_chg).abs() / std_chg
    anomalous = df[z_scores > z_threshold].copy()
    anomalous["anomaly_z_score"] = z_scores[z_scores > z_threshold]
    anomalous["anomaly_reason"] = (
        f"MoM change in {metric} exceeds {z_threshold}σ "
        f"(mean={mean_chg:.1f}%, std={std_chg:.1f}%)"
    )
    return anomalous


def trend_summary(df: pd.DataFrame) -> dict:
    """
    Compute descriptive trend statistics across the filtered dataset.
    Returns a dict with:
      - direction: "rising", "falling", or "mixed"
      - avg_monthly_volume_growth: mean MoM % change (volume)
      - avg_monthly_value_growth: mean MoM % change (value)
      - correlation_volume_value: Pearson correlation between volume and value
      - best_month: period with max volume
      - worst_month: period with min volume
    """
    if df.empty or len(df) < 2:
        return {}

    df2 = monthly_growth_rates(df)
    vol_growth = df2["volume_mom_growth"].dropna()
    val_growth = df2["value_mom_growth"].dropna() if "value_mom_growth" in df2.columns else pd.Series([], dtype=float)

    avg_vol_growth = float(vol_growth.mean()) if not vol_growth.empty else np.nan
    avg_val_growth = float(val_growth.mean()) if not val_growth.empty else np.nan

    if avg_vol_growth > 1:
        direction = "rising"
    elif avg_vol_growth < -1:
        direction = "falling"
    else:
        direction = "mixed"

    corr = np.nan
    if (
        "transaction_volume" in df.columns
        and "transaction_value" in df.columns
        and len(df) > 2
    ):
        corr = float(df["transaction_volume"].corr(df["transaction_value"]))

    best = highest_volume_period(df)
    worst = lowest_volume_period(df)

    return {
        "direction": direction,
        "avg_monthly_volume_growth": avg_vol_growth,
        "avg_monthly_value_growth": avg_val_growth,
        "correlation_volume_value": corr,
        "best_month": best["period"],
        "best_month_volume": best["value"],
        "worst_month": worst["period"],
        "worst_month_volume": worst["value"],
    }


# ---------------------------------------------------------------------------
# Forecasting (simple linear baseline — only used when data ≥ 24 months)
# ---------------------------------------------------------------------------

def simple_forecast(
    df: pd.DataFrame,
    periods_ahead: int = 6,
    metric: str = "transaction_volume",
) -> Optional[pd.DataFrame]:
    """
    Fit a simple linear regression on the numeric index to produce a
    short-term trend baseline forecast. This is intentionally naive;
    it does NOT model seasonality or external factors.

    Returns a DataFrame with columns ['period', 'forecast', 'lower', 'upper']
    or None if insufficient data.

    Uncertainty bands use ±1 standard deviation of residuals — NOT prediction
    intervals. This is for illustrative trend extrapolation only.
    """
    if df.empty or len(df) < 24 or metric not in df.columns:
        return None

    series = df[metric].dropna().values
    n = len(series)
    X = np.arange(n, dtype=float)

    # OLS via numpy — no external dependency
    X_mean = X.mean()
    y_mean = series.mean()
    denom = np.sum((X - X_mean) ** 2)
    if denom == 0:
        return None
    slope = np.sum((X - X_mean) * (series - y_mean)) / denom
    intercept = y_mean - slope * X_mean

    def _predict(x_arr):
        return intercept + slope * x_arr

    residuals = series - _predict(X)
    residual_std = float(np.std(residuals))

    future_X = np.arange(n, n + periods_ahead, dtype=float)
    forecasts = _predict(future_X)

    # Generate future period labels
    last_year = int(df["year"].iloc[-1])
    last_month_num = int(df["month_num"].iloc[-1])
    from src.data_loader import MONTH_ORDER
    future_periods = []
    for i in range(periods_ahead):
        month_num = (last_month_num + i) % 12
        year_offset = (last_month_num + i) // 12
        future_periods.append(
            f"{MONTH_ORDER[month_num]}-{last_year + year_offset}"
        )

    return pd.DataFrame({
        "period": future_periods,
        "forecast": np.maximum(forecasts, 0),
        "lower": np.maximum(forecasts - residual_std, 0),
        "upper": forecasts + residual_std,
        "model": "Linear trend baseline (illustrative only)",
    })
