"""
data_loader.py
--------------
Handles loading, validation, and preprocessing of UPI transaction data.
All data originates from user-supplied CSVs or the bundled demo file.
No statistics are fabricated or assumed.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MONTH_ORDER = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
MONTH_NUM = {m: i + 1 for i, m in enumerate(MONTH_ORDER)}

REQUIRED_COLUMNS = {"month", "year", "transaction_volume", "transaction_value"}
OPTIONAL_COLUMNS = {"reporting_source", "state", "payment_app"}

DATA_DIR = Path(__file__).parent.parent / "data"
SAMPLE_PATH = DATA_DIR / "upi_transactions_sample.csv"
OFFICIAL_PATH = DATA_DIR / "upi_transactions_official.csv"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_data(uploaded_file=None) -> tuple[pd.DataFrame, dict]:
    """
    Load UPI transaction data from one of three sources:
    1. An uploaded file object (Streamlit UploadedFile).
    2. The official CSV at data/upi_transactions_official.csv.
    3. The bundled demo CSV at data/upi_transactions_sample.csv.

    Returns
    -------
    df : pd.DataFrame  — cleaned, validated dataframe
    meta : dict        — source info, warnings, quality summary
    """
    meta = {
        "source": None,
        "is_demo": False,
        "warnings": [],
        "rows_raw": 0,
        "rows_clean": 0,
        "date_range": None,
    }

    # --- Determine file source ---
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            meta["source"] = f"Uploaded file: {uploaded_file.name}"
        except Exception as exc:
            meta["warnings"].append(f"Could not read uploaded file: {exc}")
            df_raw = pd.DataFrame()
    elif OFFICIAL_PATH.exists():
        df_raw = pd.read_csv(OFFICIAL_PATH)
        meta["source"] = "data/upi_transactions_official.csv"
    else:
        df_raw = pd.read_csv(SAMPLE_PATH)
        meta["source"] = "data/upi_transactions_sample.csv (DEMO DATA)"
        meta["is_demo"] = True

    meta["rows_raw"] = len(df_raw)

    if df_raw.empty:
        meta["warnings"].append("Loaded file is empty.")
        return df_raw, meta

    # --- Validate and clean ---
    df, warnings = _validate_and_clean(df_raw)
    meta["warnings"].extend(warnings)
    meta["rows_clean"] = len(df)

    if not df.empty:
        meta["date_range"] = _date_range_str(df)

    return df, meta


def _validate_and_clean(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Validate required columns, coerce numeric types, drop unrecoverable rows,
    flag duplicates, and sort chronologically.
    """
    warnings = []
    df = df_raw.copy()

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Check required columns
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        warnings.append(
            f"Missing required columns: {missing_cols}. "
            "Expected: month, year, transaction_volume, transaction_value."
        )
        return pd.DataFrame(), warnings

    # Coerce numerics
    for col in ("transaction_volume", "transaction_value", "year"):
        original_nulls = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        new_nulls = df[col].isna().sum()
        if new_nulls > original_nulls:
            warnings.append(
                f"Column '{col}': {new_nulls - original_nulls} non-numeric "
                "value(s) converted to NaN."
            )

    # Drop rows where critical numerics are NaN
    before = len(df)
    df = df.dropna(subset=["transaction_volume", "transaction_value", "year"])
    dropped = before - len(df)
    if dropped:
        warnings.append(f"Dropped {dropped} row(s) with missing numeric values.")

    # Validate year is reasonable
    df["year"] = df["year"].astype(int)
    invalid_years = df[(df["year"] < 2016) | (df["year"] > 2030)]
    if not invalid_years.empty:
        warnings.append(
            f"Found {len(invalid_years)} row(s) with year outside 2016–2030. "
            "These are retained but may be erroneous."
        )

    # Normalise month abbreviations
    df["month"] = df["month"].str.strip().str.capitalize()
    invalid_months = ~df["month"].isin(MONTH_ORDER)
    if invalid_months.any():
        bad = df.loc[invalid_months, "month"].unique().tolist()
        warnings.append(f"Unrecognised month value(s): {bad}. Rows retained.")

    # Add month number for sorting
    df["month_num"] = df["month"].map(MONTH_NUM).fillna(0).astype(int)

    # Build period label (e.g. "Apr-2023")
    df["period"] = df["month"] + "-" + df["year"].astype(str)

    # Check duplicates
    dup_mask = df.duplicated(subset=["month", "year"], keep="first")
    dup_count = dup_mask.sum()
    if dup_count:
        warnings.append(
            f"Found {dup_count} duplicate reporting period(s). "
            "Keeping first occurrence."
        )
        df = df[~dup_mask]

    # Sort chronologically
    df = df.sort_values(["year", "month_num"]).reset_index(drop=True)

    # Derived columns
    df["avg_transaction_value"] = np.where(
        df["transaction_volume"] > 0,
        (df["transaction_value"] * 1e5) / (df["transaction_volume"] * 1e7),
        np.nan,
    )
    # avg_transaction_value is in ₹
    # transaction_value in lakh crore → × 1e12 rupees
    # transaction_volume in crore → × 1e7 transactions
    df["avg_transaction_value"] = np.where(
        df["transaction_volume"] > 0,
        (df["transaction_value"] * 1e12) / (df["transaction_volume"] * 1e7),
        np.nan,
    )

    # Monthly growth rate (volume)
    df["volume_mom_growth"] = df["transaction_volume"].pct_change() * 100
    # YoY growth
    df["volume_yoy_growth"] = df["transaction_volume"].pct_change(12) * 100
    df["value_yoy_growth"] = df["transaction_value"].pct_change(12) * 100

    return df, warnings


def _date_range_str(df: pd.DataFrame) -> str:
    first = df.iloc[0]
    last = df.iloc[-1]
    return f"{first['month']} {int(first['year'])} – {last['month']} {int(last['year'])}"


def filter_data(
    df: pd.DataFrame,
    years: list[int] | None = None,
    start_period: str | None = None,
    end_period: str | None = None,
) -> pd.DataFrame:
    """
    Filter the dataframe by year list and/or period range.
    Periods are strings like 'Apr-2016'.
    """
    if df.empty:
        return df

    result = df.copy()

    if years:
        result = result[result["year"].isin(years)]

    if start_period and start_period in result["period"].values:
        idx_start = result[result["period"] == start_period].index[0]
        result = result.loc[idx_start:]

    if end_period and end_period in result["period"].values:
        idx_end = result[result["period"] == end_period].index[-1]
        result = result.loc[:idx_end]

    return result.reset_index(drop=True)


def get_annual_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly data into annual totals and averages.
    """
    if df.empty:
        return df

    annual = (
        df.groupby("year")
        .agg(
            total_volume=("transaction_volume", "sum"),
            total_value=("transaction_value", "sum"),
            avg_monthly_volume=("transaction_volume", "mean"),
            avg_monthly_value=("transaction_value", "mean"),
            months_reported=("month", "count"),
        )
        .reset_index()
    )
    annual["avg_transaction_value"] = np.where(
        annual["total_volume"] > 0,
        (annual["total_value"] * 1e12) / (annual["total_volume"] * 1e7),
        np.nan,
    )
    annual["yoy_volume_growth"] = annual["total_volume"].pct_change() * 100
    annual["yoy_value_growth"] = annual["total_value"].pct_change() * 100
    return annual
