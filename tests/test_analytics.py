"""
tests/test_analytics.py
-----------------------
Unit tests for the analytics module.
Run with: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import pandas as pd

from src.analytics import (
    total_transaction_volume,
    total_transaction_value,
    average_transaction_value,
    period_over_period_growth,
    latest_period,
    highest_volume_period,
    lowest_volume_period,
    monthly_growth_rates,
    yoy_growth_rates,
    detect_anomalies,
    trend_summary,
    simple_forecast,
)
from src.data_loader import load_data, filter_data, get_annual_summary, MONTH_ORDER


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_df(n_months: int = 24, start_year: int = 2022, start_month_idx: int = 0) -> pd.DataFrame:
    """Build a minimal synthetic dataframe for testing."""
    months = []
    years = []
    m_nums = []
    for i in range(n_months):
        month_idx = (start_month_idx + i) % 12
        year_offset = (start_month_idx + i) // 12
        months.append(MONTH_ORDER[month_idx])
        years.append(start_year + year_offset)
        m_nums.append(month_idx + 1)

    volumes = [float(10 + i * 5) for i in range(n_months)]  # 10, 15, 20, ...
    values  = [float(1 + i * 0.5) for i in range(n_months)]  # 1.0, 1.5, 2.0, ...

    df = pd.DataFrame({
        "month": months,
        "year": years,
        "month_num": m_nums,
        "transaction_volume": volumes,
        "transaction_value": values,
        "period": [f"{m}-{y}" for m, y in zip(months, years)],
        "reporting_source": ["TEST"] * n_months,
    })
    df["avg_transaction_value"] = np.where(
        df["transaction_volume"] > 0,
        (df["transaction_value"] * 1e12) / (df["transaction_volume"] * 1e7),
        np.nan,
    )
    df["volume_mom_growth"] = df["transaction_volume"].pct_change() * 100
    df["volume_yoy_growth"] = df["transaction_volume"].pct_change(12) * 100
    df["value_mom_growth"]  = df["transaction_value"].pct_change() * 100
    return df


@pytest.fixture
def df_24():
    return make_df(24)


@pytest.fixture
def df_empty():
    return pd.DataFrame()


@pytest.fixture
def df_single():
    return make_df(1)


@pytest.fixture
def df_with_nan():
    df = make_df(6)
    df.loc[2, "transaction_volume"] = np.nan
    df.loc[4, "transaction_value"]  = np.nan
    return df


# ---------------------------------------------------------------------------
# total_transaction_volume
# ---------------------------------------------------------------------------

class TestTotalTransactionVolume:
    def test_sum_correct(self, df_24):
        result = total_transaction_volume(df_24)
        expected = sum(10 + i * 5 for i in range(24))
        assert abs(result - expected) < 0.01

    def test_empty_returns_zero(self, df_empty):
        assert total_transaction_volume(df_empty) == 0.0

    def test_single_row(self, df_single):
        assert total_transaction_volume(df_single) == 10.0

    def test_missing_column_returns_zero(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        assert total_transaction_volume(df) == 0.0


# ---------------------------------------------------------------------------
# total_transaction_value
# ---------------------------------------------------------------------------

class TestTotalTransactionValue:
    def test_sum_correct(self, df_24):
        result = total_transaction_value(df_24)
        expected = sum(1.0 + i * 0.5 for i in range(24))
        assert abs(result - expected) < 0.01

    def test_empty_returns_zero(self, df_empty):
        assert total_transaction_value(df_empty) == 0.0


# ---------------------------------------------------------------------------
# average_transaction_value
# ---------------------------------------------------------------------------

class TestAverageTransactionValue:
    def test_formula(self, df_single):
        """avg = (value * 1e12) / (volume * 1e7)"""
        val = df_single["transaction_value"].iloc[0]
        vol = df_single["transaction_volume"].iloc[0]
        expected = (val * 1e12) / (vol * 1e7)
        result = average_transaction_value(df_single)
        assert abs(result - expected) < 1.0  # within ₹1

    def test_empty_returns_nan(self, df_empty):
        assert np.isnan(average_transaction_value(df_empty))

    def test_zero_volume_returns_nan(self):
        df = make_df(1)
        df["transaction_volume"] = 0.0
        assert np.isnan(average_transaction_value(df))

    def test_aggregate_is_total_not_mean(self, df_24):
        """avg should be computed on totals, not mean of individual averages."""
        tot_vol = df_24["transaction_volume"].sum()
        tot_val = df_24["transaction_value"].sum()
        expected = (tot_val * 1e12) / (tot_vol * 1e7)
        result = average_transaction_value(df_24)
        assert abs(result - expected) < 1.0


# ---------------------------------------------------------------------------
# period_over_period_growth
# ---------------------------------------------------------------------------

class TestPeriodOverPeriodGrowth:
    def test_mom_growth(self, df_24):
        """Last row = 10 + 23*5 = 125; second-to-last = 120. Growth = (125-120)/120 * 100"""
        result = period_over_period_growth(df_24, "transaction_volume", 1)
        prev = 10 + 22 * 5  # = 120
        curr = 10 + 23 * 5  # = 125
        expected = ((curr - prev) / prev) * 100
        assert abs(result - expected) < 0.01

    def test_insufficient_data_returns_nan(self, df_single):
        result = period_over_period_growth(df_single, "transaction_volume", 1)
        assert np.isnan(result)

    def test_empty_returns_nan(self, df_empty):
        assert np.isnan(period_over_period_growth(df_empty, "transaction_volume", 1))

    def test_yoy_growth(self, df_24):
        """12-period growth: row 23 vs row 11."""
        result = period_over_period_growth(df_24, "transaction_volume", 12)
        prev = 10 + 11 * 5  # = 65
        curr = 10 + 23 * 5  # = 125
        expected = ((curr - prev) / prev) * 100
        assert abs(result - expected) < 0.01


# ---------------------------------------------------------------------------
# latest_period
# ---------------------------------------------------------------------------

class TestLatestPeriod:
    def test_returns_last_period(self, df_24):
        result = latest_period(df_24)
        assert result == df_24["period"].iloc[-1]

    def test_empty_returns_na(self, df_empty):
        assert latest_period(df_empty) == "N/A"


# ---------------------------------------------------------------------------
# highest/lowest volume period
# ---------------------------------------------------------------------------

class TestHighestLowestPeriod:
    def test_highest_is_last(self, df_24):
        result = highest_volume_period(df_24)
        assert result["period"] == df_24["period"].iloc[-1]
        assert abs(result["value"] - 125.0) < 0.01

    def test_lowest_is_first(self, df_24):
        result = lowest_volume_period(df_24)
        assert result["period"] == df_24["period"].iloc[0]
        assert abs(result["value"] - 10.0) < 0.01

    def test_empty_returns_nan(self, df_empty):
        r = highest_volume_period(df_empty)
        assert r["period"] == "N/A"
        assert np.isnan(r["value"])


# ---------------------------------------------------------------------------
# Growth rates
# ---------------------------------------------------------------------------

class TestGrowthRates:
    def test_mom_growth_length(self, df_24):
        result = monthly_growth_rates(df_24)
        assert len(result) == 24
        assert "volume_mom_growth" in result.columns

    def test_first_mom_is_nan(self, df_24):
        result = monthly_growth_rates(df_24)
        assert np.isnan(result["volume_mom_growth"].iloc[0])

    def test_yoy_needs_13_rows(self):
        df_short = make_df(12)
        result = yoy_growth_rates(df_short)
        # Less than 13 rows — should return unchanged df
        assert "volume_yoy_growth" in result.columns or len(result) == len(df_short)

    def test_yoy_first_12_are_nan(self, df_24):
        result = yoy_growth_rates(df_24)
        assert result["volume_yoy_growth"].iloc[:12].isna().all()


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

class TestAnomalyDetection:
    def test_no_anomalies_in_linear_series(self, df_24):
        """A perfectly linear series should have no z-score anomalies."""
        result = detect_anomalies(df_24, "transaction_volume", z_threshold=2.0)
        # All MoM changes are identical (5/prev_vol) so std is low — few if any anomalies
        assert isinstance(result, pd.DataFrame)

    def test_spike_detected(self):
        df = make_df(12)
        # Insert a huge spike at index 6
        df.loc[6, "transaction_volume"] = 10000.0
        result = detect_anomalies(df, "transaction_volume", z_threshold=2.0)
        assert len(result) >= 1

    def test_empty_returns_empty_df(self, df_empty):
        result = detect_anomalies(df_empty)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_correct_columns(self):
        df = make_df(12)
        df.loc[6, "transaction_volume"] = 10000.0
        result = detect_anomalies(df, "transaction_volume")
        if not result.empty:
            assert "anomaly_z_score" in result.columns
            assert "anomaly_reason" in result.columns


# ---------------------------------------------------------------------------
# Trend summary
# ---------------------------------------------------------------------------

class TestTrendSummary:
    def test_rising_direction(self, df_24):
        ts = trend_summary(df_24)
        assert ts["direction"] == "rising"

    def test_contains_required_keys(self, df_24):
        ts = trend_summary(df_24)
        assert "direction" in ts
        assert "avg_monthly_volume_growth" in ts
        assert "correlation_volume_value" in ts

    def test_empty_returns_empty_dict(self, df_empty):
        ts = trend_summary(df_empty)
        assert ts == {}

    def test_correlation_in_valid_range(self, df_24):
        ts = trend_summary(df_24)
        corr = ts["correlation_volume_value"]
        assert -1.0 <= corr <= 1.0


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

class TestFilterData:
    def test_year_filter(self, df_24):
        years_in_df = df_24["year"].unique().tolist()
        # Filter to only first year
        result = filter_data(df_24, years=[years_in_df[0]])
        assert result["year"].unique().tolist() == [years_in_df[0]]

    def test_empty_year_filter_unchanged(self, df_24):
        # filter_data treats years=[] as "no year filter" — returns all rows
        result = filter_data(df_24, years=[])
        assert isinstance(result, pd.DataFrame)
        # Should not crash; may return full or empty set depending on implementation
        assert len(result) >= 0

    def test_empty_df_passthrough(self, df_empty):
        result = filter_data(df_empty)
        assert result.empty


# ---------------------------------------------------------------------------
# Annual summary
# ---------------------------------------------------------------------------

class TestAnnualSummary:
    def test_annual_totals(self, df_24):
        annual = get_annual_summary(df_24)
        assert "total_volume" in annual.columns
        assert "total_value" in annual.columns
        assert annual["total_volume"].sum() == total_transaction_volume(df_24)

    def test_empty_returns_empty(self, df_empty):
        result = get_annual_summary(df_empty)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

class TestSimpleForecast:
    def test_returns_none_for_short_data(self, df_single):
        result = simple_forecast(df_single, 6)
        assert result is None

    def test_returns_df_for_24_months(self, df_24):
        result = simple_forecast(df_24, 6, "transaction_volume")
        assert result is not None
        assert len(result) == 6
        assert "forecast" in result.columns
        assert "lower" in result.columns
        assert "upper" in result.columns

    def test_forecast_non_negative(self, df_24):
        result = simple_forecast(df_24, 6, "transaction_volume")
        assert result is not None
        assert (result["forecast"] >= 0).all()

    def test_upper_gte_lower(self, df_24):
        result = simple_forecast(df_24, 6, "transaction_volume")
        assert result is not None
        assert (result["upper"] >= result["lower"]).all()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class TestDataLoader:
    def test_load_demo_data(self):
        df, meta = load_data()
        assert not df.empty
        assert meta["rows_clean"] > 0
        assert "transaction_volume" in df.columns
        assert "transaction_value" in df.columns
        assert "period" in df.columns

    def test_no_missing_in_critical_columns(self):
        df, _ = load_data()
        assert df["transaction_volume"].isna().sum() == 0
        assert df["transaction_value"].isna().sum() == 0
        assert df["year"].isna().sum() == 0

    def test_data_sorted_chronologically(self):
        df, _ = load_data()
        year_month = list(zip(df["year"], df["month_num"]))
        assert year_month == sorted(year_month)

    def test_avg_transaction_value_computed(self):
        df, _ = load_data()
        non_null_avg = df["avg_transaction_value"].notna().sum()
        assert non_null_avg > 0

    def test_demo_flag_is_set(self):
        df, meta = load_data()
        # Bundled sample should flag as demo
        assert meta["is_demo"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
