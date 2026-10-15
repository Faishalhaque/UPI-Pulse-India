"""
insights.py
-----------
Rule-based natural-language insights engine for UPI Pulse India.
Generates human-readable summaries grounded in the currently loaded data.
All numerical statements are derived from the dataframe passed in — never
from hardcoded assumptions.

Optional: if OPENAI_API_KEY (or WATSONX_API_KEY) is present in environment,
the local summary is optionally enhanced via the configured LLM.
All AI-generated content is labelled "AI-assisted insight" — it is NOT
an official statement from NPCI or RBI.
"""

import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.analytics import (
    total_transaction_volume,
    total_transaction_value,
    average_transaction_value,
    period_over_period_growth,
    latest_period,
    trend_summary,
    detect_anomalies,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Local rule-based insights engine
# ---------------------------------------------------------------------------

def generate_local_insights(df: pd.DataFrame, annual_df: pd.DataFrame = None) -> dict:
    """
    Generate a structured insights dict from the filtered dataframe.
    Returns sections that the UI can render independently.
    """
    if df.empty:
        return {"error": "No data available for the selected filters."}

    tot_vol = total_transaction_volume(df)
    tot_val = total_transaction_value(df)
    avg_val = average_transaction_value(df)
    latest = latest_period(df)
    mom_vol_growth = period_over_period_growth(df, "transaction_volume", 1)
    yoy_vol_growth = period_over_period_growth(df, "transaction_volume", 12)
    ts = trend_summary(df)
    anomalies = detect_anomalies(df, "transaction_volume", z_threshold=2.0)

    # --- Section: KPI Summary ---
    kpi_lines = [
        f"The selected period covers **{len(df)} months** from "
        f"**{df['period'].iloc[0]}** to **{df['period'].iloc[-1]}**.",
        f"Total UPI transactions: **{tot_vol:,.2f} crore**.",
        f"Total transaction value: **₹{tot_val:,.2f} lakh crore**.",
        f"Average per-transaction value: **₹{avg_val:,.0f}**." if not np.isnan(avg_val) else "",
        f"Latest reporting period: **{latest}**.",
    ]

    # --- Section: Trend Narrative ---
    trend_lines = []
    if ts:
        direction = ts.get("direction", "mixed")
        avg_growth = ts.get("avg_monthly_volume_growth", np.nan)

        if direction == "rising":
            trend_lines.append(
                f"Transaction volume shows a broadly **rising trend** over the selected period, "
                f"with an average month-over-month growth of **{avg_growth:.1f}%**."
            )
        elif direction == "falling":
            trend_lines.append(
                f"Transaction volume shows a broadly **falling trend** over the selected period, "
                f"with an average month-over-month change of **{avg_growth:.1f}%**."
            )
        else:
            trend_lines.append(
                f"Transaction volume shows a **mixed trend** over the selected period, "
                f"with an average month-over-month change of **{avg_growth:.1f}%**."
            )

        best = ts.get("best_month", "N/A")
        best_vol = ts.get("best_month_volume", np.nan)
        worst = ts.get("worst_month", "N/A")
        worst_vol = ts.get("worst_month_volume", np.nan)

        if not np.isnan(best_vol):
            trend_lines.append(
                f"The **highest transaction month** in this period was **{best}** "
                f"({best_vol:,.2f} crore transactions)."
            )
        if not np.isnan(worst_vol):
            trend_lines.append(
                f"The **lowest transaction month** was **{worst}** "
                f"({worst_vol:,.2f} crore transactions)."
            )

        corr = ts.get("correlation_volume_value", np.nan)
        if not np.isnan(corr):
            if corr > 0.95:
                trend_lines.append(
                    f"Transaction **volume and value are highly correlated** "
                    f"(r = {corr:.3f}), meaning they grew largely in tandem."
                )
            elif corr > 0.80:
                trend_lines.append(
                    f"Transaction volume and value show a **strong correlation** "
                    f"(r = {corr:.3f})."
                )
            else:
                trend_lines.append(
                    f"Transaction volume and value show a **moderate correlation** "
                    f"(r = {corr:.3f}), suggesting some divergence in patterns."
                )

    # --- Section: Growth ---
    growth_lines = []
    if not np.isnan(mom_vol_growth):
        sign = "↑" if mom_vol_growth >= 0 else "↓"
        growth_lines.append(
            f"Month-over-month volume change (last two periods): "
            f"**{sign} {abs(mom_vol_growth):.1f}%**."
        )
    if not np.isnan(yoy_vol_growth):
        sign = "↑" if yoy_vol_growth >= 0 else "↓"
        growth_lines.append(
            f"Year-over-year volume change (last vs same month prior year): "
            f"**{sign} {abs(yoy_vol_growth):.1f}%**."
        )

    # --- Section: Anomalies ---
    anomaly_lines = []
    if not anomalies.empty:
        anomaly_lines.append(
            f"**{len(anomalies)} period(s)** showed unusually large month-over-month changes "
            f"(exceeding 2 standard deviations from the mean change):"
        )
        for _, row in anomalies.iterrows():
            anomaly_lines.append(
                f"- **{row['period']}**: {row['transaction_volume']:,.2f} crore transactions "
                f"(z-score: {row['anomaly_z_score']:.1f}σ)"
            )
    else:
        anomaly_lines.append(
            "No statistically unusual periods detected in transaction volume "
            "(within 2 standard deviations of mean MoM change)."
        )

    # --- Section: Further Questions ---
    question_lines = [
        "**Suggested questions for further analysis:**",
        "1. Which months show the strongest seasonal upticks and why?",
        "2. How did demonetisation and COVID-19 periods affect UPI adoption?",
        "3. Did average transaction value increase or decrease as user base expanded?",
        "4. Which states or payment apps drove the most growth?",
        "5. Is UPI volume growth decelerating or accelerating in recent quarters?",
    ]

    # --- Executive Summary ---
    exec_summary = _build_executive_summary(df, tot_vol, tot_val, avg_val, ts, mom_vol_growth)

    return {
        "kpi_summary": "\n\n".join([l for l in kpi_lines if l]),
        "trend_narrative": "\n\n".join(trend_lines),
        "growth_analysis": "\n\n".join(growth_lines) if growth_lines else "Insufficient data for growth calculation.",
        "anomaly_detection": "\n\n".join(anomaly_lines),
        "further_questions": "\n\n".join(question_lines),
        "executive_summary": exec_summary,
        "ai_enhanced": False,
    }


def _build_executive_summary(df, tot_vol, tot_val, avg_val, ts, mom_growth) -> str:
    direction = ts.get("direction", "mixed") if ts else "mixed"
    avg_growth = ts.get("avg_monthly_volume_growth", np.nan) if ts else np.nan
    best = ts.get("best_month", "N/A") if ts else "N/A"

    avg_val_str = f"₹{avg_val:,.0f}" if not np.isnan(avg_val) else "N/A"
    avg_growth_str = f"{avg_growth:.1f}%" if not np.isnan(avg_growth) else "N/A"
    mom_str = f"{mom_growth:+.1f}%" if not np.isnan(mom_growth) else "N/A"

    return f"""**Executive Summary — UPI Pulse India (AI-assisted insight)**

The selected dataset covers **{len(df)} months** of UPI transaction data from \
**{df['period'].iloc[0]}** to **{df['period'].iloc[-1]}**.

Over this period, India processed a cumulative **{tot_vol:,.2f} crore UPI transactions** \
valued at **₹{tot_val:,.2f} lakh crore**. The average transaction value was **{avg_val_str}**.

The overall trend in transaction volume was **{direction}**, with an average monthly \
growth rate of **{avg_growth_str}**. The peak transaction period was **{best}**.

The most recent month-over-month change in volume was **{mom_str}**, indicating \
{"continued momentum" if not np.isnan(mom_growth) and mom_growth > 0 else "a slight moderation"} \
in UPI adoption.

*This summary is generated algorithmically from the loaded dataset. \
It is an educational analytical observation and does not constitute an official \
statement from NPCI, RBI, or any financial institution.*"""


# ---------------------------------------------------------------------------
# Optional LLM Enhancement
# ---------------------------------------------------------------------------

def enhance_with_llm(local_insights: dict, df: pd.DataFrame) -> dict:
    """
    Attempt to enhance the local insights using an LLM API if configured.
    Falls back silently to local insights if no API key is available or
    if the API call fails.

    Supports OpenAI API (OPENAI_API_KEY) as an example integration.
    Never hardcodes keys.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return local_insights  # No key → return local insights unchanged

    try:
        import requests

        context = (
            f"UPI data summary: {len(df)} months, "
            f"total volume {total_transaction_volume(df):.2f} crore, "
            f"total value ₹{total_transaction_value(df):.2f} lakh crore. "
            f"Trend: {local_insights.get('trend_narrative', '')[:300]}"
        )

        prompt = (
            "You are a data analyst summarising Indian UPI payment trends for an "
            "educational internship report. Based on the following statistics, "
            "write a concise 3-sentence professional summary. "
            "Label it as AI-assisted insight. "
            "Do not invent numbers beyond what is provided.\n\n"
            f"Context: {context}"
        )

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0.3,
            },
            timeout=15,
        )
        response.raise_for_status()
        llm_text = response.json()["choices"][0]["message"]["content"]
        enhanced = local_insights.copy()
        enhanced["executive_summary"] = (
            f"**AI-Enhanced Summary (GPT-3.5)**\n\n{llm_text}\n\n"
            f"---\n*Rule-based summary also available above.*"
        )
        enhanced["ai_enhanced"] = True
        return enhanced

    except Exception:
        # API call failed — silently fall back, never crash the dashboard
        return local_insights
