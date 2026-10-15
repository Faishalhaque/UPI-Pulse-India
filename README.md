# 💳 UPI Pulse India

**Digital Payments and UPI Transaction Trend Analysis in India**

> AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026  
> Author: Mo Faishal Haque

---

## Overview

UPI Pulse India is an interactive data analytics dashboard that helps users explore India's Unified Payments Interface (UPI) transaction trends, including transaction volumes, values, growth rates, anomalies, and AI-assisted insights.

The application is built with Python and Streamlit, processes NPCI monthly statistics, and provides a polished fintech-themed dashboard with 6 interactive chart types, rule-based AI insights, anomaly detection, and optional linear trend forecasting.

---

## Features

| Feature | Description |
|---------|-------------|
| 📊 Executive Overview | KPI cards for total volume, value, avg transaction value, MoM growth |
| 📈 Transaction Trends | 6 interactive Plotly charts across tabbed views |
| 💡 Payment Insights | Rule-based observations, peak/trough analysis, anomaly detection |
| 🔍 Data Explorer | Searchable, filterable data table with CSV download |
| 🤖 AI Insights | Rule-based summaries + optional LLM enhancement + linear forecast |
| ℹ️ About Project | Full documentation, methodology, and reproduction instructions |
| 🎛️ Sidebar Filters | Year filter, date range selector, CSV upload |
| ⚠️ Data Quality | Missing value, duplicate, and unit validation with visible warnings |

---

## Dataset Source

**Primary:** National Payments Corporation of India (NPCI)
- UPI Product Statistics: https://www.npci.org.in/product/upi/product-statistics
- UPI Ecosystem Statistics: https://www.npci.org.in/what-we-do/upi/upi-ecosystem-statistics

> ⚠️ The bundled `data/upi_transactions_sample.csv` contains **illustrative synthetic figures** for demonstration only. Replace it with the official NPCI CSV for certified analysis.

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip
- Git

### Steps (Windows)

```powershell
# Clone the repository
git clone https://github.com/<your-username>/upi-pulse-india.git
cd upi-pulse-india

# Create a virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run the Dashboard

```powershell
# From the upi-pulse-india/ directory
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

**To use official NPCI data:**
1. Download the monthly UPI statistics CSV from https://www.npci.org.in/product/upi/product-statistics
2. Rename/save as `data/upi_transactions_official.csv`
3. Ensure columns: `month`, `year`, `transaction_volume`, `transaction_value`
4. Reload the dashboard — it will automatically detect and use the official file

**Optional: AI Enhancement**
```powershell
# Create .env file and add your OpenAI API key
echo OPENAI_API_KEY=sk-your-key-here > .env
```
The dashboard works fully without an API key using the built-in rule-based engine.

---

## How to Run the Jupyter Notebook

```powershell
# Install Jupyter if not already available
pip install jupyter

# Launch notebook
cd upi-pulse-india
jupyter notebook MoFaishalHaque_UPIPulseIndia.ipynb
```

Run all cells in order. The notebook will use the demo dataset unless `data/upi_transactions_official.csv` is present.

---

## How to Run Tests

```powershell
# From the upi-pulse-india/ directory
python -m pytest tests/ -v
```

---

## Repository Structure

```
upi-pulse-india/
├── app.py                              ← Main Streamlit dashboard
├── requirements.txt                    ← Python dependencies
├── README.md                           ← This file
├── MoFaishalHaque_UPIPulseIndia.ipynb  ← Jupyter Notebook (analytics workflow)
├── MoFaishalHaque_ProjectReport.docx   ← Project report (Word)
├── data/
│   ├── README.md                       ← Data documentation
│   └── upi_transactions_sample.csv     ← DEMO DATA (illustrative only)
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  ← Data loading, validation, cleaning
│   ├── analytics.py                    ← KPI calculations, anomaly detection, forecast
│   └── insights.py                     ← Rule-based + optional LLM insights engine
├── assets/                             ← Chart exports (created by notebook)
└── tests/
    └── test_analytics.py               ← Unit tests for analytics functions
```

---

## Key Analytical Questions

1. How has UPI transaction volume grown since April 2016?
2. What is the trend in average transaction value over time?
3. Which months showed the strongest year-over-year growth?
4. Did COVID-19 lockdowns (April–May 2020) visibly affect UPI volumes?
5. Are volume and value growth correlated, or do they diverge?
6. Which reporting periods showed statistically unusual changes?

---

## Limitations

- **Demo data is synthetic.** The bundled CSV contains illustrative figures. For accurate analysis, use official NPCI data.
- **No live API.** NPCI does not expose a stable public API; manual CSV download is required.
- **No state/app breakdown.** Geographic and payment-app granularity requires additional NPCI ecosystem data.
- **Naive forecasting.** The linear model does not account for seasonality or economic shocks.
- **Educational project.** Not a banking, investment, or payment service.

---

## Attribution

- **Data source:** National Payments Corporation of India (NPCI) — https://www.npci.org.in
- **Project context:** AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026
- **Author:** Mo Faishal Haque
- **Technologies:** Python, Streamlit, Pandas, NumPy, Plotly, Scikit-learn

---

*This project is for educational purposes. All demo data is synthetic and must not be cited as official statistics.*
