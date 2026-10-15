# Data Directory

## Primary Dataset

**Source:** National Payments Corporation of India (NPCI)
- UPI Product Statistics: https://www.npci.org.in/product/upi/product-statistics
- UPI Ecosystem Statistics: https://www.npci.org.in/what-we-do/upi/upi-ecosystem-statistics

## Files

- `upi_transactions_sample.csv` — Illustrative **DEMO DATA** file used when no official dataset is present.
  This file contains synthetic data for testing and demonstration only.
  **Do not mix with official NPCI figures.**

## How to use official data

1. Visit https://www.npci.org.in/product/upi/product-statistics
2. Download the monthly UPI transaction statistics table.
3. Save as `upi_transactions_official.csv` in this directory.
4. Ensure columns match: `month`, `year`, `transaction_volume`, `transaction_value`
   - `transaction_volume`: Number of UPI transactions (in crore)
   - `transaction_value`: Total value of UPI transactions (in ₹ lakh crore)

## Column Definitions

| Column              | Unit         | Description                              |
|---------------------|--------------|------------------------------------------|
| month               | String (MMM) | Month abbreviation e.g. Jan, Feb         |
| year                | Integer      | Calendar year e.g. 2023                  |
| transaction_volume  | Crore        | Number of UPI transactions               |
| transaction_value   | ₹ Lakh Crore | Aggregate transaction value              |
| reporting_source    | String       | Source attribution (e.g. NPCI)           |

## Data Quality

The application will display warnings for:
- Missing values
- Duplicate reporting periods
- Non-numeric entries in volume/value columns
- Gaps in monthly continuity
