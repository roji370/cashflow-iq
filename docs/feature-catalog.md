# Cashflow IQ — Feature Catalog

All features computed by the behavioral feature engineering pipeline. Each feature
is stored in long-format in the `features` table (one row per customer × feature × date).

## Anomaly Detector Features

| Feature Name | Description | Computation Logic | Owner | Version |
|---|---|---|---|---|
| `subscription_cleansing_flag` | Boolean: customer cancelled 3+ recurring discretionary subscriptions within a 60-day window while income stayed stable | Identifies recurring debit series (same `merchant_category`, `recurring_group_id`, ≥3 occurrences), checks if 3+ stopped within 60 days of each other, and verifies income CV < 30% over the period | Engineer 2 | 1.0.0 |
| `subscription_cleansing_score` | Continuous (0–1): proportion of recurring series that stopped, capped at 1.0 | `len(stopped_series) / len(total_recurring_series)` | Engineer 2 | 1.0.0 |
| `liquidity_pooling_flag` | Boolean: FD/MF maturity credit not reinvested within 45 days | Finds large investment credits (> 5× median transaction), checks for investment debits within 45 days after each maturity event | Engineer 2 | 1.0.0 |
| `liquidity_pooling_score` | Continuous (0–1): normalized gap from maturity to data end (max 90 days → 1.0) | `min(1.0, days_since_maturity / 90)` | Engineer 2 | 1.0.0 |
| `bill_shift_flag` | Boolean: recurring bill payment day drifting toward month-end | For each recurring bill series (utility/rent), compares mean payment day of first half vs. second half of history; flags if drift > 3 days and IsolationForest marks later payments as anomalous | Engineer 2 | 1.0.0 |
| `bill_shift_score` | Continuous (0–1): IsolationForest anomaly score for drifting payments, normalized | `max(0, min(1, -iso_decision_function + 0.5))` | Engineer 2 | 1.0.0 |

## Income Stability Features

| Feature Name | Description | Computation Logic | Owner | Version |
|---|---|---|---|---|
| `salary_variance_12m` | Coefficient of variation of monthly salary credits over trailing 12 months | Identify salary credits (merchant_category="salary"), aggregate monthly, compute `std / mean` | Engineer 2 | 1.0.0 |
| `income_growth_pct` | Linear trend slope of monthly income as a percentage | Fit OLS on monthly income totals, slope as % of mean income | Engineer 2 | 1.0.0 |
| `num_income_sources` | Count of distinct credit categories | `nunique(category)` across all credits | Engineer 2 | 1.0.0 |
| `income_source_tenure_months` | Months since first salary credit | `(max_date - min_salary_credit_date).days / 30` | Engineer 2 | 1.0.0 |
| `salary_amount_median` | Median monthly salary amount (not mean, to avoid bonus distortion) | Median of monthly salary credit totals | Engineer 2 | 1.0.0 |

## Cash Flow Features

| Feature Name | Description | Computation Logic | Owner | Version |
|---|---|---|---|---|
| `avg_monthly_income` | Trailing 6-month median of monthly credit totals | Median of last 6 months' total credits | Engineer 2 | 1.0.0 |
| `avg_monthly_outflow` | Trailing 6-month median of monthly debit totals | Median of last 6 months' total debits | Engineer 2 | 1.0.0 |
| `savings_ratio_6m` | Trailing 6-month median savings ratio | Median of `(monthly_income - monthly_outflow) / monthly_income` | Engineer 2 | 1.0.0 |
| `liquidity_buffer` | Minimum monthly net balance over trailing 3 months | `min(monthly_credits - monthly_debits)` over last 3 months | Engineer 2 | 1.0.0 |
| `expense_volatility` | CV of monthly debits over trailing 6 months | `std(monthly_debits) / mean(monthly_debits)` | Engineer 2 | 1.0.0 |

## Debt Behavior Features

| Feature Name | Description | Computation Logic | Owner | Version |
|---|---|---|---|---|
| `emi_punctuality_score` | Fraction of EMI-like payments made on time (≤ day 5 of month) | Identify recurring debit series matching EMI profile (monthly, consistent amount ±10%), check payment day ≤ 5 | Engineer 2 | 1.0.0 |
| `current_dti` | Estimated debt-to-income ratio | `sum(monthly_recurring_debits) / monthly_income` using trailing 3-month medians | Engineer 2 | 1.0.0 |
| `debt_trend_direction` | Slope of monthly total debt payments — positive = increasing debt | OLS slope on monthly EMI/recurring debit totals, normalized by mean | Engineer 2 | 1.0.0 |
| `num_active_emis` | Count of distinct active EMI-like recurring debit series | Count recurring debit groups with consistent monthly cadence and amount variance < 10% | Engineer 2 | 1.0.0 |

## Data Completeness

All features include a `data_completeness_score` (0–1) stored alongside each feature value:
- `1.0` = full 12-month history available
- `months_available / 12` for shorter histories
- Trend features (`income_growth_pct`, `debt_trend_direction`) return `None` and are not stored when < 3 months of data are available, to avoid falsely precise values
