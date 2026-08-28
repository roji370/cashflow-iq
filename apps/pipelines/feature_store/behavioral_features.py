"""Core behavioral feature engineering for Cashflow IQ.

Replaces the three hand-picked features in compute_minimal_features.py with
a full suite of income stability, cash flow, and debt behavior features.

Every function is pure: takes a transactions DataFrame + customer_id, returns
a flat dict of feature_name → value. No side effects, no database access.

Sparse data handling: if a customer has < 3 months of history, trend features
return None and data_completeness_score is set to months / 12.
"""

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _monthly_aggregates(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Aggregate credits and debits by month.

    Returns:
        Tuple of (monthly_credits, monthly_debits) as pd.Series indexed by
        month-start Timestamp.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    credits = df[df["type"] == "credit"].groupby("month")["amount"].sum()
    debits = df[df["type"] == "debit"].groupby("month")["amount"].sum()

    # Reindex to cover all months in the data range
    all_months = pd.date_range(
        df["month"].min(), df["month"].max(), freq="MS",
    )
    credits = credits.reindex(all_months, fill_value=0.0)
    debits = debits.reindex(all_months, fill_value=0.0)

    return credits, debits


def _data_completeness(num_months: int, ideal: int = 12) -> float:
    """Compute data completeness score."""
    return round(min(1.0, num_months / ideal), 4)


def _ols_slope(values: np.ndarray) -> Optional[float]:
    """Fit OLS and return slope. Returns None if too few points."""
    if len(values) < 3:
        return None
    x = np.arange(len(values), dtype=float)
    # Simple OLS: slope = cov(x, y) / var(x)
    x_mean = x.mean()
    y_mean = values.mean()
    cov_xy = ((x - x_mean) * (values - y_mean)).sum()
    var_x = ((x - x_mean) ** 2).sum()
    if var_x == 0:
        return 0.0
    return float(cov_xy / var_x)


# ---------------------------------------------------------------------------
# Income stability features
# ---------------------------------------------------------------------------

def compute_income_features(
    txns_df: pd.DataFrame,
    customer_id: str,
) -> dict[str, Any]:
    """Compute income stability features.

    Features:
        salary_variance_12m: CV of monthly salary credits
        income_growth_pct: linear trend slope as % of mean
        num_income_sources: count of distinct credit categories
        income_source_tenure_months: months since first salary credit
        salary_amount_median: median monthly salary amount

    Args:
        txns_df: Transactions for one customer.
        customer_id: The customer ID (for context, not used in computation).

    Returns:
        Dict of feature_name → value. Trend features are None if < 3 months.
    """
    df = txns_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    credits = df[df["type"] == "credit"]

    num_months = df["date"].dt.to_period("M").nunique()
    completeness = _data_completeness(num_months)

    result: dict[str, Any] = {"_completeness": completeness}

    if credits.empty:
        result.update({
            "salary_variance_12m": None,
            "income_growth_pct": None,
            "num_income_sources": 0,
            "income_source_tenure_months": 0,
            "salary_amount_median": 0.0,
        })
        return result

    # Monthly salary credits
    salary_credits = credits[credits["merchant_category"] == "salary"]
    if not salary_credits.empty:
        monthly_salary = salary_credits.groupby(
            salary_credits["date"].dt.to_period("M"),
        )["amount"].sum()
        salary_mean = monthly_salary.mean()
        salary_std = monthly_salary.std() if len(monthly_salary) > 1 else 0.0
        result["salary_variance_12m"] = round(
            salary_std / salary_mean if salary_mean > 0 else 0.0, 4,
        )
        result["salary_amount_median"] = round(float(monthly_salary.median()), 2)
    else:
        result["salary_variance_12m"] = None
        result["salary_amount_median"] = 0.0

    # Income growth — linear trend
    monthly_income, _ = _monthly_aggregates(df)
    if len(monthly_income) >= 3:
        slope = _ols_slope(monthly_income.values)
        mean_income = monthly_income.mean()
        result["income_growth_pct"] = round(
            (slope / mean_income * 100) if mean_income > 0 and slope is not None else 0.0, 4,
        )
    else:
        result["income_growth_pct"] = None

    # Number of distinct income sources
    result["num_income_sources"] = int(credits["category"].nunique())

    # Income source tenure
    if not salary_credits.empty:
        first_salary = salary_credits["date"].min()
        last_date = df["date"].max()
        result["income_source_tenure_months"] = round(
            (last_date - first_salary).days / 30.0, 1,
        )
    else:
        result["income_source_tenure_months"] = 0

    return result


# ---------------------------------------------------------------------------
# Cash flow features
# ---------------------------------------------------------------------------

def compute_cashflow_features(
    txns_df: pd.DataFrame,
    customer_id: str,
) -> dict[str, Any]:
    """Compute cash flow features using trailing window medians.

    Features:
        avg_monthly_income: trailing 6-month median of monthly credits
        avg_monthly_outflow: trailing 6-month median of monthly debits
        savings_ratio_6m: trailing 6-month median savings ratio
        liquidity_buffer: min monthly net balance over trailing 3 months
        expense_volatility: CV of monthly debits over trailing 6 months

    Args:
        txns_df: Transactions for one customer.
        customer_id: The customer ID.

    Returns:
        Dict of feature_name → value.
    """
    monthly_credits, monthly_debits = _monthly_aggregates(txns_df)
    num_months = len(monthly_credits)
    completeness = _data_completeness(num_months)

    result: dict[str, Any] = {"_completeness": completeness}

    if num_months == 0:
        result.update({
            "avg_monthly_income": 0.0,
            "avg_monthly_outflow": 0.0,
            "savings_ratio_6m": 0.0,
            "liquidity_buffer": 0.0,
            "expense_volatility": 0.0,
        })
        return result

    # Trailing 6-month windows (or all data if < 6 months)
    trail_6 = min(6, num_months)
    recent_credits = monthly_credits.iloc[-trail_6:]
    recent_debits = monthly_debits.iloc[-trail_6:]

    result["avg_monthly_income"] = round(float(recent_credits.median()), 2)
    result["avg_monthly_outflow"] = round(float(recent_debits.median()), 2)

    # Savings ratio per month, then median
    monthly_savings_ratio = []
    for c, d in zip(recent_credits, recent_debits):
        if c > 0:
            monthly_savings_ratio.append((c - d) / c)
        else:
            monthly_savings_ratio.append(0.0)
    result["savings_ratio_6m"] = round(
        float(np.median(monthly_savings_ratio)) if monthly_savings_ratio else 0.0, 4,
    )

    # Liquidity buffer — min monthly net over trailing 3 months
    trail_3 = min(3, num_months)
    recent_net = monthly_credits.iloc[-trail_3:] - monthly_debits.iloc[-trail_3:]
    result["liquidity_buffer"] = round(float(recent_net.min()), 2)

    # Expense volatility
    if recent_debits.mean() > 0 and len(recent_debits) > 1:
        result["expense_volatility"] = round(
            float(recent_debits.std() / recent_debits.mean()), 4,
        )
    else:
        result["expense_volatility"] = 0.0

    return result


# ---------------------------------------------------------------------------
# Debt behavior features
# ---------------------------------------------------------------------------

def compute_debt_features(
    txns_df: pd.DataFrame,
    customer_id: str,
) -> dict[str, Any]:
    """Compute debt behavior features.

    Features:
        emi_punctuality_score: fraction of EMI-like payments on time (≤ day 5)
        current_dti: estimated monthly debt / monthly income
        debt_trend_direction: slope of monthly debt payments
        num_active_emis: count of active EMI-like recurring debit series

    Args:
        txns_df: Transactions for one customer.
        customer_id: The customer ID.

    Returns:
        Dict of feature_name → value. Trend features None if < 3 months.
    """
    df = txns_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    num_months = df["date"].dt.to_period("M").nunique()
    completeness = _data_completeness(num_months)

    result: dict[str, Any] = {"_completeness": completeness}

    debits = df[df["type"] == "debit"]
    credits = df[df["type"] == "credit"]

    if debits.empty:
        result.update({
            "emi_punctuality_score": None,
            "current_dti": 0.0,
            "debt_trend_direction": None,
            "num_active_emis": 0,
        })
        return result

    # Identify EMI-like series: recurring debits with consistent amount (CV < 0.10)
    # and monthly cadence
    recurring = debits[
        (debits["recurring_group_id"].notna())
        & (debits["recurring_group_id"] != "")
    ]

    emi_series: list[pd.DataFrame] = []
    emi_group_ids: list[str] = []
    for gid, grp in recurring.groupby("recurring_group_id"):
        if len(grp) >= 3:
            amount_cv = grp["amount"].std() / grp["amount"].mean() if grp["amount"].mean() > 0 else 1.0
            if amount_cv < 0.10:  # consistent amount = likely EMI
                emi_series.append(grp)
                emi_group_ids.append(str(gid))

    result["num_active_emis"] = len(emi_series)

    # EMI punctuality — fraction paid on or before day 5
    if emi_series:
        total_payments = 0
        on_time = 0
        for grp in emi_series:
            for _, row in grp.iterrows():
                total_payments += 1
                if row["date"].day <= 5:
                    on_time += 1
        result["emi_punctuality_score"] = round(on_time / total_payments, 4) if total_payments > 0 else None
    else:
        result["emi_punctuality_score"] = None

    # DTI — monthly EMI-like recurring obligations / monthly income.
    # IMPORTANT: Use only identified EMI series (recurring debits with consistent
    # amounts, CV < 0.10, ≥ 3 occurrences) for the numerator. Using all debits
    # would include one-time payments (e.g., builder payments, large purchases)
    # and wildly inflate DTI — a home-buyer's one-time property payment is NOT
    # a recurring monthly debt obligation.
    monthly_credits, _ = _monthly_aggregates(df)
    trail_3 = min(3, num_months)
    recent_income = monthly_credits.iloc[-trail_3:]
    median_income = float(recent_income.median()) if not recent_income.empty else 0.0

    if emi_series:
        # Sum monthly EMI obligations from identified recurring series
        emi_txns = pd.concat(emi_series)
        emi_txns_copy = emi_txns.copy()
        emi_txns_copy["month"] = pd.to_datetime(emi_txns_copy["date"]).dt.to_period("M").dt.to_timestamp()
        monthly_emi = emi_txns_copy.groupby("month")["amount"].sum()
        # Reindex to cover the same trailing window as income
        all_months = monthly_credits.index
        monthly_emi = monthly_emi.reindex(all_months, fill_value=0.0)
        recent_emi = monthly_emi.iloc[-trail_3:]
        median_emi = float(recent_emi.median())
    else:
        median_emi = 0.0  # No recurring debt obligations identified

    result["current_dti"] = round(
        median_emi / median_income if median_income > 0 else 0.0, 4,
    )

    # Debt trend direction
    if num_months >= 3:
        monthly_debt = debits.groupby(
            debits["date"].dt.to_period("M"),
        )["amount"].sum()
        # Reindex to fill gaps
        all_periods = pd.period_range(
            monthly_debt.index.min(), monthly_debt.index.max(), freq="M",
        )
        monthly_debt = monthly_debt.reindex(all_periods, fill_value=0.0)
        slope = _ols_slope(monthly_debt.values.astype(float))
        mean_debt = monthly_debt.mean()
        result["debt_trend_direction"] = round(
            slope / mean_debt if mean_debt > 0 and slope is not None else 0.0, 4,
        )
    else:
        result["debt_trend_direction"] = None

    return result
