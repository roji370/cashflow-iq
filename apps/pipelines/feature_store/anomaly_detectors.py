"""Behavioral anomaly detectors for Cashflow IQ.

Three pure-function detectors that operate on a single customer's transaction
DataFrame and return boolean flags + continuous anomaly scores (0–1).

Detectors:
  1. Subscription cleansing — 3+ recurring discretionary payments cancelled
     within a 60-day window while income stays stable.
  2. Liquidity pooling — FD/MF maturity not reinvested within 45 days.
  3. Out-of-cycle bill shift — recurring bill payment day drifting toward
     month-end, detected via IsolationForest.

Each detector compares against the customer's own historical baseline, not a
population-wide threshold.
"""

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# ---------------------------------------------------------------------------
# 1. Subscription cleansing detector
# ---------------------------------------------------------------------------

def detect_subscription_cleansing(
    txns_df: pd.DataFrame,
) -> dict[str, Any]:
    """Detect subscription cleansing — multiple recurring discretionary payments
    cancelled in a short window while income remains stable.

    Args:
        txns_df: Transaction DataFrame for a single customer. Must have columns:
            date (datetime-like), amount, type, merchant_category, recurring_group_id.

    Returns:
        Dict with keys:
            subscription_cleansing_flag: bool
            subscription_cleansing_score: float (0–1)
    """
    result = {"subscription_cleansing_flag": False, "subscription_cleansing_score": 0.0}

    df = txns_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    discretionary_cats = {
        "entertainment", "subscriptions", "dining", "travel",
        "fuel", "jewellery", "grocery",
    }

    # Identify recurring discretionary debits — same merchant_category with
    # recurring_group_id, at least 3 occurrences
    recurring_debits = df[
        (df["type"] == "debit")
        & (df["merchant_category"].isin(discretionary_cats))
        & (df["recurring_group_id"].notna())
        & (df["recurring_group_id"] != "")
    ]

    if recurring_debits.empty:
        return result

    # Group by recurring_group_id to find recurring series
    series_info: dict[str, dict] = {}
    for gid, grp in recurring_debits.groupby("recurring_group_id"):
        if len(grp) >= 3:
            series_info[str(gid)] = {
                "first": grp["date"].min(),
                "last": grp["date"].max(),
                "count": len(grp),
            }

    if len(series_info) < 3:
        return result

    # Check for salary/income stability over the period
    credits = df[df["type"] == "credit"]
    if credits.empty:
        return result

    credits_monthly = credits.set_index("date").resample("MS")["amount"].sum()
    if len(credits_monthly) < 3:
        return result

    income_cv = credits_monthly.std() / credits_monthly.mean() if credits_monthly.mean() > 0 else 1.0
    income_stable = income_cv < 0.3  # CV < 30% = relatively stable

    if not income_stable:
        return result

    # Find the overall data range
    overall_last = df["date"].max()
    # Check: did 3+ recurring series stop within a 60-day window?
    # "Stopped" = last occurrence is > 60 days before the data end
    stopped_series: list[str] = []
    stop_dates: list[pd.Timestamp] = []
    for gid, info in series_info.items():
        days_since_last = (overall_last - info["last"]).days
        if days_since_last >= 30:  # hasn't appeared in last month = likely stopped
            stopped_series.append(gid)
            stop_dates.append(info["last"])

    if len(stopped_series) < 3:
        return result

    # Check if stops cluster within a 60-day window
    stop_dates_sorted = sorted(stop_dates)
    for i in range(len(stop_dates_sorted) - 2):
        window = (stop_dates_sorted[i + 2] - stop_dates_sorted[i]).days
        if window <= 60:
            # Score: proportion of recurring series that stopped, capped at 1.0
            score = min(1.0, len(stopped_series) / max(len(series_info), 1))
            return {
                "subscription_cleansing_flag": True,
                "subscription_cleansing_score": round(score, 4),
            }

    return result


# ---------------------------------------------------------------------------
# 2. Liquidity pooling detector
# ---------------------------------------------------------------------------

def detect_liquidity_pooling(
    txns_df: pd.DataFrame,
) -> dict[str, Any]:
    """Detect liquidity pooling — FD/MF maturity credit not reinvested within
    45 days.

    Args:
        txns_df: Transaction DataFrame for a single customer.

    Returns:
        Dict with keys:
            liquidity_pooling_flag: bool
            liquidity_pooling_score: float (0–1)
    """
    result = {"liquidity_pooling_flag": False, "liquidity_pooling_score": 0.0}

    df = txns_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Identify investment maturity credits (large investment credits)
    investment_credits = df[
        (df["type"] == "credit")
        & (df["merchant_category"] == "investment")
    ].sort_values("date")

    if investment_credits.empty:
        return result

    # Identify investment debits (reinvestments)
    investment_debits = df[
        (df["type"] == "debit")
        & (df["merchant_category"] == "investment")
    ].sort_values("date")

    # For each maturity event, check if reinvested within 45 days
    overall_last = df["date"].max()
    unreinvested_events = 0
    total_maturity_events = 0
    max_gap_days = 0

    for _, maturity in investment_credits.iterrows():
        mat_date = maturity["date"]
        mat_amount = maturity["amount"]

        # Only consider significant amounts (> 10x median transaction)
        median_txn = df["amount"].median()
        if mat_amount < median_txn * 5:
            continue

        total_maturity_events += 1

        # Look for reinvestment within 45 days after maturity
        window_end = mat_date + pd.Timedelta(days=45)
        reinvestments = investment_debits[
            (investment_debits["date"] > mat_date)
            & (investment_debits["date"] <= window_end)
        ]

        if reinvestments.empty:
            unreinvested_events += 1
            # Gap = days from maturity to data end (or 45 if longer)
            gap = min((overall_last - mat_date).days, 90)
            max_gap_days = max(max_gap_days, gap)

    if total_maturity_events == 0:
        return result

    if unreinvested_events > 0:
        # Score: based on how far past 45 days the gap is, normalized
        score = min(1.0, max_gap_days / 90.0)
        return {
            "liquidity_pooling_flag": True,
            "liquidity_pooling_score": round(score, 4),
        }

    return result


# ---------------------------------------------------------------------------
# 3. Out-of-cycle bill shift detector
# ---------------------------------------------------------------------------

def detect_bill_shift(
    txns_df: pd.DataFrame,
) -> dict[str, Any]:
    """Detect out-of-cycle bill shift — recurring bill payment day drifting
    toward month-end over several months.

    Uses IsolationForest on (payment_day_of_month, amount) to flag anomalous
    drift from the customer's own historical pattern.

    Args:
        txns_df: Transaction DataFrame for a single customer.

    Returns:
        Dict with keys:
            bill_shift_flag: bool
            bill_shift_score: float (0–1)
    """
    result = {"bill_shift_flag": False, "bill_shift_score": 0.0}

    df = txns_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Filter to recurring bills — utility and rent with recurring_group_id
    bill_cats = {"utility", "rent"}
    bills = df[
        (df["type"] == "debit")
        & (df["merchant_category"].isin(bill_cats))
        & (df["recurring_group_id"].notna())
        & (df["recurring_group_id"] != "")
    ].copy()

    if len(bills) < 6:
        return result

    # Add day-of-month feature
    bills = bills.copy()
    bills["day_of_month"] = bills["date"].dt.day

    # Check each recurring series for drift
    max_score = 0.0
    flagged = False

    for gid, grp in bills.groupby("recurring_group_id"):
        if len(grp) < 5:
            continue

        grp_sorted = grp.sort_values("date")
        days = grp_sorted["day_of_month"].values
        amounts = grp_sorted["amount"].values

        # Check for systematic drift in payment day (trending later)
        first_half = days[: len(days) // 2]
        second_half = days[len(days) // 2:]

        mean_first = float(np.mean(first_half))
        mean_second = float(np.mean(second_half))
        drift = mean_second - mean_first

        if drift <= 3:  # less than 3-day drift is not significant
            continue

        # Use IsolationForest on (day_of_month, amount) to quantify anomaly
        features = np.column_stack([
            days.astype(float),
            amounts.astype(float),
        ])

        if len(features) < 4:
            continue

        iso = IsolationForest(
            contamination=0.3,
            random_state=42,
            n_estimators=50,
        )
        iso.fit(features)

        # Score the later payments — are they anomalous vs. the full history?
        later_features = features[len(features) // 2:]
        anomaly_scores = iso.decision_function(later_features)
        # decision_function: lower = more anomalous
        # Convert to 0–1 score where 1 = most anomalous
        min_score = float(np.min(anomaly_scores))
        # Normalize: typical scores range from -0.5 (anomalous) to 0.5 (normal)
        normalized = max(0.0, min(1.0, -min_score + 0.5))

        if normalized > 0.4:
            flagged = True
            max_score = max(max_score, normalized)

    if flagged:
        return {
            "bill_shift_flag": True,
            "bill_shift_score": round(max_score, 4),
        }

    return result
