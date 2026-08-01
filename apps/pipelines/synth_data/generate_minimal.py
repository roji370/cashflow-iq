"""Minimal synthetic data generator for the walking skeleton.

Generates one 'Stable Salaried' persona with ~25 transactions over 3 months.
Phase A only — will be replaced by a richer multi-persona generator in Phase B.
"""

import csv
import os
import random
from datetime import date, timedelta
from pathlib import Path


def generate_customer() -> dict[str, str]:
    """Return the single test customer record."""
    return {"customer_id": "cust_001", "name": "Test Customer"}


def generate_transactions() -> list[dict]:
    """Generate ~25 transactions for cust_001 over 3 months.

    Pattern: monthly salary credit (~70k), monthly rent debit (~20k),
    and 5–10 misc discretionary debits spread across the period.
    """
    transactions: list[dict] = []
    customer_id = "cust_001"
    base_date = date(2025, 1, 1)

    # Seed for reproducibility
    random.seed(42)

    discretionary_categories = ["groceries", "dining", "fuel", "entertainment", "shopping"]

    for month_offset in range(3):
        # Month start
        month_start = date(
            base_date.year,
            base_date.month + month_offset,
            1,
        )

        # Salary credit on the 1st
        transactions.append({
            "customer_id": customer_id,
            "date": month_start.isoformat(),
            "amount": round(70000 + random.uniform(-2000, 2000), 2),
            "category": "salary",
            "type": "credit",
        })

        # Rent debit on the 5th
        transactions.append({
            "customer_id": customer_id,
            "date": date(month_start.year, month_start.month, 5).isoformat(),
            "amount": round(20000 + random.uniform(-500, 500), 2),
            "category": "rent",
            "type": "debit",
        })

        # 5–10 misc discretionary debits spread across the month
        num_misc = random.randint(5, 7)
        for _ in range(num_misc):
            day = random.randint(2, 28)
            txn_date = date(month_start.year, month_start.month, day)
            transactions.append({
                "customer_id": customer_id,
                "date": txn_date.isoformat(),
                "amount": round(random.uniform(200, 5000), 2),
                "category": random.choice(discretionary_categories),
                "type": "debit",
            })

    return transactions


def write_csv(filepath: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write a list of dicts to a CSV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {filepath}")


def main() -> None:
    """Entry point: generate customers.csv and transactions.csv."""
    output_dir = Path("apps/pipelines/synth_data/output")

    customer = generate_customer()
    write_csv(
        output_dir / "customers.csv",
        [customer],
        fieldnames=["customer_id", "name"],
    )

    transactions = generate_transactions()
    write_csv(
        output_dir / "transactions.csv",
        transactions,
        fieldnames=["customer_id", "date", "amount", "category", "type"],
    )

    print(f"Done — {len(transactions)} transactions generated for {customer['customer_id']}")


if __name__ == "__main__":
    main()
