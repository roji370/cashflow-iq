"""Full multi-persona synthetic data generator for Cashflow IQ.

Generates 8 distinct customer personas with 12 months of transaction history,
injected behavioral signals for anomaly detector ground truth, and per-product
conversion labels.

Replaces the Phase A generate_minimal.py (which only created one persona with
3 months of data).

Usage:
    python -m apps.pipelines.synth_data.generate [--personas N] [--months M] [--seed S]
"""

import argparse
import csv
import hashlib
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Merchant categories (MCC-like tags)
# ---------------------------------------------------------------------------
MERCHANT_CATEGORIES = [
    "grocery", "dining", "travel", "entertainment", "subscriptions",
    "fuel", "jewellery", "education", "medical", "real_estate",
    "auto_dealer", "utility", "rent", "salary", "investment",
]

DISCRETIONARY_CATEGORIES = [
    "grocery", "dining", "travel", "entertainment", "subscriptions",
    "fuel", "jewellery",
]

PRODUCTS = ["personal_loan", "home_loan", "mortgage_loan", "auto_loan"]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _month_range(base: date, months: int) -> list[date]:
    """Return the first day of each month for *months* months starting at *base*."""
    result: list[date] = []
    for i in range(months):
        m = (base.month - 1 + i) % 12 + 1
        y = base.year + (base.month - 1 + i) // 12
        result.append(date(y, m, 1))
    return result


def _random_day_in_month(month_start: date, rng: random.Random) -> date:
    """Pick a random day within the given month."""
    if month_start.month == 12:
        last_day = 28  # safe for all months
    else:
        last_day = 28
    day = rng.randint(1, last_day)
    return date(month_start.year, month_start.month, day)


def _txn_id(customer_id: str, txn_date: date, amount: float,
            category: str, txn_type: str, merchant_category: str) -> str:
    """Deterministic transaction ID based on content hash.

    NOTE: hash-based ID assumes no true duplicate transactions same-day/same-amount
    — revisit if this becomes a real risk with production data.
    """
    raw = f"{customer_id}|{txn_date.isoformat()}|{amount:.2f}|{category}|{txn_type}|{merchant_category}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _make_txn(customer_id: str, txn_date: date, amount: float,
              category: str, txn_type: str, merchant_category: str,
              recurring_group_id: Optional[str] = None) -> dict:
    """Build a single transaction dict matching the expanded Transaction schema."""
    return {
        "customer_id": customer_id,
        "date": txn_date.isoformat(),
        "amount": round(amount, 2),
        "category": category,
        "type": txn_type,
        "merchant_category": merchant_category,
        "recurring_group_id": recurring_group_id or "",
    }


# ---------------------------------------------------------------------------
# Persona generators
# ---------------------------------------------------------------------------
# Each returns (customer_dict, list[txn_dicts]).  Signals are injected inline.


def _gen_stable_salaried(cid: str, months: list[date], rng: random.Random,
                         ) -> tuple[dict, list[dict]]:
    """Persona 1: Stable salaried.

    Injects subscription cleansing signal — 3 recurring subscriptions cancelled
    in months 10–11 while salary continues.
    """
    customer = {
        "customer_id": cid, "name": "Priya Sharma",
        "occupation": "Software Engineer", "employer_tenure_months": 48,
        "city": "Bangalore", "account_vintage_months": 60,
        "persona_type": "stable_salaried",
    }
    txns: list[dict] = []
    salary_base = 70000.0

    # Subscription setup — 3 recurring subscriptions active from month 0
    sub_names = ["sub_netflix", "sub_gym", "sub_news"]
    sub_amounts = [799.0, 1500.0, 299.0]

    for idx, month in enumerate(months):
        # Salary credit on 1st
        txns.append(_make_txn(
            cid, date(month.year, month.month, 1),
            salary_base + rng.uniform(-1500, 1500),
            "salary", "credit", "salary",
            recurring_group_id="sal_priya",
        ))
        # Rent on 5th
        txns.append(_make_txn(
            cid, date(month.year, month.month, 5),
            20000 + rng.uniform(-300, 300),
            "rent", "debit", "rent",
            recurring_group_id="rent_priya",
        ))
        # Utility on 10th
        txns.append(_make_txn(
            cid, date(month.year, month.month, 10),
            2500 + rng.uniform(-200, 200),
            "utility", "debit", "utility",
            recurring_group_id="util_priya",
        ))

        # Subscriptions — cancel 3 in months 10-11 (subscription cleansing signal)
        for si, (sname, samt) in enumerate(zip(sub_names, sub_amounts)):
            if idx < 9:  # months 0–8: all active
                txns.append(_make_txn(
                    cid, date(month.year, month.month, 15 + si),
                    samt + rng.uniform(-5, 5), "subscriptions", "debit",
                    "subscriptions", recurring_group_id=sname,
                ))
            # months 9–11: subscriptions stop (cleansing signal)

        # 3–5 discretionary debits
        for _ in range(rng.randint(3, 5)):
            cat = rng.choice(["grocery", "dining", "fuel"])
            txns.append(_make_txn(
                cid, _random_day_in_month(month, rng),
                rng.uniform(200, 4000), cat, "debit", cat,
            ))

    return customer, txns


def _gen_gig_worker(cid: str, months: list[date], rng: random.Random,
                    ) -> tuple[dict, list[dict]]:
    """Persona 2: Gig worker / freelancer with UPI-based income."""
    customer = {
        "customer_id": cid, "name": "Ravi Kumar",
        "occupation": "Freelance Designer", "employer_tenure_months": 0,
        "city": "Hyderabad", "account_vintage_months": 24,
        "persona_type": "gig_worker",
    }
    txns: list[dict] = []

    for month in months:
        # 5–12 irregular UPI income credits across the month
        num_credits = rng.randint(5, 12)
        for _ in range(num_credits):
            txns.append(_make_txn(
                cid, _random_day_in_month(month, rng),
                rng.uniform(2000, 25000), "freelance", "credit", "salary",
            ))
        # Rent
        txns.append(_make_txn(
            cid, date(month.year, month.month, 7),
            12000 + rng.uniform(-500, 500), "rent", "debit", "rent",
            recurring_group_id="rent_ravi",
        ))
        # 4–8 discretionary
        for _ in range(rng.randint(4, 8)):
            cat = rng.choice(DISCRETIONARY_CATEGORIES)
            txns.append(_make_txn(
                cid, _random_day_in_month(month, rng),
                rng.uniform(100, 3000), cat, "debit", cat,
            ))

    return customer, txns


def _gen_small_business(cid: str, months: list[date], rng: random.Random,
                        ) -> tuple[dict, list[dict]]:
    """Persona 3: Small business owner with round-number P2P inflows."""
    customer = {
        "customer_id": cid, "name": "Anita Patel",
        "occupation": "Business Owner", "employer_tenure_months": 0,
        "city": "Ahmedabad", "account_vintage_months": 84,
        "persona_type": "small_business_owner",
    }
    txns: list[dict] = []

    round_amounts = [10000, 25000, 50000, 75000, 100000]

    for month in months:
        # 2–4 round-number business inflows
        for _ in range(rng.randint(2, 4)):
            txns.append(_make_txn(
                cid, _random_day_in_month(month, rng),
                rng.choice(round_amounts) + rng.uniform(-500, 500),
                "business_income", "credit", "salary",
            ))
        # Business expenses
        for _ in range(rng.randint(3, 6)):
            cat = rng.choice(["grocery", "fuel", "entertainment", "education"])
            txns.append(_make_txn(
                cid, _random_day_in_month(month, rng),
                rng.uniform(500, 15000), cat, "debit", cat,
            ))
        # Rent
        txns.append(_make_txn(
            cid, date(month.year, month.month, 3),
            30000 + rng.uniform(-1000, 1000), "rent", "debit", "rent",
            recurring_group_id="rent_anita",
        ))

    return customer, txns


def _gen_over_leveraged(cid: str, months: list[date], rng: random.Random,
                        ) -> tuple[dict, list[dict]]:
    """Persona 4: Over-leveraged customer with multiple EMIs.

    Injects out-of-cycle bill shift signal — utility bill payment day drifts
    from day 5 toward day 20+ over months 7–11.
    """
    customer = {
        "customer_id": cid, "name": "Suresh Menon",
        "occupation": "Sales Executive", "employer_tenure_months": 36,
        "city": "Chennai", "account_vintage_months": 48,
        "persona_type": "over_leveraged",
    }
    txns: list[dict] = []
    salary = 55000.0

    for idx, month in enumerate(months):
        # Salary
        txns.append(_make_txn(
            cid, date(month.year, month.month, 1),
            salary + rng.uniform(-1000, 1000),
            "salary", "credit", "salary",
            recurring_group_id="sal_suresh",
        ))

        # 3 EMIs — heavy debt load
        for emi_i, (emi_amt, emi_name) in enumerate([
            (15000, "emi_car"), (12000, "emi_personal"), (8000, "emi_credit"),
        ]):
            txns.append(_make_txn(
                cid, date(month.year, month.month, 5 + emi_i),
                emi_amt + rng.uniform(-100, 100),
                "emi", "debit", "utility",
                recurring_group_id=emi_name,
            ))

        # Utility bill — inject bill-shift signal from month 7 onward
        if idx < 7:
            bill_day = 5  # stable payment day
        else:
            # Drift toward month-end: 5 → 12 → 18 → 22 → 25
            drift = min(5 + (idx - 7) * 5, 25)
            bill_day = drift
        bill_day = min(bill_day, 28)  # safety cap
        txns.append(_make_txn(
            cid, date(month.year, month.month, bill_day),
            3500 + rng.uniform(-200, 200),
            "utility", "debit", "utility",
            recurring_group_id="util_suresh",
        ))

        # Rent
        txns.append(_make_txn(
            cid, date(month.year, month.month, 3),
            18000 + rng.uniform(-500, 500),
            "rent", "debit", "rent",
            recurring_group_id="rent_suresh",
        ))

        # Minimal discretionary
        for _ in range(rng.randint(1, 3)):
            cat = rng.choice(["grocery", "fuel"])
            txns.append(_make_txn(
                cid, _random_day_in_month(month, rng),
                rng.uniform(200, 1500), cat, "debit", cat,
            ))

    return customer, txns


def _gen_hni(cid: str, months: list[date], rng: random.Random,
             ) -> tuple[dict, list[dict]]:
    """Persona 5: HNI with FD/mutual fund holdings.

    Injects liquidity pooling signal — FD maturity in month 8, not reinvested
    within 45 days.
    """
    customer = {
        "customer_id": cid, "name": "Vikram Reddy",
        "occupation": "Senior Director", "employer_tenure_months": 120,
        "city": "Mumbai", "account_vintage_months": 144,
        "persona_type": "hni",
    }
    txns: list[dict] = []
    salary = 250000.0

    for idx, month in enumerate(months):
        # Salary
        txns.append(_make_txn(
            cid, date(month.year, month.month, 1),
            salary + rng.uniform(-5000, 5000),
            "salary", "credit", "salary",
            recurring_group_id="sal_vikram",
        ))

        # Regular investments — monthly SIP (except around maturity month)
        if idx not in (7, 8, 9):  # normal months
            txns.append(_make_txn(
                cid, date(month.year, month.month, 5),
                50000 + rng.uniform(-2000, 2000),
                "sip", "debit", "investment",
                recurring_group_id="sip_vikram",
            ))

        # FD maturity in month 8 — large credit, NOT reinvested (liquidity pooling)
        if idx == 7:
            txns.append(_make_txn(
                cid, date(month.year, month.month, 15),
                500000.0, "fd_maturity", "credit", "investment",
            ))
        # Months 8, 9, 10: no reinvestment — the liquidity just sits there

        # Premium spending
        for _ in range(rng.randint(3, 7)):
            cat = rng.choice(["dining", "travel", "jewellery", "entertainment"])
            txns.append(_make_txn(
                cid, _random_day_in_month(month, rng),
                rng.uniform(2000, 30000), cat, "debit", cat,
            ))

        # Rent (premium)
        txns.append(_make_txn(
            cid, date(month.year, month.month, 3),
            60000 + rng.uniform(-2000, 2000),
            "rent", "debit", "rent",
            recurring_group_id="rent_vikram",
        ))

    return customer, txns


def _gen_young_first_jobber(cid: str, months: list[date], rng: random.Random,
                            ) -> tuple[dict, list[dict]]:
    """Persona 6: Young first-jobber with growing salary, small amounts."""
    customer = {
        "customer_id": cid, "name": "Sneha Iyer",
        "occupation": "Junior Analyst", "employer_tenure_months": 6,
        "city": "Pune", "account_vintage_months": 8,
        "persona_type": "young_first_jobber",
    }
    txns: list[dict] = []
    base_salary = 30000.0

    for idx, month in enumerate(months):
        # Salary with slight growth trend
        salary = base_salary + idx * 500 + rng.uniform(-500, 500)
        txns.append(_make_txn(
            cid, date(month.year, month.month, 1),
            salary, "salary", "credit", "salary",
            recurring_group_id="sal_sneha",
        ))

        # Rent
        txns.append(_make_txn(
            cid, date(month.year, month.month, 5),
            10000 + rng.uniform(-200, 200),
            "rent", "debit", "rent",
            recurring_group_id="rent_sneha",
        ))

        # 3–6 small discretionary
        for _ in range(rng.randint(3, 6)):
            cat = rng.choice(["grocery", "dining", "entertainment", "subscriptions"])
            txns.append(_make_txn(
                cid, _random_day_in_month(month, rng),
                rng.uniform(100, 2000), cat, "debit", cat,
            ))

    return customer, txns


def _gen_self_employed_volatile(cid: str, months: list[date], rng: random.Random,
                                ) -> tuple[dict, list[dict]]:
    """Persona 7: Self-employed with highly volatile income, seasonal patterns."""
    customer = {
        "customer_id": cid, "name": "Deepak Joshi",
        "occupation": "Consultant", "employer_tenure_months": 0,
        "city": "Delhi", "account_vintage_months": 36,
        "persona_type": "self_employed_volatile",
    }
    txns: list[dict] = []

    for idx, month in enumerate(months):
        # Volatile income — seasonal bump in Q4 (months 9-11)
        if idx >= 9:
            base = rng.uniform(80000, 150000)
        else:
            base = rng.uniform(20000, 80000)

        # 1–3 income credits
        num_credits = rng.randint(1, 3)
        for _ in range(num_credits):
            txns.append(_make_txn(
                cid, _random_day_in_month(month, rng),
                base / num_credits + rng.uniform(-3000, 3000),
                "consulting", "credit", "salary",
            ))

        # Expenses proportional to income (roughly)
        for _ in range(rng.randint(3, 7)):
            cat = rng.choice(DISCRETIONARY_CATEGORIES)
            txns.append(_make_txn(
                cid, _random_day_in_month(month, rng),
                rng.uniform(200, 8000), cat, "debit", cat,
            ))

        # Rent
        txns.append(_make_txn(
            cid, date(month.year, month.month, 4),
            15000 + rng.uniform(-500, 500),
            "rent", "debit", "rent",
            recurring_group_id="rent_deepak",
        ))

    return customer, txns


def _gen_life_event_home_buyer(cid: str, months: list[date], rng: random.Random,
                               ) -> tuple[dict, list[dict]]:
    """Persona 8: Normal customer until month 9, then home purchase sequence.

    Injects home purchase signal — property registration + builder payments
    over months 9–11.
    """
    customer = {
        "customer_id": cid, "name": "Meera Nair",
        "occupation": "Product Manager", "employer_tenure_months": 60,
        "city": "Kochi", "account_vintage_months": 72,
        "persona_type": "life_event_home_buyer",
    }
    txns: list[dict] = []
    salary = 90000.0

    for idx, month in enumerate(months):
        # Salary
        txns.append(_make_txn(
            cid, date(month.year, month.month, 1),
            salary + rng.uniform(-2000, 2000),
            "salary", "credit", "salary",
            recurring_group_id="sal_meera",
        ))

        # Rent
        txns.append(_make_txn(
            cid, date(month.year, month.month, 5),
            25000 + rng.uniform(-500, 500),
            "rent", "debit", "rent",
            recurring_group_id="rent_meera",
        ))

        # Normal discretionary
        for _ in range(rng.randint(3, 5)):
            cat = rng.choice(["grocery", "dining", "fuel", "entertainment"])
            txns.append(_make_txn(
                cid, _random_day_in_month(month, rng),
                rng.uniform(300, 5000), cat, "debit", cat,
            ))

        # Home purchase sequence — months 9, 10, 11
        if idx == 8:
            # Property registration fee
            txns.append(_make_txn(
                cid, date(month.year, month.month, 20),
                200000.0, "property_registration", "debit", "real_estate",
            ))
        if idx == 9:
            # Builder payment 1
            txns.append(_make_txn(
                cid, date(month.year, month.month, 10),
                500000.0, "builder_payment", "debit", "real_estate",
            ))
        if idx == 10:
            # Builder payment 2
            txns.append(_make_txn(
                cid, date(month.year, month.month, 15),
                300000.0, "builder_payment", "debit", "real_estate",
            ))
        if idx == 11:
            # Final builder payment
            txns.append(_make_txn(
                cid, date(month.year, month.month, 5),
                200000.0, "builder_payment", "debit", "real_estate",
            ))

    return customer, txns


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def _generate_labels(customers: list[dict], rng: random.Random) -> list[dict]:
    """Generate per-customer × per-product conversion labels.

    Labels are correlated with persona type + noise to target AUC ~0.75–0.85,
    not trivially separable.
    """
    # Base conversion probabilities by persona × product
    # These are tuned so that a model trained on these features would get
    # AUC 0.75–0.85 — some signal, not perfect separation.
    base_probs: dict[str, dict[str, float]] = {
        "stable_salaried": {
            "personal_loan": 0.55, "home_loan": 0.60,
            "mortgage_loan": 0.40, "auto_loan": 0.50,
        },
        "gig_worker": {
            "personal_loan": 0.65, "home_loan": 0.25,
            "mortgage_loan": 0.15, "auto_loan": 0.35,
        },
        "small_business_owner": {
            "personal_loan": 0.50, "home_loan": 0.35,
            "mortgage_loan": 0.30, "auto_loan": 0.40,
        },
        "over_leveraged": {
            "personal_loan": 0.70, "home_loan": 0.20,
            "mortgage_loan": 0.15, "auto_loan": 0.25,
        },
        "hni": {
            "personal_loan": 0.15, "home_loan": 0.55,
            "mortgage_loan": 0.50, "auto_loan": 0.30,
        },
        "young_first_jobber": {
            "personal_loan": 0.45, "home_loan": 0.10,
            "mortgage_loan": 0.05, "auto_loan": 0.30,
        },
        "self_employed_volatile": {
            "personal_loan": 0.55, "home_loan": 0.30,
            "mortgage_loan": 0.20, "auto_loan": 0.35,
        },
        "life_event_home_buyer": {
            "personal_loan": 0.35, "home_loan": 0.85,
            "mortgage_loan": 0.70, "auto_loan": 0.20,
        },
    }

    labels: list[dict] = []
    for cust in customers:
        persona = cust.get("persona_type", "stable_salaried")
        probs = base_probs.get(persona, base_probs["stable_salaried"])
        for product in PRODUCTS:
            prob = probs[product]
            # Add noise (±0.15) to prevent trivial separability
            noisy_prob = max(0.0, min(1.0, prob + rng.uniform(-0.15, 0.15)))
            converted = rng.random() < noisy_prob
            labels.append({
                "customer_id": cust["customer_id"],
                "product": product,
                "converted": converted,
            })

    return labels


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CUSTOMER_FIELDS = [
    "customer_id", "name", "occupation", "employer_tenure_months",
    "city", "account_vintage_months", "persona_type",
]

TRANSACTION_FIELDS = [
    "customer_id", "date", "amount", "category", "type",
    "merchant_category", "recurring_group_id",
]

LABEL_FIELDS = ["customer_id", "product", "converted"]


def _write_csv(filepath: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write a list of dicts to a CSV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows → {filepath}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

# Registry of persona generators — add new personas here.
PERSONA_GENERATORS = [
    _gen_stable_salaried,
    _gen_gig_worker,
    _gen_small_business,
    _gen_over_leveraged,
    _gen_hni,
    _gen_young_first_jobber,
    _gen_self_employed_volatile,
    _gen_life_event_home_buyer,
]


def generate(num_personas: int = 8, num_months: int = 12,
             seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]:
    """Generate synthetic customers, transactions, and labels.

    Args:
        num_personas: Number of personas to generate (max 8, will cycle if >8).
        num_months: Months of transaction history per persona.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (customers, transactions, labels) as lists of dicts.
    """
    rng = random.Random(seed)
    base_date = date(2024, 1, 1)
    months = _month_range(base_date, num_months)

    all_customers: list[dict] = []
    all_txns: list[dict] = []

    for i in range(num_personas):
        cid = f"cust_{i + 1:03d}"
        gen_fn = PERSONA_GENERATORS[i % len(PERSONA_GENERATORS)]
        customer, txns = gen_fn(cid, months, rng)
        all_customers.append(customer)
        all_txns.extend(txns)

    labels = _generate_labels(all_customers, rng)

    return all_customers, all_txns, labels


def main() -> None:
    """CLI entry point: generate customers.csv, transactions.csv, labels.csv."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for Cashflow IQ",
    )
    parser.add_argument("--personas", type=int, default=8,
                        help="Number of personas to generate (default: 8)")
    parser.add_argument("--months", type=int, default=12,
                        help="Months of transaction history (default: 12)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    print(f"Generating {args.personas} personas × {args.months} months (seed={args.seed})...")
    customers, txns, labels = generate(args.personas, args.months, args.seed)

    output_dir = Path("apps/pipelines/synth_data/output")
    _write_csv(output_dir / "customers.csv", customers, CUSTOMER_FIELDS)
    _write_csv(output_dir / "transactions.csv", txns, TRANSACTION_FIELDS)
    _write_csv(output_dir / "labels.csv", labels, LABEL_FIELDS)

    print(f"\nDone — {len(customers)} customers, {len(txns)} transactions, "
          f"{len(labels)} labels")


if __name__ == "__main__":
    main()
