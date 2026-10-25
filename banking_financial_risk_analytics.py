"""Banking & Financial Risk Analytics: an end-to-end, single-file project.

Workflow
--------
Python -> NumPy -> Pandas -> Visualization -> Data Quality ->
Feature Engineering -> Probability -> Statistics -> SQL preparation

This project is intended for learning and portfolio demonstrations. The built-in
synthetic dataset is not representative of any real bank or borrower population.
The transparent risk segment is an illustrative rule, not a credit score, lending
policy, or production decision system. Never use this script alone to approve,
reject, price, or otherwise make decisions about real applicants.

Quick start
-----------
1. Install core dependencies:
       pip install numpy pandas matplotlib
   Optional packages:
       pip install scipy duckdb

2. Run with reproducible synthetic data (default: 5,000 rows):
       python banking_financial_risk_analytics.py
       python banking_financial_risk_analytics.py --rows 20000 --seed 7

3. Run against your own CSV, using the canonical column names documented below:
       python banking_financial_risk_analytics.py --input path/to/loans.csv

4. Choose a custom output directory:
       python banking_financial_risk_analytics.py --output-dir outputs/my_analysis

Canonical input columns
-----------------------
Required analytical inputs: loan_amount, annual_income, credit_score.
Recommended columns: loan_id, customer_id, application_date, approved, default,
employment_years, debt_to_income_ratio, interest_rate, loan_term_months,
existing_monthly_debt, monthly_payment, home_ownership, loan_purpose, region,
delinquencies_2y, credit_history_years.

Aliases for several common LendingClub-style fields are supported (for example,
loan_amnt, annual_inc, fico_score, dti, int_rate, issue_d, installment, purpose,
emp_length, delinq_2yrs, and loan_status). For `default`, use 1 for default / charged
off and 0 for non-default / fully paid. Open/current loans normally have unknown
outcomes and should remain blank, not be labelled as non-default.

Percentage conventions
----------------------
Ratios such as debt_to_income_ratio should normally be decimal fractions (0.35 =
35%). Interest rates should normally be decimal fractions (0.12 = 12%). The script
attempts to normalize common percentage-style inputs such as "12.5%" or 12.5.
Please inspect the generated quality reports and confirm assumptions for your data.

Outputs
-------
The output directory contains cleaned/engineered data, data-quality reports,
probability summaries, statistical tests, CSV result tables, PNG charts, a SQL
query file, and an executive summary. DuckDB queries are executed when DuckDB is
installed; otherwise SQL is still written to sql_queries.sql for review/use.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")  # Save charts reliably in headless environments.
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "Missing visualization dependency. Install it with: pip install matplotlib"
    ) from exc

try:
    from scipy.stats import chi2_contingency, ttest_ind
    SCIPY_AVAILABLE = True
except ImportError:  # Optional; core analysis remains available without SciPy.
    SCIPY_AVAILABLE = False

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:  # Optional; SQL is still exported for external execution.
    DUCKDB_AVAILABLE = False

LOGGER = logging.getLogger("banking_risk_analytics")

RANDOM_SEED_DEFAULT = 42
ROWS_DEFAULT = 5_000
OUTPUT_DIR_DEFAULT = "outputs/banking_financial_risk"

NUMERIC_FIELDS = [
    "loan_amount",
    "annual_income",
    "employment_years",
    "credit_score",
    "debt_to_income_ratio",
    "interest_rate",
    "loan_term_months",
    "existing_monthly_debt",
    "monthly_payment",
    "delinquencies_2y",
    "credit_history_years",
]

CATEGORICAL_FIELDS = ["home_ownership", "loan_purpose", "region"]

ALIASES = {
    "loan_amnt": "loan_amount",
    "funded_amnt": "loan_amount",
    "annual_inc": "annual_income",
    "income": "annual_income",
    "fico_score": "credit_score",
    "fico_range_low": "credit_score",
    "dti": "debt_to_income_ratio",
    "int_rate": "interest_rate",
    "term": "loan_term_months",
    "installment": "monthly_payment",
    "emp_length": "employment_years",
    "issue_d": "application_date",
    "application_dt": "application_date",
    "purpose": "loan_purpose",
    "home_ownership_status": "home_ownership",
    "delinq_2yrs": "delinquencies_2y",
    "delinq_2yrs_count": "delinquencies_2y",
    "credit_history_length_years": "credit_history_years",
    "loan_status_label": "loan_status",
    "is_default": "default",
    "defaulted": "default",
    "is_approved": "approved",
}


def configure_logging() -> None:
    """Configure consistent console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def normalize_column_name(name: Any) -> str:
    """Convert column labels into stable snake_case names."""
    text = str(name).strip().lower()
    text = text.replace("%", "pct")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize headers and apply aliases without overwriting canonical fields."""
    df = frame.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]

    # If both an alias and its canonical name are present, keep the canonical field.
    rename = {
        old: new
        for old, new in ALIASES.items()
        if old in df.columns and new not in df.columns
    }
    df = df.rename(columns=rename)

    # Common LendingClub field sometimes contains values such as "36 months".
    return df


def _numeric_from_mixed_text(series: pd.Series) -> pd.Series:
    """Parse numbers from values such as '$25,000', '36 months', or '10+ years'."""
    source = series.astype("string").str.strip()
    lower = source.str.lower()
    special = {
        "< 1 year": 0.5,
        "<1 year": 0.5,
        "less than 1 year": 0.5,
        "n/a": np.nan,
        "na": np.nan,
        "none": np.nan,
        "unknown": np.nan,
        "": np.nan,
    }
    result = pd.Series(np.nan, index=series.index, dtype="float64")
    for token, number in special.items():
        result.loc[lower.eq(token).fillna(False)] = number

    remaining = result.isna() & source.notna() & lower.ne("")
    cleaned = (
        source.str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace("%", "", regex=False)
    )
    extracted = cleaned.str.extract(r"([-+]?(?:\d*\.)?\d+)", expand=False)
    parsed = pd.to_numeric(extracted, errors="coerce")
    result.loc[remaining] = parsed.loc[remaining]
    return result


def parse_fraction_or_percent(series: pd.Series) -> pd.Series:
    """Normalize a ratio/rate stored as either fraction or common percent value."""
    source = series.astype("string")
    has_percent_sign = source.str.contains("%", regex=False, na=False)
    values = _numeric_from_mixed_text(series)

    # Explicit percentage signs are row-level evidence that division by 100 is needed.
    values.loc[has_percent_sign] = values.loc[has_percent_sign] / 100.0
    non_explicit = ~has_percent_sign & values.notna()
    median_value = values.loc[non_explicit].median() if non_explicit.any() else np.nan
    if pd.notna(median_value) and 1.0 < median_value <= 100.0:
        values.loc[non_explicit] = values.loc[non_explicit] / 100.0
    return values


def parse_binary_label(series: pd.Series, field_name: str) -> pd.Series:
    """Convert common binary labels to 0/1 while preserving unknown outcomes as NaN."""
    raw = series.astype("string").str.strip().str.lower()
    numeric = pd.to_numeric(raw, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    valid_numeric = numeric.isin([0, 1])
    out.loc[valid_numeric] = numeric.loc[valid_numeric].astype(float)

    positive = {"yes", "y", "true", "t", "1", "default", "defaulted", "charged off", "charged_off"}
    negative = {"no", "n", "false", "f", "0", "not default", "non-default", "fully paid", "paid", "repaid"}
    for value in positive:
        out.loc[raw.eq(value).fillna(False)] = 1.0
    for value in negative:
        out.loc[raw.eq(value).fillna(False)] = 0.0

    if field_name == "default":
        # Treat only terminal loan-status outcomes as labels. Current/open/late cases
        # are intentionally unknown rather than being mislabeled as non-default.
        terminal_defaults = {"default", "charged off", "charged_off", "does not meet the credit policy. status:charged off"}
        terminal_paid = {"fully paid", "paid", "repaid", "does not meet the credit policy. status:fully paid"}
        for value in terminal_defaults:
            out.loc[raw.eq(value).fillna(False)] = 1.0
        for value in terminal_paid:
            out.loc[raw.eq(value).fillna(False)] = 0.0
    return out


def parse_loan_status_default(status: pd.Series) -> pd.Series:
    """Derive default outcomes from terminal statuses; unresolved statuses stay missing."""
    normalized = status.astype("string").str.strip().str.lower()
    out = pd.Series(np.nan, index=status.index, dtype="float64")
    default_terms = ("charged off", "charged_off", "default")
    paid_terms = ("fully paid", "fully_paid", "repaid")
    out.loc[normalized.apply(lambda x: any(t in str(x) for t in default_terms) if pd.notna(x) else False)] = 1.0
    out.loc[normalized.apply(lambda x: any(t in str(x) for t in paid_terms) if pd.notna(x) else False)] = 0.0
    return out


def parse_loan_status_approval(status: pd.Series) -> pd.Series:
    """Infer whether a loan was funded/active from common loan status text."""
    normalized = status.astype("string").str.strip().str.lower()
    out = pd.Series(np.nan, index=status.index, dtype="float64")
    rejected_words = ("reject", "declin", "denied", "withdrawn")
    approved_words = (
        "fully paid", "charged off", "default", "current", "late", "issued",
        "in grace period", "does not meet the credit policy. status:fully paid",
        "does not meet the credit policy. status:charged off",
    )
    out.loc[normalized.apply(lambda x: any(t in str(x) for t in rejected_words) if pd.notna(x) else False)] = 0.0
    out.loc[normalized.apply(lambda x: any(t in str(x) for t in approved_words) if pd.notna(x) else False)] = 1.0
    return out


def standardize_and_clean(raw_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Standardize schema, audit raw values, and apply conservative cleaning rules.

    Returns
    -------
    cleaned_frame, range_check_report, cleaning_actions_report
    """
    if raw_frame.empty:
        raise ValueError("The supplied dataset contains zero rows.")

    df = normalize_columns(raw_frame)

    # Convert common free-form fields to numeric values before auditing ranges.
    for field in NUMERIC_FIELDS:
        if field in df.columns:
            if field in {"debt_to_income_ratio", "interest_rate"}:
                df[field] = parse_fraction_or_percent(df[field])
            else:
                df[field] = _numeric_from_mixed_text(df[field])

    if "application_date" in df.columns:
        df["application_date"] = pd.to_datetime(df["application_date"], errors="coerce", utc=True)
    else:
        df["application_date"] = pd.NaT

    # Resolve outcomes from either binary labels or known terminal loan statuses.
    if "default" in df.columns:
        parsed_default = parse_binary_label(df["default"], "default")
        if not parsed_default.notna().any() and "loan_status" in df.columns:
            parsed_default = parse_loan_status_default(df["loan_status"])
        df["default"] = parsed_default
    elif "loan_status" in df.columns:
        df["default"] = parse_loan_status_default(df["loan_status"])
    else:
        df["default"] = np.nan

    if "approved" in df.columns:
        df["approved"] = parse_binary_label(df["approved"], "approved")
    elif "loan_status" in df.columns:
        df["approved"] = parse_loan_status_approval(df["loan_status"])
    elif df["default"].notna().any():
        # When only completed-loan outcomes are supplied, observed outcomes indicate
        # that the loan was funded. This is an analytical inference, not source data.
        df["approved"] = df["default"].notna().astype(float)
    else:
        df["approved"] = np.nan

    # Create a stable row/loan identifier when absent. Do not invent customer IDs.
    if "loan_id" not in df.columns:
        df["loan_id"] = [f"LOAN_{i:07d}" for i in range(1, len(df) + 1)]
    else:
        df["loan_id"] = df["loan_id"].astype("string").str.strip()
        missing_id = df["loan_id"].isna() | df["loan_id"].eq("")
        if missing_id.any():
            df.loc[missing_id, "loan_id"] = [f"LOAN_MISSING_{i:07d}" for i in range(1, int(missing_id.sum()) + 1)]

    for field in CATEGORICAL_FIELDS:
        if field not in df.columns:
            df[field] = "Unknown"
        df[field] = df[field].astype("string").str.strip().replace({"": pd.NA}).fillna("Unknown")

    # Optional numeric fields should exist consistently for downstream functions.
    for field in NUMERIC_FIELDS:
        if field not in df.columns:
            df[field] = np.nan

    # Audit invalid values before changing them to missing.
    checks: list[dict[str, Any]] = []
    range_rules: list[tuple[str, str, pd.Series]] = []
    if "loan_amount" in df:
        range_rules.append(("loan_amount", "must be greater than 0", df["loan_amount"].notna() & (df["loan_amount"] <= 0)))
    if "annual_income" in df:
        range_rules.append(("annual_income", "must be greater than 0", df["annual_income"].notna() & (df["annual_income"] <= 0)))
    if "credit_score" in df:
        range_rules.append(("credit_score", "expected range 300 to 850", df["credit_score"].notna() & ~df["credit_score"].between(300, 850)))
    if "employment_years" in df:
        range_rules.append(("employment_years", "expected range 0 to 80 years", df["employment_years"].notna() & ~df["employment_years"].between(0, 80)))
    if "debt_to_income_ratio" in df:
        range_rules.append(("debt_to_income_ratio", "expected range 0 to 5 (0% to 500%)", df["debt_to_income_ratio"].notna() & ~df["debt_to_income_ratio"].between(0, 5)))
    if "interest_rate" in df:
        range_rules.append(("interest_rate", "expected range 0 to 1 after normalization", df["interest_rate"].notna() & ~df["interest_rate"].between(0, 1)))
    if "loan_term_months" in df:
        range_rules.append(("loan_term_months", "expected range 1 to 600 months", df["loan_term_months"].notna() & ~df["loan_term_months"].between(1, 600)))
    if "existing_monthly_debt" in df:
        range_rules.append(("existing_monthly_debt", "must be non-negative", df["existing_monthly_debt"].notna() & (df["existing_monthly_debt"] < 0)))
    if "monthly_payment" in df:
        range_rules.append(("monthly_payment", "must be non-negative", df["monthly_payment"].notna() & (df["monthly_payment"] < 0)))
    if "delinquencies_2y" in df:
        range_rules.append(("delinquencies_2y", "must be non-negative", df["delinquencies_2y"].notna() & (df["delinquencies_2y"] < 0)))
    if "credit_history_years" in df:
        range_rules.append(("credit_history_years", "expected range 0 to 100 years", df["credit_history_years"].notna() & ~df["credit_history_years"].between(0, 100)))

    invalid_masks: dict[str, pd.Series] = {}
    for field, rule, mask in range_rules:
        invalid_masks[field] = mask.fillna(False)
        checks.append({
            "column": field,
            "rule": rule,
            "invalid_count": int(mask.sum()),
            "invalid_pct": round(float(mask.mean() * 100), 3),
        })
    range_report = pd.DataFrame(checks, columns=["column", "rule", "invalid_count", "invalid_pct"])

    # Conservative cleaning: impossible values become NaN rather than being clipped
    # to a plausible-looking value. All actions are reported for transparency.
    cleaning_actions: list[dict[str, Any]] = []
    for field, mask in invalid_masks.items():
        count = int(mask.sum())
        if count:
            df.loc[mask, field] = np.nan
            cleaning_actions.append({
                "column": field,
                "action": "Set out-of-range values to missing",
                "rows_affected": count,
            })

    before_duplicates = len(df)
    df = df.drop_duplicates().copy()
    exact_duplicates = before_duplicates - len(df)
    cleaning_actions.append({
        "column": "ALL_COLUMNS",
        "action": "Remove exact duplicate rows",
        "rows_affected": int(exact_duplicates),
    })

    # Ensure default is binary or missing after all parsing.
    df.loc[~df["default"].isin([0.0, 1.0]), "default"] = np.nan
    df.loc[~df["approved"].isin([0.0, 1.0]), "approved"] = np.nan

    return df.reset_index(drop=True), range_report, pd.DataFrame(cleaning_actions)


def generate_synthetic_dataset(n_rows: int = ROWS_DEFAULT, seed: int = RANDOM_SEED_DEFAULT) -> pd.DataFrame:
    """Generate reproducible mock loan applications with known risk relationships.

    The generator creates an illustrative dataset for code demonstration only.
    Its default rates, approval process, and relationships are not calibrated to
    any real financial institution or population.
    """
    if n_rows < 100:
        raise ValueError("Use at least 100 rows for meaningful demonstration analyses.")

    rng = np.random.default_rng(seed)
    start = np.datetime64("2021-01-01")
    end = np.datetime64("2026-01-01")
    total_days = int((end - start).astype("timedelta64[D]").astype(int))
    app_days = rng.integers(0, total_days, size=n_rows)
    app_dates = pd.to_datetime(start + app_days.astype("timedelta64[D]"), utc=True)

    annual_income = np.clip(rng.lognormal(mean=10.85, sigma=0.55, size=n_rows), 18_000, 350_000)
    credit_score = np.clip(np.rint(rng.normal(690, 55, size=n_rows)), 300, 850).astype(int)
    employment_years = np.clip(rng.gamma(shape=2.0, scale=4.0, size=n_rows), 0, 40)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], size=n_rows, p=[0.08, 0.12, 0.43, 0.15, 0.22])
    loan_to_income_draw = rng.beta(2.0, 4.0, size=n_rows) * 1.6 + 0.05
    loan_amount = np.clip(annual_income * loan_to_income_draw, 1_000, 100_000)

    interest_rate = np.clip(
        0.065 + (700 - credit_score) * 0.00055 + rng.normal(0, 0.018, n_rows),
        0.035,
        0.32,
    )
    monthly_rate = interest_rate / 12.0
    payment_factor = np.where(
        monthly_rate > 0,
        monthly_rate / (1 - np.power(1 + monthly_rate, -loan_term_months)),
        1 / loan_term_months,
    )
    monthly_payment = loan_amount * payment_factor

    monthly_income = annual_income / 12.0
    existing_debt_ratio = np.clip(rng.beta(2.0, 7.0, size=n_rows) * 0.75, 0.005, 0.75)
    existing_monthly_debt = monthly_income * existing_debt_ratio
    debt_to_income_ratio = existing_monthly_debt / monthly_income
    delinquencies_2y = np.clip(rng.poisson(0.32, size=n_rows), 0, 6).astype(int)
    credit_history_years = np.clip(rng.gamma(2.5, 3.3, size=n_rows), 0.25, 45)

    home_ownership = rng.choice(
        ["Rent", "Mortgage", "Own", "Other"], size=n_rows, p=[0.37, 0.43, 0.17, 0.03]
    )
    loan_purpose = rng.choice(
        ["Debt consolidation", "Credit card", "Home improvement", "Medical", "Auto", "Education", "Other"],
        size=n_rows,
        p=[0.31, 0.22, 0.13, 0.08, 0.11, 0.06, 0.09],
    )
    region = rng.choice(["Northeast", "Midwest", "South", "West"], size=n_rows, p=[0.18, 0.21, 0.37, 0.24])

    # Simulated application acceptance probability: deliberately transparent and synthetic.
    approval_logit = (
        1.25
        + (credit_score - 680) / 75.0
        - np.maximum(debt_to_income_ratio - 0.30, 0) * 4.0
        - delinquencies_2y * 0.35
        - np.maximum(loan_to_income_draw - 0.75, 0) * 1.2
    )
    approval_probability = 1.0 / (1.0 + np.exp(-np.clip(approval_logit, -20, 20)))
    approved = rng.binomial(1, approval_probability, size=n_rows).astype(float)

    total_dti = debt_to_income_ratio + monthly_payment / monthly_income
    purpose_adjustment = np.select(
        [loan_purpose == "Debt consolidation", loan_purpose == "Medical", loan_purpose == "Education"],
        [0.16, 0.26, 0.10],
        default=0.0,
    )
    default_logit = (
        -2.75
        + (700 - credit_score) / 62.0 * 0.72
        + total_dti * 1.8
        + np.minimum(delinquencies_2y, 4) * 0.32
        + np.maximum(loan_to_income_draw - 0.55, 0) * 1.0
        + (interest_rate - 0.12) * 2.0
        + purpose_adjustment
    )
    default_probability = 1.0 / (1.0 + np.exp(-np.clip(default_logit, -20, 20)))
    generated_default = rng.binomial(1, default_probability, size=n_rows).astype(float)
    generated_default[approved == 0] = np.nan  # Outcome is unobserved for rejected applications.

    # Create a small, controlled amount of missingness to exercise data-quality logic.
    for column, fraction in [("employment_years", 0.025), ("credit_history_years", 0.02), ("region", 0.008)]:
        missing_count = int(n_rows * fraction)
        if missing_count > 0:
            missing_indices = rng.choice(n_rows, size=missing_count, replace=False)
            if column == "region":
                region = region.astype(object)
                region[missing_indices] = None
            elif column == "employment_years":
                employment_years = employment_years.astype(float)
                employment_years[missing_indices] = np.nan
            else:
                credit_history_years = credit_history_years.astype(float)
                credit_history_years[missing_indices] = np.nan

    return pd.DataFrame({
        "loan_id": [f"LN{seed:03d}{i:07d}" for i in range(1, n_rows + 1)],
        "customer_id": [f"CUST{i:07d}" for i in rng.integers(1, max(2, int(n_rows * 0.82)), size=n_rows)],
        "application_date": app_dates,
        "loan_amount": np.round(loan_amount, 2),
        "annual_income": np.round(annual_income, 2),
        "employment_years": np.round(employment_years, 2),
        "credit_score": credit_score,
        "debt_to_income_ratio": np.round(debt_to_income_ratio, 5),
        "interest_rate": np.round(interest_rate, 5),
        "loan_term_months": loan_term_months,
        "existing_monthly_debt": np.round(existing_monthly_debt, 2),
        "monthly_payment": np.round(monthly_payment, 2),
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "region": region,
        "delinquencies_2y": delinquencies_2y,
        "credit_history_years": np.round(credit_history_years, 2),
        "approved": approved,
        "default": generated_default,
    })


def feature_engineering(frame: pd.DataFrame) -> pd.DataFrame:
    """Create transparent, documented analytical features without fitting a model."""
    df = frame.copy()

    # Ratios make borrowing size and payment burden more comparable across incomes.
    monthly_income = df["annual_income"] / 12.0
    df["loan_to_income_ratio"] = np.where(
        df["annual_income"] > 0,
        df["loan_amount"] / df["annual_income"],
        np.nan,
    )
    df["payment_to_monthly_income_ratio"] = np.where(
        monthly_income > 0,
        df["monthly_payment"] / monthly_income,
        np.nan,
    )

    # Estimate total DTI from the existing DTI plus the new payment burden where
    # possible. Fall back to existing debt + payment when that is better supported.
    total_dti_from_ratio = df["debt_to_income_ratio"] + df["payment_to_monthly_income_ratio"].fillna(0)
    total_dti_from_ratio = total_dti_from_ratio.where(df["debt_to_income_ratio"].notna())
    has_monthly_debt_information = df["existing_monthly_debt"].notna() | df["monthly_payment"].notna()
    monthly_obligations = df["existing_monthly_debt"].fillna(0) + df["monthly_payment"].fillna(0)
    total_dti_from_debt = (monthly_obligations / monthly_income).where(
        (monthly_income > 0) & has_monthly_debt_information
    )
    df["total_debt_to_income_ratio"] = total_dti_from_ratio.where(
        df["debt_to_income_ratio"].notna(), total_dti_from_debt
    )
    df["monthly_income"] = monthly_income

    df["credit_score_band"] = pd.cut(
        df["credit_score"],
        bins=[-np.inf, 579, 669, 739, np.inf],
        labels=["Below 580", "580-669", "670-739", "740+"],
        right=True,
    ).astype("string").fillna("Unknown")
    df["dti_band"] = pd.cut(
        df["total_debt_to_income_ratio"],
        bins=[-np.inf, 0.20, 0.35, 0.50, np.inf],
        labels=["Below 20%", "20%-35%", "35%-50%", "50%+"],
        right=False,
    ).astype("string").fillna("Unknown")

    # This rule-based segment is for descriptive comparisons only; it is not a
    # production score and should not be treated as a lending recommendation.
    high_risk = (
        (df["credit_score"] < 620)
        | (df["total_debt_to_income_ratio"] >= 0.50)
        | (df["delinquencies_2y"] >= 2)
    )
    medium_risk = (
        (df["credit_score"] < 680)
        | (df["total_debt_to_income_ratio"] >= 0.35)
        | (df["delinquencies_2y"] >= 1)
    )
    enough_data = df[["credit_score", "total_debt_to_income_ratio", "delinquencies_2y"]].notna().any(axis=1)
    df["illustrative_risk_segment"] = np.select(
        [~enough_data, high_risk.fillna(False), medium_risk.fillna(False)],
        ["Insufficient Data", "Higher Risk", "Moderate Risk"],
        default="Lower Risk",
    )

    df["has_any_delinquency_2y"] = np.where(
        df["delinquencies_2y"].isna(), pd.NA, df["delinquencies_2y"].gt(0)
    )
    df["high_total_dti"] = np.where(
        df["total_debt_to_income_ratio"].isna(), pd.NA, df["total_debt_to_income_ratio"].ge(0.50)
    )

    app_date = pd.to_datetime(df["application_date"], errors="coerce", utc=True)
    df["application_year"] = app_date.dt.year.astype("Int64")
    df["application_month"] = app_date.dt.tz_localize(None).dt.to_period("M").astype("string")
    df.loc[app_date.isna(), "application_month"] = pd.NA

    # Data checks should not silently turn unknown outcomes into non-defaults.
    df["default"] = pd.to_numeric(df["default"], errors="coerce")
    df.loc[~df["default"].isin([0.0, 1.0]), "default"] = np.nan
    df["approved"] = pd.to_numeric(df["approved"], errors="coerce")
    df.loc[~df["approved"].isin([0.0, 1.0]), "approved"] = np.nan

    return df


def schema_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe columns, data types, missing values, and cardinality."""
    return pd.DataFrame({
        "column": frame.columns,
        "dtype": [str(frame[col].dtype) for col in frame.columns],
        "row_count": len(frame),
        "missing_count": [int(frame[col].isna().sum()) for col in frame.columns],
        "missing_pct": [round(float(frame[col].isna().mean() * 100), 3) for col in frame.columns],
        "unique_count_including_null": [int(frame[col].nunique(dropna=False)) for col in frame.columns],
    })


def duplicate_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize exact duplicates and potential duplicate business identifiers."""
    rows = [{
        "check": "Exact duplicate rows after conservative cleaning",
        "count": int(frame.duplicated().sum()),
        "interpretation": "Exact duplicate records can inflate counts; removed during cleaning.",
    }]
    if "loan_id" in frame:
        duplicate_ids = int(frame["loan_id"].duplicated(keep=False).sum())
        rows.append({
            "check": "Rows involved in duplicate loan_id values",
            "count": duplicate_ids,
            "interpretation": "Potential identifier conflicts; reviewed, not automatically removed.",
        })
    if "customer_id" in frame:
        repeated_customers = int(frame.loc[frame["customer_id"].notna(), "customer_id"].duplicated(keep=False).sum())
        rows.append({
            "check": "Rows with a repeated customer_id",
            "count": repeated_customers,
            "interpretation": "Repeat borrowers may be valid; this is not automatically a data-quality error.",
        })
    return pd.DataFrame(rows)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion."""
    if trials <= 0:
        return (np.nan, np.nan)
    p_hat = successes / trials
    denom = 1.0 + z * z / trials
    center = (p_hat + z * z / (2.0 * trials)) / denom
    half_width = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / trials) + (z * z / (4.0 * trials * trials)))
    return max(0.0, center - half_width), min(1.0, center + half_width)


def grouped_rate_table(frame: pd.DataFrame, group_column: str, outcome_column: str = "default") -> pd.DataFrame:
    """Calculate observed default rate and Wilson confidence interval by group."""
    if group_column not in frame.columns or outcome_column not in frame.columns:
        return pd.DataFrame(columns=["group", "observations", "defaults", "default_rate", "ci_95_low", "ci_95_high"])
    data = frame.loc[frame[outcome_column].isin([0.0, 1.0]), [group_column, outcome_column]].copy()
    data[group_column] = data[group_column].astype("string").fillna("Unknown")
    rows: list[dict[str, Any]] = []
    for group_name, group in data.groupby(group_column, dropna=False, sort=False, observed=True):
        n = len(group)
        defaults = int(group[outcome_column].sum())
        low, high = wilson_interval(defaults, n)
        rows.append({
            "group": str(group_name),
            "observations": n,
            "defaults": defaults,
            "default_rate": defaults / n if n else np.nan,
            "ci_95_low": low,
            "ci_95_high": high,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("observations", ascending=False).reset_index(drop=True)
    return result


def probability_analysis(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute base and conditional empirical probabilities with denominators."""
    outcomes = frame.loc[frame["default"].isin([0.0, 1.0])].copy()
    rows: list[dict[str, Any]] = []

    def add_event(label: str, mask: pd.Series, description: str) -> None:
        selected = outcomes.loc[mask.reindex(outcomes.index, fill_value=False)]
        n = len(selected)
        defaults = int(selected["default"].sum()) if n else 0
        rows.append({
            "probability": label,
            "event_definition": description,
            "observations": n,
            "defaults": defaults,
            "empirical_probability": defaults / n if n else np.nan,
            "ci_95_low": wilson_interval(defaults, n)[0],
            "ci_95_high": wilson_interval(defaults, n)[1],
        })

    add_event("P(Default)", pd.Series(True, index=outcomes.index), "Observed defaults / observed terminal outcomes")
    if "delinquencies_2y" in outcomes:
        add_event(
            "P(Default | Any delinquency)",
            outcomes["delinquencies_2y"].gt(0) & outcomes["delinquencies_2y"].notna(),
            "Default among records with at least one reported delinquency in the prior two years",
        )
        add_event(
            "P(Default | No delinquency)",
            outcomes["delinquencies_2y"].eq(0),
            "Default among records with zero reported delinquencies in the prior two years",
        )
    if "total_debt_to_income_ratio" in outcomes:
        add_event(
            "P(Default | Total DTI >= 50%)",
            outcomes["total_debt_to_income_ratio"].ge(0.50) & outcomes["total_debt_to_income_ratio"].notna(),
            "Default among records with an available engineered total DTI of at least 50%",
        )
        add_event(
            "P(Default | Total DTI < 50%)",
            outcomes["total_debt_to_income_ratio"].lt(0.50) & outcomes["total_debt_to_income_ratio"].notna(),
            "Default among records with an available engineered total DTI below 50%",
        )
    if "credit_score" in outcomes:
        add_event(
            "P(Default | Credit score < 620)",
            outcomes["credit_score"].lt(620) & outcomes["credit_score"].notna(),
            "Default among observed outcomes with a credit score below 620",
        )
        add_event(
            "P(Default | Credit score >= 740)",
            outcomes["credit_score"].ge(740) & outcomes["credit_score"].notna(),
            "Default among observed outcomes with a credit score of at least 740",
        )

    base_p = float(outcomes["default"].mean()) if len(outcomes) else np.nan
    default_count = int(outcomes["default"].sum()) if len(outcomes) else 0
    probability_summary = pd.DataFrame([
        {"metric": "outcome_observations", "value": len(outcomes), "definition": "Records with default = 0 or 1"},
        {"metric": "observed_defaults", "value": default_count, "definition": "Records labelled default = 1"},
        {"metric": "empirical_default_probability", "value": base_p, "definition": "Observed default rate among records with known outcomes"},
        {"metric": "outcome_coverage_pct", "value": float(frame["default"].isin([0.0, 1.0]).mean() * 100), "definition": "Percent of all records with a known terminal outcome"},
    ])
    return probability_summary, pd.DataFrame(rows)


def statistical_tests(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run transparent exploratory tests; p-values do not imply causality."""
    outcomes = frame.loc[frame["default"].isin([0.0, 1.0])].copy()
    numeric_candidates = [
        "loan_amount", "annual_income", "employment_years", "credit_score",
        "debt_to_income_ratio", "interest_rate", "loan_term_months",
        "existing_monthly_debt", "monthly_payment", "delinquencies_2y",
        "credit_history_years", "loan_to_income_ratio", "payment_to_monthly_income_ratio",
        "total_debt_to_income_ratio",
    ]
    numeric_rows: list[dict[str, Any]] = []

    for field in numeric_candidates:
        if field not in outcomes:
            continue
        group_default = outcomes.loc[outcomes["default"] == 1, field].dropna().to_numpy(dtype=float)
        group_nondefault = outcomes.loc[outcomes["default"] == 0, field].dropna().to_numpy(dtype=float)
        if len(group_default) < 2 or len(group_nondefault) < 2:
            continue
        mean_default = float(np.mean(group_default))
        mean_nondefault = float(np.mean(group_nondefault))
        sd_default = float(np.std(group_default, ddof=1))
        sd_nondefault = float(np.std(group_nondefault, ddof=1))
        pooled_denominator = len(group_default) + len(group_nondefault) - 2
        pooled_sd = math.sqrt(
            (((len(group_default) - 1) * sd_default**2) + ((len(group_nondefault) - 1) * sd_nondefault**2))
            / pooled_denominator
        ) if pooled_denominator > 0 else np.nan
        effect_size = (mean_default - mean_nondefault) / pooled_sd if pooled_sd and np.isfinite(pooled_sd) else np.nan

        if SCIPY_AVAILABLE:
            statistic, p_value = ttest_ind(group_default, group_nondefault, equal_var=False, nan_policy="omit")
            test_name = "Welch independent-samples t-test"
            test_note = "Exploratory comparison of means; sensitive to outliers and distribution shape."
            statistic_value = float(statistic)
            p_value_num = float(p_value)
        else:
            test_name = "Not run: SciPy unavailable"
            test_note = "Install scipy to calculate the Welch t-test: pip install scipy. Descriptive means are still reported."
            statistic_value = np.nan
            p_value_num = np.nan

        numeric_rows.append({
            "feature": field,
            "test": test_name,
            "n_default": len(group_default),
            "mean_default": mean_default,
            "n_nondefault": len(group_nondefault),
            "mean_nondefault": mean_nondefault,
            "mean_difference_default_minus_nondefault": mean_default - mean_nondefault,
            "cohens_d_approx": effect_size,
            "test_statistic": statistic_value,
            "p_value": p_value_num,
            "interpretation_note": test_note,
        })

    categorical_candidates = ["credit_score_band", "dti_band", "illustrative_risk_segment", "loan_purpose", "home_ownership", "region", "has_any_delinquency_2y"]
    categorical_rows: list[dict[str, Any]] = []
    for field in categorical_candidates:
        if field not in outcomes:
            continue
        valid = outcomes.loc[outcomes[field].notna(), [field, "default"]].copy()
        if valid[field].nunique() < 2 or valid["default"].nunique() < 2:
            continue
        contingency = pd.crosstab(valid[field].astype(str), valid["default"])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue
        if SCIPY_AVAILABLE:
            chi2, p_value, dof, expected = chi2_contingency(contingency, correction=False)
            n = int(contingency.to_numpy().sum())
            denominator = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
            cramers_v = math.sqrt(float(chi2) / (n * denominator)) if denominator > 0 and n > 0 else np.nan
            low_expected = int((expected < 5).sum())
            categorical_rows.append({
                "feature": field,
                "test": "Chi-square test of independence",
                "observations": n,
                "categories": contingency.shape[0],
                "chi_square_statistic": float(chi2),
                "degrees_of_freedom": int(dof),
                "p_value": float(p_value),
                "cramers_v": cramers_v,
                "cells_expected_below_5": low_expected,
                "interpretation_note": "Association test only, not causation. Interpret cautiously when expected cell counts are small.",
            })
        else:
            categorical_rows.append({
                "feature": field,
                "test": "Not run: SciPy unavailable",
                "observations": int(contingency.to_numpy().sum()),
                "categories": contingency.shape[0],
                "chi_square_statistic": np.nan,
                "degrees_of_freedom": np.nan,
                "p_value": np.nan,
                "cramers_v": np.nan,
                "cells_expected_below_5": np.nan,
                "interpretation_note": "Install scipy to run a chi-square test: pip install scipy.",
            })

    return pd.DataFrame(numeric_rows), pd.DataFrame(categorical_rows)


def monthly_risk_trend(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize monthly applications, approvals, outcome coverage, and defaults."""
    df = frame.loc[frame["application_month"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["application_month", "applications", "approved", "known_outcomes", "defaults", "default_rate"])
    rows: list[dict[str, Any]] = []
    for month, group in df.groupby("application_month", sort=True, observed=True):
        known = group.loc[group["default"].isin([0.0, 1.0])]
        rows.append({
            "application_month": str(month),
            "applications": len(group),
            "approved": int(group["approved"].eq(1.0).sum()),
            "known_outcomes": len(known),
            "defaults": int(known["default"].sum()),
            "default_rate": float(known["default"].mean()) if len(known) else np.nan,
        })
    return pd.DataFrame(rows)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a UTF-8 CSV and log its shape."""
    frame.to_csv(path, index=False, encoding="utf-8")
    LOGGER.info("Saved %-42s rows=%s columns=%s", path.name, f"{len(frame):,}", len(frame.columns))


def save_bar_rate_chart(rate_table: pd.DataFrame, title: str, path: Path, ordered_labels: list[str] | None = None) -> None:
    """Save a default-rate bar chart with sample counts."""
    if rate_table.empty:
        LOGGER.warning("Skipping chart %s: no observed outcome data.", path.name)
        return
    data = rate_table.copy()
    if ordered_labels:
        data["_order"] = data["group"].apply(lambda x: ordered_labels.index(x) if x in ordered_labels else len(ordered_labels))
        data = data.sort_values("_order")
    else:
        data = data.sort_values("default_rate", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.bar(data["group"].astype(str), data["default_rate"] * 100)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_ylabel("Observed default rate (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, n, value in zip(bars, data["observations"], data["default_rate"]):
        ax.annotate(
            f"{value:.1%}\nn={n:,}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved chart %s", path.name)


def create_visualizations(frame: pd.DataFrame, output_dir: Path, credit_rate: pd.DataFrame, segment_rate: pd.DataFrame, dti_rate: pd.DataFrame, monthly: pd.DataFrame) -> None:
    """Create reusable visualizations and save them as PNG files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_bar_rate_chart(
        credit_rate,
        "Observed Default Rate by Credit Score Band",
        output_dir / "default_rate_by_credit_score_band.png",
        ["Below 580", "580-669", "670-739", "740+", "Unknown"],
    )
    save_bar_rate_chart(
        segment_rate,
        "Observed Default Rate by Illustrative Risk Segment",
        output_dir / "default_rate_by_risk_segment.png",
        ["Lower Risk", "Moderate Risk", "Higher Risk", "Insufficient Data", "Unknown"],
    )
    save_bar_rate_chart(
        dti_rate,
        "Observed Default Rate by Total Debt-to-Income Band",
        output_dir / "default_rate_by_total_dti_band.png",
        ["Below 20%", "20%-35%", "35%-50%", "50%+", "Unknown"],
    )

    known = frame.loc[frame["default"].isin([0.0, 1.0])].copy()
    if not known.empty and "loan_amount" in known:
        fig, ax = plt.subplots(figsize=(10, 5.8))
        for label, subset in [("Non-default", known.loc[known["default"] == 0, "loan_amount"]), ("Default", known.loc[known["default"] == 1, "loan_amount"])]:
            subset = subset.dropna()
            if len(subset):
                ax.hist(subset, bins=35, alpha=0.58, density=False, label=f"{label} (n={len(subset):,})")
        ax.set_title("Loan Amount Distribution by Observed Outcome", fontsize=14, weight="bold")
        ax.set_xlabel("Loan amount")
        ax.set_ylabel("Number of loans")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "loan_amount_distribution_by_outcome.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("Saved chart loan_amount_distribution_by_outcome.png")

    correlation_fields = [
        "loan_amount", "annual_income", "credit_score", "debt_to_income_ratio",
        "interest_rate", "monthly_payment", "loan_to_income_ratio",
        "payment_to_monthly_income_ratio", "total_debt_to_income_ratio", "default",
    ]
    available = [col for col in correlation_fields if col in frame and frame[col].notna().sum() >= 3]
    if len(available) >= 2:
        corr = frame[available].corr(numeric_only=True)
        fig_width = max(8, 0.85 * len(available))
        fig, ax = plt.subplots(figsize=(fig_width, max(6, 0.75 * len(available))))
        image = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, aspect="auto", cmap="coolwarm")
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr.index, fontsize=8)
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                value = corr.iat[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title("Correlation Matrix of Selected Numeric Features", fontsize=13, weight="bold")
        fig.colorbar(image, ax=ax, shrink=0.8, label="Pearson correlation")
        fig.tight_layout()
        fig.savefig(output_dir / "numeric_correlation_heatmap.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("Saved chart numeric_correlation_heatmap.png")

    if not monthly.empty and monthly["known_outcomes"].sum() > 0:
        monthly_plot = monthly.copy()
        monthly_plot["application_month"] = pd.to_datetime(monthly_plot["application_month"], errors="coerce")
        fig, ax1 = plt.subplots(figsize=(12, 5.8))
        ax1.plot(monthly_plot["application_month"], monthly_plot["applications"], marker="o", linewidth=1.7, label="Applications")
        ax1.set_xlabel("Application month")
        ax1.set_ylabel("Applications")
        ax1.grid(axis="y", alpha=0.25)
        ax2 = ax1.twinx()
        ax2.plot(monthly_plot["application_month"], monthly_plot["default_rate"] * 100, marker="s", linewidth=1.5, label="Observed default rate")
        ax2.set_ylabel("Observed default rate (%)")
        ax1.set_title("Monthly Application Volume and Observed Default Rate", fontsize=14, weight="bold")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        fig.tight_layout()
        fig.savefig(output_dir / "monthly_application_and_default_trend.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("Saved chart monthly_application_and_default_trend.png")

    # Approval rate by credit score, when approval information exists.
    if frame["approved"].isin([0.0, 1.0]).any():
        approved_rows = frame.loc[frame["approved"].isin([0.0, 1.0])]
        approval = (
            approved_rows.groupby("credit_score_band", observed=True)["approved"]
            .agg(applications="size", approvals="sum", approval_rate="mean")
            .reset_index()
        )
        order = ["Below 580", "580-669", "670-739", "740+", "Unknown"]
        approval["_order"] = approval["credit_score_band"].astype(str).apply(lambda x: order.index(x) if x in order else len(order))
        approval = approval.sort_values("_order")
        fig, ax = plt.subplots(figsize=(9.5, 5.4))
        bars = ax.bar(approval["credit_score_band"].astype(str), approval["approval_rate"] * 100)
        ax.set_title("Observed Approval Rate by Credit Score Band", fontsize=14, weight="bold")
        ax.set_ylabel("Approval rate (%)")
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(axis="y", alpha=0.25)
        for bar, n in zip(bars, approval["applications"]):
            ax.annotate(f"n={n:,}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "approval_rate_by_credit_score_band.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("Saved chart approval_rate_by_credit_score_band.png")


def sql_query_catalog() -> dict[str, str]:
    """Return DuckDB-compatible analytical SQL against a registered `loans` table."""
    return {
        "portfolio_overview": """
SELECT
    COUNT(*) AS total_applications,
    SUM(CASE WHEN approved = 1 THEN 1 ELSE 0 END) AS approvals,
    AVG(CASE WHEN approved IS NOT NULL THEN approved END) AS approval_rate,
    SUM(CASE WHEN default = 1 THEN 1 ELSE 0 END) AS observed_defaults,
    COUNT(default) AS known_outcomes,
    AVG(CASE WHEN default IS NOT NULL THEN default END) AS observed_default_rate
FROM loans;
""",
        "default_rate_by_credit_score": """
SELECT
    credit_score_band,
    COUNT(*) AS known_outcomes,
    SUM(default) AS observed_defaults,
    AVG(default) AS observed_default_rate
FROM loans
WHERE default IN (0, 1)
GROUP BY credit_score_band
ORDER BY CASE credit_score_band
    WHEN 'Below 580' THEN 1 WHEN '580-669' THEN 2
    WHEN '670-739' THEN 3 WHEN '740+' THEN 4 ELSE 5 END;
""",
        "default_rate_by_risk_segment": """
SELECT
    illustrative_risk_segment,
    COUNT(*) AS known_outcomes,
    SUM(default) AS observed_defaults,
    AVG(default) AS observed_default_rate
FROM loans
WHERE default IN (0, 1)
GROUP BY illustrative_risk_segment
ORDER BY observed_default_rate DESC;
""",
        "monthly_portfolio_trend": """
SELECT
    DATE_TRUNC('month', application_date) AS application_month,
    COUNT(*) AS applications,
    SUM(CASE WHEN approved = 1 THEN 1 ELSE 0 END) AS approvals,
    COUNT(default) AS known_outcomes,
    SUM(CASE WHEN default = 1 THEN 1 ELSE 0 END) AS observed_defaults,
    AVG(CASE WHEN default IS NOT NULL THEN default END) AS observed_default_rate
FROM loans
WHERE application_date IS NOT NULL
GROUP BY 1
ORDER BY 1;
""",
        "loan_purpose_performance": """
SELECT
    loan_purpose,
    COUNT(*) AS applications,
    AVG(loan_amount) AS average_loan_amount,
    COUNT(default) AS known_outcomes,
    AVG(CASE WHEN default IS NOT NULL THEN default END) AS observed_default_rate
FROM loans
GROUP BY loan_purpose
ORDER BY applications DESC;
""",
        "risk_feature_profile": """
SELECT
    illustrative_risk_segment,
    COUNT(*) AS applications,
    AVG(credit_score) AS average_credit_score,
    AVG(total_debt_to_income_ratio) AS average_total_dti,
    AVG(loan_amount) AS average_loan_amount,
    AVG(CASE WHEN default IS NOT NULL THEN default END) AS observed_default_rate
FROM loans
GROUP BY illustrative_risk_segment
ORDER BY applications DESC;
""",
        "data_quality_missingness": """
SELECT
    COUNT(*) AS row_count,
    SUM(CASE WHEN loan_amount IS NULL THEN 1 ELSE 0 END) AS missing_loan_amount,
    SUM(CASE WHEN annual_income IS NULL THEN 1 ELSE 0 END) AS missing_annual_income,
    SUM(CASE WHEN credit_score IS NULL THEN 1 ELSE 0 END) AS missing_credit_score,
    SUM(CASE WHEN default IS NULL THEN 1 ELSE 0 END) AS missing_default_outcome
FROM loans;
""",
        "largest_loans": """
SELECT loan_id, customer_id, loan_amount, annual_income, credit_score,
       total_debt_to_income_ratio, illustrative_risk_segment, approved, default
FROM loans
ORDER BY loan_amount DESC NULLS LAST
LIMIT 25;
""",
        "windowed_monthly_default_trend": """
WITH monthly AS (
    SELECT DATE_TRUNC('month', application_date) AS application_month,
           COUNT(default) AS known_outcomes,
           AVG(CASE WHEN default IS NOT NULL THEN default END) AS observed_default_rate
    FROM loans
    WHERE application_date IS NOT NULL
    GROUP BY 1
)
SELECT application_month, known_outcomes, observed_default_rate,
       AVG(observed_default_rate) OVER (
           ORDER BY application_month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
       ) AS rolling_3_month_average_default_rate
FROM monthly
ORDER BY application_month;
""",
    }


def execute_sql_queries(frame: pd.DataFrame, output_dir: Path, queries: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Export SQL always; execute it using DuckDB if available."""
    sql_path = output_dir / "sql_queries.sql"
    with sql_path.open("w", encoding="utf-8") as file:
        file.write("-- Banking & Financial Risk Analytics SQL catalog\n")
        file.write("-- DuckDB-compatible queries. Execute after registering/loading the engineered\n")
        file.write("-- dataset as a table named `loans`. Results are descriptive, not causal.\n\n")
        for name, query in queries.items():
            file.write(f"-- Query: {name}\n{query.strip()}\n\n")
    LOGGER.info("Saved SQL query catalog: %s", sql_path)

    results: dict[str, pd.DataFrame] = {}
    if not DUCKDB_AVAILABLE:
        LOGGER.warning("DuckDB is not installed. SQL was exported but not executed. Install with: pip install duckdb")
        return results

    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("loans", frame)
        for name, query in queries.items():
            try:
                result = connection.execute(query).df()
                results[name] = result
                save_csv(result, output_dir / f"sql_result_{name}.csv")
            except Exception as exc:  # Keep one failed query from blocking the rest.
                LOGGER.exception("SQL query failed (%s): %s", name, exc)
                results[name] = pd.DataFrame({"query_name": [name], "error": [str(exc)]})
    finally:
        connection.close()
    return results


def build_executive_summary(frame: pd.DataFrame, probability_summary: pd.DataFrame, quality: pd.DataFrame, range_report: pd.DataFrame, output_dir: Path) -> str:
    """Write a human-readable executive summary with coverage caveats."""
    total = len(frame)
    approved_known = frame.loc[frame["approved"].isin([0.0, 1.0]), "approved"]
    outcomes = frame.loc[frame["default"].isin([0.0, 1.0]), "default"]
    observed_defaults = int(outcomes.sum()) if len(outcomes) else 0
    default_rate = float(outcomes.mean()) if len(outcomes) else np.nan
    approval_rate = float(approved_known.mean()) if len(approved_known) else np.nan
    missing_outcomes = total - len(outcomes)
    duplicate_loan_ids = int(frame["loan_id"].duplicated(keep=False).sum()) if "loan_id" in frame else 0
    invalid_count = int(range_report["invalid_count"].sum()) if not range_report.empty else 0

    def fmt_pct(value: float) -> str:
        return f"{value:.2%}" if pd.notna(value) else "Not available"

    lines = [
        "BANKING & FINANCIAL RISK ANALYTICS | EXECUTIVE SUMMARY",
        "=" * 60,
        "",
        "DATASET SCOPE",
        f"Rows after exact-duplicate removal: {total:,}",
        f"Columns after feature engineering: {len(frame.columns):,}",
        f"Records with a known approval status: {len(approved_known):,}",
        f"Records with a known terminal default outcome: {len(outcomes):,}",
        f"Records with an unknown/missing default outcome: {missing_outcomes:,}",
        "",
        "PORTFOLIO INDICATORS",
        f"Observed approval rate among known approval labels: {fmt_pct(approval_rate)}",
        f"Observed default rate among known terminal outcomes: {fmt_pct(default_rate)}",
        f"Observed default count: {observed_defaults:,}",
        "",
        "DATA QUALITY",
        f"Fields with at least one missing value: {int((quality['missing_count'] > 0).sum()) if not quality.empty else 0}",
        f"Invalid range values flagged before conservative cleaning: {invalid_count:,}",
        f"Rows involved in duplicate loan_id values: {duplicate_loan_ids:,}",
        "Exact duplicate rows are removed; duplicate loan IDs are flagged but not silently removed.",
        "",
        "INTERPRETATION NOTES",
        "1. The default rate uses only records with terminal outcomes coded 0 or 1.",
        "   Open, current, rejected, or otherwise unresolved outcomes should not be treated as non-defaults.",
        "2. Confidence intervals and group rates are empirical estimates; small groups can be unstable.",
        "3. Statistical associations do not establish cause and effect.",
        "4. Illustrative risk segments are transparent heuristics, not a credit score or underwriting policy.",
        "5. Synthetic-demo figures are for software validation only and are not real-world risk benchmarks.",
        "6. Review source definitions, units, outcome timing, missingness, and regulatory requirements before production use.",
        "",
        "ARTIFACTS",
        "See the output directory for CSV tables, PNG visualizations, sql_queries.sql, and engineered data.",
    ]
    text = "\n".join(lines)
    (output_dir / "executive_summary.txt").write_text(text, encoding="utf-8")
    return text


def run_pipeline(args: argparse.Namespace) -> Path:
    """Execute the complete analytical workflow and write all artifacts."""
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        input_path = Path(args.input).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_path}")
        LOGGER.info("Loading input CSV: %s", input_path)
        raw = pd.read_csv(input_path, low_memory=False)
        source_description = f"User-supplied CSV: {input_path}"
    else:
        LOGGER.info("No input CSV supplied; generating reproducible synthetic data.")
        raw = generate_synthetic_dataset(n_rows=args.rows, seed=args.seed)
        synthetic_path = output_dir / "synthetic_input_data.csv"
        raw.to_csv(synthetic_path, index=False)
        source_description = f"Synthetic demo data, rows={args.rows:,}, seed={args.seed}"
        LOGGER.info("Saved synthetic source data: %s", synthetic_path)

    if not {"loan_amount", "annual_income", "credit_score"}.issubset({ALIASES.get(c, c) for c in [normalize_column_name(x) for x in raw.columns]}):
        normalized_preview = sorted({ALIASES.get(normalize_column_name(c), normalize_column_name(c)) for c in raw.columns})
        raise ValueError(
            "Input is missing one or more required analytical columns: loan_amount, annual_income, credit_score. "
            f"Detected columns after alias mapping: {normalized_preview}"
        )

    cleaned, range_report, cleaning_actions = standardize_and_clean(raw)
    if cleaned.empty:
        raise ValueError("No rows remain after cleaning.")

    engineered = feature_engineering(cleaned)
    quality_report = schema_audit(engineered)
    duplicates = duplicate_summary(engineered)

    LOGGER.info("Running probability analysis.")
    prob_summary, conditional_probabilities = probability_analysis(engineered)
    credit_rate = grouped_rate_table(engineered, "credit_score_band")
    segment_rate = grouped_rate_table(engineered, "illustrative_risk_segment")
    dti_rate = grouped_rate_table(engineered, "dti_band")
    purpose_rate = grouped_rate_table(engineered, "loan_purpose")
    ownership_rate = grouped_rate_table(engineered, "home_ownership")
    region_rate = grouped_rate_table(engineered, "region")
    monthly = monthly_risk_trend(engineered)

    # Descriptive statistics are kept separate from inferential tests.
    numeric_descriptives = engineered.select_dtypes(include=[np.number]).describe().transpose().reset_index(names="field")
    numeric_descriptives["missing_count"] = [int(engineered[field].isna().sum()) for field in numeric_descriptives["field"]]
    numeric_descriptives["missing_pct"] = [float(engineered[field].isna().mean() * 100) for field in numeric_descriptives["field"]]
    correlation_fields = [
        "loan_amount", "annual_income", "credit_score", "debt_to_income_ratio",
        "interest_rate", "monthly_payment", "loan_to_income_ratio",
        "payment_to_monthly_income_ratio", "total_debt_to_income_ratio", "default",
    ]
    correlation_fields = [c for c in correlation_fields if c in engineered and engineered[c].notna().sum() >= 3]
    correlation = engineered[correlation_fields].corr(numeric_only=True) if len(correlation_fields) >= 2 else pd.DataFrame()

    LOGGER.info("Running exploratory statistical tests.")
    welch_results, chi_square_results = statistical_tests(engineered)

    # Output all key artifacts. Keep raw source (for synthetic mode), cleaned features,
    # quality audits, analytical tables, and plots distinct for reproducibility.
    save_csv(quality_report, output_dir / "data_quality_schema_audit.csv")
    save_csv(duplicates, output_dir / "data_quality_duplicate_summary.csv")
    save_csv(range_report, output_dir / "data_quality_range_checks.csv")
    save_csv(cleaning_actions, output_dir / "data_quality_cleaning_actions.csv")
    save_csv(engineered, output_dir / "engineered_loan_dataset.csv")
    save_csv(numeric_descriptives, output_dir / "numeric_descriptive_statistics.csv")
    save_csv(prob_summary, output_dir / "probability_summary.csv")
    save_csv(conditional_probabilities, output_dir / "conditional_probabilities.csv")
    save_csv(credit_rate, output_dir / "default_rates_by_credit_score_band.csv")
    save_csv(segment_rate, output_dir / "default_rates_by_risk_segment.csv")
    save_csv(dti_rate, output_dir / "default_rates_by_total_dti_band.csv")
    save_csv(purpose_rate, output_dir / "default_rates_by_loan_purpose.csv")
    save_csv(ownership_rate, output_dir / "default_rates_by_home_ownership.csv")
    save_csv(region_rate, output_dir / "default_rates_by_region.csv")
    save_csv(monthly, output_dir / "monthly_risk_trend.csv")
    save_csv(welch_results, output_dir / "statistics_welch_tests.csv")
    save_csv(chi_square_results, output_dir / "statistics_chi_square_tests.csv")
    if not correlation.empty:
        correlation.to_csv(output_dir / "numeric_correlation_matrix.csv", encoding="utf-8")
        LOGGER.info("Saved numeric_correlation_matrix.csv")

    create_visualizations(engineered, output_dir, credit_rate, segment_rate, dti_rate, monthly)
    query_results = execute_sql_queries(engineered, output_dir, sql_query_catalog())

    summary = build_executive_summary(engineered, prob_summary, quality_report, range_report, output_dir)
    metadata = [
        "PROJECT RUN METADATA",
        "====================",
        f"Source: {source_description}",
        f"Rows after cleaning: {len(engineered):,}",
        f"Seed: {args.seed if not args.input else 'not applicable'}",
        f"SciPy available: {SCIPY_AVAILABLE}",
        f"DuckDB available: {DUCKDB_AVAILABLE}",
        f"SQL query results exported: {len(query_results)}",
        "Synthetic-data caution: if the source is synthetic, its risk rates are not real-world benchmarks.",
        "",
        summary,
    ]
    (output_dir / "run_report.txt").write_text("\n".join(metadata), encoding="utf-8")

    LOGGER.info("Pipeline complete. Artifacts written to: %s", output_dir)
    print("\n" + summary)
    print(f"\nOutput directory: {output_dir}")
    return output_dir


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="End-to-end Banking & Financial Risk Analytics project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, default=None, help="Optional input loan CSV. If omitted, synthetic data is generated.")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT, help="Directory for CSV outputs, charts, SQL, and reports.")
    parser.add_argument("--rows", type=int, default=ROWS_DEFAULT, help="Number of synthetic records when --input is not supplied.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED_DEFAULT, help="Random seed for reproducible synthetic data.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point with actionable error messages."""
    configure_logging()
    args = parse_args(argv)
    try:
        run_pipeline(args)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        LOGGER.error("Could not complete analysis: %s", exc)
        return 2
    except Exception as exc:  # Keep unexpected failures visible with a traceback in logs.
        LOGGER.exception("Unexpected pipeline failure: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
