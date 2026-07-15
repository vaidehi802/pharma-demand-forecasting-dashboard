"""
generate_data.py
----------------
Creates a realistic SYNTHETIC pharma commercial dataset:
  1. sales_data.csv        - monthly unit sales by Region x Product (48 months / 4 years)
  2. physician_data.csv    - physician-level prescribing behavior features + adherence/
                             under-prescribing labels, for the ML classification model.

This mimics the structure of real-world sources like CMS Medicare Part D Public Use
Files and commercial pharma syndicated data (IQVIA-style), without using any real
patient or prescriber information. All numbers are simulated.
"""

import numpy as np
import pandas as pd
from datetime import datetime

RNG = np.random.default_rng(42)

REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
PRODUCTS = ["CardiaFlow", "GlucoBalance", "NeuroCalm", "OncoShield", "RespirEase"]

N_MONTHS = 48  # 4 years of monthly history
START = "2022-01-01"


def generate_sales_data() -> pd.DataFrame:
    dates = pd.date_range(START, periods=N_MONTHS, freq="MS")
    rows = []

    for region in REGIONS:
        region_base = RNG.uniform(8000, 20000)   # base demand level per region
        region_trend = RNG.uniform(0.002, 0.012)  # slow secular growth
        for product in PRODUCTS:
            product_scale = RNG.uniform(0.6, 1.4)
            seasonality_amp = RNG.uniform(0.05, 0.18)
            noise_scale = RNG.uniform(0.03, 0.08)

            for i, d in enumerate(dates):
                trend = region_base * product_scale * (1 + region_trend) ** i
                # flu-season / refill-cycle style seasonality (peaks in winter months)
                season = 1 + seasonality_amp * np.sin(2 * np.pi * (d.month - 1) / 12 + 1.0)
                noise = RNG.normal(1, noise_scale)
                units = max(0, trend * season * noise)

                rows.append({
                    "date": d,
                    "region": region,
                    "product": product,
                    "units_sold": round(units),
                    "net_sales_usd": round(units * RNG.uniform(28, 65), 2),
                })

    df = pd.DataFrame(rows)
    return df.sort_values(["region", "product", "date"]).reset_index(drop=True)


def generate_physician_data(n_physicians: int = 1500) -> pd.DataFrame:
    """
    One row per physician per quarter, with features used to predict
    UNDER-PRESCRIBING (physician segment whose prescribing rate for a target
    therapy is well below what their patient panel would predict) - a real,
    ZS-style commercial analytics use case (HCP segmentation / call-plan targeting).
    """
    specialties = ["Cardiology", "Endocrinology", "Primary Care", "Neurology", "Pulmonology"]
    quarters = [f"{y}-Q{q}" for y in [2023, 2024, 2025] for q in range(1, 5)]

    rows = []
    for pid in range(1, n_physicians + 1):
        region = RNG.choice(REGIONS)
        specialty = RNG.choice(specialties)
        panel_size = int(np.clip(RNG.normal(1200, 350), 200, 3000))
        years_experience = int(RNG.uniform(1, 35))
        digital_engagement = RNG.uniform(0, 1)     # opens emails, uses portal, etc.
        rep_visit_freq = RNG.poisson(3)             # rep visits per quarter
        formulary_access = RNG.uniform(0.4, 1.0)    # % patients with covered access
        peer_influence_score = RNG.uniform(0, 1)    # KOL network proximity

        # Epidemiology-driven benchmark: how many scripts this panel SHOULD generate,
        # independent of the physician's own behavior (disease prevalence * panel size).
        expected_scripts_base = panel_size * 0.04

        # Commercial engagement/access features drive ACTUAL behavior relative to that
        # benchmark - this is the learnable signal for the classifier.
        engagement_index = (
            0.35 * digital_engagement
            + 0.30 * (rep_visit_freq / 6)
            + 0.25 * formulary_access
            + 0.10 * peer_influence_score
        )  # roughly in [0, 1], higher = more aligned with expected prescribing

        for q in quarters:
            # actual scripts scale with engagement_index around the epidemiology benchmark,
            # plus quarter-to-quarter noise (patient mix, sampling variation, etc.)
            behavior_multiplier = 0.55 + 0.9 * engagement_index  # ranges ~0.55x - 1.45x
            expected_scripts = expected_scripts_base
            actual_scripts = max(
                0, RNG.normal(expected_scripts_base * behavior_multiplier, expected_scripts_base * 0.15)
            )

            gap_ratio = actual_scripts / (expected_scripts + 1e-6)
            under_prescriber = int(gap_ratio < 0.75)

            rows.append({
                "physician_id": pid,
                "quarter": q,
                "region": region,
                "specialty": specialty,
                "panel_size": panel_size,
                "years_experience": years_experience,
                "digital_engagement_score": round(digital_engagement, 3),
                "rep_visits_per_quarter": rep_visit_freq,
                "formulary_access_pct": round(formulary_access, 3),
                "peer_influence_score": round(peer_influence_score, 3),
                "expected_scripts": round(expected_scripts, 1),
                "actual_scripts": round(actual_scripts, 1),
                "under_prescriber": under_prescriber,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    sales = generate_sales_data()
    physicians = generate_physician_data()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sales.to_csv(os.path.join(base_dir, "data", "sales_data.csv"), index=False)
    physicians.to_csv(os.path.join(base_dir, "data", "physician_data.csv"), index=False)

    print(f"sales_data.csv        -> {sales.shape[0]:,} rows, {sales['date'].min().date()} to {sales['date'].max().date()}")
    print(f"physician_data.csv    -> {physicians.shape[0]:,} rows, {physicians['physician_id'].nunique()} physicians")
    print(f"under_prescriber rate -> {physicians['under_prescriber'].mean():.1%}")
