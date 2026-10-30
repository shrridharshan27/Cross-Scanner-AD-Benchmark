"""Audit script for OASIS-1 dataset.

Inspects raw Excel file, verifies eligibility rules, checks CDR distributions,
null counts, and verifies acceptance counts:
  436 total, 198 eligible, 98 CN, 30 AD primary
"""

from pathlib import Path
import pandas as pd
import numpy as np

from ad_generalization_benchmark.configs import resolve_dataset_path


def audit_oasis1(excel_path=None) -> dict:
    """Run full audit on OASIS-1 cross-sectional dataset."""
    if excel_path is None:
        excel_path = resolve_dataset_path("oasis1_xlsx")
    else:
        excel_path = Path(excel_path)

    print(f"Reading OASIS-1 dataset from: {excel_path}")
    raw_df = pd.read_excel(excel_path)

    n_total = len(raw_df)

    # 1. Eligibility Rule: Age >= 60 and CDR not NaN
    eligible_mask = (raw_df["Age"] >= 60) & (raw_df["CDR"].notna())
    df_eligible = raw_df[eligible_mask]
    n_eligible = len(df_eligible)

    # CDR distribution in eligible subjects
    cdr_counts = df_eligible["CDR"].value_counts().to_dict()

    # 2. Primary Label Rule:
    # CDR == 0.0 -> CN (0)
    # CDR >= 1.0 -> AD (1)
    # CDR == 0.5 -> Excluded (MCI)
    n_cn_primary = int((df_eligible["CDR"] == 0.0).sum())
    n_ad_primary = int((df_eligible["CDR"] >= 1.0).sum())
    n_mci_excluded = int((df_eligible["CDR"] == 0.5).sum())
    n_primary = n_cn_primary + n_ad_primary

    # 3. Secondary Label Rule:
    # CDR == 0.0 -> CN (0)
    # CDR >= 0.5 -> AD (1)
    n_secondary = len(df_eligible)
    n_ad_secondary = int((df_eligible["CDR"] >= 0.5).sum())

    # 4. Null value checks on structural features
    null_counts = {
        "nWBV": int(df_eligible["nWBV"].isna().sum()),
        "eTIV": int(df_eligible["eTIV"].isna().sum()),
        "Age": int(df_eligible["Age"].isna().sum()),
        "M/F": int(df_eligible["M/F"].isna().sum()),
        "MMSE": int(df_eligible["MMSE"].isna().sum()),
        "Educ": int(df_eligible["Educ"].isna().sum()),
        "SES": int(df_eligible["SES"].isna().sum()),
    }

    # Print exact required acceptance string
    print(f"\nAudit Summary: {n_total} total, {n_eligible} eligible, {n_cn_primary} CN, {n_ad_primary} AD primary\n")

    print("Detailed Audit Report:")
    print("-" * 50)
    print(f"Total subjects in raw file:        {n_total}")
    print(f"Eligible subjects (Age >= 60, CDR): {n_eligible}")
    print(f"Primary strict binary:              {n_primary} (98 CN + 30 AD)")
    print(f"  - Cognitively Normal (CDR=0.0):  {n_cn_primary}")
    print(f"  - Alzheimer's Disease (CDR>=1.0):{n_ad_primary} (CDR 1.0: {cdr_counts.get(1.0, 0)}, CDR 2.0: {cdr_counts.get(2.0, 0)})")
    print(f"  - MCI Excluded (CDR=0.5):        {n_mci_excluded}")
    print(f"Secondary broad binary:             {n_secondary} (98 CN + {n_ad_secondary} AD)")
    print("-" * 50)
    print("CDR Distribution in Eligible Cohort:")
    for cdr_val, count in sorted(cdr_counts.items()):
        print(f"  CDR = {cdr_val}: {count}")
    print("-" * 50)
    print("Null value counts in Eligible Cohort:")
    for col, nulls in null_counts.items():
        print(f"  {col}: {nulls} missing")
    print("-" * 50)

    # Acceptance criteria assertions
    assert n_total == 436, f"Expected 436 total subjects, found {n_total}"
    assert n_eligible == 198, f"Expected 198 eligible subjects, found {n_eligible}"
    assert n_cn_primary == 98, f"Expected 98 primary CN subjects, found {n_cn_primary}"
    assert n_ad_primary == 30, f"Expected 30 primary AD subjects, found {n_ad_primary}"

    return {
        "n_total": n_total,
        "n_eligible": n_eligible,
        "n_cn_primary": n_cn_primary,
        "n_ad_primary": n_ad_primary,
        "n_mci_excluded": n_mci_excluded,
        "n_primary": n_primary,
        "null_counts": null_counts,
        "cdr_counts": cdr_counts,
    }


if __name__ == "__main__":
    audit_oasis1()
