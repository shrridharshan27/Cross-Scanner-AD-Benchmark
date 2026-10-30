"""OASIS-1 Dataset Loader.

Loads oasis_cross-sectional xlsx file, applies prespecified eligibility and
labeling rules, extracts allowed features, and prevents data leakage.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd

from ad_generalization_benchmark.configs import resolve_dataset_path

FORBIDDEN_FEATURES = ["MMSE", "CDR", "ASF", "Educ", "SES", "Delay", "Hand"]


def apply_primary_label_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Apply primary binary label rule to a DataFrame with CDR column.

    CDR == 0.0 -> label 0 (CN)
    CDR >= 1.0 -> label 1 (AD)
    CDR == 0.5 -> excluded (dropped)
    Returns new DataFrame with 'label' column added, CDR=0.5 rows removed.
    """
    df = df.copy()
    df = df[df["CDR"] != 0.5].copy()
    df["label"] = df["CDR"].apply(lambda x: 0 if x == 0.0 else 1)
    return df


def load_oasis1(
    excel_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load OASIS-1 dataset and return primary and secondary DataFrames.

    Eligibility rule:
        Include if: Age >= 60 AND CDR is not NaN (198 subjects)

    Label rules:
        Primary (strict):
            CDR == 0.0 -> CN (label=0)
            CDR >= 1.0 -> AD (label=1)
            CDR == 0.5 -> Excluded (MCI)
            Total: 128 subjects (98 CN, 30 AD)

        Secondary (broad):
            CDR == 0.0 -> CN (label=0)
            CDR >= 0.5 -> AD (label=1)
            Total: 198 subjects (98 CN, 100 AD)

    Features:
        nWBV (normalized whole brain volume)
        eTIV (estimated total intracranial volume)
        age  (age at scan)
        sex  (M/F encoded as F=0, M=1)

    Returns:
        tuple (df_primary, df_secondary):
            df_primary: DataFrame with 128 rows, columns [subject_id, label, nWBV, eTIV, age, sex]
            df_secondary: DataFrame with 198 rows, columns [subject_id, label, nWBV, eTIV, age, sex]
    """
    if excel_path is None:
        excel_path = resolve_dataset_path("oasis1_xlsx")
    else:
        excel_path = Path(excel_path)

    if not excel_path.exists():
        raise FileNotFoundError(f"OASIS-1 Excel file not found at: {excel_path}")

    raw_df = pd.read_excel(excel_path)
    n_total = len(raw_df)

    # 1. Eligibility filtering: Age >= 60 and CDR not NaN
    eligible_mask = (raw_df["Age"] >= 60) & (raw_df["CDR"].notna())
    df_eligible = raw_df[eligible_mask].copy()
    n_eligible = len(df_eligible)

    # 2. Encode sex: F -> 0, M -> 1
    df_eligible["sex"] = df_eligible["M/F"].map({"F": 0, "M": 1})
    if df_eligible["sex"].isna().any():
        raise ValueError("Encountered unmapped sex value in OASIS-1 dataset")

    # Rename age and subject ID
    df_eligible["age"] = df_eligible["Age"]
    df_eligible["subject_id"] = df_eligible["ID"].astype(str).str.strip()

    # 3. Primary label rule (strict):
    # Using standalone apply_primary_label_rule
    df_primary_labeled = apply_primary_label_rule(df_eligible)
    # Filter to valid CDR in primary (CDR 0, 1, 2)
    df_primary = df_primary_labeled[df_primary_labeled["CDR"].isin([0.0, 1.0, 2.0])].copy()

    n_cn_primary = int((df_primary["label"] == 0).sum())
    n_ad_primary = int((df_primary["label"] == 1).sum())
    n_excluded_primary = n_total - len(df_primary)

    df_primary = df_primary[["subject_id", "label", "nWBV", "eTIV", "age", "sex"]].reset_index(drop=True)

    # 4. Secondary label rule (broad):
    # CDR == 0.0 -> CN (0), CDR >= 0.5 -> AD (1)
    df_secondary = df_eligible.copy()
    df_secondary["label"] = df_secondary["CDR"].apply(lambda cdr: 0 if cdr == 0.0 else 1)
    df_secondary = df_secondary[["subject_id", "label", "nWBV", "eTIV", "age", "sex"]].reset_index(drop=True)

    # 5. Leakage check on returned DataFrames
    for forbidden in FORBIDDEN_FEATURES:
        assert forbidden not in df_primary.columns, f"Leakage violation: '{forbidden}' in primary features"
        assert forbidden not in df_secondary.columns, f"Leakage violation: '{forbidden}' in secondary features"

    # Verify no nulls in required columns
    assert df_primary.isna().sum().sum() == 0, "Null values detected in OASIS-1 primary DataFrame"
    assert df_secondary.isna().sum().sum() == 0, "Null values detected in OASIS-1 secondary DataFrame"

    if verbose:
        print("=== OASIS-1 Loader Audit Summary ===")
        print(f"Total subjects in file: {n_total}")
        print(f"Eligible subjects (Age >= 60 & CDR not null): {n_eligible}")
        print(f"Primary cohort (strict): {len(df_primary)} subjects ({n_cn_primary} CN, {n_ad_primary} AD)")
        print(f"Excluded from primary (Age < 60, CDR NaN, or CDR=0.5 MCI): {n_excluded_primary}")
        print(f"Secondary cohort (broad): {len(df_secondary)} subjects")
        print(f"Columns returned: {list(df_primary.columns)}")
        print("====================================")

    return df_primary, df_secondary


if __name__ == "__main__":
    df_p, df_s = load_oasis1(verbose=True)
    print(f"Primary DataFrame shape: {df_p.shape}")
    print(f"Secondary DataFrame shape: {df_s.shape}")
    print(df_p.head())
