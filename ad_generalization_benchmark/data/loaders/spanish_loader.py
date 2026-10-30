"""Spanish Target Cohort Loader.

Loads clinical metadata from Clinical_data.zip (clinical_data_id_age_gender.csv),
filters diagnostic groups (ea -> 1 [AD], crl -> 0 [HC], tb -> excluded),
maps standardized demographic features, lists T1 image paths from T1_preprocessed.zip,
and matches clinical IDs to image files.
"""

from pathlib import Path
import tempfile
from typing import Optional, Tuple, Union
import zipfile
import pandas as pd

from ad_generalization_benchmark.configs import resolve_dataset_path


def apply_spanish_label_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Apply label rule to Spanish cohort clinical DataFrame.

    ea  -> label 1 (Alzheimer's disease)
    crl -> label 0 (healthy control)
    tb  -> excluded (bipolar disorder)
    Returns new DataFrame with 'label' column, tb rows removed.
    """
    df = df.copy()
    col = "group" if "group" in df.columns else "Subject"
    df = df[df[col] != "tb"].copy()
    df["label"] = df[col].map({"ea": 1, "crl": 0})
    return df


def load_spanish_clinical(
    clinical_zip_path: Optional[Union[str, Path]] = None,
    mri_zip_or_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load and prepare Spanish cohort clinical data and matched image paths.

    Requirements:
      - Reads clinical_data_id_age_gender.csv
      - Filter: keep only ea (label=1) and crl (label=0), drop tb
      - Return subject IDs, labels, age, sex (F=0, M=1)
      - List T1 image paths and match clinical IDs
      - Print audit: n_AD, n_HC, n_bipolar_excluded, n_matched_images
      - STOP with a clear error if fewer than 10 subjects match

    Returns:
      pd.DataFrame with columns: [subject_id, label, age, sex, image_path, diagnostic_group]
    """
    if clinical_zip_path is None:
        clinical_zip_path = resolve_dataset_path("spanish_clinical_zip")
    else:
        clinical_zip_path = Path(clinical_zip_path)

    if not clinical_zip_path.exists():
        raise FileNotFoundError(f"Spanish clinical zip not found at: {clinical_zip_path}")

    # Read clinical_data_id_age_gender.csv directly from zip
    with zipfile.ZipFile(clinical_zip_path, "r") as z:
        if "clinical_data_id_age_gender.csv" not in z.namelist():
            raise FileNotFoundError("clinical_data_id_age_gender.csv not found in clinical zip")
        with z.open("clinical_data_id_age_gender.csv") as f:
            raw_df = pd.read_csv(f)

    n_total_clinical = len(raw_df)

    # Count bipolar and diagnostic groups
    group_counts = raw_df["Subject"].value_counts().to_dict()
    n_bipolar_excluded = group_counts.get("tb", 0)

    # Apply standalone Spanish label rule
    df = apply_spanish_label_rule(raw_df)

    # Create standardized subject_id (e.g. ea_217, crl_104)
    df["subject_id"] = df["Subject"].astype(str) + "_" + df["id"].astype(str)
    df["diagnostic_group"] = df["Subject"]

    # Standardize age
    df["age"] = df["Age"].astype(float)

    # Standardize sex: gender=1 is mujer (Female=0), gender=2 is hombre (Male=1)
    df["sex"] = df["gender"].map({1: 0, 2: 1})
    if df["sex"].isna().any():
        raise ValueError("Invalid gender values found in Spanish clinical dataset")

    # Match image paths from T1_preprocessed.zip or extracted directory
    if mri_zip_or_dir is None:
        mri_path = resolve_dataset_path("spanish_mri_zip")
    else:
        mri_path = Path(mri_zip_or_dir)

    matched_images = {}
    if mri_path.is_file() and mri_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(mri_path, "r") as z:
            for item in z.namelist():
                if item.startswith("T1_original/") and item.endswith(".nii.gz"):
                    fname = item.split("/")[-1]
                    subj_key = fname.replace(".nii.gz", "")
                    matched_images[subj_key] = item
    elif mri_path.is_dir():
        for p in mri_path.glob("**/T1_original/*.nii*"):
            fname = p.name
            subj_key = fname.split(".")[0]
            matched_images[subj_key] = str(p)

    df["image_path"] = df["subject_id"].map(matched_images)
    n_matched_images = int(df["image_path"].notna().sum())

    n_ad = int((df["label"] == 1).sum())
    n_hc = int((df["label"] == 0).sum())

    if verbose:
        print("=== Spanish Cohort Loader Audit Summary ===")
        print(f"Total clinical records: {n_total_clinical}")
        print(f"AD subjects (ea): {n_ad} (label=1)")
        print(f"HC subjects (crl): {n_hc} (label=0)")
        print(f"Bipolar subjects excluded (tb): {n_bipolar_excluded}")
        print(f"Matched T1 image paths: {n_matched_images} / {len(df)}")
        print("===========================================")

    # Acceptance / Safety constraint: STOP if fewer than 10 subjects match
    if n_matched_images < 10:
        raise ValueError(
            f"Insufficient matched subjects: only {n_matched_images} matched. "
            "At least 10 required to proceed."
        )

    # Return tidy DataFrame
    cols = ["subject_id", "label", "age", "sex", "image_path", "diagnostic_group"]
    return df[cols].reset_index(drop=True)


if __name__ == "__main__":
    df_spanish = load_spanish_clinical(verbose=True)
    print(f"\nSpanish loader returned {len(df_spanish)} valid subjects:")
    print(df_spanish.head(10))
