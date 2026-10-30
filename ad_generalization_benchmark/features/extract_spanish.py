"""Spanish Cohort Feature Extraction Pipeline (Option C: SPM Segmentations).

Extracts brain volumes and ICV estimates from pre-computed SPM 1.5mm tissue
segmentations (m0wrp1 GM maps, m0wrp2 WM maps, wmrf skull-stripped brains)
matched to clinical records from clinical_data_id_age_gender.csv.
"""

from pathlib import Path
import gc
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

from ad_generalization_benchmark.configs import resolve_dataset_path


def get_spanish_paths(
    spm_base_dir: Optional[Union[str, Path]] = None,
    clinical_csv_path: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path, Path, Path]:
    """Resolve and validate Spanish cohort directories and clinical CSV."""
    if clinical_csv_path is None:
        clin_candidates = [
            Path("datasets/Clinical_data/clinical_data_id_age_gender.csv"),
            Path("C:/Project/ML Project/datasets/Clinical_data/clinical_data_id_age_gender.csv"),
        ]
        clinical_csv_path = next((c for c in clin_candidates if c.exists()), clin_candidates[0])
    else:
        clinical_csv_path = Path(clinical_csv_path)

    if spm_base_dir is None:
        spm_candidates = [
            Path("datasets/spanish_extracted/T1_preprocessed_SPM_1.5mm"),
            Path("C:/Project/ML Project/datasets/spanish_extracted/T1_preprocessed_SPM_1.5mm"),
        ]
        spm_base_dir = next((c for c in spm_candidates if c.exists()), spm_candidates[0])
    else:
        spm_base_dir = Path(spm_base_dir)

    gm_wm_dir = spm_base_dir / "T1_GM_WM_segmented"
    brain_dir = spm_base_dir / "T1_brain"

    return clinical_csv_path, spm_base_dir, gm_wm_dir, brain_dir


def step1_identify_naming_pattern(gm_wm_dir: Path) -> Dict[str, str]:
    """Step 1: Identify GM and WM file naming pattern in SPM directory."""
    print("=" * 60)
    print("STEP 1: IDENTIFYING GM AND WM SPM FILE NAMING PATTERNS")
    print("=" * 60)
    gm_files = sorted(list(gm_wm_dir.glob("m0wrp1*.nii*")))
    wm_files = sorted(list(gm_wm_dir.glob("m0wrp2*.nii*")))

    print(f"Total GM map candidates (m0wrp1*.nii): {len(gm_files)}")
    print(f"Total WM map candidates (m0wrp2*.nii): {len(wm_files)}")

    print("\n5 Example GM maps (c1 / Gray Matter):")
    for f in gm_files[:5]:
        print(f"  GM: {f.name}")

    print("\n5 Example WM maps (c2 / White Matter):")
    for f in wm_files[:5]:
        print(f"  WM: {f.name}")

    pattern = {
        "gm_prefix": "m0wrp1f",
        "wm_prefix": "m0wrp2f",
        "brain_prefix": "wmrf",
        "gm_desc": "Modulated warped Gray Matter tissue segmentation (c1)",
        "wm_desc": "Modulated warped White Matter tissue segmentation (c2)",
    }
    print(f"\nDetected pattern: GM={pattern['gm_prefix']}*.nii, WM={pattern['wm_prefix']}*.nii")
    print("Pattern identification complete.\n")
    return pattern


def step2_load_clinical_data(clinical_csv_path: Path) -> pd.DataFrame:
    """Step 2: Load clinical data and filter to primary study cohorts (ea and crl)."""
    print("=" * 60)
    print("STEP 2: LOADING CLINICAL DATA & FILTERING COHORTS")
    print("=" * 60)
    if not clinical_csv_path.exists():
        raise FileNotFoundError(f"Clinical CSV not found at: {clinical_csv_path}")

    raw_df = pd.read_csv(clinical_csv_path)
    print(f"Loaded clinical data with {len(raw_df)} total entries.")

    group_counts = raw_df["Subject"].value_counts().to_dict()
    print(f"Raw diagnostic groups: {group_counts}")

    # Filter: keep ea (AD=1) and crl (HC=0), exclude tb (bipolar=24)
    study_df = raw_df[raw_df["Subject"].isin(["ea", "crl"])].copy()

    # Standardize subject_id: e.g. crl_104, ea_146
    study_df["subject_id"] = study_df["Subject"].astype(str) + "_" + study_df["id"].astype(str)
    study_df["label"] = study_df["Subject"].map({"crl": 0, "ea": 1})
    study_df["age"] = study_df["Age"].astype(float)
    # Sex encoding: gender=1 -> Female -> sex=0, gender=2 -> Male -> sex=1
    study_df["sex"] = study_df["gender"].map({1: 0, 2: 1})

    n_ad = int((study_df["label"] == 1).sum())
    n_hc = int((study_df["label"] == 0).sum())
    print(f"Filtered study subjects: {len(study_df)} (AD={n_ad}, HC={n_hc})")
    assert n_ad == 35, f"Expected 35 AD subjects, got {n_ad}"
    assert n_hc == 19, f"Expected 19 HC subjects, got {n_hc}"
    print("Clinical cohort verified.\n")
    return study_df.reset_index(drop=True)


def step3_match_clinical_to_spm(
    study_df: pd.DataFrame,
    gm_wm_dir: Path,
    brain_dir: Path,
) -> pd.DataFrame:
    """Step 3: Match clinical IDs to SPM segmentation files."""
    print("=" * 60)
    print("STEP 3: MATCHING CLINICAL IDS TO SPM SEGMENTATION FILES")
    print("=" * 60)
    matched_records = []
    unmatched_ids = []

    for _, row in study_df.iterrows():
        sid = row["subject_id"]
        gm_file = gm_wm_dir / f"m0wrp1f{sid}.nii"
        wm_file = gm_wm_dir / f"m0wrp2f{sid}.nii"
        brain_file = brain_dir / f"wmrf{sid}.nii"

        if gm_file.exists() and wm_file.exists():
            matched_records.append({
                "subject_id": sid,
                "label": int(row["label"]),
                "age": float(row["age"]),
                "sex": int(row["sex"]),
                "gm_path": str(gm_file),
                "wm_path": str(wm_file),
                "brain_path": str(brain_file) if brain_file.exists() else None,
            })
        else:
            unmatched_ids.append(sid)

    n_matched = len(matched_records)
    n_unmatched = len(unmatched_ids)
    print(f"Matching Results: {n_matched} matched, {n_unmatched} unmatched")
    if unmatched_ids:
        print(f"Unmatched Subject IDs: {unmatched_ids}")

    if n_matched < 40:
        raise ValueError(f"CRITICAL ERROR: Only {n_matched} subjects matched (<40 threshold). Aborting extraction.")

    print(f"[PASS] Successfully matched {n_matched} / {len(study_df)} subjects.\n")
    return pd.DataFrame(matched_records)


def step4_compute_spm_volumes(matched_df: pd.DataFrame) -> pd.DataFrame:
    """Step 4: Compute GM, WM, brain volume, and ICV estimate using nibabel."""
    print("=" * 60)
    print("STEP 4: COMPUTING TISSUE VOLUMES FROM SPM MAPS (ONE SUBJECT AT A TIME)")
    print("=" * 60)

    extracted_rows = []

    for _, row in tqdm(matched_df.iterrows(), total=len(matched_df), desc="Extracting Spanish Volumes"):
        sid = row["subject_id"]
        gm_path = row["gm_path"]
        wm_path = row["wm_path"]
        brain_path = row["brain_path"]

        # Load GM
        img_gm = nib.load(gm_path)
        data_gm = img_gm.get_fdata()
        zooms = img_gm.header.get_zooms()[:3]
        voxel_vol_mm3 = float(np.prod(zooms))

        # Load WM
        img_wm = nib.load(wm_path)
        data_wm = img_wm.get_fdata()

        # SPM probability maps contain out-of-brain NaNs; use np.nansum
        gm_vol = float(np.nansum(data_gm) * voxel_vol_mm3)
        wm_vol = float(np.nansum(data_wm) * voxel_vol_mm3)
        total_brain_mm3 = gm_vol + wm_vol

        # ICV estimate from skull-stripped brain if available
        if brain_path and Path(brain_path).exists():
            img_br = nib.load(brain_path)
            data_br = img_br.get_fdata()
            icv_estimate_mm3 = float(np.nansum(data_br > 0) * voxel_vol_mm3)
            del img_br, data_br
        else:
            # Fallback approximation if brain file is not available
            icv_estimate_mm3 = float(total_brain_mm3 / 0.82)

        brain_vol_ratio = float(total_brain_mm3 / icv_estimate_mm3)

        extracted_rows.append({
            "subject_id": sid,
            "label": row["label"],
            "brain_vol_ratio": round(brain_vol_ratio, 6),
            "icv_estimate": round(icv_estimate_mm3, 2),
            "age": row["age"],
            "sex": row["sex"],
        })

        # Memory hygiene: delete loaded volume arrays and run gc
        del img_gm, data_gm, img_wm, data_wm
        gc.collect()

    features_df = pd.DataFrame(extracted_rows)
    return features_df


def step6_validate_features(df: pd.DataFrame) -> None:
    """Step 6: Validate feature ranges, nulls, and distributions."""
    print("\n" + "=" * 60)
    print("STEP 6: VALIDATING EXTRACTED SPANISH FEATURES")
    print("=" * 60)
    print("\nSummary Statistics of Spanish Features:")
    print(df[["brain_vol_ratio", "icv_estimate", "age", "sex"]].describe().round(4))

    n_ad = int((df["label"] == 1).sum())
    n_hc = int((df["label"] == 0).sum())
    print(f"\nLabel counts: {n_ad} AD (label=1), {n_hc} HC (label=0)")

    # Assertions
    assert df.isna().sum().sum() == 0, f"NaN values detected in Spanish features:\n{df.isna().sum()}"
    assert (df["brain_vol_ratio"] >= 0.50).all() and (df["brain_vol_ratio"] <= 1.00).all(), (
        f"brain_vol_ratio out of expected range [0.5, 1.0]: min={df['brain_vol_ratio'].min()}, max={df['brain_vol_ratio'].max()}"
    )
    assert (df["icv_estimate"] >= 800000).all() and (df["icv_estimate"] <= 2500000).all(), (
        f"icv_estimate out of expected range [800k, 2.5M]: min={df['icv_estimate'].min()}, max={df['icv_estimate'].max()}"
    )
    assert len(df) == 54, f"Expected 54 Spanish subjects, got {len(df)}"
    print("[PASS] All feature validation assertions passed successfully.\n")


def step7_save_cached_features(
    df: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """Step 7: Save extracted features to CSV and NPY cache."""
    print("=" * 60)
    print("STEP 7: SAVING CACHED FEATURES")
    print("=" * 60)
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "feature_cache"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "spanish_features.csv"
    npy_path = output_dir / "spanish_features.npy"

    df.to_csv(csv_path, index=False)
    # Save the 4 model features in standard order [brain_vol_ratio, icv_estimate, age, sex]
    features_array = df[["brain_vol_ratio", "icv_estimate", "age", "sex"]].to_numpy(dtype=float)
    np.save(npy_path, features_array)

    print(f"Spanish features cached. Shape: {df.shape}")
    print(f"  - CSV: {csv_path}")
    print(f"  - NPY: {npy_path}")
    return csv_path, npy_path


def run_spanish_extraction() -> pd.DataFrame:
    """Run full Spanish feature extraction pipeline."""
    clin_csv, _, gm_wm_dir, brain_dir = get_spanish_paths()
    step1_identify_naming_pattern(gm_wm_dir)
    study_df = step2_load_clinical_data(clin_csv)
    matched_df = step3_match_clinical_to_spm(study_df, gm_wm_dir, brain_dir)
    features_df = step4_compute_spm_volumes(matched_df)
    step6_validate_features(features_df)
    step7_save_cached_features(features_df)
    return features_df


if __name__ == "__main__":
    run_spanish_extraction()
