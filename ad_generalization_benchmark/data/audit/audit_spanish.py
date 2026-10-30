"""Audit script for Spanish Target Cohort.

Inspects Clinical_data.zip and T1_preprocessed.zip manifests without
extracting large raw MRI archives, verifies clinical labels and counts,
and checks image-to-clinical matching.
"""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union
import zipfile
import pandas as pd

from ad_generalization_benchmark.configs import resolve_dataset_path


def audit_spanish_clinical(clinical_zip_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Audit Clinical_data.zip and return filtered clinical DataFrame."""
    if clinical_zip_path is None:
        clinical_zip_path = resolve_dataset_path("spanish_clinical_zip")
    else:
        clinical_zip_path = Path(clinical_zip_path)

    print(f"Auditing Spanish clinical archive: {clinical_zip_path}")
    if not clinical_zip_path.exists():
        raise FileNotFoundError(f"Spanish clinical zip not found at: {clinical_zip_path}")

    with zipfile.ZipFile(clinical_zip_path, "r") as z:
        files = z.namelist()
        print("\nFiles in Clinical_data.zip:")
        for fname in sorted(files):
            if fname and not fname.endswith("/"):
                info = z.getinfo(fname)
                print(f"  - {fname} ({info.file_size:,} bytes)")

        target_csv = "clinical_data_id_age_gender.csv"
        if target_csv not in files:
            raise FileNotFoundError(f"Expected '{target_csv}' not found in {clinical_zip_path}")

        with z.open(target_csv) as f:
            df = pd.read_csv(f)

    print("\nClinical Data Raw Summary:")
    print(f"Total entries in {target_csv}: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    subject_counts = df["Subject"].value_counts().to_dict()
    print("\nDiagnostic Group Distribution:")
    for grp, cnt in subject_counts.items():
        print(f"  {grp}: {cnt}")

    n_ad = subject_counts.get("ea", 0)
    n_hc = subject_counts.get("crl", 0)
    n_bipolar = subject_counts.get("tb", 0)

    print(f"\nAudit Check:")
    print(f"  Alzheimer's Disease (ea): {n_ad} (expected: 35)")
    print(f"  Healthy Control (crl):    {n_hc} (expected: 19)")
    print(f"  Bipolar Disorder (tb):    {n_bipolar} (expected: 24, EXCLUDED)")
    print(f"  Total Valid Binary:       {n_ad + n_hc}")

    assert n_ad == 35, f"Expected 35 AD subjects, found {n_ad}"
    assert n_hc == 19, f"Expected 19 HC subjects, found {n_hc}"
    assert n_bipolar == 24, f"Expected 24 bipolar subjects, found {n_bipolar}"

    return df


def audit_spanish_mri_manifest(mri_zip_path: Optional[Union[str, Path]] = None) -> Dict[str, List[str]]:
    """Inspect T1_preprocessed.zip manifest without extracting raw MRI data."""
    if mri_zip_path is None:
        mri_zip_path = resolve_dataset_path("spanish_mri_zip")
    else:
        mri_zip_path = Path(mri_zip_path)

    print(f"\nAuditing Spanish MRI archive manifest: {mri_zip_path}")
    if not mri_zip_path.exists():
        raise FileNotFoundError(f"Spanish MRI zip not found at: {mri_zip_path}")

    dir_file_counts = defaultdict(int)
    t1_original_files = []
    file_extensions = Counter()

    with zipfile.ZipFile(mri_zip_path, "r") as z:
        for info in z.infolist():
            fname = info.filename
            if fname.endswith("/"):
                continue
            parent = "/".join(fname.split("/")[:-1])
            dir_file_counts[parent] += 1
            if "." in fname:
                ext = "." + ".".join(fname.split("/")[-1].split(".")[1:])
                file_extensions[ext] += 1
            if fname.startswith("T1_original/"):
                t1_original_files.append(fname)

    print("\nDirectory breakdown in T1_preprocessed.zip:")
    for dir_name, count in sorted(dir_file_counts.items()):
        print(f"  {dir_name or '[root]'}: {count} files")

    print("\nFile extension breakdown:")
    for ext, cnt in file_extensions.items():
        print(f"  {ext}: {cnt} files")

    print(f"\nTotal T1_original anatomical scans found: {len(t1_original_files)}")
    return {"t1_original": t1_original_files, "dirs": list(dir_file_counts.keys())}


def audit_spanish():
    """Run complete Spanish cohort audit."""
    print("=" * 60)
    print("STARTING SPANISH COHORT AUDIT")
    print("=" * 60)
    df_clinical = audit_spanish_clinical()
    mri_manifest = audit_spanish_mri_manifest()

    # Match check
    t1_filenames = set(f.split("/")[-1] for f in mri_manifest["t1_original"])
    df_clinical["expected_filename"] = df_clinical["Subject"] + "_" + df_clinical["id"].astype(str) + ".nii.gz"
    matched = df_clinical["expected_filename"].isin(t1_filenames)

    n_matched = int(matched.sum())
    print("\nSubject-to-Image Matching Results:")
    print(f"  Total clinical subjects: {len(df_clinical)}")
    print(f"  Matched T1_original images: {n_matched} / {len(df_clinical)}")

    # Filter to primary study subjects (ea and crl)
    study_subjects = df_clinical[df_clinical["Subject"].isin(["ea", "crl"])]
    study_matched = study_subjects["expected_filename"].isin(t1_filenames).sum()
    print(f"  Primary study subjects matched: {study_matched} / {len(study_subjects)}")

    assert study_matched == 54, f"Expected 54 matched study subjects, found {study_matched}"
    print("\n[SUCCESS] Spanish Cohort Audit passed all checks.")
    print("=" * 60)


if __name__ == "__main__":
    audit_spanish()
