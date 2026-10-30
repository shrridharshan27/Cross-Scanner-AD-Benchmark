"""Feature Unit and Scale Audit.

Audits volumetric measurement units between OASIS-1 (eTIV) and the Spanish cohort (icv_estimate).
Confirms unit mismatch (eTIV ~1100-2000 cm3/scaled vs icv_estimate ~1.8M mm3) and applies
the standardized unit correction (dividing Spanish icv_estimate by 1000 to match OASIS-1 scale).
"""

from pathlib import Path
import shutil
import numpy as np
import pandas as pd

from ad_generalization_benchmark.data.loaders.oasis1_loader import load_oasis1


def run_unit_audit(apply_fix: bool = True) -> dict:
    """Run full unit audit across OASIS-1 and Spanish cohorts."""
    print("=" * 70)
    print("TASK 0: VOLUMETRIC FEATURE UNIT AUDIT")
    print("=" * 70)

    repo_dir = Path(__file__).resolve().parent.parent.parent
    spanish_cache = repo_dir / "features" / "feature_cache" / "spanish_features.csv"

    # Step 1: Check OASIS-1 eTIV units
    oasis_df, _ = load_oasis1(verbose=False)
    etiv = oasis_df["eTIV"]
    print("\n--- OASIS-1 Source Domain (eTIV) ---")
    print(f"Count: {len(etiv)}")
    print(f"Mean:  {etiv.mean():.2f}")
    print(f"Std:   {etiv.std():.2f}")
    print(f"Min:   {etiv.min():.2f}")
    print(f"Max:   {etiv.max():.2f}")

    if 1000 <= etiv.mean() <= 2500:
        oasis_scale = "scaled_cm3_mL"
        print("Range check: eTIV matches 1100-2000 range (~cm^3 / mL). NOT raw mm^3.")
    elif 1000000 <= etiv.mean() <= 2500000:
        oasis_scale = "raw_mm3"
        print("Range check: eTIV matches 1.1M-2.5M range (raw mm^3).")
    else:
        oasis_scale = "unknown"
        print(f"Range check: Unknown scale ({etiv.mean():.2f})")

    # Step 2: Check Spanish icv_estimate units
    if not spanish_cache.exists():
        raise FileNotFoundError(f"Spanish feature cache not found: {spanish_cache}")

    spanish_df = pd.read_csv(spanish_cache)
    icv = spanish_df["icv_estimate"]
    print("\n--- Spanish Cohort Target Domain (icv_estimate) ---")
    print(f"Count: {len(icv)}")
    print(f"Mean:  {icv.mean():.2f}")
    print(f"Std:   {icv.std():.2f}")
    print(f"Min:   {icv.min():.2f}")
    print(f"Max:   {icv.max():.2f}")

    if 1000000 <= icv.mean() <= 2500000:
        spanish_scale = "raw_mm3"
        print("Range check: icv_estimate matches ~1.8M range (raw mm^3).")
    elif 1000 <= icv.mean() <= 2500:
        spanish_scale = "scaled_cm3_mL"
        print("Range check: icv_estimate already matches ~1000-2000 range.")
    else:
        spanish_scale = "unknown"

    # Step 3: Confirm mismatch and determine correction
    mismatch_detected = (oasis_scale != spanish_scale)
    print(f"\nMismatch Status: {'CONFIRMED' if mismatch_detected else 'RESOLVED / NONE'}")

    if mismatch_detected and apply_fix:
        print("\n--- Applying Mandatory Fix ---")
        print("Reference scale: OASIS-1 eTIV (~1100-2000 cm^3 / mL).")
        print("Converting Spanish icv_estimate: dividing by 1000.0 to match OASIS-1 units.")

        # Backup original raw mm3 file if not already backed up
        backup_path = spanish_cache.parent / "spanish_features_raw_mm3.csv"
        if not backup_path.exists():
            shutil.copyfile(spanish_cache, backup_path)
            print(f"Preserved raw mm^3 features to: {backup_path.name}")

        # Correct icv_estimate
        corrected_df = spanish_df.copy()
        corrected_df["icv_estimate"] = corrected_df["icv_estimate"] / 1000.0

        # brain_vol_ratio is a ratio of two mm^3 quantities, so it remains unchanged
        print(f"brain_vol_ratio: mean={corrected_df['brain_vol_ratio'].mean():.4f} (unaffected dimensionless ratio).")
        print(f"Corrected icv_estimate: mean={corrected_df['icv_estimate'].mean():.2f}, "
              f"min={corrected_df['icv_estimate'].min():.2f}, max={corrected_df['icv_estimate'].max():.2f}")

        # Save corrected files
        corrected_df.to_csv(spanish_cache, index=False)
        npy_path = spanish_cache.with_suffix(".npy")
        np.save(npy_path, corrected_df[["brain_vol_ratio", "icv_estimate", "age", "sex"]].to_numpy(dtype=float))
        print("Saved unit-aligned features to spanish_features.csv and .npy")

    print("=" * 70)
    return {
        "oasis_scale": oasis_scale,
        "spanish_scale": spanish_scale,
        "mismatch_detected": mismatch_detected,
        "fixed": apply_fix and mismatch_detected,
    }


if __name__ == "__main__":
    run_unit_audit(apply_fix=True)
