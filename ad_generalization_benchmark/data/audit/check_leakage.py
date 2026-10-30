"""Leakage Verification Script.

Inspects all saved split JSON files and verifies:
1. No participant appears in multiple partitions (train/val/test).
2. No participant ID collision exists between OASIS-1 and Spanish cohorts.
3. Track B calibration and evaluation sets are strictly disjoint.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

from ad_generalization_benchmark.configs import load_experiment_config


def check_split_file_integrity(split_path: Path) -> Dict[str, Any]:
    """Check a single split JSON file for partition disjointness."""
    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "train" in data and "val" in data and "test" in data:
        train = set(data["train"])
        val = set(data["val"])
        test = set(data["test"])

        overlap_tv = train & val
        overlap_tt = train & test
        overlap_vt = val & test

        if overlap_tv or overlap_tt or overlap_vt:
            raise AssertionError(
                f"LEAKAGE in {split_path.name}: "
                f"train-val: {len(overlap_tv)}, train-test: {len(overlap_tt)}, val-test: {len(overlap_vt)}"
            )
        return {"status": "PASS", "type": "3-way", "total": len(train | val | test)}

    elif "calibration" in data and "evaluation" in data:
        calib = set(data["calibration"])
        ev = set(data["evaluation"])
        overlap = calib & ev
        if overlap:
            raise AssertionError(f"LEAKAGE in {split_path.name}: calib-eval overlap: {len(overlap)}")
        return {"status": "PASS", "type": "calibration-eval", "total": len(calib | ev)}

    else:
        raise ValueError(f"Unrecognized split structure in {split_path.name}")


def run_full_leakage_audit(splits_dir: Optional[Path] = None) -> bool:
    """Run full leakage audit across all saved splits."""
    if splits_dir is None:
        splits_dir = Path(__file__).resolve().parent.parent.parent / "splits" / "saved"

    split_files = list(splits_dir.glob("*.json"))
    if not split_files:
        raise FileNotFoundError(f"No split JSON files found in {splits_dir}")

    print(f"Auditing {len(split_files)} split files in {splits_dir}...")
    oasis_primary_all_ids: Set[str] = set()
    spanish_all_ids: Set[str] = set()

    for sf in split_files:
        res = check_split_file_integrity(sf)
        print(f"  [PASS] {sf.name}: Disjoint partitions verified (N={res['total']})")

        with open(sf, "r", encoding="utf-8") as f:
            d = json.load(f)

        if "oasis1_primary" in sf.name:
            oasis_primary_all_ids.update(d["train"] + d["val"] + d["test"])
        elif "spanish" in sf.name:
            spanish_all_ids.update(d["calibration"] + d["evaluation"])

    # Check cross-cohort ID collision
    cross_overlap = oasis_primary_all_ids & spanish_all_ids
    if cross_overlap:
        raise AssertionError(f"CROSS-COHORT LEAKAGE: {len(cross_overlap)} shared IDs: {cross_overlap}")

    print(f"  [PASS] Zero ID collision between OASIS-1 ({len(oasis_primary_all_ids)}) and Spanish ({len(spanish_all_ids)}) cohorts.")
    print("=" * 60)
    print("ALL LEAKAGE CHECKS PASSED SUCCESSFULLY.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    run_full_leakage_audit()
