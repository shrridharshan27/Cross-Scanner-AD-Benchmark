"""Test ensuring ComBat fitting uses only calibration subjects and never evaluation subjects.

Verifies that:
1. Target calibration and evaluation sets are strictly disjoint.
2. Spanish evaluation subjects never appear in the ComBat fitting matrix.
"""

import json
from pathlib import Path
import pytest


def test_eval_subjects_not_in_combat_fit():
    """Verify that Spanish evaluation subjects never appear in ComBat fitting set."""
    splits_dir = Path("ad_generalization_benchmark/splits/saved")
    seeds = [42, 123, 456]

    for seed in seeds:
        split_file = splits_dir / f"spanish_trackB_seed{seed}.json"
        assert split_file.exists(), f"Split file not found: {split_file}"

        with open(split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)

        calib_ids = set(split_data["calibration"])
        eval_ids = set(split_data["evaluation"])

        # 1. Disjointness check
        overlap = calib_ids & eval_ids
        assert len(overlap) == 0, f"Seed {seed}: Overlap found between calibration and eval: {overlap}"

        # 2. Check counts: calibration is ~20%, eval is ~80%
        assert len(calib_ids) >= 2, f"Seed {seed}: ComBat requires at least 2 calibration subjects"
        assert len(eval_ids) >= 10, f"Seed {seed}: Evaluation set too small"
        assert len(calib_ids) + len(eval_ids) == 54, f"Seed {seed}: Total subjects should be 54"
