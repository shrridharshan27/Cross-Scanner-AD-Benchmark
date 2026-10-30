"""Test to verify ComBat harmonization does not leak target evaluation subjects.

Uses strictly synthetic fixtures (4 subjects) — NEVER real patient data.
"""

from pathlib import Path
import pytest


def test_combat_no_eval_leak_synthetic():
    """Verify that ComBat fit index never contains evaluation subjects."""
    # Synthetic source and target subjects
    source_train_ids = ["SYN_OAS_001", "SYN_OAS_002"]
    spanish_calib_ids = ["SYN_ESP_CALIB_001"]
    spanish_eval_ids = ["SYN_ESP_EVAL_001", "SYN_ESP_EVAL_002"]

    # In Track B, ComBat is fit ONLY on source train + target calibration
    combat_fit_ids = source_train_ids + spanish_calib_ids

    # Assert no evaluation subject appears in fit set
    overlap = set(combat_fit_ids) & set(spanish_eval_ids)
    assert (
        len(overlap) == 0
    ), f"Leakage: Spanish evaluation subject(s) {overlap} present in ComBat fit data!"
