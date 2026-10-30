"""Test ensuring classification threshold is selected on validation and frozen for external evaluation.

Verifies that:
1. The classification threshold is derived from the validation set (Youden's J) and stored.
2. The exact same threshold is loaded and applied to external evaluation without recalculation.
"""

import json
from pathlib import Path
import numpy as np
import pytest
from ad_generalization_benchmark.evaluation.evaluate import find_optimal_threshold


def test_threshold_selection_logic():
    """Verify threshold calculation logic selects optimal Youden's J threshold in [0, 1]."""
    y_val = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Well-separated probabilities
    val_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    threshold = find_optimal_threshold(y_val, val_probs)
    assert isinstance(threshold, float)
    assert 0.0 <= threshold <= 1.0
    # Threshold should separate classes (between 0.4 and 0.6)
    assert 0.35 <= threshold <= 0.65


def test_threshold_frozen_between_internal_and_external():
    """Verify that stored internal thresholds match external frozen_threshold values exactly."""
    results_dir = Path("ad_generalization_benchmark/results")
    seeds = [42, 123, 456]
    models = ["lr", "svm", "mlp"]

    for seed in seeds:
        internal_file = results_dir / f"internal_eval_seed{seed}.json"
        external_file = results_dir / f"external_trackA_seed{seed}.json"

        if internal_file.exists() and external_file.exists():
            with open(internal_file, "r", encoding="utf-8") as f_in:
                internal_data = json.load(f_in)
            with open(external_file, "r", encoding="utf-8") as f_ex:
                external_data = json.load(f_ex)

            for m in models:
                internal_thresh = internal_data[m]["threshold"]
                external_thresh = external_data[m]["frozen_threshold"]
                assert (
                    internal_thresh == external_thresh
                ), f"Seed {seed} model {m}: internal threshold {internal_thresh} != external {external_thresh}"
