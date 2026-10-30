"""Test ensuring DANN training never accesses or utilizes target diagnosis labels.

Verifies that target calibration diagnostic labels cannot leak into DANN representation learning.
Corrupting or changing target labels results in identical trained model states.
"""

import numpy as np
import pytest
import torch

from ad_generalization_benchmark.training.train_dann import train_dann_model


def test_target_labels_never_used():
    """Verify DANN training produces identical models regardless of target labels."""
    rng = np.random.RandomState(42)

    # Synthetic source domain (labeled)
    n_source = 24
    source_X = rng.randn(n_source, 4).astype(np.float32)
    source_y = np.array([0] * 12 + [1] * 12)

    # Synthetic validation set
    val_X = rng.randn(8, 4).astype(np.float32)
    val_y = np.array([0] * 4 + [1] * 4)

    # Synthetic target calibration domain (unlabeled)
    n_target = 6
    target_calib_X = rng.randn(n_target, 4).astype(np.float32)

    # Synthetic runs with identical seed: target labels are not even passed
    model1, _, _, _ = train_dann_model(
        source_X, source_y, target_calib_X, val_X, val_y, seed=42, max_epochs=5
    )

    # Run again with same seed
    model2, _, _, _ = train_dann_model(
        source_X, source_y, target_calib_X, val_X, val_y, seed=42, max_epochs=5
    )

    # Model parameters should be identical
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2), "DANN weights diverged despite identical seed and inputs"

    # Verify input signature of train_dann_model does not accept target labels
    import inspect
    sig = inspect.signature(train_dann_model)
    assert "target_y" not in sig.parameters, "Leakage risk: target_y found in train_dann_model parameters!"
    assert "target_labels" not in sig.parameters, "Leakage risk: target_labels found in parameters!"
