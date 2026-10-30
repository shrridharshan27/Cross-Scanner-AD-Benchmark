"""Test ensuring StandardScaler is never fitted on test or external data.

Verifies that scaler parameters (mean_, var_) match only the training partition statistics,
and that test or external cohort features never influence the fitted scaler.
"""

import numpy as np
import pytest
from ad_generalization_benchmark.features.align_features import fit_and_apply_scaler


def test_scaler_not_fitted_on_test():
    """Verify scaler was NOT fitted on test or Spanish data; mean matches training only."""
    # Synthetic training set with known mean
    X_train = np.array([
        [0.70, 1400.0, 70.0, 0.0],
        [0.80, 1600.0, 80.0, 1.0],
    ])
    # Expected mean of continuous features [0:3]: [0.75, 1500.0, 75.0]
    expected_train_mean = np.array([0.75, 1500.0, 75.0])

    # Distinct test and external arrays with very different means
    X_test = np.array([
        [0.60, 1200.0, 60.0, 1.0],
        [0.62, 1220.0, 62.0, 0.0],
    ])
    X_external = np.array([
        [0.85, 1900.0, 90.0, 0.0],
        [0.87, 1950.0, 92.0, 1.0],
    ])

    scaler, [scaled_tr, scaled_te, scaled_ext] = fit_and_apply_scaler(X_train, X_test, X_external)

    # Scaler mean must match training set statistics exactly
    np.testing.assert_allclose(scaler.mean_, expected_train_mean, rtol=1e-5)

    # Verify test data did NOT shift the scaler mean
    combined_train_test_mean = np.mean(np.vstack([X_train[:, :3], X_test[:, :3]]), axis=0)
    assert not np.allclose(scaler.mean_, combined_train_test_mean)

    # Verify external data did NOT shift the scaler mean
    combined_all_mean = np.mean(np.vstack([X_train[:, :3], X_external[:, :3]]), axis=0)
    assert not np.allclose(scaler.mean_, combined_all_mean)

    # Scaled training continuous features must have mean ~ 0
    np.testing.assert_allclose(np.mean(scaled_tr[:, :3], axis=0), np.zeros(3), atol=1e-7)

    # Scaled test continuous features must NOT have mean 0 (because fitted on train only)
    assert not np.allclose(np.mean(scaled_te[:, :3], axis=0), np.zeros(3), atol=1e-2)
