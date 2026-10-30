"""Test ensuring volumetric feature units (eTIV vs icv_estimate) are scale-consistent.

Verifies that OASIS-1 eTIV and Spanish cohort icv_estimate are within the same order of
magnitude (~1000 - 2000 cm3/mL) and not mismatched by 1000x (mm3 vs cm3).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest


def test_etiv_scale_consistent(synthetic_oasis_df, synthetic_spanish_df):
    """Assert OASIS eTIV and Spanish icv_estimate are in the same scale range (~1000-2000)."""
    # 1. Test on synthetic fixture
    assert 1000.0 <= synthetic_oasis_df["eTIV"].mean() <= 2500.0
    assert 1000.0 <= synthetic_spanish_df["icv_estimate"].mean() <= 2500.0
    ratio_synthetic = synthetic_spanish_df["icv_estimate"].mean() / synthetic_oasis_df["eTIV"].mean()
    assert 0.5 <= ratio_synthetic <= 2.0, f"Synthetic scale ratio outside [0.5, 2.0]: {ratio_synthetic}"

    # 2. Test on cached features if available on disk
    spanish_cache = Path("ad_generalization_benchmark/features/feature_cache/spanish_features.csv")
    if spanish_cache.exists():
        df_sp = pd.read_csv(spanish_cache)
        mean_sp = df_sp["icv_estimate"].mean()
        assert 1000.0 <= mean_sp <= 2500.0, (
            f"Spanish icv_estimate mean {mean_sp:.2f} is not in ~1000-2500 range. "
            f"Did unit audit fix run?"
        )
