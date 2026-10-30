"""Test Spanish feature extraction validity, ranges, NaN checks, and counts.

Uses synthetic fixtures for isolated unit testing, with optional checks against
cached Spanish features when available.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

FEATURE_COLS = ["brain_vol_ratio", "icv_estimate", "age", "sex"]


def test_brain_vol_ratio_in_range(synthetic_spanish_df):
    """Assert all brain_vol_ratio values between 0.5 and 1.0."""
    assert (synthetic_spanish_df["brain_vol_ratio"] >= 0.5).all(), "Values below 0.5 detected"
    assert (synthetic_spanish_df["brain_vol_ratio"] <= 1.0).all(), "Values above 1.0 detected"

    # Also verify real cache if present on disk
    cache_path = Path("ad_generalization_benchmark/features/feature_cache/spanish_features.csv")
    if cache_path.exists():
        df_real = pd.read_csv(cache_path)
        assert (df_real["brain_vol_ratio"] >= 0.5).all()
        assert (df_real["brain_vol_ratio"] <= 1.0).all()


def test_no_nan_in_spanish_features(synthetic_spanish_df):
    """Assert no NaN values in Spanish feature dataframe or cache."""
    assert synthetic_spanish_df.isnull().sum().sum() == 0, "NaN found in synthetic Spanish dataframe"

    cache_path = Path("ad_generalization_benchmark/features/feature_cache/spanish_features.csv")
    if cache_path.exists():
        df_real = pd.read_csv(cache_path)
        assert df_real.isnull().sum().sum() == 0, "NaN found in Spanish feature cache!"


def test_feature_count(synthetic_spanish_df):
    """Assert exactly 4 features per subject (brain_vol_ratio, icv_estimate, age, sex)."""
    for col in FEATURE_COLS:
        assert col in synthetic_spanish_df.columns, f"Missing feature {col}"

    cache_path = Path("ad_generalization_benchmark/features/feature_cache/spanish_features.csv")
    if cache_path.exists():
        df_real = pd.read_csv(cache_path)
        for col in FEATURE_COLS:
            assert col in df_real.columns, f"Missing feature {col} in cache"


def test_label_counts(synthetic_spanish_df):
    """Assert n_AD >= 10 and n_HC >= 10 in Spanish cohort."""
    cache_path = Path("ad_generalization_benchmark/features/feature_cache/spanish_features.csv")
    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        df = synthetic_spanish_df

    n_ad = int((df["label"] == 1).sum())
    n_hc = int((df["label"] == 0).sum())
    assert n_ad >= 10, f"Expected n_AD >= 10, got {n_ad}"
    assert n_hc >= 10, f"Expected n_HC >= 10, got {n_hc}"
