import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def synthetic_oasis_df():
    """Tiny synthetic OASIS-like DataFrame for tests. Never use real data."""
    return pd.DataFrame({
        'subject_id': ['S001', 'S002', 'S003', 'S004'],
        'label': [0, 0, 1, 1],
        'nWBV': [0.75, 0.78, 0.65, 0.68],
        'eTIV': [1400, 1500, 1300, 1350],
        'age': [72, 68, 75, 80],
        'sex': [0, 1, 0, 1]
    })

@pytest.fixture
def synthetic_spanish_df():
    """Synthetic Spanish-like DataFrame for tests (10 HC, 10 AD)."""
    n_hc = 10
    n_ad = 10
    ids = [f'ES_HC_{i:02d}' for i in range(n_hc)] + [f'ES_AD_{i:02d}' for i in range(n_ad)]
    labels = [0] * n_hc + [1] * n_ad
    ratios = [0.75 + (0.01 * (i % 5)) for i in range(n_hc)] + [0.65 + (0.01 * (i % 5)) for i in range(n_ad)]
    icvs = [1400.0 + i * 10 for i in range(n_hc + n_ad)]
    ages = [70 + (i % 10) for i in range(n_hc + n_ad)]
    sexes = [i % 2 for i in range(n_hc + n_ad)]

    return pd.DataFrame({
        'subject_id': ids,
        'label': labels,
        'brain_vol_ratio': ratios,
        'icv_estimate': icvs,
        'age': ages,
        'sex': sexes
    })
