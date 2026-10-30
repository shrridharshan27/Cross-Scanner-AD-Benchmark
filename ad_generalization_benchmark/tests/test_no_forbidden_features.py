import pytest

FORBIDDEN_FEATURES = ['MMSE', 'CDR', 'ASF', 'Delay', 'SES', 'Educ']

def test_no_forbidden_features_in_oasis(synthetic_oasis_df):
    """Forbidden features must never appear in the feature matrix."""
    cols = list(synthetic_oasis_df.columns)
    for feat in FORBIDDEN_FEATURES:
        assert feat not in cols, f"Forbidden feature '{feat}' found in columns!"

def test_no_nan_in_features(synthetic_oasis_df):
    """Feature matrix must have no NaN or Inf values."""
    feature_cols = ['nWBV', 'eTIV', 'age', 'sex']
    for col in feature_cols:
        assert synthetic_oasis_df[col].isnull().sum() == 0, \
            f"NaN found in column '{col}'"

def test_labels_are_binary(synthetic_oasis_df):
    """Labels must be 0 or 1 only."""
    valid_labels = {0, 1}
    actual_labels = set(synthetic_oasis_df['label'].unique())
    assert actual_labels.issubset(valid_labels), \
        f"Non-binary labels found: {actual_labels}"
