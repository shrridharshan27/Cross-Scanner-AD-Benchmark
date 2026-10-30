import pytest

def test_no_overlap_between_train_and_test(synthetic_oasis_df):
    """Train and test subject IDs must never overlap."""
    all_ids = list(synthetic_oasis_df['subject_id'])
    train_ids = set(all_ids[:2])
    test_ids = set(all_ids[2:])
    overlap = train_ids.intersection(test_ids)
    assert len(overlap) == 0, f"Subject overlap found: {overlap}"

def test_no_overlap_between_cohorts(synthetic_oasis_df, synthetic_spanish_df):
    """OASIS and Spanish subject IDs must never overlap."""
    oasis_ids = set(synthetic_oasis_df['subject_id'])
    spanish_ids = set(synthetic_spanish_df['subject_id'])
    overlap = oasis_ids.intersection(spanish_ids)
    assert len(overlap) == 0, f"Cross-cohort overlap found: {overlap}"
