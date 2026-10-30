import pytest
import pandas as pd

def test_cdr_to_label_primary():
    """Primary rule: CDR=0 -> 0, CDR>=1 -> 1, CDR=0.5 -> excluded."""
    from ad_generalization_benchmark.data.loaders.oasis1_loader import apply_primary_label_rule
    
    df = pd.DataFrame({'CDR': [0.0, 0.5, 1.0, 2.0, 0.0]})
    result = apply_primary_label_rule(df)
    
    # CDR=0.5 should be excluded
    assert 0.5 not in result['CDR'].values, "CDR=0.5 should be excluded"
    # CDR=0 -> label 0
    assert all(result[result['CDR'] == 0.0]['label'] == 0)
    # CDR>=1 -> label 1
    assert all(result[result['CDR'] >= 1.0]['label'] == 1)

def test_spanish_label_mapping():
    """Spanish labels: ea->1, crl->0, tb->excluded."""
    from ad_generalization_benchmark.data.loaders.spanish_loader import apply_spanish_label_rule
    
    df = pd.DataFrame({'group': ['ea', 'crl', 'tb', 'ea', 'crl']})
    result = apply_spanish_label_rule(df)
    
    assert 'tb' not in result['group'].values, "Bipolar (tb) should be excluded"
    assert all(result[result['group'] == 'ea']['label'] == 1)
    assert all(result[result['group'] == 'crl']['label'] == 0)
