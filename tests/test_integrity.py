import numpy as np
import pandas as pd
import pytest
from src.data.splits import make_splits, assert_disjoint
from scripts.audit_data import canonical_hash

def test_patient_leakage_rejected():
    with pytest.raises(ValueError): assert_disjoint(['a'],['a'],['b'])

def test_splits_deterministic_and_complete():
    f=pd.DataFrame([dict(patient_id=f'p{p}',image_id=f'{p}_{s}',label=p%3) for p in range(45) for s in range(2)])
    a=make_splits(f);pd.testing.assert_frame_equal(a,make_splits(f))
    assert a[a.role=='test'].image_id.value_counts().eq(1).all()
    for _,g in a.groupby('fold'):
        assert_disjoint(*(g[g.role==role].patient_id for role in ['train','validation','test']))

def test_hash_shape_and_value_sensitive():
    a=np.arange(12,dtype=np.uint16).reshape(3,4)
    assert canonical_hash(a)==canonical_hash(a.copy())
    assert canonical_hash(a)!=canonical_hash(a.reshape(4,3))
    assert canonical_hash(a)!=canonical_hash(a+1)
