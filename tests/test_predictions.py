import pandas as pd
import pytest
from src.evaluation.predictions import ensemble,patients

def frame():
    return pd.DataFrame([dict(dataset='f',image_id=str(i),patient_id='p',fold=0,seed=s,true_label=0,mask_available=True,
        p_class0=p[0],p_class1=p[1],p_class2=p[2]) for i,p in enumerate([(0.8,0.1,0.1),(0.4,0.5,0.1)]) for s in [42,43,44]])

def test_ensemble_and_patient_probabilities():
    e=ensemble(frame());assert len(e)==2
    p=patients(e);assert len(p)==1
    assert p.p_class0.iloc[0]==pytest.approx(0.6)
    assert p.predicted_label.iloc[0]==0

def test_missing_seed_and_duplicate_rejected():
    f=frame()
    with pytest.raises(ValueError):ensemble(f.iloc[1:])
    with pytest.raises(ValueError):ensemble(pd.concat([f,f.iloc[:1]]))

def test_cross_fold_rejected():
    f=frame();f.loc[0,'fold']=1
    with pytest.raises(ValueError):ensemble(f)
