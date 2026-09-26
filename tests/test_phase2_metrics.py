import numpy as np
import pandas as pd
import pytest
from src.evaluation.patient_aggregation import aggregate_patients
from src.evaluation.patient_metrics import metrics,patient_metrics
from src.evaluation.calibration import calibration
from src.statistics.patient_bootstrap import bootstrap_patients
from src.evaluation.predictions import ensemble

def table():
    rows=[]
    for patient in range(6):
        for image in range(2):
            p=np.full(3,.1);p[patient%3]=.8
            rows.append(dict(dataset='synthetic',patient_id=str(patient),image_id=f'{patient}_{image}',fold=patient%2,true_label=patient%3,**dict(zip(['p_class0','p_class1','p_class2'],p))))
    return pd.DataFrame(rows)

def test_patient_mean_determinism():
    a=aggregate_patients(table());assert len(a)==6
    pd.testing.assert_frame_equal(a,aggregate_patients(table().sample(frac=1,random_state=42)))
    assert a.patient_id.is_unique

def test_seed_average_not_best_seed():
    f=table().iloc[:1].copy();f['mask_available']=True
    rows=[]
    for seed,p in [(42,[.99,.005,.005]),(43,[.01,.98,.01]),(44,[.01,.98,.01])]:
        r=f.copy();r['seed']=seed;r[['p_class0','p_class1','p_class2']]=p;rows.append(r)
    e=ensemble(pd.concat(rows));assert e.predicted_label.iloc[0]==1
    assert e.p_class0.iloc[0]==pytest.approx(1.01/3)

def test_auc_and_absent_class():
    f=aggregate_patients(table());p=f[['p_class0','p_class1','p_class2']]
    assert metrics(f.true_label,p)['ovr_macro_auc']==1
    absent=metrics([0,0],[[.8,.1,.1],[.7,.2,.1]])
    assert absent['ovr_macro_auc'] is None and absent['balanced_accuracy'] is None
    assert absent['per_class'][1]['auc_undefined_reason']

def test_patient_bootstrap_and_undefined_replicates():
    f=aggregate_patients(table())
    a=bootstrap_patients(f,draws=30);assert a==bootstrap_patients(f,draws=30)
    assert a['unit']=='patient' and a['intervals']['accuracy']['low']==1
    with pytest.raises(ValueError):bootstrap_patients(pd.concat([f,f.iloc[:1]]),draws=2)
    tiny=f.iloc[:3];b=bootstrap_patients(tiny,draws=50,stratified=False)
    assert b['intervals']['ovr_macro_auc']['undefined_draws']>0

def test_calibration_perfect_and_bad_probabilities():
    r=calibration([0,1,2],np.eye(3));assert r['ece']==0 and r['brier_multiclass_sum']==0
    with pytest.raises(ValueError):metrics([0],[[.2,.2,.2]])

def test_majority_is_sensitivity_with_explicit_ties():
    f=table().iloc[:2].copy();f.loc[f.index[0],['p_class0','p_class1','p_class2']]=[.9,.05,.05]
    f.loc[f.index[1],['p_class0','p_class1','p_class2']]=[.1,.8,.1]
    assert aggregate_patients(f,'majority').predicted_label.iloc[0]==0

def test_majority_metrics_use_vote_labels():
    f=table().iloc[:2].copy();extra=f.iloc[:1].copy();extra['image_id']='extra';f=pd.concat([f,extra],ignore_index=True)
    f[['p_class0','p_class1','p_class2']]=[[.51,.49,0],[.51,.49,0],[.01,.99,0]]
    a=aggregate_patients(f,'mean');b=aggregate_patients(f,'majority')
    assert a.predicted_label.iloc[0]==1 and b.predicted_label.iloc[0]==0
    assert patient_metrics(a)['accuracy']==0 and patient_metrics(b)['accuracy']==1
