import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def assert_disjoint(train, validation, test):
    a,b,c=map(set,(train,validation,test))
    if a & b or a & c or b & c:
        raise ValueError('Patient leakage')

def make_splits(frame, seed=2026):
    if frame.patient_id.isna().any() or frame.image_id.duplicated().any():
        raise ValueError('Missing patient IDs or duplicate image IDs')
    patients=frame[['patient_id','label']].drop_duplicates()
    if patients.patient_id.duplicated().any(): raise ValueError('Conflicting patient labels')
    if patients.label.value_counts().min()<5: raise ValueError('Insufficient patients per class')
    outputs=[]
    outer=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
    for fold,(training,test) in enumerate(outer.split(patients,patients.label,patients.patient_id)):
        pool=patients.iloc[training]
        inner=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
        a,b=next(inner.split(pool,pool.label,pool.patient_id))
        train_ids=pool.iloc[a].patient_id;val_ids=pool.iloc[b].patient_id;test_ids=patients.iloc[test].patient_id
        assert_disjoint(train_ids,val_ids,test_ids)
        for role,ids in [('train',train_ids),('validation',val_ids),('test',test_ids)]:
            part=frame[frame.patient_id.isin(ids)].copy();part['fold']=fold;part['role']=role;outputs.append(part)
    return pd.concat(outputs,ignore_index=True)
