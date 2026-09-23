import numpy as np
import pandas as pd
from src.evaluation.calibration import validate

def aggregate_patients(frame,method='mean'):
    if method not in ('mean','majority'):raise ValueError('Unknown aggregation')
    required=['dataset','patient_id','image_id','fold','true_label']
    if frame.empty or frame[required].isna().any().any():raise ValueError('Missing identifiers/labels')
    if frame.duplicated(['dataset','image_id']).any():raise ValueError('Duplicate canonical image')
    probs=['p_class0','p_class1','p_class2'];validate(frame.true_label,frame[probs])
    rows=[]
    for (dataset,pid),g in frame.groupby(['dataset','patient_id'],sort=True):
        g=g.sort_values('image_id')
        if g.true_label.nunique()!=1 or g.fold.nunique()!=1:raise ValueError('Patient crosses diagnosis/fold')
        p=g[probs].mean().to_numpy();prediction=int(p.argmax())
        if method=='majority':
            counts=np.bincount(g[probs].to_numpy().argmax(1),minlength=3);candidates=np.flatnonzero(counts==counts.max())
            prediction=int(candidates[np.argmax(p[candidates])])
        rows.append(dict(dataset=dataset,patient_id=pid,fold=int(g.fold.iloc[0]),true_label=int(g.true_label.iloc[0]),predicted_label=prediction,n_slices=len(g),aggregation=method,**dict(zip(probs,p))))
    return pd.DataFrame(rows)
