import numpy as np
import pandas as pd

PROBS=['p_class0','p_class1','p_class2']

def ensemble(frame):
    keys=['dataset','image_id']
    if frame.empty or frame[keys+['patient_id','fold','seed','true_label']].isna().any().any():raise ValueError('Missing OOF identifiers')
    if frame.duplicated(keys+['seed']).any(): raise ValueError('Duplicate OOF rows')
    values=frame[PROBS].to_numpy()
    if not np.isfinite(values).all() or (values<0).any() or not np.allclose(values.sum(1),1):
        raise ValueError('Invalid probabilities')
    rows=[]
    for _,g in frame.groupby(keys,dropna=False):
        g=g.sort_values('seed')
        if set(g.seed)!={42,43,44}: raise ValueError('Missing or unexpected seed')
        for field in ['patient_id','fold','true_label','mask_available']:
            if g[field].isna().any() or g[field].nunique()!=1: raise ValueError('Inconsistent '+field)
        row=g.iloc[0][keys+['patient_id','fold','true_label','mask_available']].to_dict()
        p=g[PROBS].mean().to_numpy();row.update(zip(PROBS,p))
        row['predicted_label']=int(p.argmax());row['correct']=row['predicted_label']==row['true_label']
        for field in ['track','model','protocol_sha256','manifest_sha256','environment_sha256','git_commit']:
            if field in g:
                if g[field].nunique()!=1:raise ValueError('Mixed prediction provenance: '+field)
                row[field]=g[field].iloc[0]
        if 'checkpoint' in g:row['source_checkpoints']=g.checkpoint.tolist()
        if 'checkpoint_sha256' in g:row['source_checkpoint_sha256']=g.checkpoint_sha256.tolist()
        rows.append(row)
    return pd.DataFrame(rows)

def patients(frame):
    if frame.duplicated(['dataset','image_id']).any():raise ValueError('Not a canonical slice ensemble')
    rows=[]
    for (dataset,pid),g in frame.groupby(['dataset','patient_id']):
        if g.true_label.nunique()!=1 or g.fold.nunique()!=1:raise ValueError('Patient label/fold conflict')
        p=g[PROBS].mean().to_numpy()
        rows.append(dict(dataset=dataset,patient_id=pid,true_label=int(g.true_label.iloc[0]),predicted_label=int(p.argmax()),**dict(zip(PROBS,p))))
    return pd.DataFrame(rows)
