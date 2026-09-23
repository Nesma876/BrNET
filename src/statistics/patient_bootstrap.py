"""Each sampled row is one patient, never one slice. Undefined replicates are recorded."""
import numpy as np
from src.evaluation.patient_metrics import metrics

def bootstrap_patients(frame,draws=10000,seed=2026,stratified=True):
    if draws<1 or frame.empty or frame.patient_id.isna().any() or frame.duplicated(['dataset','patient_id']).any():raise ValueError('Requires one complete row per patient')
    y=frame.true_label.to_numpy();p=frame[['p_class0','p_class1','p_class2']].to_numpy()
    predicted=frame.predicted_label.to_numpy() if 'predicted_label' in frame else p.argmax(1)
    point=metrics(y,p,predicted);keys=['accuracy','balanced_accuracy','macro_f1','ovr_macro_auc','brier_multiclass_sum','ece']
    def flatten(result):
        values={key:result[key] for key in keys}
        for row in result['per_class']:
            for key in ['sensitivity','specificity','precision_ppv','npv','f1','auc']:
                values[f"class{row['class_index']}_{key}"]=row[key]
        return values
    all_keys=list(flatten(point))
    rng=np.random.default_rng(seed);values={k:[] for k in all_keys};groups=[np.flatnonzero(y==c) for c in np.unique(y)]
    for _ in range(draws):
        idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]) if stratified else rng.integers(0,len(y),len(y))
        result=flatten(metrics(y[idx],p[idx],predicted[idx]))
        for key in all_keys:
            if result[key] is not None:values[key].append(result[key])
    return dict(point=point,unit='patient',draws=draws,seed=seed,stratified=stratified,interval='percentile 95%',
        intervals={k:dict(low=float(np.quantile(v,.025)) if v else None,high=float(np.quantile(v,.975)) if v else None,valid_draws=len(v),undefined_draws=draws-len(v)) for k,v in values.items()})
