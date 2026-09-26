import numpy as np
from sklearn.metrics import confusion_matrix,roc_auc_score
from src.evaluation.calibration import validate,calibration

def metrics(y,p,predicted=None):
    y,p=validate(y,p);k=p.shape[1];pred=p.argmax(1) if predicted is None else np.asarray(predicted)
    if pred.shape!=y.shape or not np.isin(pred,np.arange(k)).all():raise ValueError('Invalid predicted labels')
    cm=confusion_matrix(y,pred,labels=np.arange(k));total=cm.sum();per=[]
    def ratio(n,d):return float(n/d) if d else None
    for c in range(k):
        tp=int(cm[c,c]);fn=int(cm[c].sum()-tp);fp=int(cm[:,c].sum()-tp);tn=int(total-tp-fn-fp)
        truth=y==c
        auc=float(roc_auc_score(truth,p[:,c])) if truth.any() and (~truth).any() else None
        per.append(dict(class_index=c,sensitivity=ratio(tp,tp+fn),specificity=ratio(tn,tn+fp),precision_ppv=ratio(tp,tp+fp),npv=ratio(tn,tn+fn),f1=ratio(2*tp,2*tp+fp+fn),auc=auc,
            auc_undefined_reason=None if auc is not None else 'positive or negative class absent'))
    def mean_complete(field):
        vals=[r[field] for r in per];return float(np.mean(vals)) if all(v is not None for v in vals) else None
    return dict(n=len(y),accuracy=float((y==pred).mean()),balanced_accuracy=mean_complete('sensitivity'),macro_f1=mean_complete('f1'),ovr_macro_auc=mean_complete('auc'),
        confusion_matrix=cm.tolist(),per_class=per,**calibration(y,p))

def patient_metrics(frame):
    if frame.empty or frame.patient_id.isna().any() or frame.duplicated(['dataset','patient_id']).any():raise ValueError('Requires one row per patient')
    return metrics(frame.true_label,frame[['p_class0','p_class1','p_class2']],frame.predicted_label)
