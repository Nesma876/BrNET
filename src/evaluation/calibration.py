import numpy as np

def validate(y,p):
    y=np.asarray(y);p=np.asarray(p,dtype=float)
    if p.ndim!=2 or len(y)!=len(p) or not len(y):raise ValueError('Invalid/empty prediction shape')
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1,atol=1e-7):raise ValueError('Invalid probabilities')
    if not np.isin(y,np.arange(p.shape[1])).all():raise ValueError('Invalid class index')
    return y.astype(int),p

def calibration(y,p,bins=15):
    y,p=validate(y,p)
    if bins<1:raise ValueError('bins must be positive')
    brier=float(np.mean(np.sum((p-np.eye(p.shape[1])[y])**2,axis=1)))
    confidence=p.max(1);correct=p.argmax(1)==y
    bucket=np.minimum((confidence*bins).astype(int),bins-1)
    ece=sum(np.mean(bucket==i)*abs(float(correct[bucket==i].mean())-float(confidence[bucket==i].mean())) for i in range(bins) if (bucket==i).any())
    return {'brier_multiclass_sum':brier,'ece':float(ece),'ece_bins':bins}
