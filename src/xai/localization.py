import numpy as np
from sklearn.metrics import average_precision_score

def localization(saliency,mask,thresholds=(.1,.2,.3,.4,.5,.6,.7,.8,.9)):
    s=np.asarray(saliency,dtype=float);m=np.asarray(mask)
    if s.shape!=m.shape or s.ndim!=2 or not np.isfinite(s).all() or (s<0).any() or not np.isin(m,[0,1]).all():raise ValueError('Invalid aligned saliency/mask')
    m=m.astype(bool)
    if not m.any():raise ValueError('Empty tumor mask')
    total=s.sum();fraction=float(m.mean());mass=float(s[m].sum()/total) if total else None
    span=np.ptp(s);normalized=(s-s.min())/span if span else np.zeros_like(s)
    def iou(binary):return float((binary&m).sum()/(binary|m).sum())
    sweep={str(t):iou(normalized>=t) for t in thresholds}
    # Evaluate tied score groups together: linear-memory exact threshold oracle.
    order=np.argsort(-s.ravel(),kind='stable');values=s.ravel()[order];truth=m.ravel()[order]
    ends=np.r_[np.flatnonzero(values[:-1]!=values[1:]),len(values)-1]
    tp=np.cumsum(truth)[ends];area=ends+1;maximum=float(np.max(tp/(area+m.sum()-tp)))
    maxima=s==s.max()
    return dict(saliency_mass=mass,tumor_area_fraction=fraction,saliency_enrichment=mass/fraction if mass is not None else None,
        pointing_game=float(m[maxima].mean()) if total else None,pointing_tie_rule='fraction of tied maxima in tumor',
        iou_threshold_sweep=sweep,maximum_iou=maximum,auprc=float(average_precision_score(m.ravel(),s.ravel())),
        constant_map=bool(span==0),zero_mass=bool(total==0))

def null_maps(image,seed=2026):
    x=np.asarray(image,dtype=float)
    if x.ndim!=2:raise ValueError('Expected grayscale 2D image')
    h,w=x.shape;y,xx=np.mgrid[:h,:w];rng=np.random.default_rng(seed)
    gy,gx=np.gradient(x)
    return dict(uniform=np.ones_like(x),random=rng.random(x.shape),center=np.exp(-((y-(h-1)/2)**2+(xx-(w-1)/2)**2)/(2*(min(h,w)/4)**2)),edge=np.hypot(gx,gy))
