import pandas as pd
import pytest
import torch
from scripts.canonical_oof import validate_manifest
from src.models.baselines import build_model,training_mode

def test_oof_manifest_contract():
    manifest=pd.DataFrame([dict(image_id='1',patient_id='p',role='test',fold=0,label='glioma')])
    f=pd.DataFrame([dict(dataset='figshare_v8',image_id='1',patient_id='p',fold=0,seed=seed,true_label=0,mask_available=True,p_class0=.8,p_class1=.1,p_class2=.1) for seed in [42,43,44]])
    assert len(validate_manifest(f,manifest))==1
    f.loc[0,'patient_id']='other'
    with pytest.raises(ValueError):validate_manifest(f,manifest)

def test_frozen_baseline_parameters_and_batchnorm():
    model=build_model('mobilenet_v2','frozen',False);training_mode(model)
    assert all(not p.requires_grad for p in model.features.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())
    assert all(not m.training for m in model.features.modules() if isinstance(m,torch.nn.BatchNorm2d))

def test_no_track_head_mixing():
    with pytest.raises(ValueError):build_model('brnet',classes=4)

def test_frozen_manifest_hashes_and_patient_leakage():
    import json,hashlib
    from pathlib import Path
    from src.data.splits import assert_disjoint
    freeze=json.loads(Path('results/folds/design_freeze.json').read_text())
    for path,expected in freeze['sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
    f=pd.read_csv('data/manifests/figshare_patient_fold_manifest.csv')
    assert f[f.subset=='test'].patient_id.value_counts().eq(1).all()
    for _,g in f.groupby('outer_fold'):assert_disjoint(*(g[g.subset==role].patient_id for role in ['train','validation','test']))
