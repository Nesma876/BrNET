"""Validate complete manifest-linked seed predictions and archive immutably. No model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from src.evaluation.predictions import ensemble
from src.evaluation.patient_aggregation import aggregate_patients

def validate_manifest(frame,manifest):
    if set(frame.dataset)!={'figshare_v8'}:raise ValueError('TRACK_A requires Figshare only')
    held=manifest[manifest.role=='test'].copy()
    if set(frame.image_id)!=set(held.image_id):raise ValueError('OOF image coverage mismatch')
    mapping={'glioma':0,'meningioma':1,'pituitary':2}
    expected=held.set_index('image_id')
    for r in frame.itertuples():
        x=expected.loc[r.image_id]
        if r.fold!=x.fold or r.patient_id!=x.patient_id or r.true_label!=mapping[x.label]:raise ValueError('OOF manifest mismatch')
    return ensemble(frame)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output-dir',required=True);args=ap.parse_args()
    out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Output directory must be new or empty')
    f=pd.read_parquet(args.input)
    for field in ['track','model','git_commit','environment_sha256']:
        if field not in f or f[field].isna().any() or f[field].nunique()!=1:raise ValueError('Missing or mixed provenance: '+field)
    if set(f.track)!={'TRACK_A_PRIMARY'}:raise ValueError('Wrong track')
    for col in ['checkpoint','git_commit','protocol_sha256','manifest_sha256','environment_sha256','checkpoint_sha256']:
        if col not in f or f[col].isna().any() or f[col].eq('').any():raise ValueError('Missing provenance '+col)
    for field,path in [('protocol_sha256','configs/track_a_primary.yaml'),('manifest_sha256','data/manifests/figshare_image_fold_manifest.csv')]:
        if set(f[field])!={hashlib.sha256(Path(path).read_bytes()).hexdigest()}:raise ValueError('Frozen provenance mismatch: '+field)
    for checkpoint,g in f.groupby('checkpoint'):
        path=Path(checkpoint)
        if not path.is_file() or set(g.checkpoint_sha256)!={hashlib.sha256(path.read_bytes()).hexdigest()}:raise ValueError('Checkpoint checksum mismatch')
    for _,g in f.groupby(['fold','seed']):
        if g.checkpoint.nunique()!=1:raise ValueError('Multiple checkpoints within fold/seed')
    manifest=pd.read_csv('data/manifests/figshare_image_fold_manifest.csv',dtype={'patient_id':str,'image_id':str})
    f['image_id']=f.image_id.astype(str);f['patient_id']=f.patient_id.astype(str)
    e=validate_manifest(f,manifest);patients=aggregate_patients(e)
    out.mkdir(parents=True,exist_ok=True)
    f.to_parquet(out/'figshare_oof_by_seed.parquet',index=False);e.to_parquet(out/'figshare_oof_ensemble.parquet',index=False)
    patients.to_parquet(out/'figshare_patient_ensemble.parquet',index=False)
    reconciliation=dict(total=len(e),correct=int(e.correct.sum()),accuracy=float(e.correct.mean()),with_masks=int(e.mask_available.sum()),correct_with_masks=int((e.correct&e.mask_available).sum()),class_counts={int(c):dict(total=len(g),correct=int(g.correct.sum()),with_masks=int(g.mask_available.sum()),correct_with_masks=int((g.correct&g.mask_available).sum())) for c,g in e.groupby('true_label')})
    (out/'reconciliation.json').write_text(json.dumps(reconciliation,indent=2))
    (out/'artifact_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},indent=2))

if __name__=='__main__':main()
