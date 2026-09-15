import csv
import json
from collections import defaultdict
from pathlib import Path

def run():
    rows=[]
    for dataset,path in [('figshare_v8','data/manifests/figshare_images.csv'),('mendeley_v2','data/manifests/external_images.csv')]:
        with open(path,newline='',encoding='utf-8') as f:
            for r in csv.DictReader(f):r['dataset']=dataset;rows.append(r)
    pairs=[]
    for field in ['raw_sha256','canonical_sha256']:
        buckets=defaultdict(list)
        for r in rows:buckets[r[field]].append(r)
        for key,group in buckets.items():
            for i,a in enumerate(group):
                for b in group[i+1:]:pairs.append(dict(level=field,hash=key,dataset_a=a['dataset'],image_a=a['image_id'],patient_a=a['patient_id'],dataset_b=b['dataset'],image_b=b['image_id'],patient_b=b['patient_id']))
    out=Path('results/duplicate_audit')
    with (out/'exact_duplicates.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['level','hash','dataset_a','image_a','patient_a','dataset_b','image_b','patient_b']);writer.writeheader();writer.writerows(pairs)
    summary=dict(status='partial_exact_only',images=len(rows),pair_rows=len(pairs),
        cross_source_pairs=sum(p['dataset_a']!=p['dataset_b'] for p in pairs),
        caveat='Native decoded hashes preserve dtype/channels; canonical cross-format grayscale, perceptual and embedding review remain pending. No independence conclusion.',
        perceptual_status='pending',embedding_status='pending',manual_review_status='pending',aggregate_status='unavailable')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':run()
