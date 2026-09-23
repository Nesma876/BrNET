"""Independent ingestion audit. Never trains or selects a model."""
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import h5py
import numpy as np

def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def canonical_hash(array):
    array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder('<')))
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()

def load_mat(path):
    with h5py.File(path, 'r') as f:
        d = f['cjdata']
        # MATLAB HDF5 reverses dimension order. Apply the SAME transpose to image and mask.
        image = np.asarray(d['image']).T
        mask = np.asarray(d['tumorMask']).T
        pid = ''.join(chr(int(c)) for c in np.asarray(d['PID']).ravel()).strip('\x00')
        label = int(np.asarray(d['label']).item())
    if image.ndim != 2 or image.shape != mask.shape or not pid or label not in (1,2,3):
        raise ValueError('Invalid dimensions, patient ID, or class')
    if not np.isfinite(image).all() or not np.isin(mask, [0,1]).all() or not mask.any():
        raise ValueError('Nonfinite image or invalid/empty binary mask')
    return image, mask, pid, {1:'meningioma',2:'glioma',3:'pituitary'}[label]

def run():
    raw = Path('data/raw/figshare_v8')
    target = Path('data/interim/figshare_v8')
    target.mkdir(parents=True, exist_ok=True)
    meta = json.loads(Path('data/manifests/figshare_v8_metadata.json').read_text())
    registry, records, errors = [], [], []
    for item in meta['files']:
        path = raw / item['name']
        if not path.exists():
            errors.append({'file':str(path), 'error':'missing'}); continue
        if path.stat().st_size != item['size']:
            errors.append({'file':str(path), 'error':'size mismatch'}); continue
        digest = sha(path)
        with open(path, 'rb') as f:
            md5 = hashlib.file_digest(f, 'md5').hexdigest()
        if md5 != item['computed_md5']:
            errors.append({'file':str(path), 'error':'publisher MD5 mismatch'}); continue
        count, valid = 0, 0
        if path.suffix == '.zip':
            with zipfile.ZipFile(path) as z:
                for info in z.infolist():
                    if info.is_dir(): continue
                    count += 1
                    output = target / path.stem / info.filename
                    if not output.resolve().is_relative_to(target.resolve()):
                        raise ValueError('Unsafe archive path')
                    output.parent.mkdir(parents=True, exist_ok=True)
                    if not output.exists(): output.write_bytes(z.read(info))
                    if output.suffix.lower() != '.mat': continue
                    try:
                        image, mask, pid, label = load_mat(output)
                        records.append(dict(dataset='figshare_v8', image_id=output.stem, path=output.as_posix(),
                            patient_id=pid, label=label, height=image.shape[0], width=image.shape[1], channels=1,
                            dtype=str(image.dtype), mask_available=True, tumor_area=int(mask.sum()),
                            raw_sha256=sha(output), canonical_sha256=canonical_hash(image)))
                        valid += 1
                    except Exception as exc:
                        errors.append({'file':str(output),'error':str(exc)})
        registry.append(dict(dataset_id='figshare_v8',dataset_name=meta['title'],version=8,
            source=item['download_url'],doi=meta['doi'],download_date=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),
            archive_filename=path.name,byte_size=path.stat().st_size,archive_sha256=digest,license=meta['license']['name'],
            raw_path=path.as_posix(),n_files=count,n_valid_images=valid,notes='Publisher MD5 verified; README/cvind are auxiliary, not patient-disjoint splits'))
    for name, rows in [('data/manifests/dataset_registry.csv',registry),('data/manifests/figshare_images.csv',records)]:
        if rows:
            with open(name,'w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    groups=defaultdict(list)
    for r in records: groups[r['patient_id']].append(r)
    summary=dict(status='audited' if len(records)==3064 and not errors else 'incomplete', n_images=len(records),
        n_patients=len(groups),classes=dict(Counter(r['label'] for r in records)),
        dimensions=dict(Counter(f"{r['height']}x{r['width']}" for r in records)),
        dtypes=dict(Counter(r['dtype'] for r in records)),channels=dict(Counter(r['channels'] for r in records)),
        mask_count=sum(r['mask_available'] for r in records),images_per_patient={p:len(v) for p,v in groups.items()},
        conflicting_patient_labels=[p for p,v in groups.items() if len(set(r['label'] for r in v))>1],
        duplicate_filenames=[k for k,v in Counter(r['image_id'] for r in records).items() if v>1],errors=errors,
        aggregate_status='unavailable; authentication absent',external_status='unavailable; endpoint unresolved')
    duplicate_rows=[]
    for field in ['raw_sha256','canonical_sha256']:
        buckets=defaultdict(list)
        for r in records: buckets[r[field]].append(r)
        for key, values in buckets.items():
            for i,a in enumerate(values):
                for b in values[i+1:]:
                    duplicate_rows.append(dict(level=field,hash=key,image_a=a['image_id'],image_b=b['image_id'],patient_a=a['patient_id'],patient_b=b['patient_id']))
    out=Path('results/duplicate_audit');out.mkdir(exist_ok=True)
    with (out/'exact_duplicates.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['level','hash','image_a','image_b','patient_a','patient_b']);writer.writeheader();writer.writerows(duplicate_rows)
    summary['exact_duplicate_pairs']=len(duplicate_rows)
    Path('results/data_audit.json').write_text(json.dumps(summary,indent=2))
    Path('reports/DATASET_AUDIT.md').write_text('# Dataset integrity audit\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n\nExpected counts are not substituted for observations. Mask geometry is checked structurally; anatomical visual alignment remains a gate. No exclusions were silently made.\n')
    return summary

if __name__=='__main__': run()
