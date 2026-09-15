import csv
import io
import json
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from scripts.audit_data import sha, canonical_hash

def run():
    path=Path('data/raw/mendeley_v2/Brain Cancer MRI Dataset.zip')
    metadata=json.loads(Path('data/manifests/mendeley_files.json').read_text())[0]['content_details']
    if sha(path)!=metadata['sha256_hash']: raise ValueError('Mendeley checksum mismatch')
    records,errors,other=[],[],[]
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.is_dir():continue
            if Path(info.filename).suffix.lower() not in ['.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp']:
                other.append(info.filename);continue
            try:
                payload=z.read(info)
                with Image.open(io.BytesIO(payload)) as im:
                    im.load();im=ImageOps.exif_transpose(im)
                    a=np.array(im)
                parts=info.filename.lower()
                labels=[c for c in ['glioma','healthy','meningioma','pituitary'] if c in parts]
                if len(labels)!=1: raise ValueError('Unresolved label mapping')
                import hashlib
                records.append(dict(image_id=info.filename,label=labels[0],height=a.shape[0],width=a.shape[1],
                    channels=1 if a.ndim==2 else a.shape[2],dtype=str(a.dtype),raw_sha256=hashlib.sha256(payload).hexdigest(),
                    canonical_sha256=canonical_hash(a),patient_id=None,mask_available=False))
            except Exception as e: errors.append({'file':info.filename,'error':str(e)})
    with open('data/manifests/external_images.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    summary=dict(status='audited',n_valid_images=len(records),classes=dict(Counter(r['label'] for r in records)),
        errors=errors,non_image_files=other,patient_ids='not provided in image paths; not inferred',mask_count=0,
        duplicate_filenames=[k for k,v in Counter(Path(r['image_id']).name for r in records).items() if v>1],
        dimensions=dict(Counter(f"{r['height']}x{r['width']}" for r in records)),
        channels=dict(Counter(r['channels'] for r in records)),dtypes=dict(Counter(r['dtype'] for r in records)),
        archive_sha256=sha(path),advertised_count=1200,prior_user_observed_count=1137,
        independence='unverified; cross-source duplicate audit pending')
    Path('results/external_data_audit.json').write_text(json.dumps(summary,indent=2))
    with open('data/manifests/dataset_registry.csv',newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    rows=[r for r in rows if r['dataset_id']!='mendeley_v2']
    from datetime import datetime,timezone
    rows.append(dict(dataset_id='mendeley_v2',dataset_name='Multi-Class Brain Tumor MRI Dataset',version=2,
        source=metadata['download_url'],doi='10.17632/82mtzd8x72.2',download_date=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),
        archive_filename=path.name,byte_size=path.stat().st_size,archive_sha256=sha(path),license='CC BY 4.0',raw_path=path.as_posix(),
        n_files=len(records)+len(errors)+len(other),n_valid_images=len(records),notes='No independence claim; 1200 advertised vs archive audit'))
    with open('data/manifests/dataset_registry.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    with open('reports/DATASET_AUDIT.md','a',encoding='utf-8') as f:
        f.write('\n## External-source V2 audit\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
    combined=json.loads(Path('results/data_audit.json').read_text());combined['external']=summary
    combined['external_status']='archive audited; provenance unresolved'
    Path('results/data_audit.json').write_text(json.dumps(combined,indent=2))

if __name__=='__main__':run()
