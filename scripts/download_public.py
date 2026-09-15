"""Official metadata-driven downloads; atomic completion and checksum verification."""
import concurrent.futures
import hashlib
import json
import urllib.request
import subprocess
from pathlib import Path

def download(job):
    url, dest, size, algorithm, expected = job
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.stat().st_size==size:
        with dest.open('rb') as f:
            if hashlib.file_digest(f,algorithm).hexdigest()==expected: return str(dest)
    partial=dest.with_suffix(dest.suffix+'.part')
    # Native curl is more reliable for these public redirect endpoints on this host.
    subprocess.run(['curl.exe','--silent','--show-error','--fail','--location',
        '--max-time','600','--output',str(partial),url],check=True)
    if partial.stat().st_size!=size: raise ValueError('Download size mismatch: '+str(dest))
    with partial.open('rb') as f:
        if hashlib.file_digest(f,algorithm).hexdigest()!=expected: raise ValueError('Checksum mismatch: '+str(dest))
    partial.replace(dest)
    return str(dest)

if __name__=='__main__':
    jobs=[]
    meta=json.loads(Path('data/manifests/figshare_v8_metadata.json').read_text())
    for f in meta['files']:
        jobs.append((f['download_url'],'data/raw/figshare_v8/'+f['name'],f['size'],'md5',f['computed_md5']))
    for f in json.loads(Path('data/manifests/mendeley_files.json').read_text()):
        d=f['content_details'];jobs.append((d['download_url'],'data/raw/mendeley_v2/'+f['filename'],d['size'],'sha256',d['sha256_hash']))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for value in executor.map(download,jobs): print(value,flush=True)
