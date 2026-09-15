import importlib.metadata as md
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
import psutil
import torch

def run():
    def command(args):
        try:
            p=subprocess.run(args,capture_output=True,text=True,timeout=30)
            return {'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
        except (OSError,subprocess.TimeoutExpired) as e: return {'unavailable':str(e)}
    record={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
        'ram_bytes':psutil.virtual_memory().total,'disk_free_bytes':shutil.disk_usage('.').free,
        'cuda_available':torch.cuda.is_available(),'torch_cuda_build':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'gpu_vram':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        'versions':{},'git_status':command(['git','status','--short']),
        'branch':command(['git','branch','--show-current']),'commit':command(['git','rev-parse','HEAD']),
        'nvidia_smi':command(['nvidia-smi']),'pip_check':command([sys.executable,'-m','pip','check'])}
    for package in ['torch','torchvision','tensorflow','keras','scikit-learn','numpy','pandas']:
        try: record['versions'][package]=md.version(package)
        except md.PackageNotFoundError: record['versions'][package]=None
    Path('environment/requirements-lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    Path('results/environment_audit.json').write_text(json.dumps(record,indent=2))
    Path('reports/ENVIRONMENT_AUDIT.md').write_text('# Environment audit\n\n'
        'Initial repository: empty, master, no commits. Revision branch created. No existing datasets, '
        'checkpoints, predictions, fold manifests, BrNet, XAI or baselines in repository. '
        'Filename search of Downloads and Documents found no datasets; this does not prove absence elsewhere. '
        'Host query: Intel UHD Graphics, driver 31.0.101.2140; reported adapter RAM 1 GiB is not dedicated CUDA VRAM. '
        'No Kaggle environment tokens or standard credential files found; no HF_TOKEN. '
        'Public Figshare/Mendeley and torchvision weights do not require HuggingFace credentials.\n\n'
        '```json\n'+json.dumps(record,indent=2)+'\n```\n\n'
        'CPU setup environment only. TensorFlow/Keras unnecessary for a new PyTorch implementation. '
        'GPU environment must be separately locked on the selected training host. Deterministic algorithms '
        'raise errors for unsupported operations; no cross-device/version bitwise reproducibility claim. '
        'Set PYTHONHASHSEED before process launch and seed DataLoader workers/generator before training. '
        'No scientific training run has started.\n')

if __name__=='__main__':run()
