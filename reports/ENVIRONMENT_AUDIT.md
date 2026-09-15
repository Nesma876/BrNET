# Environment audit

Initial repository: empty, master, no commits. Revision branch created. No existing datasets, checkpoints, predictions, fold manifests, BrNet, XAI or baselines in repository. Filename search of Downloads and Documents found no datasets; this does not prove absence elsewhere. Host query: Intel UHD Graphics, driver 31.0.101.2140; reported adapter RAM 1 GiB is not dedicated CUDA VRAM. No Kaggle environment tokens or standard credential files found; no HF_TOKEN. Public Figshare/Mendeley and torchvision weights do not require HuggingFace credentials.

```json
{
  "python": "3.12.14 (main, Aug 25 2026, 14:01:42) [MSC v.1944 64 bit (AMD64)]",
  "executable": "C:\\Users\\ISEN\\OneDrive - yncr\u00e9a\\Documents\\ChatGPT\\BrNET\\.venv\\Scripts\\python.exe",
  "platform": "Windows-11-10.0.26200-SP0",
  "ram_bytes": 16942211072,
  "disk_free_bytes": 67704524800,
  "cuda_available": false,
  "torch_cuda_build": null,
  "gpu": null,
  "gpu_vram": null,
  "versions": {
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "tensorflow": null,
    "keras": null,
    "scikit-learn": "1.6.1",
    "numpy": "2.2.3",
    "pandas": "2.2.3"
  },
  "git_status": {
    "exit_code": 0,
    "stdout": "?? .gitignore\n?? README.md\n?? configs/\n?? data/\n?? environment/\n?? pytest.ini\n?? reports/\n?? results/\n?? scripts/\n?? src/\n?? tests/\n",
    "stderr": ""
  },
  "branch": {
    "exit_code": 0,
    "stdout": "revision/major-rerun-2026\n",
    "stderr": ""
  },
  "commit": {
    "exit_code": 128,
    "stdout": "HEAD\n",
    "stderr": "fatal: ambiguous argument 'HEAD': unknown revision or path not in the working tree.\nUse '--' to separate paths from revisions, like this:\n'git <command> [<revision>...] -- [<file>...]'\n"
  },
  "nvidia_smi": {
    "unavailable": "[WinError 2] Le fichier sp\u00e9cifi\u00e9 est introuvable"
  },
  "pip_check": {
    "exit_code": 0,
    "stdout": "No broken requirements found.\n",
    "stderr": ""
  }
}
```

CPU setup environment only. TensorFlow/Keras unnecessary for a new PyTorch implementation. GPU environment must be separately locked on the selected training host. Deterministic algorithms raise errors for unsupported operations; no cross-device/version bitwise reproducibility claim. Set PYTHONHASHSEED before process launch and seed DataLoader workers/generator before training. No scientific training run has started.
