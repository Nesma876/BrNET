# Environment

The local `.venv` is an isolated Python 3.12 setup/audit environment. Install from the generated `requirements-lock.txt` using `python -m pip install -r environment/requirements-lock.txt`; verify with `python -m pip check`. `initial-pip-freeze.txt` records the bundled environment before this work and is not the revision lock.

The lock pins transitive package versions for Windows CPU setup. It is not a GPU lock or a wheel-hash lock. Generate a separate validated CUDA lock on the future training host; preserve both. TensorFlow, transformers and timm are unnecessary for the current torchvision model choices. XAI/plotting/perturbation dependencies will be added and locked when those implementations are built, before protocol freeze.

Set `$env:PYTHONHASHSEED='42'` before launching a seeded process. `seed_everything` enables strict deterministic PyTorch algorithms and seeds Python, NumPy and CUDA. Future DataLoaders must use explicitly seeded generators/workers; unsupported nondeterministic operations must fail rather than silently fall back. Determinism across different hardware or package versions is not assumed.
